# Upstream Generator — first-principles half of the binary TDB pipeline

`sqs2tdb_pipeline.py` (one directory up) is the **downstream** consumer: it
assumes every `sqs_lev=*` directory already contains `energy`,
`str_relax.out`, and optionally `svib_ht`, then runs `sqs2tdb -fit`. This
`upstream/` package is the **producer** that creates those inputs:

```
run_upstream.py
  ├─ sqsgen     generate SQS (sqs2tdb -cp [-l=*_small]); randomspin (DLM);
  │             SIGMA lev=3 → lev=0 ±spin endmember conversion
  ├─ potcar     ENMAX from POTCARs → ENCUT/KPPRA sweep grids
  ├─ vaspwrap   vasp.wrap (INCAR) for static / relax / phonon modes
  ├─ converge   ENCUT + KPPRA convergence (static, 1 meV/atom selection)
  ├─ relax      robustrelax_vasp  (normal)  or  -id  (infdet)
  └─ phonon     fitfc workflow + DLM spin-suffix fixup  → svib_ht
                                    │
                                    ▼
                      sqs2tdb_pipeline.py  (Stage 1/2 fit)
```

It **drives VASP on the node** (generate-and-submit/poll): it calls
`runstruct_vasp`, `pollmach`, `robustrelax_vasp`, and `fitfc` directly. Run it
inside a PBS job on a compute node with ATAT + VASP on `PATH`
(`submit_upstream_template.pbs`).

> **Advisor-review upgrades (2026-07, see `../../REVIEW_RESPONSE.md`):**
> spin polarization is auto-on for magnetic 3d elements (F1); ENCUT/KPPRA
> are converged once per phase and reused for all its SQS (F4); ISIF=3
> relaxations get a Pulay-safe ENCUT floor of 1.3×max(ENMAX) (F2); every
> relaxation is checked for lattice drift and flagged with
> `checkrelax.out`/`relaxaway.flag` at 0.1 (F3). All decisions land in
> the manifest and live log.

## Procedure

For each SQS (and each SIGMA endmember):

1. **Static runs**, starting at `lev=1` for each binary single-sublattice
   phase.
2. **Convergence testing** — the probe protocol (2026-07-17 default,
   `--convergence-scope system`): randomly pick ONE single-sublattice
   phase (seeded, `--probe-seed`), sweep one randomly chosen SQS on
   **each element-rich side** (>50% of that element; endmembers count;
   lev irrelevant), take the elementwise **max** over the probes, fold
   in the Pulay floor — and use that **single (ENCUT, KPPRA) for every
   energy, relaxation, inflection-detection and phonon run** in the
   job. Rich-side sampling matters because basis demand is
   element-dependent (Cr_pv ≫ Co); the max is conservative for the
   other side. Picks + per-probe results land in the manifest.
   `MAX_ENMAX` is the largest `ENMAX` across the POTCARs.
   - **KPPRA** swept `4000 … 10000` (step 1000, extends to 20000 if
     needed) at a fixed `ENCUT = 1.125 × MAX_ENMAX`; the converged
     KPPRA is frozen.
   - **ENCUT** starts at `1.00 × … 1.25 × MAX_ENMAX` (5 points) at the
     converged KPPRA and **extends upward with no convergence ceiling**
     (runaway guard at 3 × ENMAX) until the criterion is met — the
     2026-07-16 e2e sweep was still moving 0.4–1.4 meV/atom at the old
     1.25× ceiling.
   - *Converged* (successive-difference + confirmation, 2026-07-16): the
     smallest setting whose energy changed by less than `--tol-ev`
     (default **0.1 meV/atom**) from the PREVIOUS point AND whose NEXT
     point — the deliberately "unneeded" confirmation run — also agrees
     to within tolerance. Energy *differences* converge faster than
     totals (VASP wiki), so this is conservative for mixing energies.
   - *Noise-floor fallback* (2026-07-17, from the real 31-point FCC-Co
     sweep): sweep statics run **PREC=Accurate + LREAL=.FALSE.** so the
     point-to-point noise (~0.3–0.5 meV/atom at PREC=Normal) stays
     below the tolerance; if the pointwise rule still finds nothing,
     the first **4 consecutive points spanning < 0.5 meV/atom**
     terminate the sweep (`--plateau-band`; the table reports which
     rule fired). Without this, that sweep wandered to 760 eV and
     "converged" on a noise coincidence; the plateau rule stops it at
     488 eV and picks 437.
   - `ALGO = All` by default.
3. **Relaxation** — `--relax-method infdet` is the DEFAULT (user
   decision 2026-07-15): `robustrelax_vasp -mk` then
   `robustrelax_vasp -id -c 0.05 <launcher>` — inflection detection
   with the 5% strain cutoff it requires to engage (override the
   cutoff via `--relax-opts "-c 0.10"`). `normal` (plain robustrelax)
   and `runstruct` remain available. Uses the converged KPPRA and the
   Pulay-floored ENCUT; the result is VALIDATED (parseable cell +
   atoms) before the pipeline moves on, and the energy is taken from
   infdet's `energy_end` when the plain `energy` file is empty.
   ISIF=3 / IBRION=2 / ISMEAR=1 / NSW=100 per the production INCAR.
4. **Phonons** — the `fitfc` workflow, verified against the
   `$atatdir/src/fitfc.c++` source. Default recipe is the sqs2tdb
   vibrational (harmonic) one:

   ```
   fitfc -si=str_relax.out -ernn=4 -ns=1 -dr=0.04 -nrr   # ONE call
   pollmach runstruct_vasp -lu -w vaspf.wrap <launcher>  # force.out per p*
   [DLM fixup]
   fitfc -f -frnn=2 -si=str_relax.out                    # → vol_0/svib_ht
   robustrelax_vasp -vib                                 # paper step (iv):
                                                         # svib_ht where
                                                         # sqs2tdb reads it
                                                         # (cp fallback kept)
   ```

   These are the PUBLISHED values (van de Walle et al., Calphad 58
   (2017) 70, §3.3; "-ernn/-frnn may need to be adjusted depending on
   the alloy system" — override with `--fitfc-ernn/--fitfc-frnn/`
   `--fitfc-dr`). `vaspf.wrap` (the `-mk` name) is REwritten by the
   pipeline after generation so its MAGMOM/NCORE/KPAR match the
   perturbation SUPERCELL. **Scope: endmembers only by default**
   (`--phonon-scope`, per the paper) — sqs2tdb -fit fits svib linearly
   across composition; use `all` for nonideal vibrational entropy.

   With `-nrr` (default at `ns=1`) the volume dir is the already-relaxed
   input, so generation is a single invocation. A quasiharmonic strain
   series (`ns>1` via `phonon.run_fitfc`) follows fitfc's two-invocation
   contract instead: generate → relax each `vol_*` ions-only at fixed
   strained cell (per-vol `ISIF=2` wrap, removed afterwards) → re-run
   fitfc with the *same* options → force runs → fit.

   **Unstable modes** (`fitfc -f` prints `Unstable modes found.` and
   aborts *before* writing `svib_ht` unless `-fn`/`-rl` is set) are
   handled by policy — `--fitfc-on-unstable` / PBS `FITFC_ON_UNSTABLE`:

   | Policy | What happens |
   |---|---|
   | `mark` (default) | Record `unstable_modes.log`; leave the SQS **energy-only**. Honest: svib from a fit that drops imaginary branches is biased, and downstream (`sqs2tdb -fit`, the Stage-2 svib gates) handles a missing `svib_ht` cleanly. |
   | `escalate` | Regenerate perturbations at a **1.5× larger displacement radius** (`-ernn` 2→3; `--fitfc-escalate-ernn` to override) and refit — rules out the finite-supercell artifact, the most common *fixable* cause. Only the new `p*` dirs get VASP force runs; the old equations stay in the refit. Resolved → escalated `svib_ht` promoted. Persists → likely **genuine dynamical instability**: SQS stays energy-only and the marker names the manual options (tighter re-relax, `fitfc -rl`, `-fu`/`-gu` mode-following). |
   | `force` | Retry once with fitfc's `-fn`. The resulting `svib_ht` **omits** the unstable branches (a lower bound); provenance recorded. |

   `--fitfc-rl <len>` (PBS `FITFC_RL`) passes fitfc's robust-length
   soft-mode treatment (beta) on the first attempt, which also prevents
   the abort. Every instability leaves `<sqs>/unstable_modes.log`
   (evidence + disposition), a `STAGE 3/3 UNSTABLE MODES` stamp in the
   live log, and `svib_ht_present` / `unstable_modes` fields in the
   manifest. Stale fit outputs are always cleared before a (re)fit, so
   an aborted refit can never promote an old `svib_ht` as fresh.

   Recommended for SIGMA endmembers (whose `svib_ht` you need for
   normalization): `FITFC_ON_UNSTABLE=escalate`. If you want
   temperature-dependent `fvib`, drop a `Trange.in` (e.g. `2000 21`) in
   the **phase directory** — fitfc reads `../Trange.in` relative to each
   SQS dir; `svib_ht` itself is the T-independent high-T limit.

Spin-polarized (non-DLM) runs write an explicit
`MAGMOM = <natoms>*3` line (uniform 3 μB initialization, the
production-INCAR convention; `--magmom-init` to change) plus the
magnetic-mixing keys — VASP's "did not specify MAGMOM" warning means a
wrap predates this and should be regenerated. fitfc force runs use a
separate `fvasp.wrap` (PREC=Accurate, ISIF=2 frozen statics) selected
with `runstruct_vasp -w fvasp.wrap`, so they never clobber the relax
wrap.

With `--dlm`, random spins are applied after SQS generation: `randomspin` is
run inside each `*_small` directory so `str.out` gains `+2` / `-2` tags, which
become per-site `MAGMOM` (ISPIN=2). After the phonon force runs, the
spin-suffix **fixup** strips every `±N` tag from `str_relax.out` /
`str_unpert.out` (top level and recursively) so `fitfc -f` can parse plain
element symbols — the element-agnostic generalisation of the
`sed -e s/Co+2/Co/g … ; foreachfile -d 2 …` recipe.

## Caveats handled

1. **`HCP_A3_small`, `FCC_A1_small`, `BCC_A2_small`** — the single-sublattice
   systems are copied from `--template-root` before generation
   (`sqsgen.copy_small_systems`), and `randomspin` runs inside them for DLM.
2. **`SIGMA_D8B`** — endmembers only. For DLM we generate at `lev=3`
   (randomises each site among two species) and convert each to a `lev=0`
   endmember where one element fills the sublattice but its equivalent sites
   are split into a spin-up (`_A`, `+2`) and spin-down (`_B`, `-2`)
   pseudo-species — `sqsgen.sigma_lev3_to_lev0_dlm`. **This is the piece that
   was previously "not implemented."**
3. **fitfc DLM fixup** — `phonon.dlm_fixup` performs the spin-suffix stripping
   for any element pair.

> **Note on `--sqs-level`:** the spec flags that the installed `sqs2tdb` may or
> may not honour generation of *only* a specified level (it has historically
> generated *up to* that level). `--sqs-level N` passes `-lev=N`; the
> orchestrator then discovers whatever `lev=*` directories actually appear, so
> it is correct either way — but verify the generated levels in the log.

## Usage

```bash
python3 run_upstream.py \
  --element1 Co --element2 Cr \
  --work-root /scratch/CoCr_upstream \
  --potcars $VASP_PP/Co/POTCAR,$VASP_PP/Cr/POTCAR \
  ##internally this is --potcars /home1/zwu6/vasp/POTPAW_PBE.64/Cr_pv,/home1/zwu6/vasp/POTPAW_PBE.64/Co
  --template-root /home/you/atat_small_templates \
  --phases FCC_A1,BCC_A2,HCP_A3,SIGMA_D8B \
  --relax-method normal \
  --tol-ev 0.001
```

Add `--dlm` for a disordered-local-moment run, `--relax-method infdet` for
inflection-detection relaxation, `--skip-phonon` for an energy-only pass.
Phonon-stage knobs: `--fitfc-on-unstable {mark,escalate,force}` (unstable-mode
policy, see Procedure step 4), `--fitfc-escalate-ernn <x>` (radius for the
escalate retry), `--fitfc-rl <len>` (fitfc robust soft-mode treatment).

The resulting tree is then fed straight into the downstream pipeline:

```bash
python3 ../select_endmembers.py --element1 Co --element2 Cr \
  --data-roots /scratch/CoCr_upstream --out endmembers.yaml
python3 ../sqs2tdb_pipeline.py --endmembers-yaml endmembers.yaml \
  --data-roots /scratch/CoCr_upstream ...
```

## PBS fan-out mode (`--submit pbs`)

By default (`--submit node`) the whole pipeline runs inside ONE PBS job and
every VASP execution shares that job's allocation serially. `--submit pbs`
inverts this: the orchestrator runs on a **front end** (pfe) doing only
bookkeeping + light ATAT glue (sqs2tdb, fitfc, checkrelax), and every long
VASP execution is submitted as its **own right-sized qsub job** via
`pbsjobs.Broker`. This is the paper's own cluster pattern
(`foreachfile wait sbatch jobfile.in`, Calphad 58 (2017) 70 §3.2) plus
sizing, throttling, retries and restart-safety.

Job shapes produced:

| Work | Shape | Parallelism |
|---|---|---|
| convergence probes (system scope) | 1 job per probe SQS (`--probe-worker` submode) | 2 rich-side probes run simultaneously |
| relaxation (runstruct / robustrelax `-id`) | 1 single job per SQS | all SQS of a phase concurrently (needs `--preset-*`) |
| fitfc force runs | **1 PBS job array per SQS** — one array element per perturbation dir | all perturbation statics wall-parallel for the same SBUs |
| fitfc strain runs (`ns>1`) | loop job over `vol_*` dirs | per SQS |

Resource sizing lives in `pbsjobs.SIZING` (ncpus/walltime per atom-count
tier and job kind), and the trailing `mpiexec -n K` of every command is
**rewritten to the job's ncpus**, so the select-line/rank-count mismatch
class of failure cannot occur in this mode. In-flight jobs are capped by
`--job-max-inflight` (default 16); jobs that leave the queue without their
output files are resubmitted up to `--job-retries` times. State is kept in
`.qjob_<tag>` markers + the work tree itself, so killing and rerunning the
orchestrator **adopts** still-running jobs instead of resubmitting.

Launch with the template (edit the USER CONFIG block first):

```bash
# on pfe, NOT inside a job:
bash submit_orchestrator_template.sh
tail -f <WORK_ROOT>/upstream_live.log     # pipeline progress
qstat -u $USER                            # the job fan-out
```

The template generates `<WORK_ROOT>/job_env.sh` (modules/venv/PATH) that
every submitted job sources — pass your own with `--job-env` if launching
`run_upstream.py --submit pbs` by hand. Requirements enforced at startup:
`--job-env` is mandatory, and either `--convergence-scope system` (default)
or explicit `--preset-encut/--preset-kppra`, because per-SQS parallelism is
only unlocked once a single global (ENCUT, KPPRA) exists. Sanity-check the
rendered scripts without submitting via `--job-dry-run`; disable arrays on
queues that reject `#PBS -J` with `--no-job-arrays` (falls back to a loop
job). **This mode has not yet been exercised on real NAS hardware** — do a
`--job-dry-run` inspection of the `qjob_*.pbs` scripts first, and keep the
single-job `--submit node` path as the fallback.

## What is and isn't tested

The pure logic — ENMAX parsing, ENCUT/KPPRA grids, `vasp.wrap` generation,
**1 meV/atom convergence selection**, SIGMA lev=3→lev=0 spin conversion, and
the DLM fixup — is covered by `tests/` (`pytest`). The VASP-driving glue
(`runner`, `converge.run_static_point`, `relax`, `phonon.run_fitfc`) can only
be exercised on a real ATAT + VASP node.

## Monitoring a running job (live logs)

PBS only delivers `#PBS -o` output **after** the job ends, so the live
view comes from files the pipeline writes as it goes:

```bash
# Step-level index — which phase / SQS / stage is running right now.
# Every line timestamped; written by run_upstream.py itself:
tail -f <WORK_ROOT>/upstream_live.log

# Same content, captured by the PBS wrapper in the submit dir:
tail -f upstream_live_<AB>_<timestamp>.log
```

Stage markers look like:

```
[2026-07-09 18:20:11] [sqsdb_lev=2_a_Co=0.5,...] STAGE 1/3 convergence sweep starting ...
[2026-07-09 18:40:03] [sqsdb_lev=2_a_Co=0.5,...] STAGE 2/3 relaxation starting (method=runstruct; ...)
[2026-07-09 19:55:47] [sqsdb_lev=2_a_Co=0.5,...] STAGE 2/3 relaxation done (str_relax.out present: True)
[2026-07-09 19:55:48] [sqsdb_lev=2_a_Co=0.5,...] STAGE 3/3 fitfc phonons starting
```

Per-command detail (full VASP/ATAT output) streams live into per-step
logs under `WORK_ROOT` — each STAGE marker names the one to watch:

| Log | What's in it |
|---|---|
| `sqs2tdb_cp_<PHASE>.log`, `.2.log` | SQS generation, pass 1 / pass 2 |
| `<sqs>/convergence/*/vasp.log` | each ENCUT/KPPRA sweep point |
| `<sqs>/runstruct.log` | `pollmach runstruct_vasp` relaxation |
| `<sqs>/robustrelax_mk.log` | robustrelax input generation (`-mk`) |
| `<sqs>/robustrelax_{normal,infdet}.log` | robustrelax / infdet relaxation |
| `<sqs>/fitfc_gen.log` | fitfc perturbation generation |
| `<sqs>/fitfc_strain_runs.log` | `vol_*` relaxations (quasiharmonic `ns>1` only) |
| `<sqs>/fitfc_force_runs.log` | `pollmach runstruct_vasp` force runs in `vol_*/p*` |
| `<sqs>/fitfc_fit.log` | the `fitfc -f` fit itself |
| `<sqs>/fitfc_{gen,force_runs,fit}_escalated.log` | the `escalate` retry stages |
| `<sqs>/fitfc_fit_forced.log` | the `-fn` retry (`force` policy) |
| `<sqs>/unstable_modes.log` | unstable-mode evidence + disposition (see step 4) |

Quick health checks while it runs:

```bash
grep STAGE <WORK_ROOT>/upstream_live.log | tail -20   # recent stage history
find <WORK_ROOT> -name str_relax.out | wc -l          # relaxations finished
find <WORK_ROOT> -name energy | wc -l                 # energies produced
find <WORK_ROOT> -maxdepth 3 -name svib_ht | wc -l    # phonon fits promoted
find <WORK_ROOT> -name unstable_modes.log             # SQS flagged unstable
```
