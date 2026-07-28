# Upstream Pipeline — Hand-Review Guide

One section per module: what it does, the algorithm step by step, the
constants that tune behavior, and **review checkpoints** — the places
where a logic edit is most likely to be wanted and the invariants an
edit must not break. Modules are ordered by data flow; reviewing top to
bottom follows one SQS through the pipeline.

```
                 ┌──────────── run_upstream.py (orchestrator) ────────────┐
                 │  CLI → config → per-phase loop → manifest              │
                 └────────────────────────────────────────────────────────┘
 phases.py      sqsgen.py     potcar.py     converge.py    relax.py    phonon.py   refine.py
 (phase DB)  →  (sqs2tdb   →  (ENCUT/    →  (sweeps,    →  (robust- →  (fitfc   →  (adaptive
                two-pass)     KPPRA        early stop,    relax -id    force        mesh +
                              grids)       caching)       + wraps)     runs)        svib)
                     │             │            │             │            │
                     └──────┬──────┴────────────┴──────┬──────┴────────────┘
                        strfile.py                 vaspwrap.py
                        (str.out parsing,          (INCAR/wrap writer,
                         drift metric,              spin, NCORE/KPAR)
                         stub detection)
                     execution layer:  runner.py ──(set_backend)── pbsjobs.py
                     diagnostics:      vasp_triage.py
```

Line counts (2026-07-27): run_upstream 1245, vasp_triage 578, phonon
575, refine 426, vaspwrap 374, converge 371, sqsgen 322, pbsjobs 306,
relax 268, strfile 239, runner 226, potcar 194, phases 133.

---

## 1. `phases.py` — phase/lattice knowledge base (133 lines)

**Purpose.** Static tables describing each CALPHAD phase and the two
dataclasses configuring DLM runs. No I/O, no algorithms.

**Contents.**
- `PHASE_SITE` — sublattice letter per phase (FCC/BCC `a`, HCP `c`);
  `PHASE_MULT` — site multiplicities (SIGMA 10/4/16).
- `SINGLE_SUBLATTICE_PHASES`, `ENDMEMBER_ONLY_PHASES` (SIGMA),
  `ALL_PHASES`, `SMALL_SYSTEM` (phase → `<PHASE>_small` sqsdb target).
- `DLM_SPIN_UP/DOWN` (`"+2"`/`"-2"` pseudo-species suffixes).
- `DLMConfig` (enabled, subatom map, moment), `SigmaDLMSpec`.
- Helpers `is_single_sublattice`, `site_for`.

**Review checkpoints.**
- Adding a phase = add to all four tables + confirm the lattice exists
  in `$atatdir/data/sqsdb` (the HCP incident) — nothing else needs
  editing anywhere.
- `SMALL_SYSTEM` values must exactly match sqsdb directory names.
- HCP's sublattice letter is `c`, not `a` — anything parsing dir names
  must go through the generic regexes, never assume `a_`.

---

## 2. `sqsgen.py` — SQS generation via sqs2tdb (322 lines)

**Purpose.** Drives `sqs2tdb -cp` (the ONLY sanctioned way to obtain
SQS — never copy the raw database), plus DLM decorations.

**Algorithm — `generate_phase_sqs(work_root, phase, elements, level,
dlm, use_small, env_bin)`:**
1. Resolve target: `SMALL_SYSTEM[phase]` if `use_small` else phase.
2. Build `sqs2tdb -cp -l=<target> [-lv=N] [-sp=El1,El2]`.
3. **Quarantine** any work-root `species.in` → `species.in.stray`
   (sqs2tdb's own passes drop one there; a leftover stalls the
   handshake — the missing-HCP root cause).
4. Pass 1 (`runner.run_logged`, cwd=work_root): creates
   `<target>/species.in` + prompts.
5. Optional `species_edit` hook (DLM writes spin pseudo-species here).
6. Pass 2: actually copies `sqs_lev=*` dirs.
7. **Verification (fail-loud, 2026-07-22):**
   - pass-2 log still says "Edit the file" → RuntimeError (lattice
     missing from sqsdb / handshake broken);
   - `<target>` dir absent → RuntimeError (the work-root fallback that
     aliased HCP onto FCC+BCC is FORBIDDEN);
   - no element-DECORATED dir (`DECORATED_SQS_RE`) and no `link` files
     → RuntimeError.
8. Return the phase root.

**Other functions.** `apply_randomspin` (DLM decoration in *_small),
`sigma_lev3_to_lev0_dlm` (builds SIGMA DLM endmembers from a lev=3
generation: splits equivalent sites into ±spin pseudo-species),
`copy_small_systems` (DEPRECATED, kept as a no-op warning).

**Review checkpoints.**
- `DECORATED_SQS_RE = lev=\d+.*_[A-Z][a-z]?[+-]?\d*=` separates real
  calc dirs from raw database entries (`sqsdb_lev=*`); if sqs2tdb's
  naming ever changes, this and `run_upstream._DECORATED_RE` /
  `_COMP_TOKEN_RE` must change TOGETHER.
- `-lv=N` is CUMULATIVE (copies levels 0..N) and re-runs skip dirs
  that already have `str.out` — refinement relies on both facts.

---

## 3. `potcar.py` — POTCAR parsing and grid policy (194 lines)

**Purpose.** ENMAX extraction and every numeric policy for the
convergence sweeps.

**Constants (the tuning surface).**
`ENCUT_LOW/HIGH_FACTOR` (1.00/1.25 × max ENMAX → base grid),
`ENCUT_N_POINTS=5`, `ENCUT_GUARD_FACTOR=3.0` (extension runaway guard),
`KPPRA_MIN/MAX/STEP = 4000/10000/1000`, `KPPRA_EXTEND_MAX=20000`,
`KPPRA_PROBE_ENCUT_FACTOR=1.125` (KPPRA sweeps run at 1.125×ENMAX),
`PULAY_ENCUT_FACTOR=1.3` (ISIF=3 floor).

**Functions.** `parse_enmax`/`max_enmax` (regex over POTCAR, one value
per species block); `encut_grid`, `kppra_grid`, `kppra_probe_encut`;
`pulay_safe_encut(chosen, enmax) = max(chosen, ceil(1.3*enmax))`;
`find_potcars` / `element_potcar_map` (locate per-element POTCARs).

**Review checkpoints.**
- Physics knobs live HERE, not in converge.py: to change grid density,
  Pulay floor, or extension ceilings, edit these constants only.
- `pulay_safe_encut` applies to RELAXATIONS only (ISIF=3 stress);
  statics/phonons deliberately keep the sweep-chosen ENCUT.

---

## 4. `vaspwrap.py` — vasp.wrap / INCAR writer (374 lines)

**Purpose.** Single source of truth for every INCAR the pipeline emits
(`runstruct_vasp` consumes the `[INCAR]` section of vasp.wrap).

**Key structures.**
- `_BASE_INCAR`: LREAL=Auto, LWAVE/LCHARG=F, NCORE=8, KPAR=4, ALGO
  (overridden), PREC=Normal, LORBIT=11, ISMEAR=1, SIGMA=0.08,
  NELM=100, EDIFF=1e-6.
- `_MODE_INCAR`: `static` (NSW=0, IBRION=-1, +DOSTATIC), `relax`
  (NSW=100, IBRION=2, ISIF=3, EDIFFG=-0.01, +DOSTATIC), `phonon`
  (NSW=0, ISIF=2, PREC=Accurate, no DOSTATIC). NO volrelax/static
  modes for robustrelax — `-mk` DERIVES those from vasp.wrap.
- `_SPIN_INCAR` (AMIX .03 / BMIX 1e-4 / AMIX_MAG .8 / BMIX_MAG 1e-4),
  `_DLM_INCAR` (ISYM=0, NUPDOWN=0, its own mixing).
- `MAGNETIC_3D`, `DEFAULT_SPIN` (set True by run_upstream when any
  element is magnetic), `DEFAULT_MAGMOM_INIT=3.0`.

**Algorithm — `build_vasp_wrap(mode, encut, kppra, dlm, algo, usepot,
spin, natoms, magmom_init, ranks, extra)`:**
1. Start from `_BASE_INCAR`; `put()` de-duplicates keys in order.
2. Overlay ALGO, ENCUT, then `_MODE_INCAR[mode]` (`_dostatic` is a
   pseudo-key toggling the `DOSTATIC` line = ezvasp's ISMEAR=-5
   final static).
3. Overlay `parallel_overrides(natoms, ranks)`:
   - ≤4 atoms → NCORE=1, KPAR = largest divisor of `ranks` ≤ 8;
   - ≤12 atoms → NCORE=2, KPAR = divisor ≤ 4;
   - else keep NCORE=8×KPAR=4 (the validated 32-rank layout).
4. KPPRA, USEPOT; spin ON (non-DLM) → ISPIN=2 +
   `MAGMOM = <natoms>*<magmom_init>` + `_SPIN_INCAR`; DLM → MAGATOM/
   SUBATOM path instead.
5. `extra` dict overlays LAST (highest priority — sweeps use it for
   PREC=Accurate/LREAL=.FALSE.).
6. `ranks_from_prefix("mpiexec -n 32") → 32` feeds step 3.

**Review checkpoints.**
- MAGMOM must have exactly `natoms` entries (VASP 6.6 hard-errors) —
  any caller adding a new run type must pass the atom count OF THE
  CELL THAT RUNS (the fvasp.wrap bug was sizing from the SQS instead
  of the perturbation supercell).
- The ≤4/≤12/>12 tier boundaries are MIRRORED in `pbsjobs.SIZING`;
  change both together (2026-07-27 dry-run catch).
- KPAR must divide total ranks or VASP aborts — `_kpar_dividing`
  guarantees it; keep that property in any edit.

---

## 5. `converge.py` — ENCUT/KPPRA sweeps (371 lines)

**Purpose.** Static-energy convergence with a noise-robust stopping
rule and restart caching.

**Selection rule — `select_converged(settings, e_pa, tol_ev,
plateau_band_ev)`:**
1. Sort points ascending; scan triples: chosen = FIRST `S_i` with
   `|E_i−E_{i−1}| < tol` AND `|E_{i+1}−E_i| < tol` (the confirming
   point is the deliberately-unneeded extra run). rule="successive".
2. Fallback (noise floor): first window of `PLATEAU_WINDOW=4` points
   spanning ≤ `PLATEAU_BAND_EV=0.5 meV` → chosen = window start,
   rule="plateau". NEVER preempts a successive hit because it is only
   consulted when the pointwise scan found nothing on the FULL grid.
3. Nothing → (max setting, converged=False).

**Sweep — `run_sweep(parameter, …, extend_step, extend_max)`:**
1. Evaluate grid points ONE AT A TIME; after each, run
   `select_converged` on the prefix; STOP as soon as rule=="successive"
   (trailing points cannot change the answer — first-triple property).
   Plateau is deliberately not allowed to early-stop (would have
   mispicked Co KPPRA 4000 over 7000 on real data).
2. If not converged and `extend_step>0`: append `last+step` points
   until converged or `extend_max` (guard, not ceiling — warns loudly).

**Point — `run_static_point(src_sqs, dst, encut, kppra, …)`:**
1. **Cache**: if `dst/energy` parses → return it (restart fast-forward).
2. Copy str.out/POTCAR into dst; write static wrap with
   `extra={PREC:Accurate, LREAL:.FALSE.}` (sweep noise ≈0.3–0.5
   meV/atom at PREC=Normal swamps the 0.1 meV criterion) and
   rank-aware KPAR.
3. `runner.run_logged(runstruct_vasp + launcher)`; return energy/atom.

**`converge_sqs`** = KPPRA sweep (at 1.125×ENMAX, extendable to 20000)
then ENCUT sweep (at chosen KPPRA, extendable to 3×ENMAX). Returns
`(encut, kppra, kres, eres)`.

**Review checkpoints.**
- tol semantics: successive-DIFFERENCE with confirmation, per user
  spec; don't reintroduce compare-to-highest (it declared victory on
  moving curves).
- The plateau band (0.5 meV) is the measured PREC=Accurate noise
  envelope; tighten only with evidence from a real sweep table.

---

## 6. `relax.py` — robustrelax/infdet driver (268 lines)

**Purpose.** Produce `str_relax.out` + `energy` per SQS via the method
hierarchy: `infdet` (default) | `normal` | `runstruct`.

**Key facts from the robustrelax_vasp SOURCE (verified 2026-07-22):**
- `-mk` REQUIRES vasp.wrap and derives vaspvol/vaspstatic/vaspid/
  vaspf/vaspneb.wrap from it by grep-transform → tuned settings
  propagate automatically. Write vasp.wrap FIRST, then `-mk`.
- Flow: full relax (vasp.wrap) → `checkrelax -1` vs `-c` cutoff →
  stable: rescale energy via `energy_sup`; unstable: `energy` →
  `energy_end` (the DECAYED energy — never the result), volume relax
  in `00/`, infdet in `01/`, statics, inflection energy → `energy`,
  inflection geometry → `str_relax.out`.

**Functions.**
- `write_relax_wrap(dir, encut, kppra, dlm, algo, ranks)`.
- `robustrelax_complete(dir)` — completion predicate:
  `energy_sup+energy` (stable) OR `energy_end + 01/cstr_relax.out +
  energy` (unstable). The step-1 transient matches NEITHER — this is
  what stopped the poller from murdering robustrelax mid-workflow.
- `infdet_status(dir) → (engaged, ok, detail)` — engaged =
  `01/`|`energy_end` exists; ok = last line of `01/infdet.log` is
  `INFDET_NORMAL_TERMINATION` ("infdet terminated normally") AND
  `energy` present. checkrelax magnitude is NOT part of the criterion.
- `relax_structure(...)`:
  1. `runstruct` method: write wrap → `run_polled(pollmach
     runstruct_vasp, done_when=str_relax.out)` (runstruct writes it
     LAST, so that predicate is safe here — and only here).
  2. robustrelax methods: write wrap → `-mk` → delete stale
     `error`/`stop` (stale error makes robustrelax bail after step 1)
     → build cmd (`-id -c 0.05` unless caller overrides via
     relax_opts; `-idop` for infdet options) →
     `run_polled(done_when=robustrelax_complete)`.

**Review checkpoints.**
- NEVER use `str_relax.out` existence as robustrelax completion.
- NEVER adopt `energy_end` as a result.
- `INFDET_STRAIN_CUTOFF=0.05` (the -c) is the engage threshold —
  distinct from run_upstream's drift GATE (0.1), which applies only to
  runstruct.

---

## 7. `phonon.py` — fitfc phonon stage (575 lines)

**Purpose.** Published recipe (Calphad 58 (2017) 70 §3.3):
`fitfc -si=str_relax.out -ernn=4 -ns=1 -dr=0.04 -nrr` (generate
perturbations) → force statics in each `vol_*/p*` → `fitfc -f -frnn=2`
(fit) → `robustrelax_vasp -vib` scaling → top-level `svib_ht`.

**Algorithm — `run_fitfc(sqs_dir, encut, kppra, …, on_unstable, rl)`:**
1. `_clear_stale_fit_outputs` (old svib/fitfc.out/fvib must not mask a
   failed refit).
2. Generation: `fitfc` with `build_fitfc_gen_args` (ernn=4.0, ns=1,
   dr=0.04, -nrr) via run_logged.
3. `_write_force_wrap` AFTER generation: phonon-mode wrap sized from
   the PERTURBATION SUPERCELL atom count (MAGMOM=N*3 — the NIONS=8
   crash fix), written as `vaspf.wrap`.
4. Strain runs (`ns>1` only): run_polled over `vol_*`.
5. Force runs: `run_polled(pollmach runstruct_vasp -lu -w vaspf.wrap,
   done_when=all_force_runs_done(pert_dirs), kind="force")` — in PBS
   mode this becomes ONE JOB ARRAY, one element per `p*` dir.
6. Fit: `fitfc -f -frnn=2 -si=str_relax.out`; `detect_unstable_modes`
   greps the log for `UNSTABLE_MARKERS`.
7. Unstable policy: `mark` (record `unstable_modes.log`, energy-only) |
   `escalate` (regenerate at 1.5×dr radius, rerun forces, refit) |
   `force` (retry `fitfc -fn`; svib omits unstable branches — lower
   bound, flagged). Optional `-rl=<len>` soft-mode treatment.
8. `robustrelax_vasp -vib` then `promote_svib_ht` fallback (copy
   `vol_0/svib_ht` up with the cellcvrt -pn atom-ratio scaling) —
   `sqs2tdb -fit` reads ONLY the top-level svib_ht.
9. DLM: `dlm_fixup` strips ±spin suffixes from str_relax/str_unpert
   before fitfc parses them.

**Review checkpoints.**
- Force-run reuse: a `p*` dir with `force.out` is skipped — after any
  crash-era corruption, DELETE `vol_*` before refitting (2026-07-27).
- Phonon ENCUT = the relax (Pulay-floored) value by pipeline
  convention; phonons at endmembers only unless scope says otherwise.

---

## 8. `refine.py` — adaptive compute placement (426 lines)

**Purpose.** Two refiners deciding WHERE to spend next (2026-07-22).

**A. Energy mesh (`refine_energy_mesh` / CLI `energy`):**
1. `parse_fit_energy` — rows of ≥3 floats: (x = col0, E_dft = col[-2],
   E_fit = col[-1]) from `sqs2tdb -fit`'s fit_energy.out.
2. `worst_fit_point` — max |E_dft−E_fit| → x*.
3. Below `--min-err-ev` (1 meV default): plan says "none", stop.
4. `refinement_targets` — midpoints between x* and its nearest sampled
   neighbor on EACH side.
5. Generate next `-lv=N` (cumulative, existing dirs skipped).
6. `select_new_dirs` — among fresh dirs (str.out, no energy), pick
   nearest composition to each target → `refine_pick`; every other
   fresh dir → `refine_skip` (discovery + downstream both honor it).
7. Write `refinement_plan.json`; rerun the upstream job to compute.

**B. Adaptive svib (`adaptive_svib_phase`, driven by
`--phonon-scope adaptive`):**
1. Collect svib PER ATOM (`svib_ht`/natoms — fitfc writes per-cell) at
   x=0, 0.5, 1.
2. Linear test: dev = svib(0.5) − mean(endmembers); |dev| ≤ tol
   (`--svib-linear-tol`, k_B/atom, default 0.1) → record linear model,
   STOP (no lev=2 purchase).
3. Refuted → order lev=2 sides by mixing energy/atom (more stable
   first); run phonons on side 1; least-squares quadratic through the
   4 points (`_quad_lstsq`).
4. Keep if RMSE ≤ tol (NOTE: the literal "RMSE higher than lev=1"
   comparison can't fire — a 4-point LSQ always beats the lev-1
   quadratic on the same points — so the spec is operationalized
   against the same tol as the linearity test; lev1_prediction_error
   is recorded as a diagnostic). Else buy the other side; keep the
   lower-RMSE fit. Trail → `svib_adaptive.json`.

**Review checkpoints.**
- `composition_fraction` ignores site multiplicities — fine for
  single-sublattice phases, WRONG for SIGMA (which is endmember-only
  anyway). Extend with `PHASE_MULT` weighting before using on
  multi-sublattice meshes.
- fit_energy.out column convention is asserted only loosely (first /
  last-two); verify once against a real file from your sqs2tdb build.

---

## 9. `runner.py` — subprocess layer (226 lines)

**Purpose.** The ONLY two ways the pipeline executes external
commands; also the pluggable PBS backend hook.

**`run_logged(cmd, cwd, log, timeout, check)`** — blocking, tees to
log, raises on rc≠0 when check.

**`run_polled(cmd, cwd, log, done_when, stop_sentinel, …)`:**
1. If a backend is installed (`set_backend(broker)`) → delegate to
   `broker.run_as_job` with the work_dirs/natoms/kind/done_file
   metadata. Control flow identical; only WHERE it runs changes.
2. Local path: unlink stale sentinel → `Popen(start_new_session=True)`
   (own process GROUP) → poll: done_when? proc exited? timeout?
3. Teardown: touch sentinel ONLY if proc still alive (pollmach needs
   it; self-terminating robustrelax must not get litter); wait; on
   refusal `os.killpg` TERM then KILL (parent-only terminate orphaned
   the mpiexec children → the UCX node-exhaustion incident); unlink
   sentinel.

**Review checkpoints.**
- Every long VASP execution MUST flow through run_polled — that is the
  single chokepoint the PBS mode relies on.
- done_when predicates must describe FINAL state of the whole command
  (see relax.robustrelax_complete), not an intermediate file.

---

## 10. `pbsjobs.py` — PBS fan-out broker (306 lines)

**Purpose.** `--submit pbs`: every VASP execution becomes its own
right-sized qsub job (the paper's `foreachfile wait sbatch` pattern +
sizing, throttling, retries, restart safety).

**`SIZING`** — (max_natoms, ncpus, walltime) per kind; tier boundaries
(≤4/≤12/>12) MIRROR `vaspwrap.parallel_overrides`. relax: 8/2h,
16/4h, 32/6h, 64/12h; force: 16/30m, 32/2h, 64/4h; probe 32/6h.
**`retarget_launcher`** rewrites trailing `mpiexec -n K` to the job's
ncpus (select-line/rank mismatch structurally impossible).

**`Broker` fields:** work_root, group_list=a1485, model=mil_ait,
queue=normal, site_env (job_env.sh source line), max_inflight=16,
max_retries=1, poll_interval=60, use_arrays, dry_run.

**Job shapes:** `render_single` (one command, one dir),
`render_loop` (serial fallback over dirs, skip if done_file present),
`render_array` (`#PBS -J 0-(N-1)`, manifest file indexed by
`$PBS_ARRAY_INDEX` — all perturbation statics wall-parallel).

**`run_as_job(tag, cwd, cmd, done_when, …)`:**
1. Size from (natoms, kind); array iff use_arrays ∧ ≥2 work_dirs ∧
   cmd[0]=="pollmach" (pollmach itself is STRIPPED — a dedicated job
   has no machine pool to poll).
2. Marker `.qjob_<tag>` (JSON: job_id, attempt, script): a live
   recorded job is ADOPTED on restart, never resubmitted.
3. Loop: done_when → 0; throttle (`n_inflight < max_inflight`); qsub;
   dry_run returns immediately after rendering; job left queue without
   outputs → retry ≤ max_retries → −1.
4. Completion is judged ONLY by done_when (files in the tree), never
   job exit codes.

**Review checkpoints.**
- SIZING boundaries ↔ parallel_overrides tiers (test-asserted).
- Walltimes are scheduling requests; NAS bills actual usage — keep
  them generous.

---

## 11. `strfile.py` — ATAT structure files (239 lines)

**Purpose.** Parse `str.out`-family files; the drift metric; stub
detection.

- `read_structure` — header (3 coord rows + 3 lattice rows, or
  a,b,c,α,β,γ form) + atom lines → `Structure`.
- `parse_cell` — real-space cell = lattice·coord matrices (pure-python
  3×3 helpers, no numpy).
- `cell_distortion(A,B)` — polar-decomposition-free strain metric:
  ‖(B·A⁻¹)ᵀ(B·A⁻¹) − I‖/2-ish; `lattice_drift(ideal, relaxed)` wraps
  it (ATAT checkrelax analogue).
- `validate_structure_file` — catches the DEGENERATE STUB robustrelax
  leaves after an inner VASP crash (identity cell + atom lines without
  coordinates) — "exists" is never trusted.
- `strip_spin_suffix_text` — `Cr+2`→`Cr` for DLM fixups.

**Review checkpoints.** If checkrelax-vs-`lattice_drift` values ever
disagree systematically with ATAT's own `checkrelax`, recalibrate here
(same metric, different norm conventions ≈ few %).

---

## 12. `vasp_triage.py` — failure catalog (578 lines)

**Purpose.** Scan a run tree (gz-aware, `.relax`/`.static` suffixes),
match OUTCAR/vasp.out text against `CATALOG: List[Signature]`
(error_id, category, regex, stream, fix text), report causally —
input/chain errors first, because later errors are consequences
(empty_poscar → XC_FOCK_READER red herring).

**Signatures include:** empty_poscar, magmom_count, sick_job,
mpi_init_failure (UCX QP exhaustion — orphaned ranks), infdet_crash
(malloc tcache abort), EDDDAV/EDDRMM/ZHEEV/EDWAV/subspacematrix/
BRMIX/not_converged_scf, ZBRENT, truncated_outcar.

**Review checkpoint.** New failure mode → add a Signature with the FIX
in the message; keep "genuine failure" regexes narrow (the BZINTS
false-positive lesson).

---

## 13. `run_upstream.py` — orchestrator (1245 lines)

**Purpose.** CLI + per-phase driver + manifest. Review it LAST — it
only sequences the modules above.

**main() flow:**
1. Parse args (see below); resolve POTCARs → `max_enmax`; set
   `vaspwrap.DEFAULT_SPIN` if any element magnetic; build DLMConfig.
2. `--submit pbs`: construct `pbsjobs.Broker` (requires `--job-env`,
   and system scope or presets), `runner.set_backend(broker)`.
3. Convergence settings: presets given → skip sweeps; else scope
   `system` → `system_probe_convergence`:
   - seeded RNG picks ONE single-sublattice phase, generates it, then
     ONE random SQS per element-RICH side (>50 %, endmembers count,
     lev irrelevant — `pick_probe_dirs`/`site_fractions`);
   - node mode: `converge_sqs` inline; pbs mode: `--probe-worker`
     self-invocations as parallel jobs writing `probe_result.json`;
   - global (ENCUT, KPPRA) = elementwise MAX over probes (+ Pulay
     handled at relax time); recorded in manifest["system_probe"].
4. Per phase: `process_phase` → SIGMA branches to `process_sigma`
   (endmember corners only, DLM via sigma_lev3_to_lev0_dlm).
5. `process_phase`: generate → `discover_sqs_dirs` (decorated dirs,
   skipping `refine_skip`) → `_one(d)` per SQS = `process_one_sqs`;
   ThreadPool over SQS when pbs+preset; adaptive-svib hook afterwards
   (`--phonon-scope adaptive`).
6. `process_one_sqs` (the per-SQS state machine):
   STAGE 1 convergence (preset or sweep) → Pulay floor for relax →
   STAGE 2 `relax.relax_structure` → `validate_structure_file` →
   `infdet_status` bookkeeping (engaged/ok; failure ⇒ relax_ok=False,
   energy_end NEVER adopted; success ⇒ `infdet_ok.flag`, stale
   relaxaway cleared) → wait-marker removal on success → checkrelax:
   value always recorded; `relaxaway.flag` ONLY for method=runstruct
   (drift is not a robustrelax failure signal) → STAGE 3 phonons per
   scope (endmem markers / lev=0; adaptive adds lev=1) → result dict
   into the manifest.
7. Manifest JSON written after every phase.

**CLI groups.** Identity/paths (`--element1/2, --work-root, --potcars,
--phases, --sqs-level, --env-bin, --cmd-prefix`); convergence
(`--tol-ev, --convergence-scope, --preset-encut/kppra, --probe-seed,
--plateau-band`); relax (`--relax-method, --relax-opts`); spin/DLM
(`--no-spin, --magmom-init, --dlm, --dlm-moment`); phonons
(`--skip-phonon, --phonon-scope, --svib-linear-tol,
--fitfc-on-unstable/-escalate-ernn/-rl, --fitfc-ernn/frnn/dr`); gates
(`--max-checkrelax`); PBS (`--submit, --job-env/model/queue/
group-list/max-inflight/retries, --no-job-arrays, --job-dry-run`,
hidden `--probe-worker`).

**Review checkpoints.**
- The keep/throw semantics live in `process_one_sqs` — edits to what
  counts as success/failure belong there and in relax.infdet_status,
  nowhere else.
- `_probe_worker_argv` strips orchestration flags before re-invoking
  itself — new PBS flags must be added to its VALUED/FLAGS sets.

---

## 14. Templates, smoke tests, downstream touchpoints

- `submit_upstream_template.pbs` — single-job mode (whole pipeline in
  one 128-core job, `mpiexec -n $NCPUS`, ulimit -c 0). Proven path.
- `submit_orchestrator_template.sh` — fan-out mode: generates
  `job_env.sh` (self-initializing module system, ulimit -c 0), nohups
  `run_upstream.py --submit pbs` with presets/dry-run passthrough.
- `nas_smoke/` — `run_smoke.py` T1–T5 (env, runstruct, robustrelax,
  fvasp force pair, convergence mini-sweep) and
  `run_endmember_e2e.py` + PBS wrappers (graded FCC endmember e2e).
- Downstream (`../sqs2tdb_pipeline.py`) honors: `energy`,
  `checkrelax.out`/`relaxaway.flag` (skipped when `infdet_ok.flag`),
  `refine_skip`, top-level `svib_ht`, oszicar quality score.

---

## 15. Cross-cutting invariants (break one of these and the failure
is silent — check them after ANY edit)

| Invariant | Where enforced |
|---|---|
| Wrap NCORE/KPAR tiers (≤4/≤12/>12) == broker cpu tiers | vaspwrap.parallel_overrides ↔ pbsjobs.SIZING (+ test) |
| KPAR divides total ranks | vaspwrap._kpar_dividing |
| MAGMOM count == atoms of the cell that runs | every build_vasp_wrap caller |
| robustrelax completion ≠ str_relax.out existence | relax.robustrelax_complete |
| energy_end is never a result | run_upstream.process_one_sqs |
| checkrelax gates runstruct only | run_upstream drift block |
| infdet success == "infdet terminated normally" + energy | relax.infdet_status |
| sqs2tdb two-pass must not see a work-root species.in | sqsgen quarantine |
| phase dir fallback to work root is forbidden | sqsgen verification |
| decorated-name regexes stay in sync | sqsgen.DECORATED_SQS_RE ↔ run_upstream._DECORATED_RE/_COMP_TOKEN_RE ↔ refine._COMP_TOKEN_RE |
| ONE global (ENCUT,KPPRA) for all energy/relax/phonon runs | probe protocol + presets |
| sweep statics PREC=Accurate + LREAL=.FALSE. | converge.run_static_point |
| done_when = final state of the whole command | every run_polled call site |
| completion judged by files, never exit codes | pbsjobs.run_as_job |
| all temp state lives in the tree (restart = reconcile) | markers: .qjob_*, wait, endmem, *.flag, refine_* |
```
