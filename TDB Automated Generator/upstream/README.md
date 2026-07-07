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

## Procedure

For each SQS (and each SIGMA endmember):

1. **Static runs**, starting at `lev=1` for each binary single-sublattice
   phase.
2. **Convergence testing** — `MAX_ENMAX` is the largest `ENMAX` across the
   POTCARs.
   - **KPPRA** swept `4000 … 10000` (step 1000) at a fixed
     `ENCUT = 1.125 × MAX_ENMAX`; the converged KPPRA is frozen.
   - **ENCUT** swept `1.00 × … 1.25 × MAX_ENMAX` (5 points) at the converged
     KPPRA.
   - *Converged* = the smallest setting whose energy/atom — and every larger
     setting's — is within `--tol-ev` (default **1 meV/atom**) of the
     highest-setting reference.
   - `ALGO = All` by default.
3. **Relaxation** — `--relax-method normal` (robustrelax) or `infdet`
   (inflection detection), using the converged ENCUT/KPPRA. Produces
   `str_relax.out`.
4. **Phonons** — full `fitfc` workflow → `svib_ht`.

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

The resulting tree is then fed straight into the downstream pipeline:

```bash
python3 ../select_endmembers.py --element1 Co --element2 Cr \
  --data-roots /scratch/CoCr_upstream --out endmembers.yaml
python3 ../sqs2tdb_pipeline.py --endmembers-yaml endmembers.yaml \
  --data-roots /scratch/CoCr_upstream ...
```

## What is and isn't tested

The pure logic — ENMAX parsing, ENCUT/KPPRA grids, `vasp.wrap` generation,
**1 meV/atom convergence selection**, SIGMA lev=3→lev=0 spin conversion, and
the DLM fixup — is covered by `tests/` (`pytest`). The VASP-driving glue
(`runner`, `converge.run_static_point`, `relax`, `phonon.run_fitfc`) can only
be exercised on a real ATAT + VASP node.
