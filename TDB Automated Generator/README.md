# TDB Automated Generator — Binary (A–B) Pipeline

High-throughput automation for generating a CALPHAD thermodynamic database
(`.tdb`) for a **binary A–B alloy** from first-principles SQS data, by
orchestrating ATAT's `sqs2tdb`. Three stages run in sequence:

```
select_endmembers.py  →  sqs2tdb_pipeline.py  →  score_tdb_combinations.py
   (pick corners)         (fit: Stage 1 + 2)        (combine + score)
   endmembers.yaml        tdb_manifest.json         BEST_A_B.tdb
```

---

## Prerequisites

- **ATAT** installed with `sqs2tdb` on your `PATH` (`which sqs2tdb`).
- Per-SQS DFT data already computed (VASP workflow), i.e. each `sqs_lev=*`
  directory contains at least `energy` and `str.out` (and optionally
  `svib_ht`, `str_relax.out`).
- **Python 3.8+** and:
  - `pyyaml` — Steps 1 & 2.
  - `pycalphad`, `xarray`, `numpy` — Step 3 only.

```bash
pip install pyyaml pycalphad xarray numpy
```

Supported phases: `FCC_A1`, `BCC_A2`, `HCP_A3` (single sublattice) and
`SIGMA_D8B` (3 sublattices: `aj`/`g`/`ii` = 10/4/16). Site labels and
multiplicities must match your installed `$ATATDIR/data/sqsdb/<PHASE>/rndstr.skel`.

---

## Step 1 — Select endmembers (`select_endmembers.py`)

Scans your data roots for the pure-element endmember SQS (`sqs_lev=0_*`),
validates each has the files `sqs2tdb -fit` needs, and lets you interactively
pick the A-rich and B-rich endmember per phase. Writes `endmembers.yaml`.

```bash
python select_endmembers.py \
  --element1 Co \
  --element2 Cr \
  --data-roots /data/CoCr/fcc,/data/CoCr/bcc,/data/CoCr/hcp \
  --out endmembers.yaml
```

| Flag | Default | Meaning |
|---|---|---|
| `--element1` | *(required)* | First element symbol |
| `--element2` | *(required)* | Second element symbol |
| `--data-roots` | *(required)* | Comma-separated directories to scan |
| `--scan-depth` | `6` | Max `os.walk` depth |
| `--out` | `endmembers.yaml` | Output YAML path |

The script prints candidates per phase with composition, energy, and whether
`svib_ht` is present, then prompts for the two indices. (Interactive — needs a
terminal.) SIGMA_D8B endmembers, if present, are auto-deduplicated by site
occupation and written under an `ALL:` list.

## Step 2 — Fit (`sqs2tdb_pipeline.py`)

Reads `endmembers.yaml`, discovers the mixing SQS (`lev>0`) per phase, and runs:

- **Stage 1 (energy-only):** every SQS subset (size `min..max`) × `terms.in`
  order (`2,0`/`2,1`/`2,2`); runs `sqs2tdb -fit`; prunes on max |error| in
  `fit_energy.out` and an overfit guard.
- **Stage 2 (vibrational):** layers `svib_ht` onto Stage-1 survivors; prunes on
  `fit_svib_ht.out`. Skipped for SIGMA and with `--skip-svib`.

```bash
python sqs2tdb_pipeline.py \
  --endmembers-yaml endmembers.yaml \
  --data-roots /data/CoCr/fcc,/data/CoCr/bcc,/data/CoCr/hcp \
  --energy-cutoff 0.10 \
  --svib-cutoff 10.0 \
  --n-workers 8 \
  --phases FCC_A1,HCP_A3
```

| Flag | Default | Meaning |
|---|---|---|
| `--endmembers-yaml` | *(required)* | YAML from Step 1 |
| `--data-roots` | *(required)* | Comma-separated dirs to scan for SQS |
| `--workdir-prefix` | `<A>-<B>_automate` | Base name for the work directory |
| `--min-sqs` | `3` | Min mixing SQS per fit |
| `--max-sqs` | `7` | Max mixing SQS per fit |
| `--energy-cutoff` | `0.10` | Max \|error\| in `fit_energy.out` col 5 (eV) |
| `--svib-cutoff` | `10.0` | Max \|error\| in `fit_svib_ht.out` col 5 |
| `--n-workers` | `4` | Parallel fit processes |
| `--skip-svib` | off | Skip Stage 2 |
| `--phases` | all in YAML | Comma-separated subset of phases |

**Outputs** (in the auto-created `<prefix>_N/` workdir):
- `fit_results.json` — full per-task results for both stages.
- `tdb_manifest.json` — surviving per-phase `.tdb` paths (input to Step 3).

## Step 3 — Combine & score (`score_tdb_combinations.py`)

Reads the manifest, takes the Cartesian product of per-phase TDB candidates,
combines each with `sqs2tdb -tdb`, and scores the combined TDB against a
reference database using pycalphad (L1 phase-fraction distance + a
phase-boundary-misplacement penalty). Ranks combos, copies the best to
`BEST_<A>_<B>.tdb`.

```bash
python score_tdb_combinations.py \
  --manifest <prefix>_0/tdb_manifest.json \
  --ref-tdb /refs/CoCr_reference.tdb \
  --comp-element Co \
  --T-range 500,1200,50 \
  --X-grid 0.005 \
  --n-workers 1
```

| Flag | Default | Meaning |
|---|---|---|
| `--manifest` | *(required)* | `tdb_manifest.json` from Step 2 |
| `--ref-tdb` | *(required)* | Reference TDB to score against |
| `--comp-element` | *(required)* | Element for the composition (X) axis |
| `--eq-phases` | from manifest | Phases for the equilibrium calc |
| `--T-range` | `500,1200,50` | `T_min,T_max,T_step` (K) |
| `--X-grid` | `0.005` | Composition step |
| `--P` | `101325` | Pressure (Pa) |
| `--stable-tol` | `1e-6` | Phase-fraction threshold for "stable" |
| `--boundary-weight` | `0.25` | Weight of the boundary penalty |
| `--boundary-power` | `1.0` | Exponent on the boundary penalty |
| `--n-workers` | `4` | Parallel scoring (use `1` if pycalphad isn't process-safe) |
| `--max-combos` | `0` | Cap combinations (`0` = unlimited; samples if exceeded) |
| `--workdir` | next to manifest | Scoring work directory |

**Outputs** (in `<workdir>/stage3_scoring/` by default):
- `scoring_results.json` — every combination ranked by `final_score`.
- `BEST_<A>_<B>.tdb` — the top-scoring combined database.

---

## End-to-end example (Co–Cr)

```bash
cd "TDB Automated Generator"

# 1. pick endmembers (interactive)
python select_endmembers.py --element1 Co --element2 Cr \
  --data-roots /data/CoCr --out CoCr.yaml

# 2. fit
python sqs2tdb_pipeline.py --endmembers-yaml CoCr.yaml \
  --data-roots /data/CoCr --n-workers 8
# -> creates Co-Cr_automate_0/ with tdb_manifest.json

# 3. score against a reference
python score_tdb_combinations.py \
  --manifest Co-Cr_automate_0/tdb_manifest.json \
  --ref-tdb /refs/CoCr.tdb --comp-element Co --n-workers 1
# -> Co-Cr_automate_0/stage3_scoring/BEST_Co_Cr.tdb
```

## Notes & troubleshooting

- **Element case matters.** `sqs2tdb -fit` matches `species.in` against the
  directory names, so symbols are normalized to chemical case (`Co`, `Cr`).
- **"0 SQS found."** Check that your directories are named `sqs_lev=N_...`
  (both `sqs_lev=` and `sqsdb_lev=` are accepted) and contain `energy` +
  `str.out`. Discovery prints a skip summary explaining what it rejected.
- **Failed fits** are logged to `<workdir>/<PHASE>/failures.log` (the raw
  `sqs2tdb` output is preserved before the temp dir is cleaned).
- **Stage 3 speed.** The reference equilibrium is computed once and reused;
  the per-combo cost is one test-TDB equilibrium. Use `--max-combos` to bound
  large cross-phase products.

For the N-component (ternary+) extension, see
`../TDB Automated Generator Multicomponent/DESIGN.md`.

## HPC submission

`submit_CoCr.pbs` is a ready-to-submit PBS script that runs Steps 2 and 3
end-to-end on a 128-core Milan/AIT node (24 h walltime, queue `long`,
group `a1485`). All paths, cutoffs, and parallelism settings live in a
clearly delineated **USER CONFIG** block at the top of the file.

```bash
qsub submit_CoCr.pbs
tail -f pipeline_Co-Cr_*.log   # live progress while the job runs
```

To run for a different binary, copy the file (e.g.
`cp submit_CoCr.pbs submit_FeNi.pbs`) and edit the USER CONFIG block —
nothing below that block should need to change.
