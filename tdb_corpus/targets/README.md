# targets/ — consensus-target JSON files (one per system × phase)

Files in this directory are consumed by `../sqs_target_gate.py` and (via
`--target-dir`) by `../../TDB Automated Generator/sqs2tdb_pipeline.py`
to decide which SQS calculations should be **included in** vs **excluded
from** the sqs2tdb fit.

## What's in a target file

Produced by `reverse_engineer_targets.ipynb` (the Fable notebook), one
JSON per (system, phase) combination. Shape (matches the notebook's
`consensus_data`):

```json
{
  "x": [0.0, 0.04, ..., 1.0],
  "E_form": {
    "mean":  [...],     // eV/atom, vs SER references
    "sigma": [...],
    "n_contributing": [...]
  },
  "svib_ht": { "mean": [...], "sigma": [...], "n_contributing": [...] },
  "curves":  { "E_form": [[...], ...], "svib_ht": [[...], ...], "sources": [...] },
  "meta": {
    "T0": 298.15, "T_hi": 1000.0,
    "els": ["CO", "CR"],
    "phase_name": "FCC_A1",
    "units": { "E_form": "eV/atom", "svib_ht": "kB/atom" }
  }
}
```

## Naming convention

`<elsA>_<elsB>_<phase>_consensus.json`, alphabetical element order:

```
CO_CR_FCC_A1_consensus.json
CO_CR_BCC_A2_consensus.json
CO_CR_HCP_A3_consensus.json
CO_CR_SIGMA_consensus.json     # SIGMA, not SIGMA_D8B — the notebook's
                               # phase aliasing collapses dialect labels
```

The gate is case-insensitive on element order; it'll also look for
`<elsB>_<elsA>_<phase>_consensus.json` if the canonical name is missing.

## How the gate uses it

For each candidate SQS at composition `x_B`:

1. Refits Redlich-Kister on the consensus `E_form` mean (sigma-weighted,
   matching cell 12 of the notebook).
2. Evaluates the **RK excess** at `x_B` — this is the target value the
   SQS's DFT formation energy should reproduce.
3. Computes `gate_sigma = sqrt(consensus_sigma^2 + dft_noise_floor^2)`
   at `x_B` (dft_noise_floor default 5 meV/atom — sane lower bound for
   converged-DFT + SQS-vs-random-alloy error budget).
4. z-score the SQS's DFT formation energy: `z = (E_DFT − target) / gate_sigma`.
5. Accept if `|z| ≤ n_sigma` (default 3.0), else reject and log.

Excess vs total: we compare the **excess** part only. The DFT side uses
same-phase pure-element references (e.g. for FCC_A1 mixing fits, both
pure-Co and pure-Cr endmembers in FCC), so lattice-stability terms
cancel and there's nothing to match against the SER-referenced RK
baseline. The gate's `evaluate()` returns the RK-excess target directly,
not the baseline+excess total.

## Regenerating

Open `../reverse_engineer_targets.ipynb` (the Fable notebook). For
each (system, phase) of interest, set the `elements` and `phase_name`
variables in the main cell and re-run; it writes the JSON into the
notebook's `OUTPUT_DIR` (configure to point here, or move output files
in by hand). The gate picks up whichever JSONs are present.

## Status

This directory ships empty in git; populate it by running the notebook.
The gate module's behaviour when no JSON is present for a given phase
is to **skip the gate cleanly** (no rejection — equivalent to disabled).

## Honesty caveats (advisor review, 2026-07)

Two systematic risks in reverse-engineering DFT targets from assessed
TDBs — both must be kept in mind when tightening the gate:

1. **Metastable extrapolation.** An assessed TDB constrains G(x,T) only
   where the phase is stable or measured. Where a phase is metastable
   (e.g. FCC across the Cr-rich half), several assessments often inherit
   the *same* SGTE/PURE lattice stabilities, so the cross-TDB spread
   shrinks *artificially* — agreement there is provenance, not evidence.
   A tight 3σ gate in such regions rejects perfectly correct DFT. The
   gate therefore floors its sigma at `--target-min-sigma` (default
   10 meV/atom, `min_sigma_ev` in `sqs_target_gate.py`); do not lower it
   below the DFT noise floor, and prefer widening it for phases known to
   be metastable over most of the composition range.

2. **Magnetic entropy contamination of svib targets.** The `*T`
   coefficients of a TDB mix vibrational, electronic, and — for Co/Cr/Ni
   — sizeable magnetic contributions, with the magnetic part *sometimes*
   split into the IHJ (TC/BMAGN) model and sometimes folded into the
   excess terms, differently per assessment. Mapping consensus `*T`
   terms directly onto `svib_ht` (a purely vibrational, high-T harmonic
   quantity) is therefore a category error unless the notebook excludes
   the IHJ contribution and flags assessments that fold magnetism into
   L-parameters. Until the notebook does that decomposition, treat
   svib_ht targets as SOFT guidance (wide sigma), never as a hard gate.

Every gate rejection is recorded in the pipeline's
`discovery_rejects.json` (dir, reason, z-score, target, sigma) — review
it after each run; a phase losing many SQS to the gate usually means the
target, not the DFT, is wrong there.
