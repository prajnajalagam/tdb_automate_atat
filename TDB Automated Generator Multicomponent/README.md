# TDB Automated Generator — Multicomponent (N ≥ 2)

Generalization of the binary `sqs2tdb` pipeline to ternary, quaternary, ...
alloys. The binary pipeline in `../TDB Automated Generator/` is **not
modified** by anything here; this directory contains a parallel, self-
contained implementation.

For the full design rationale (subsystem-extrapolation strategy,
`terms.in` generalization, feasibility-first CS-AL search), see
**[`DESIGN.md`](DESIGN.md)**.

## Status

| Phase | Module | Status |
|---|---|---|
| P0 | `DESIGN.md` | ✅ design signed off |
| P1 | `subsystems.py`, `select_endmembers_mc.py`, `test_subsystems.py` | ✅ this commit |
| P2 | `sqs2tdb_pipeline_mc.py` (Route B + complexity ladder) | ⬜ |
| P2.5 | `csal_search.py` (CS-AL surrogate, Section 6a) | ⬜ |
| P3 | ternary-term fitting + full-simplex Route A | ⬜ |
| P4 | `score_tdb_combinations_mc.py` (isopleth + CS→MOO) | ⬜ |
| P5 | Co-Cr-Ni validation, regression vs binary pipeline | ⬜ |

## P1 — what's here

### `subsystems.py`
The DFT-free foundation. Phase constants, regex parsers, helpers lifted
from the binary code, plus the new multicomponent pieces:

- `parse_occupation(dirname)` — parse `sqs[db]_lev=N_...` into a structured
  `Occupation` with per-sublattice species/fractions.
- `composition_on_phase(occ, phase, elements)` — N-element composition on
  the mixing sublattice(s), with foreign-element rejection (the same fix
  applied in the binary script).
- `subsystem_for_occupation(occ, phase, elements)` — tag each SQS by the
  subsystem it populates (`('CO','CR')` for a Co-Cr binary edge,
  `('CO','CR','NI')` for a ternary interior, ...).
- `sigma_corner_key(occ, elements)` — identify SIGMA lev=0 corners
  (one species per sublattice).
- `enumerate_subsystems(elements)` — all binary, ternary, ... subsystems.
- `scan_sqs(roots, elements, ...)` — walk data roots, yield
  `SQSCandidate`s tagged with phase, composition, subsystem, energy,
  svib_ht.
- `n_params_for_terms(species_per_sublattice, terms)` — generalized
  CALPHAD parameter count, replacing the binary `n_params = order + 3`.

### `select_endmembers_mc.py` (Stage 0)
Interactive endmember selection:

```bash
python select_endmembers_mc.py \
    --elements Co,Cr,Ni \
    --data-roots /path/to/CoCrNi_data,/path/to/CoCr_data \
    --out system.yaml
```

- Single-sublattice phases (FCC/BCC/HCP): one endmember per element
  (N corners, not just 2 like the binary script).
- SIGMA_D8B: one endmember per unique site-occupation corner. Up to
  N³ corners for a single-sublattice 3-site SIGMA (e.g. 27 for ternary).
- Multi-candidate prompts match the binary fix: list every DFT run for
  the same corner/element, default to recommended (svib-present →
  lowest energy → shortest path), `Enter` accepts default.
- `--auto-sigma` skips per-corner prompts (for PBS/batch).
- Output: `system.yaml` per **DESIGN.md §4**.

### `test_subsystems.py`
Stdlib-only unit tests (no pytest, no ATAT) — build synthetic SQS dirs
in a tempdir and verify parsing, composition, subsystem-tagging, SIGMA
corner identification, parameter counting, and end-to-end scanning.

```bash
python test_subsystems.py
```

Exit code 0 = all pass. **38 tests currently passing.**

## Workflow (when P2+ ship)

```
select_endmembers_mc.py  →  sqs2tdb_pipeline_mc.py  →  score_tdb_combinations_mc.py
       (Stage 0)                 (Stage 1 + 2)              (Stage 3)
       system.yaml               tdb_manifest.json         BEST_<system>.tdb
```

For the binary case, this directory's pipeline degenerates to the binary
one (Route B with one binary edge); when P2 ships we'll add a regression
test that the multicomponent path with N=2 reproduces the binary
pipeline's TDB.

## What's missing vs the binary pipeline

- **Fitting orchestration (P2)** — multicomponent `sqs2tdb_pipeline_mc.py`
  using Route B (assess each binary edge with the existing binary
  scripts, then combine).
- **CS-AL surrogate search (P2.5)** — feasibility-first GPC/GPR loop per
  DESIGN.md §6a, replacing brute-force subset enumeration where the
  grid is too large to enumerate.
- **Stage 3 multicomponent scoring (P4)** — isopleth/section conditions
  instead of a single `v.X(comp_el)` axis, with the CS→MOO hybrid for
  combo selection (§6a.7).
- **HPC submission template** for the MC pipeline (will be added
  alongside P2).
