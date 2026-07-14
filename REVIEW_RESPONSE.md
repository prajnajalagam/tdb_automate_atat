# Advisor review → response record

**Date:** 2026-07-14 · **Branch:** `claude/elegant-maxwell-09ZHT`
**Scope reviewed:** upstream VASP/ATAT generator, sqs2tdb fitting pipeline
(Stages 1–2), Stage-3 phase-diagram scoring, TDBDB consensus-target
reverse-engineering loop.

This file is the audit trail for the July 2026 workflow review: every
finding, its severity, what was done, and the commit that did it. Open
items are listed with the same numbering so later work can reference
them. One finding per commit; test counts refer to
`TDB Automated Generator/upstream/tests/`.

## Findings fixed in this pass

| # | Severity | Finding | Fix | Commit |
|---|---|---|---|---|
| F0 | **Critical** (process) | The consensus-target gate wired in `76ee1f4` was silently LOST when later whole-file uploads (`bfc0b5e`, `ff0ecdc`) replaced `sqs2tdb_pipeline.py` with a pre-wiring version. Every "gated" run since then was actually ungated. | Wiring restored (gate block, per-phase endmember refs, CLI); verified end-to-end against a synthetic consensus. **Process rule going forward: never overwrite a diverged file wholesale — diff first.** | `28cdcc5` |
| F1 | **Critical** (physics) | No `ISPIN` in any non-DLM INCAR → Co/Cr/Ni/Fe/Mn computed **non-magnetic**. NM errors are tens of meV/atom — 10–30× the 1 meV/atom convergence tolerance the pipeline enforces. Formation energies and lattice stabilities built on them are not physical. | ISPIN=2 auto-enabled for magnetic-3d elements (non-DLM), `--no-spin`/`--spin` overrides, decision recorded in banner + manifest. | `2f4ff85` |
| F2 | **High** (physics) | ISIF=3 cell relaxations ran at the *energy*-converged ENCUT; the stress tensor (Pulay) converges far slower → systematically small volumes. | Relaxation ENCUT floored at 1.3×max(ENMAX) (`potcar.pulay_safe_encut`); statics/phonons keep the sweep value. | `036ad5c` |
| F3 | **High** (physics) | No check that an SQS *stayed on its parent lattice* after full relaxation. Mechanically unstable compositions slide to another lattice; their energy then enters the wrong phase's fit. ATAT's own workflow prescribes `checkrelax` ≲ 0.1; the automation skipped it. | numpy-free checkrelax analogue (`strfile.cell_distortion`, rotation/volume invariant); upstream writes `checkrelax.out` + `relaxaway.flag`, manifest records; downstream discovery rejects flagged/over-threshold SQS (`--max-checkrelax`, default 0.1). | `44d2e68` |
| F4 | **Medium** (methodology) | Convergence swept **per SQS** → structures within one phase carried different (ENCUT, KPPRA). Mixing energies subtract eV-scale totals; the subtraction only cancels basis/k-mesh error at *uniform* settings. Also ~N× wasted sweep compute. | `--convergence-scope phase` (new default): sweep once per phase, reuse for siblings; `sqs` restores old behaviour for diagnostics. | `036ad5c` |
| F5 | **Medium** (methodology) | (a) Consensus σ from cross-TDB spread **underestimates uncertainty where a phase is metastable** — assessments there often inherit the same SGTE lattice stabilities, so agreement is provenance, not evidence; a 3σ gate rejects correct DFT. (b) Gate rejections were silent — data exclusion without a record. | (a) `min_sigma_ev` floor (`--target-min-sigma`, default 10 meV/atom). (b) Every discovery rejection (gate z/target/σ, drift, OSZICAR, missing energy) recorded to `discovery_rejects.json`. Caveats added to `tdb_corpus/targets/README.md`. | `28cdcc5` |

Earlier in the same review conversation (previous session days), also
driven by source-verification rather than trust-the-wrapper:

| # | Finding | Commit |
|---|---|---|
| — | fitfc orchestration did not match `fitfc.c++` (wrong defaults `-er=11.5 -ns=3`, no `-si/-nrr`, `svib_ht` never promoted to where `sqs2tdb -fit` reads it) | `b3e6e7a` |
| — | No handling of `fitfc -f` unstable-mode aborts; stale `svib_ht` could be promoted after an aborted refit | `ec2a9b1` |
| — | Dynamically-unstable-SQS workflow automated (`escalate` policy: larger `-ernn` retry → genuine-instability classification) | `8518bac` |

## Open items (flagged, deliberately not auto-fixed)

- **O1 — Cross-phase settings consistency.** F4 unifies settings *within*
  a phase; different phases can still land on different (ENCUT, KPPRA).
  Lattice-stability comparisons across phases inherit that inconsistency.
  Candidate fix: binary-level scope taking the max over phases; needs a
  re-run of affected statics, so it's a planned-run decision, not a
  silent default change.
- **O2 — Initial-moment quality for Cr/Mn.** ISPIN=2 with VASP-default
  1 μB starts reliably finds FM Co/Ni/Fe but can trap Cr/Mn in
  low-moment minima. Proper fix is per-element MAGMOM through ezvasp's
  MAGATOM/SUBATOM tag machinery (as the DLM path already does); needs
  care because tags flow into downstream str.out parsers.
- **O3 — Final-static protocol.** Energies come from the relax-run
  DOSTATIC at ISMEAR=1/PREC=Normal/LREAL=Auto. For publication-grade
  numbers: separate static on the relaxed geometry with ISMEAR=-5 (mesh
  permitting), PREC=Accurate, LREAL=.FALSE. for <~30-atom cells.
- **O4 — Subset-selection bias (Stages 1–2).** Choosing SQS *subsets*
  to minimize RK fit error is data snooping: it discards valid DFT that
  disagrees with a low-order polynomial, biasing toward smoothness.
  With F5's records the exclusions are at least auditable; the cleaner
  estimator is keep-all-validated-data + model-order selection by CV.
- **O5 — Stage-3 scoring metric.** NP-matching on a T,x grid (and the
  pairwise mode) can rank a TDB well while a third phase wrongly
  intrudes, and misses invariant-reaction temperatures. Scoring on
  phase-boundary positions / invariant reactions would be the stronger
  metric; `--scoring-mode both` is the current mitigation.
- **O6 — svib targets vs magnetism.** TDB `*T` coefficients mix
  vibrational, electronic and magnetic entropy; the notebook must
  exclude the IHJ (TC/BMAGN) part before svib_ht targets become gates.
  Until then: soft guidance only (documented in targets/README.md).
- **O7 — fitfc displacement sensitivity.** `-dr` rides on fitfc's 0.2 Å
  default; harmonic force constants may carry anharmonic contamination.
  Worth a one-off dr ∈ {0.05, 0.1, 0.2} sensitivity check per phase.
- **O8 — No uncertainty propagation into the TDB.** Point estimates
  end-to-end; ESPEI/MCMC integration was deferred by an earlier project
  decision (corpus-first). Revisit after the ternary pipeline lands.
- **O9 — Endmember drift.** The `relaxaway` gate applies to mixing SQS
  in discovery; endmembers are only warned about (dropping one kills the
  phase). An endmember that genuinely won't stay on its lattice (e.g.
  FCC Cr) is *the* signal to use DLM or accept the SGTE lattice
  stability instead of a broken DFT number — human decision, so the
  pipeline surfaces it rather than deciding.
- **O10 — Epistemic status of the gated TDB.** With the consensus gate
  active, agreement with assessed phase diagrams is partly *by
  construction*. That is the project's stated intent
  (reverse-engineering), but any resulting database must be labeled
  semi-empirical, and an ungated control fit should be reported next to
  the gated one. `discovery_rejects.json` + this file are the audit
  trail that makes that comparison possible.

## How to audit a run after these changes

1. `upstream_manifest.json` — spin policy, convergence scope, per-SQS
   `checkrelax` / `relaxed_away` / `svib_ht_present` / `unstable_modes`.
2. `<sqs>/checkrelax.out`, `relaxaway.flag`, `unstable_modes.log` — per-
   structure physics flags with dispositions.
3. `<workdir>/discovery_rejects.json` — every SQS the fit never saw,
   with the reason and the gate numbers.
4. `fit_results.json` / `tdb_manifest.json` — surviving fits (existing).
