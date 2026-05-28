# Multicomponent TDB Automated Generator — Design Document

**Status:** Draft v0.1 (design only — no implementation yet)
**Scope:** Extend the binary A–B `sqs2tdb` automation pipeline to N-component
(ternary, quaternary, …) alloys, while leaving the validated binary pipeline
in `../TDB Automated Generator/` untouched.

---

## 0. TL;DR

The binary pipeline hardcodes "2 elements" in ~6 places. The clean way to go
multicomponent is **not** to rewrite those scripts in place, but to add a thin
orchestration layer on top that:

1. Enumerates the **lower-order subsystems** (all binary edges, all ternary
   faces, …) of an N-component system.
2. **Reuses the existing binary pipeline** to assess every binary edge
   (CALPHAD databases are built bottom-up — Muggianu/Kohler extrapolation from
   assessed binaries is the standard).
3. Adds a new **ternary-term fitting** capability for systems where binary
   extrapolation is insufficient.
4. Generalizes Stage-3 scoring from a 1-D composition axis to **isopleths /
   simplex sections**.
5. Replaces Stage-1's brute-force "all SQS subsets" search with an
   **escalating-model-complexity** search, because subset enumeration explodes
   combinatorially in the multicomponent SQS pool.

This leans on ATAT's native multicomponent support (`sqs2tdb`, `mcsqs`) rather
than reimplementing it.

---

## 1. Background: what the binary pipeline does and where it is binary-locked

The existing three-stage pipeline (`../TDB Automated Generator/`):

| Script | Role |
|---|---|
| `select_endmembers.py` | Pick the pure-element endmember SQS per phase → `endmembers.yaml` |
| `sqs2tdb_pipeline.py` | Stage 1 (energy fits) + Stage 2 (svib_ht) → `tdb_manifest.json` |
| `score_tdb_combinations.py` | Stage 3 — combine per-phase TDBs, score vs reference |

**Binary-locked assumptions to generalize:**

1. `binary: A-B` / `el1, el2 = binary.split("-")` — exactly two elements.
2. `select_endmembers.py` prompts for an **A-rich** and **B-rich** endmember —
   exactly two corners per single-sublattice phase.
3. `write_species_mult()` writes `site=El1,El2` — two species per sublattice.
4. `write_terms()` writes `1,0\n2,{order}\n` — a single binary Redlich–Kister
   (RK) series, no ternary terms.
5. `gen_stage1_tasks()` enumerates **every SQS subset** × `order ∈ {0,1,2}`;
   overfit guard uses `n_params = order + 3` (the binary single-sublattice
   count).
6. `score_tdb_combinations.py` scores along **one** composition axis
   (`v.X(comp_el)`); the boundary penalty is 1-D.

**Already-general machinery worth keeping:**
- `robust_copytree()` (resolves ATAT/VASP symlinks — still needed).
- The isolated per-task working-dir pattern + `ProcessPoolExecutor`.
- The svib_ht discovery / removal logic.
- The SIGMA multi-sublattice handling (colon-joined `terms.in`, per-site
  `species.in`/`mult.in`) — this is *already* the multi-sublattice
  generalization and is the template for the rest.

---

## 2. ATAT multicomponent capabilities (grounded in the manual + papers)

- **`sqs2tdb` and `mcsqs` are fully multicomponent / multi-sublattice.** mcsqs
  is "implemented in the most general framework of multicomponent
  multisublattice systems" (van de Walle, Calphad 42 (2013) 13). The sqsdb
  database covers 30+ multi-sublattice structures.
- **`terms.in` grammar (verified):** each line is `order,level`:
  - `order` = interaction order — `1` = endmember/linear, `2` = binary,
    `3` = ternary, …
  - `level` = Redlich–Kister polynomial degree for that order.
  - **Sublattices are colon-joined** on a single line
    (e.g. `1,0:1,0:1,0` = endmember term on 3 sublattices). The binary
    pipeline's `SIGMA_TERMS = "1,0:1,0:1,0\n2,0:2,0:2,0\n"` is exactly this.
- **`species.in`** can list all N elements per sublattice (`El1,El2,…,ElN`).
- **Composition mesh / `lev`:** `-lv=N` controls mesh fineness on each
  sublattice. `lev=0` = simplex corners (endmembers), `lev=1` = edge midpoints,
  higher = finer interior mesh. Directory names encode per-site occupations:
  `sqs_lev=N_<site>_<El>=<conc>,...` (confirmed prefix: `sqs_lev=`).
- **Useful flags the binary pipeline does not yet use:**
  - `-ew=[real]` (default 5): endmember weight in the fit — lets us fit *all*
    SQS at once while anchoring endmembers, instead of subset-pruning.
  - `-sro`: CVM short-range-order correction (Samanta & van de Walle, JPED 2024).
  - `-tdb -oc`: Open-Calphad-portable output.
  - `func` files: supply full F(T) instead of `energy` + `svib_ht`.
- **High-throughput / validation tools:** `pollmach` + `robustrelax_vasp`
  (DFT orchestration), `fitfc`/`felec` (free-energy terms), and
  `emc2` + `phb` (Monte-Carlo / phase-boundary tracing — an *independent*
  check on the fitted TDB's phase diagram).

> **To verify against the installed `sqs2tdb` when implementing:** exactly how
> `-fit` partitions binary-edge vs interior SQS across `2,L`/`3,L` terms, and
> the precise per-sublattice delimiter in `species.in` for >2 elements.

---

## 3. Key theoretical decision: subsystem extrapolation vs. full-simplex fit

CALPHAD solution models for N components are conventionally **built from
assessed lower-order subsystems** and extrapolated into the interior
(Muggianu/Kohler). The Gibbs energy of a single-sublattice solution phase:

```
G = Σ_i x_i °G_i                                  (endmembers / order 1)
  + Σ_{i<j} x_i x_j Σ_ν L^ν_ij (x_i − x_j)^ν       (binary RK / order 2)
  + Σ_{i<j<k} x_i x_j x_k L_ijk                     (ternary / order 3)
  + ... + ideal & magnetic terms
```

Two implementation routes:

- **(A) Full-simplex fit:** one `species.in` with all N elements, SQS spanning
  the whole simplex, a single `sqs2tdb -fit` with `2,L` (all binaries) +
  `3,L` (all ternaries) terms.
- **(B) Subsystem extrapolation (recommended primary path):** assess each
  binary edge independently (reuse the binary pipeline verbatim), then fit
  ternary corrections only where binary extrapolation misses, then combine.

**Recommendation: a hybrid that defaults to (B).** Reasons:
- Directly **reuses the validated binary work** (the user's explicit goal).
- Controls combinatorial blow-up (Section 6) — each binary is small.
- Matches how `sqs2tdb -tdb` combines per-system models.
- (A) remains available as a `--full-simplex` mode for small systems or when
  strong ternary interactions are expected from the start.

---

## 4. Data model

```yaml
# system.yaml  (generalizes endmembers.yaml)
system: Co-Cr-Ni            # N elements, sorted
elements: [Co, Cr, Ni]
phases:
  FCC_A1:
    sublattices: [{site: a, mult: 1, species: [Co, Cr, Ni]}]
    endmembers:             # one per simplex corner (pure element)
      Co: /path/sqs_lev=0_a_Co=1
      Cr: /path/sqs_lev=0_a_Cr=1
      Ni: /path/sqs_lev=0_a_Ni=1
  HCP_A3:
    sublattices: [{site: c, mult: 2, species: [Co, Cr, Ni]}]
    endmembers: {...}
  SIGMA_D8B:
    sublattices:            # confirmed from rndstr.skel: 10/4/16 = 30 atoms
      - {site: aj, mult: 10, species: [Co, Cr, Ni]}
      - {site: g,  mult: 4,  species: [Co, Cr, Ni]}
      - {site: ii, mult: 16, species: [Co, Cr, Ni]}
    endmembers: {ALL: [...]}   # config-keyed, as today
subsystems:                 # auto-derived, drives the pipeline
  binary:  [Co-Cr, Co-Ni, Cr-Ni]
  ternary: [Co-Cr-Ni]
```

Composition becomes an **N-vector** (point in the (N−1)-simplex) instead of a
scalar `x1`. `SQSData.x1/x2` → `SQSData.comp: Dict[str, float]`. Deduplication
keys on the rounded composition tuple over all elements.

---

## 5. Stage 0 — endmember & subsystem selection (`select_endmembers_mc.py`)

- Scan as today, but parse **N-element** compositions from directory names
  (the existing `re.findall(r"([a-z]+)_([A-Za-z]+)=([0-9.]+)", ...)` already
  returns all site/element/value tokens — just stop assuming two).
- For single-sublattice phases, require **one endmember per element** (N corners,
  not 2). Interactive selection loops over elements.
- Derive the subsystem list (all C(N,2) binaries, C(N,3) ternaries, …) and
  record which binary-edge / interior SQS exist for each.
- Emit `system.yaml` (Section 4).

## 6. `terms.in` generalization and the combinatorial problem (the crux)

**Generalized parameter count** (single sublattice, K species on it):

```
n_params = K                                   # endmembers (order 1)
         + C(K,2) * (L2 + 1)                    # binary RK, degree L2
         + C(K,3) * (L3 + 1)                    # ternary,  degree L3
         + ...
```

For multi-sublattice phases this is summed/combined across sublattices (the
colon-joined `terms.in` lines), and the endmember count is the product of
per-sublattice species counts. The overfit guard becomes
`n_data > n_params` using this generalized count (replacing `order + 3`).

**Why brute-force subset enumeration must go:** Stage 1 currently tries every
SQS subset of size `min..max`. For a binary that's `Σ C(≤7, k)` — fine. For a
ternary the SQS pool (3 binary edges × a few comps + interior points) is much
larger and `C(pool, k)` explodes. **Replacement strategy:**

1. Fit **all** valid SQS in one shot, using `-ew` to anchor endmembers
   (no subset search).
2. Search only over **model complexity**: escalate `terms.in` from
   `{2,0}` → `{2,1}` → `{2,2}` (binary RK degree), and independently decide
   whether to add `3,0`/`3,1` (ternary) terms.
3. Accept the lowest-complexity model whose max |error| ≤ cutoff (parsimony /
   AIC-like preference), subject to the overfit guard.

This turns an exponential search into a small, ordered ladder per phase.

## 7. SQS discovery in the simplex (`discover_sqs` generalization)

- Parse the full N-element occupation; tag each SQS by which subsystem it
  populates (binary edge `i-j` if only two elements present, ternary face if
  three, …). This tagging lets Route (B) feed edge SQS to binary fits and
  interior SQS to ternary corrections.
- Dedup on the rounded N-tuple per sublattice.
- Keep the symlink-resolving copy and `svib_ht` discovery unchanged; also honor
  ATAT **`link`** files (symmetry-equivalent redirection) — the binary code's
  symlink resolution mostly covers this but should be made explicit.

## 8. Stage 2 — svib_ht: unchanged in spirit

Endmember svib always included; per-SQS svib subsets explored. The only change
is that "endmembers" is now N corners (and, for multi-sublattice, the product
set). Same pruning on `fit_svib_ht.out`.

## 9. Stage 3 — multicomponent scoring (`score_tdb_combinations_mc.py`)

The hard part: a ternary equilibrium grid is 2-D in composition, quaternary 3-D.

- **Conditions:** replace the single `v.X(comp_el)` with N−1 composition
  conditions, or **fixed isopleths/sections** (recommended) — e.g. constant
  X(Ni) lines, or vertices→edge sections — to keep equilibrium cost bounded.
- **Phase-fraction array:** `build_phase_fraction_array` generalizes directly
  (it already operates on whatever dims pycalphad returns).
- **Boundary penalty:** `boundary_misplacement_penalty` currently infers a
  single `X_*` dim. Generalize to compute boundary indicators per section/line;
  for a full 2-D grid this is a boundary-curve distance (expensive) — prefer the
  isopleth route first.
- **Reference equilibrium:** keep the just-fixed "compute once, reuse" pattern.
- **Combinatorics of cross-phase combos** is unchanged (Cartesian product of
  per-phase survivors); `--max-combos` sampling still applies.
- **Independent validation:** optionally cross-check the winning TDB's phase
  diagram against ATAT `emc2`/`phb` for the same system.

## 10. Combinatorial scaling — summary & mitigations

| Quantity | Binary | N-component (single sublattice, K=N) |
|---|---|---|
| Endmembers / corners | 2 | N |
| Binary subsystems | 1 | C(N,2) |
| Ternary subsystems | 0 | C(N,3) |
| `n_params` | order+3 | K + ΣC(K,r)(Lr+1) |
| Stage-1 search | subsets × 3 | **complexity ladder** (Section 6) |

Mitigations: subsystem decomposition (Route B), escalating-complexity search,
`-ew` weighting instead of subset pruning, `--max-combos` sampling, and
parsimony selection.

## 11. Proposed file layout (this directory)

```
TDB Automated Generator Multicomponent/
  DESIGN.md                       # this document
  select_endmembers_mc.py         # Stage 0  (N corners + subsystem map)
  sqs2tdb_pipeline_mc.py          # Stage 1/2 (complexity ladder, N-element terms.in)
  score_tdb_combinations_mc.py    # Stage 3  (isopleth/section scoring)
  subsystems.py                   # shared: enumerate binaries/ternaries, parse N-comp
  README.md
```

Shared, phase-agnostic helpers (`robust_copytree`, `find_svib_ht`,
`element_case`, fit-file parsing) should be lifted into `subsystems.py` (or a
small `common.py`) and imported by both the binary and multicomponent scripts,
so logic isn't duplicated. The binary scripts stay where they are; we only
*import from* them or copy the helpers — no edits to the binary directory.

## 12. Validation plan

1. **Sanity:** run the MC pipeline on a binary (N=2) and confirm it reproduces
   the binary pipeline's TDB bit-for-bit (regression guard).
2. **Ternary test case:** Co-Cr-Ni (FCC/HCP/SIGMA) end-to-end on the full ATAT
   setup — the user has confirmed access.
3. **Cross-check:** compare the scored phase diagram against a reference TDB and,
   independently, against `emc2`/`phb`.

## 13. Open questions / to confirm during implementation

- Exact `sqs2tdb -fit` behavior when `species.in` has >2 elements and
  `terms.in` mixes `2,L` and `3,L` lines (does it auto-expand all pairs/triples,
  or must each be listed?). **Verify on the installed version.**
- Per-sublattice species delimiter in `species.in` for multi-element sites.
- Whether `func` files should replace `energy`+`svib_ht` for the MC workflow
  (cleaner F(T), but changes the data-prep step).
- Selection criterion for "binary extrapolation insufficient → add ternary
  terms" (e.g. residual at interior SQS above cutoff).

## 14. Phased roadmap

- **P0 (this doc):** design sign-off.
- **P1:** `subsystems.py` + `select_endmembers_mc.py` (N corners, subsystem map,
  `system.yaml`). Pure parsing/IO — testable without DFT.
- **P2:** `sqs2tdb_pipeline_mc.py` Route B (orchestrate binary pipeline per edge)
  + generalized `terms.in`/param-count + complexity ladder.
- **P3:** ternary-term fitting (interior SQS) and `--full-simplex` Route A.
- **P4:** `score_tdb_combinations_mc.py` isopleth scoring + `emc2`/`phb` check.
- **P5:** Co-Cr-Ni validation, docs, regression test vs binary pipeline.
