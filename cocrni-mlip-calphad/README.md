# Co-Cr-Ni ternary phase diagram from first principles
## Magnetic MTP + universal GNN + ATAT → TDB → pycalphad

A reproducible pipeline for constructing the Co-Cr-Ni ternary phase diagram with **no experimental thermochemistry as input**. The energetics come from spin-polarized VASP + soft-constrained DFT, learned by a magnetic Moment Tensor Potential (mMTP) and a fine-tuned universal GNN. The magnetic ordering contribution to the Gibbs energy comes from your refined T_C/T_N and β(x) pipeline via the Inden–Hillert–Jarl (IHJ) model. Equilibrium phase diagrams are computed with pycalphad from the generated TDB.

---

## 0. Why this design

| Choice | Rationale |
|---|---|
| **mMTP** (Burov et al. 2026, spin-MLIP) | Explicit collinear m_i in the descriptor → captures FM(Co)/AFM(Cr)/PM(Ni) and excited-moment energetics. Active learning with soft-constrained VASP keeps training-set cost bounded. |
| **Fine-tuned universal GNN** (MACE or ORB, Zhu et al. 2025) | Cheap 0 K formation-energy screening across the full FCC/HCP/BCC/σ composition grid. Orders of magnitude faster than DFT, ~10 meV/atom accuracy. Used to *seed* mMTP and to compute config-averaged enthalpies of mixing on dense grids. |
| **SQS + collinear DLM** | The Co-Cr FCC and HCP solid solutions are paramagnetic at the temperatures where the phase diagram matters. ±m_i randomized over SQS, averaged → exchange-correlation-consistent paramagnetic enthalpy. mMTP is built to handle the resulting non-equilibrium m_i. |
| **ATAT mcsqs → MLIP free energy → Redlich–Kister fit → TDB** | Follows Zhu et al. exactly. Avoids manually picking interaction parameters; RK fit is least-squares over a dense composition grid. |
| **IHJ magnetic Gibbs term in TDB** | Your refined β(x) and T_C/T_N(x) become explicit parameters in `TYPE_DEFINITION ... PARA ...`. Keeps the magnetic order-disorder contribution thermodynamically self-consistent. |
| **pycalphad for plotting** | Open, scriptable, gives full ternary sections + isopleths + phase fractions. |

---

## 1. System-specific facts you should bake in

**Phases to include.** At minimum:

- `FCC_A1` (disordered solid solution; ground state of Ni and γ-Co, the matrix of CrCoNi MEAs)
- `HCP_A3` (ε-Co, Co-rich corner)
- `BCC_A2` (α-Cr, AFM)
- `SIGMA` (D8b, CoCr — the one that will give you grief)
- `LIQUID` (regular-solution model from MD-derived G_liq via thermodynamic integration)

Optional but recommended:

- `CHI` if you find evidence in your DFT screening (Cr-rich, low-T metastable)

**Magnetism per phase, baseline expectations.**

| Phase | Ground-state magnetism | DLM needed? |
|---|---|---|
| BCC Cr | AFM (SDW; collinear AFM is the standard CALPHAD approximation) | Yes, above T_N(Cr) ~311 K |
| FCC Ni | FM, T_C ~627 K | Yes, above T_C |
| FCC Co | FM, T_C ~1394 K (metastable above HCP→FCC at ~700 K) | Yes |
| HCP Co | FM, T_C ~1394 K | Yes |
| FCC CrCoNi solid solution | competing PM/AFM correlations | **Always DLM** |
| σ-CoCr | site-dependent moments | DLM on sublattices |

**Composition grid.** Δx = 0.05 across the Gibbs triangle = 231 ternary compositions per phase. mMTP fine-tunes for 2 days on the highest-energy 5% by GNN uncertainty.

---

## 2. Pipeline overview

```
                  ┌────────────────────────────────────────────────────────────┐
                  │  YOUR pipeline: refined collinear ordered/disordered m_i,  │
                  │  T_C(x), T_N(x), β(x) per phase                            │
                  └────────────────┬───────────────────────────────────────────┘
                                   │ (β, T_C/T_N feed IHJ in step 6)
                                   ▼
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │ 01_sqs   │──▶│ 02_dft   │──▶│ 03_mmtp  │──▶│ 05_freeE │──▶│ 06_tdb   │──▶ pycalphad ternary
 │ ATAT     │   │ VASP +   │   │ spin-    │   │ MD + TI  │   │ RK fit + │     plots
 │ + DLM    │   │ softcons │   │ MLIP AL  │   │ on mMTP  │   │ IHJ      │
 └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                     │              ▲
                     │              │ (seeds training set, screening grid)
                     ▼              │
                  ┌──────────┐      │
                  │ 04_gnn   │──────┘
                  │ MACE FT  │
                  └──────────┘
```

---

## 3. Step-by-step, with the script that runs it

### Step 1 — SQS + DLM configurations (`01_sqs/generate_sqs_dlm.py`)

For each phase × composition grid point:

1. Run `mcsqs` (ATAT) to get a 32-, 64-, or 108-atom SQS matching pair and triplet correlations out to 3rd nearest neighbor.
2. Generate N_DLM = 8 collinear ±m_i realizations per SQS (per-element sign assignment, charge-neutral within tolerance).
3. Generate "ordered" (FM-aligned) variants for reference.

This step uses ATAT externally; the script wraps it and emits VASP `POSCAR` + initial `MAGMOM` strings.

### Step 2 — Spin-polarized + soft-constrained DFT (`02_dft/vasp_constrained_driver.py`)

Two calculation modes per config:

- **Equilibrium (ordered + DLM-realization)**: standard collinear `ISPIN=2`, gets E_DFT, F, σ for the ground-state moments of that arrangement.
- **Soft-constrained on selected configs from active learning**: forces non-equilibrium m_i targets requested by the mMTP AL selector. Follows the Burov SI scheme exactly (Wigner-Seitz rescale, linear MWint↔Mint fit, λ ladder, refinement).

VASP INCAR templates in `02_dft/incar_templates/`.

### Step 3 — mMTP active learning (`03_mmtp/train_and_al.py`)

1. Pretrain a level-12 mMTP (N_ψ=2, N_φ=8) on the Step 2 equilibrium set.
2. Run NPT MD with LAMMPS + spin-MLIP interface across the composition grid and a T-range covering the phases of interest (300–2200 K).
3. Configurations with extrapolation grade γ ∈ (3, 5) are selected by maxvol, calculated with soft-constrained DFT (Step 2 mode 2), added to training set, retrain.
4. Stop when an AL pass yields zero selections.

Target training errors: ≤2 meV/atom (E), ≤120 meV/Å (F), ≤0.5 GPa (σ).

### Step 4 — GNN fine-tuning (`04_gnn/finetune_mace_magnetic.py`)

MACE foundation model (mp_0 or mace_mp_0a) fine-tuned on the same training set, with magnetic moments injected as **node attributes** (the cleanest way to make MACE magnetism-aware without modifying its architecture). Loss includes `m_i` MAE as a regularizer.

This GNN is used in Step 5 to:
- Screen formation energies across a dense composition grid before deciding which compositions to actually simulate.
- Provide an independent cross-check on mMTP — if they disagree by more than the larger error bar, that composition gets re-flagged for AL.

### Step 5 — Free energies on the composition grid (`05_free_energy/compute_free_energy.py`)

Per (phase, composition, T_range):

1. **Vibrational + configurational free energy** via thermodynamic integration in LAMMPS using mMTP:
   - λ-coupling from Einstein crystal reference (Frenkel-Ladd) → ideal-gas reference for liquid.
   - 8 DLM realizations averaged for paramagnetic state.
2. **Quasi-harmonic** check on selected compositions (phonopy + mMTP) as a sanity check on TI in the low-T regime.
3. Outputs G_phase(x_Co, x_Cr, T) on a (composition × temperature) grid.

### Step 6 — Redlich-Kister fit + IHJ magnetic term + TDB (`06_calphad/fit_rk_and_tdb.py`)

For each phase:

1. Fit G_mix(x, T) = ideal mixing + sum over endmembers + sum over binaries (L^0, L^1, L^2 Redlich-Kister) + ternary L^123.
2. Compose magnetic Gibbs contribution G_mag = R T ln(β + 1) f(τ) with **β(x), T_C/T_N(x) from your pipeline**, interpolated via Redlich-Kister-like polynomials of composition.
3. Emit a `.tdb` file with proper `TYPE_DEFINITION` magnetic block, all `PARAMETER` lines, and `LIST_OF_REFERENCES` pointing to your DFT/mMTP provenance (no SGTE).

### Step 7 — Plot (`07_plot/plot_ternary.py`)

- Isothermal sections at 600, 800, 1000, 1300, 1600 K.
- Liquidus projection.
- Optional: phase fraction vs T at selected ternary compositions (e.g., equiatomic CrCoNi).

---

## 4. Run order

```bash
# One-time
conda env create -f 00_config/environment.yml
conda activate cocrni
# (install spin-MLIP from gitlab.com/ivannovikov/spin-mlip, ATAT, LAMMPS w/ MLIP-LAMMPS interface)

# Per-phase loop (FCC_A1 first; then HCP_A3, BCC_A2, SIGMA, LIQUID)
python 01_sqs/generate_sqs_dlm.py --phase FCC_A1 --grid-step 0.05
python 02_dft/vasp_constrained_driver.py --inputs 01_sqs/out/FCC_A1/ --mode equilibrium
python 03_mmtp/train_and_al.py --pretrain --inputs 02_dft/out/FCC_A1/
python 03_mmtp/train_and_al.py --active-learn --max-iters 12
python 04_gnn/finetune_mace_magnetic.py --train 03_mmtp/training_set.cfg
python 05_free_energy/compute_free_energy.py --phase FCC_A1 --T 300:2200:50

# After all phases:
python 06_calphad/fit_rk_and_tdb.py --phases FCC_A1,HCP_A3,BCC_A2,SIGMA,LIQUID \
    --magnetic-params 00_config/magnetic_params.yaml \
    --output cocrni.tdb
python 07_plot/plot_ternary.py --tdb cocrni.tdb --T 1000
```

---

## 5. Known traps to budget for

1. **SGTE-free unary lattice stabilities.** You need G_Cr^FCC, G_Co^BCC, etc. These are mechanically unstable for some elements (FCC Cr has imaginary phonons). Get them from: small-volume strain extrapolation, ZPE corrected, with finite-T from QHA only down to where γ_modes go negative; *then* extrapolate using your mMTP-MD G(T) where stable. Document the protocol — this is the area most likely to be challenged in review.

2. **σ-CoCr.** 30-atom unit cell, 5 inequivalent sites. mMTP needs *explicit* σ configs in training. Generate ordered + DLM σ structures in Step 1 and run them. Don't rely on AL to find σ — it won't.

3. **Cr antiferromagnetism is incommensurate (SDW).** Collinear AFM-(001) is the standard CALPHAD approximation but you'll get a slightly wrong G_BCC for pure Cr. Either accept the ~5 meV/atom error or include a small non-collinear correction term (out of mMTP scope, easy in DFT).

4. **DLM averaging convergence.** 8 realizations per composition is a starting point. Check std-dev of E across realizations; if > 5 meV/atom, increase to 16. Co-Cr-Ni FCC at equiatomic is well-behaved here; Cr-rich BCC less so.

5. **GNN-mMTP disagreement diagnostic.** Don't trust either model alone in regions where they disagree by > 2× max(σ_GNN, σ_mMTP). Those points get re-flagged for AL.

6. **Liquid phase.** TI from Einstein → ideal-gas reference is two stages and slow. Budget ~50 LAMMPS runs per liquid composition. Or use the Bocklund-style two-state approach in ESPEI as a fallback.

---

## 6. Where your magnetic-parameter pipeline plugs in

The IHJ magnetic Gibbs term needs two phase-and-composition-dependent quantities:

- **β(x)** — mean magnetic moment per mole of formula unit
- **T*(x)** — Curie temperature (FM phases) or Néel temperature (AFM phases)

You already produce both from your refined collinear ordered/disordered moment work. The interface is `00_config/magnetic_params.yaml` (template included). Step 6 reads it and writes the `TYPE_DEFINITION` magnetic block and per-phase `PARAMETER TC` and `PARAMETER BMAGN` entries. If your pipeline gives full β_i(x) per site for the σ phase, the script will emit sublattice-resolved magnetic parameters.

Worth noting: this is one of the few places where your work is genuinely additive to what the three papers already do. Burov et al.'s mMTP gives you m_i locally but doesn't separately resolve the long-range ordering temperature; Zhu et al. and Shen review don't address magnetism in TDBs at all. Your refined magnetic params + their MLIP/CALPHAD machinery is a defensible "fully first-principles" loop.
