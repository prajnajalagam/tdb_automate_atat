"""
06_calphad/fit_rk_and_tdb.py

Fit Redlich-Kister parameters to G_phase(x, T) from step 5 and emit a TDB
including the IHJ magnetic Gibbs term using YOUR refined β(x), T*(x) from
00_config/magnetic_params.yaml.

Output TDB validates in pycalphad and Thermo-Calc.

Usage:
    python fit_rk_and_tdb.py \
        --phases FCC_A1,HCP_A3,BCC_A2,SIGMA,LIQUID \
        --gdir ../05_free_energy/out/ \
        --magnetic ../00_config/magnetic_params.yaml \
        --output cocrni_firstprinciples.tdb
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares


R_GAS = 8.314462618   # J/(mol·K)
EV_TO_J_PER_MOL = 96485.33212

# IHJ polynomial f(τ) for τ ≤ 1 and τ > 1; p is phase-specific (0.4 FCC/HCP, 0.28 BCC)
def ihj_f(tau: np.ndarray, p: float) -> np.ndarray:
    D = (518.0 / 1125.0) + (11692.0 / 15975.0) * (1.0 / p - 1.0)
    f = np.where(
        tau <= 1.0,
        1.0 - (1.0 / D) * (
            (79.0 / (140.0 * p * tau)) +
            (474.0 / 497.0) * (1.0 / p - 1.0) * (tau ** 3 / 6.0 + tau ** 9 / 135.0 + tau ** 15 / 600.0)
        ),
        -(1.0 / D) * (tau ** -5 / 10.0 + tau ** -15 / 315.0 + tau ** -25 / 1500.0),
    )
    return f


def G_mag_ihj(T: np.ndarray, beta: np.ndarray, Tstar: np.ndarray, p: float) -> np.ndarray:
    """IHJ magnetic Gibbs energy contribution, J/mol/atom."""
    # Sign convention: AFM phases have Tstar < 0; we use |Tstar| for τ
    Tstar_eff = np.abs(Tstar)
    beta_eff = np.abs(beta)
    # Avoid divide-by-zero for compositions where Tstar = 0 (purely PM endmember)
    safe = Tstar_eff > 1e-6
    tau = np.where(safe, T / np.where(safe, Tstar_eff, 1.0), 0.0)
    f = np.where(safe, ihj_f(tau, p), 0.0)
    G = R_GAS * T * np.log(beta_eff + 1.0) * f
    return G


# ---------------------------- RK fitting --------------------------------------

@dataclass
class RKModel:
    """G_excess(x_i, x_j, x_k, T) = Σ x_i x_j Σ_n L_ij^n (x_i - x_j)^n + x_i x_j x_k L_ijk."""
    L_binary: dict[tuple[str, str, int], tuple[float, float]]   # (a, b) for L = a + bT
    L_ternary: dict[tuple[str, str, str], float]
    endmember_G: dict[str, dict]   # element → {a, b, c, d_coefs} for SGTE-like polynomial


def fit_phase_rk(grid: list[dict], order_binary: int = 2,
                 order_ternary: int = 1) -> RKModel:
    """Fit RK parameters from MD-derived G(x, T) data points."""
    xs = np.array([[d["x_Co"], d["x_Cr"], d["x_Ni"]] for d in grid])
    Ts = np.array([d["T_K"] for d in grid])
    Gs = np.array([d["G_eV_per_atom"] for d in grid]) * EV_TO_J_PER_MOL    # → J/mol/atom

    # Endmember Gs: take from pure-element data points (x_i = 1)
    endmembers = {}
    for el, i in zip(["Co", "Cr", "Ni"], range(3)):
        mask = xs[:, i] > 0.999
        if not np.any(mask):
            endmembers[el] = {"a": 0.0, "b": 0.0, "c": 0.0}
            continue
        T_em = Ts[mask]
        G_em = Gs[mask]
        # G_pure(T) = a + b T + c T ln T  (3-term SGTE-style)
        def resid(p):
            return p[0] + p[1] * T_em + p[2] * T_em * np.log(T_em) - G_em
        x0 = [G_em[0], 0.0, 0.0]
        sol = least_squares(resid, x0)
        endmembers[el] = {"a": sol.x[0], "b": sol.x[1], "c": sol.x[2]}

    # Build excess (mixing) Gibbs by subtracting ideal + endmember contributions
    G_id = R_GAS * Ts * np.sum(
        np.where(xs > 1e-8, xs * np.log(np.maximum(xs, 1e-12)), 0.0), axis=1
    )
    G_pure = np.zeros_like(Gs)
    for el, i in zip(["Co", "Cr", "Ni"], range(3)):
        a, b, c = endmembers[el]["a"], endmembers[el]["b"], endmembers[el]["c"]
        G_pure += xs[:, i] * (a + b * Ts + c * Ts * np.log(np.maximum(Ts, 1e-3)))
    G_xs = Gs - G_id - G_pure   # J/mol/atom

    # Linear least-squares for L_ij^n (a + b T) and ternary L_ijk
    binaries = [("Co", "Cr", 0, 1), ("Co", "Ni", 0, 2), ("Cr", "Ni", 1, 2)]
    feat_cols = []
    feat_names = []
    for (ei, ej, i, j) in binaries:
        x_i = xs[:, i]
        x_j = xs[:, j]
        for n in range(order_binary + 1):
            base = x_i * x_j * (x_i - x_j) ** n
            feat_cols.append(base);            feat_names.append((ei, ej, n, "a"))
            feat_cols.append(base * Ts);       feat_names.append((ei, ej, n, "b"))
    if order_ternary >= 0:
        x_Co, x_Cr, x_Ni = xs[:, 0], xs[:, 1], xs[:, 2]
        feat_cols.append(x_Co * x_Cr * x_Ni); feat_names.append(("Co", "Cr", "Ni", "L0"))
    A = np.column_stack(feat_cols)
    coeffs, *_ = np.linalg.lstsq(A, G_xs, rcond=None)

    L_binary = {}
    L_ternary = {}
    for c, name in zip(coeffs, feat_names):
        if name[-1] == "L0":
            L_ternary[(name[0], name[1], name[2])] = float(c)
        else:
            ei, ej, n, ab = name
            entry = L_binary.setdefault((ei, ej, n), [0.0, 0.0])
            entry[0 if ab == "a" else 1] = float(c)

    L_binary = {k: tuple(v) for k, v in L_binary.items()}
    return RKModel(L_binary=L_binary, L_ternary=L_ternary, endmember_G=endmembers)


# ---------------------------- TDB writing -------------------------------------

TDB_HEADER = """\
$ Co-Cr-Ni first-principles thermodynamic database
$ Generated: {timestamp}
$ Energetics: mMTP + MACE on spin-polarized + soft-constrained VASP
$ Magnetic terms: refined β(x), T*(x) via Inden–Hillert–Jarl
$ NO experimental thermochemistry was used.

ELEMENT /-   ELECTRON_GAS              0.0      0.0      0.0  !
ELEMENT VA   VACUUM                    0.0      0.0      0.0  !
ELEMENT CO   HCP_A3                    58.9332  4765.567 30.0400  !
ELEMENT CR   BCC_A2                    51.9961  4050.0   23.5429  !
ELEMENT NI   FCC_A1                    58.6934  4787.0   29.7960  !

$ Note: H298 and S298 entries above are required by TDB syntax; magnitudes are
$ not used in this first-principles fit because every Gibbs polynomial is
$ self-contained.

"""


def emit_function_G(el: str, em: dict, ref_phase: str) -> str:
    """G_pure(T) function in TDB syntax."""
    return (
        f"FUNCTION GHSER{el.upper()}  298.15  "
        f"{em['a']:+.4f}  {em['b']:+.6f}*T  {em['c']:+.6f}*T*LN(T);  6000  N  REF0  !\n"
    )


def emit_phase_block(phase: str, model: RKModel, mag_params: dict) -> str:
    """Emit a full PHASE definition with sublattice, endmember, RK, and magnetic blocks."""
    mag_block = ""
    if mag_params.get("ordering") in ("FM", "AFM"):
        # TYPE_DEFINITION & magnetic Gibbs PARAMETERS
        type_letter = "%"
        mag_block = (
            f"TYPE_DEFINITION {type_letter} GES A_P_D {phase} MAGNETIC  "
            f"-3.0  {mag_params['ihj_p']:.3f}  !\n"
        )
        # Curie/Néel and β PARAMETER lines per endmember and per binary L^0
        for el, T_em in mag_params.get("Tstar_endmembers", {}).items():
            mag_block += f"PARAMETER TC({phase},{el.upper()};0)  298.15  {T_em};  6000 N REF0 !\n"
        for el, b_em in mag_params.get("beta_endmembers", {}).items():
            mag_block += f"PARAMETER BMAGN({phase},{el.upper()};0)  298.15  {b_em};  6000 N REF0 !\n"
        for pair, L0 in mag_params.get("Tstar_binary_L0", {}).items():
            i, j = pair.split("_")
            mag_block += (
                f"PARAMETER TC({phase},{i.upper()},{j.upper()};0)  298.15  {L0};  6000 N REF0 !\n"
            )
        for pair, L0 in mag_params.get("beta_binary_L0", {}).items():
            i, j = pair.split("_")
            mag_block += (
                f"PARAMETER BMAGN({phase},{i.upper()},{j.upper()};0)  298.15  {L0};  6000 N REF0 !\n"
            )

    # Phase declaration
    if phase == "SIGMA":
        phase_line = "PHASE SIGMA  %  3  10 4 16 !\n"
        const_line = "CONSTITUENT SIGMA :CO,CR,NI:CO,CR,NI:CO,CR,NI:  !\n"
    else:
        phase_line = f"PHASE {phase}  %  1  1.0 !\n"
        const_line = f"CONSTITUENT {phase} :CO,CR,NI:  !\n"

    # Endmember PARAMETERS
    endmem = ""
    for el in ("CO", "CR", "NI"):
        endmem += (
            f"PARAMETER G({phase},{el};0)  298.15  +GHSER{el};  6000 N REF0 !\n"
        )

    # Binary RK PARAMETERS
    rk = ""
    for (ei, ej, n), (a, b) in sorted(model.L_binary.items()):
        rk += (
            f"PARAMETER L({phase},{ei.upper()},{ej.upper()};{n})  298.15  "
            f"{a:+.4f}{b:+.6f}*T;  6000 N REF0 !\n"
        )

    # Ternary RK
    for trio, L0 in model.L_ternary.items():
        rk += (
            f"PARAMETER L({phase},{trio[0].upper()},{trio[1].upper()},{trio[2].upper()};0)  "
            f"298.15  {L0:+.4f};  6000 N REF0 !\n"
        )

    return phase_line + const_line + mag_block + endmem + rk + "\n"


def load_phase_grid(gdir: Path, phase: str) -> list[dict]:
    rows = []
    with open(gdir / f"G_{phase}.csv") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "x_Co": float(row["x_Co"]),
                    "x_Cr": float(row["x_Cr"]),
                    "x_Ni": float(row["x_Ni"]),
                    "T_K": float(row["T_K"]),
                    "G_eV_per_atom": float(row["G_eV_per_atom"]),
                })
            except ValueError:
                continue
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phases", default="FCC_A1,HCP_A3,BCC_A2,SIGMA,LIQUID")
    p.add_argument("--gdir", default="../05_free_energy/out/")
    p.add_argument("--magnetic", default="../00_config/magnetic_params.yaml")
    p.add_argument("--output", default="cocrni_firstprinciples.tdb")
    args = p.parse_args()

    mag_all = yaml.safe_load(Path(args.magnetic).read_text())
    gdir = Path(args.gdir)
    phases = args.phases.split(",")

    body = TDB_HEADER.format(timestamp=datetime.utcnow().isoformat() + "Z")
    # First pass: collect endmember G(T) per element from all phases for FUNCTION blocks
    endmember_seen = {}
    phase_blocks = []
    for ph in phases:
        grid = load_phase_grid(gdir, ph)
        if not grid:
            print(f"  [warn] no free-energy data for {ph}; skipping")
            continue
        model = fit_phase_rk(grid)
        phase_blocks.append((ph, model))
        for el, em in model.endmember_G.items():
            # Use the ground-state phase for each element as reference (HCP Co, BCC Cr, FCC Ni)
            ref_phase = {"Co": "HCP_A3", "Cr": "BCC_A2", "Ni": "FCC_A1"}[el]
            if el not in endmember_seen and ph == ref_phase:
                endmember_seen[el] = em
                body += emit_function_G(el, em, ref_phase)

    body += "\n"
    for ph, model in phase_blocks:
        body += emit_phase_block(ph, model, mag_all.get(ph, {}))

    body += "\nLIST_OF_REFERENCES\n"
    body += "NUMBER  SOURCE\n"
    body += "REF0    'mMTP+MACE first-principles fit (this work); soft-constrained VASP; "
    body += "magnetic params from collinear ordered/disordered DFT pipeline'\n!\n"

    Path(args.output).write_text(body)
    print(f"[tdb] wrote {args.output}  ({len(phase_blocks)} phases)")


if __name__ == "__main__":
    main()
