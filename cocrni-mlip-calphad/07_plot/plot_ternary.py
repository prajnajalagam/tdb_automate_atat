"""
07_plot/plot_ternary.py

pycalphad-driven ternary isothermal sections and phase fraction diagrams
from the generated TDB.

Usage:
    python plot_ternary.py --tdb ../06_calphad/cocrni_firstprinciples.tdb --T 1000
    python plot_ternary.py --tdb ../06_calphad/cocrni_firstprinciples.tdb --isopleth equiatomic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pycalphad import Database, ternplot, equilibrium, variables as v


PHASES = ["LIQUID", "FCC_A1", "HCP_A3", "BCC_A2", "SIGMA"]


def plot_isothermal_section(tdb_path: Path, T: float, ax=None):
    db = Database(str(tdb_path))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    conds = {v.T: T, v.P: 101325, v.N: 1,
             v.X("CR"): (0, 1, 0.025), v.X("NI"): (0, 1, 0.025)}
    ternplot(db, ["CO", "CR", "NI", "VA"], PHASES, conds,
             x=v.X("CR"), y=v.X("NI"), ax=ax)
    ax.set_title(f"Co–Cr–Ni isothermal section at {T:.0f} K\n(first-principles, no exp. input)")
    return ax


def plot_phase_fractions(tdb_path: Path, x: dict[str, float],
                         T_range=(300, 2000), n_T=70, ax=None):
    db = Database(str(tdb_path))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    Ts = np.linspace(*T_range, n_T)
    conds = {v.X("CR"): x["Cr"], v.X("NI"): x["Ni"], v.P: 101325, v.N: 1, v.T: Ts}
    eq = equilibrium(db, ["CO", "CR", "NI", "VA"], PHASES, conds)
    for phase in PHASES:
        # Sum NP over all vertices belonging to this phase
        np_phase = eq.NP.where(eq.Phase == phase).sum(dim="vertex")
        ax.plot(Ts, np_phase.values.flatten(), label=phase, lw=2)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("phase mole fraction")
    title = f"Phase fractions at x_Cr={x['Cr']:.2f}, x_Ni={x['Ni']:.2f}"
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return ax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tdb", required=True)
    p.add_argument("--T", type=float, default=1000)
    p.add_argument("--isopleth", default=None,
                   help="composition tag, e.g. 'equiatomic' for x_Co=x_Cr=x_Ni=1/3, "
                        "or 'Cr_rich' for x_Cr=0.6, x_Ni=0.2")
    p.add_argument("--out", default="phasediagram.png")
    args = p.parse_args()

    if args.isopleth is None:
        ax = plot_isothermal_section(Path(args.tdb), args.T)
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)
        print(f"[plot] isothermal {args.T} K → {args.out}")
    else:
        comps = {
            "equiatomic": {"Co": 1/3, "Cr": 1/3, "Ni": 1/3},
            "Cr_rich":    {"Co": 0.2, "Cr": 0.6, "Ni": 0.2},
            "Ni_rich":    {"Co": 0.2, "Cr": 0.2, "Ni": 0.6},
        }
        if args.isopleth not in comps:
            raise SystemExit(f"Unknown isopleth tag: {args.isopleth}")
        ax = plot_phase_fractions(Path(args.tdb), comps[args.isopleth])
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)
        print(f"[plot] phase fractions for {args.isopleth} → {args.out}")


if __name__ == "__main__":
    main()
