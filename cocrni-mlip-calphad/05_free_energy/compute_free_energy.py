"""
05_free_energy/compute_free_energy.py

For each (phase, composition) compute G_phase(T) using Frenkel-Ladd
thermodynamic integration with the trained mMTP in LAMMPS, then average
across N_DLM realizations of magnetic disorder.

G_phase(x, T) = E_DFT_0(x) + ΔF_FL(x, T) + ⟨ΔE_mag⟩_DLM(x, T)

ΔF_FL is from λ-coupling between the Einstein crystal and the real potential.

Outputs:
  out/G_<phase>.csv with columns [x_Co, x_Cr, x_Ni, T_K, G_eV_per_atom, sigma]

This is the most expensive step in the pipeline; budget ~50 LAMMPS jobs
per (composition, temperature) point. Use the --grid-subset flag for testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class FreeEnergyPoint:
    x_Co: float
    x_Cr: float
    x_Ni: float
    T_K: float
    G_eV_per_atom: float
    sigma_eV_per_atom: float
    n_dlm: int


def write_frenkel_ladd_input(path: Path, T_K: float, mtp_path: str, data_path: str,
                             k_einstein: float = 5.0,
                             n_lambda: int = 21,
                             eq_ps: float = 20.0, prod_ps: float = 100.0):
    """LAMMPS script: Einstein → real-potential adiabatic switch."""
    eq_steps = int(eq_ps * 1000)
    prod_steps = int(prod_ps * 1000)
    body = textwrap.dedent(f"""\
        units           metal
        atom_style      atomic
        boundary        p p p
        read_data       {data_path}

        # Reference Einstein crystal — k_einstein chosen from RMSD of unconstrained MD
        # (see Freitas et al. 2016 for k tuning)
        variable        k_e equal {k_einstein}
        variable        T   equal {T_K}

        pair_style      hybrid/overlay mlip_spin {mtp_path} zero 0.0
        pair_coeff      * * mlip_spin Co Cr Ni
        pair_coeff      * * zero

        # Einstein tethering
        fix             ein all spring/self ${{k_e}}

        # NVT equilibration in the real potential
        velocity        all create ${{T}} 42 mom yes rot yes
        fix             nvt all nvt temp ${{T}} ${{T}} 0.1
        timestep        0.001
        thermo          1000
        run             {eq_steps}
        unfix           nvt

        # Adiabatic switch: λ from 1 (Einstein only) to 0 (real only)
        variable        lambda  equal ramp(1.0,0.0)
        fix             nvt all nvt temp ${{T}} ${{T}} 0.1
        fix             ein all adapt 1 fix ein scale v_lambda
        fix             real all adapt 1 pair mlip_spin scale * * v_complement
        variable        complement equal 1.0-v_lambda
        compute         pe all pe pair
        fix             work_avg all ave/time 10 1 10 c_pe v_lambda v_complement file work_forward.dat

        run             {prod_steps}
        unfix           work_avg
        unfix           real
        unfix           ein
        unfix           nvt

        # Reverse switch for hysteresis correction
        velocity        all create ${{T}} 1337 mom yes rot yes
        variable        lambda  equal ramp(0.0,1.0)
        fix             nvt all nvt temp ${{T}} ${{T}} 0.1
        fix             ein all spring/self ${{k_e}}
        fix             ein_ad all adapt 1 fix ein scale v_lambda
        fix             real all adapt 1 pair mlip_spin scale * * v_complement
        fix             work_avg all ave/time 10 1 10 c_pe v_lambda v_complement file work_reverse.dat
        run             {prod_steps}
        """)
    path.write_text(body)


def parse_work(file_forward: Path, file_reverse: Path) -> tuple[float, float]:
    """
    Integrate forward & reverse switching work, return ⟨W⟩ and uncertainty.
    Free energy difference ΔF = (W_forward + W_reverse)/2 with hysteresis = (W_f - W_r)/2.
    """
    fwd = np.loadtxt(file_forward, comments="#")
    rev = np.loadtxt(file_reverse, comments="#")
    # Columns: time pe lambda complement
    W_f = np.trapz(fwd[:, 1], fwd[:, 3])    # pe dλ_complement
    W_r = np.trapz(rev[:, 1], rev[:, 3])
    return 0.5 * (W_f + W_r), 0.5 * abs(W_f - W_r)


def F_einstein(T_K: float, k_einstein: float, n_atoms: int) -> float:
    """Free energy of harmonically tethered atoms (Frenkel-Ladd reference)."""
    kB = 8.617333262e-5     # eV/K
    hbar = 6.582119569e-16  # eV·s
    # ω = sqrt(k/m); average mass for Co-Cr-Ni ≈ 58 u → 9.62e-26 kg
    m_kg = 9.62e-26
    omega = np.sqrt(k_einstein * 1.602e-19 / 1e-20 / m_kg)   # k in eV/Å² → SI
    beta = 1.0 / (kB * T_K)
    return 3 * n_atoms * kB * T_K * np.log(beta * hbar * omega)


def compute_one(phase: str, x: dict[str, float], T_K: float, mtp_path: Path,
                cfg: dict, dlm_data_paths: list[Path]) -> FreeEnergyPoint:
    """Average ΔF across DLM realizations at fixed (x, T)."""
    Gs = []
    for data in dlm_data_paths:
        work = Path(f"work_{phase}_T{int(T_K)}_{data.stem}")
        work.mkdir(parents=True, exist_ok=True)
        lmp_in = work / "ti.in"
        write_frenkel_ladd_input(lmp_in, T_K=T_K, mtp_path=str(mtp_path), data_path=str(data),
                                 n_lambda=cfg["free_energy"]["lambda_steps"],
                                 eq_ps=cfg["free_energy"]["md_equilibration_ps"],
                                 prod_ps=cfg["free_energy"]["md_production_ps"])
        rc = subprocess.run([cfg["paths"]["lammps_executable"], "-in", str(lmp_in)],
                            cwd=work, capture_output=True).returncode
        if rc != 0:
            print(f"  LAMMPS TI failed for {data.stem} at T={T_K}")
            continue
        dF, hysteresis = parse_work(work / "work_forward.dat", work / "work_reverse.dat")
        # Get n_atoms from data file
        n_atoms = int([l for l in data.read_text().splitlines() if "atoms" in l.lower()][0].split()[0])
        F_ein = F_einstein(T_K, cfg["free_energy"].get("k_einstein", 5.0), n_atoms)
        # Add static DFT/mMTP energy of the reference (E_0)
        # (placeholder: read from a precomputed E_0 table; in production link to step 2 output)
        E0 = 0.0
        F_total = (F_ein + dF) / n_atoms + E0
        Gs.append(F_total)
        print(f"   {data.stem}  T={T_K}K  ΔF={dF/n_atoms:.4f} eV/atom  hyst={hysteresis/n_atoms:.4f}")

    if not Gs:
        return FreeEnergyPoint(x["Co"], x["Cr"], x["Ni"], T_K, float("nan"), float("nan"), 0)
    return FreeEnergyPoint(x["Co"], x["Cr"], x["Ni"], T_K, float(np.mean(Gs)),
                           float(np.std(Gs) / np.sqrt(len(Gs))), len(Gs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../00_config/config.yaml")
    p.add_argument("--phase", required=True)
    p.add_argument("--mtp", default="../03_mmtp/mtp.mtp")
    p.add_argument("--T-list", default="600,800,1000,1200,1400,1600,1800,2000",
                   help="comma-separated K values, or 'all' for the config grid")
    p.add_argument("--grid-subset", type=int, default=None,
                   help="limit composition grid to first N points (for testing)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.T_list == "all":
        T_list = cfg["free_energy"]["T_grid_K"]
    else:
        T_list = [float(t) for t in args.T_list.split(",")]

    # Composition grid: enumerate from md_seeds/ which carries the LAMMPS data files
    seeds = sorted(Path(f"../03_mmtp/md_seeds/{args.phase}").glob("*/"))
    if args.grid_subset:
        seeds = seeds[: args.grid_subset]

    out_csv = Path(f"out/G_{args.phase}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_Co", "x_Cr", "x_Ni", "T_K", "G_eV_per_atom", "sigma_eV_per_atom", "n_dlm"])
        for comp_dir in seeds:
            # composition label encodes the fractions
            name = comp_dir.name
            try:
                x_Co = int(name.split("Co")[1][:4]) / 1000
                x_Cr = int(name.split("Cr")[1][:4]) / 1000
                x_Ni = int(name.split("Ni")[1][:4]) / 1000
            except (ValueError, IndexError):
                continue
            dlm_data = sorted(comp_dir.glob("dlm_*.data"))
            for T in T_list:
                pt = compute_one(args.phase, {"Co": x_Co, "Cr": x_Cr, "Ni": x_Ni}, T,
                                 Path(args.mtp), cfg, dlm_data)
                w.writerow([pt.x_Co, pt.x_Cr, pt.x_Ni, pt.T_K, pt.G_eV_per_atom,
                            pt.sigma_eV_per_atom, pt.n_dlm])
                f.flush()
    print(f"[free-energy] wrote {out_csv}")


if __name__ == "__main__":
    main()
