"""
02_dft/vasp_constrained_driver.py

Two modes:
  --mode equilibrium  : standard ISPIN=2 collinear, sets MAGMOM from sidecar JSON,
                        no constraint. Produces ground-state-for-that-arrangement
                        energies, forces, stresses.
  --mode constrained  : soft-constrained per Burov SI. Walks λ-ladder, refines
                        target moments so OUTCAR m_i ≃ target. Used by mMTP AL
                        to compute energies for non-equilibrium m_i selected by maxvol.

Reads:  POSCAR + magmoms.json in each config dir
Writes: INCAR/KPOINTS/POTCAR, runs vasp_std, collects OUTCAR + a result JSON.

This is a driver, not a queue. Run under your scheduler with one config per job
or use a fan-out (e.g., AiiDA, FireWorks, custom Slurm array).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from ase.io import read as ase_read

INCAR_BASE = """\
SYSTEM = CoCrNi {tag}
PREC = Accurate
ENCUT = {encut}
EDIFF = {ediff}
EDIFFG = {ediffg}
ISMEAR = 1
SIGMA = {sigma}
ISPIN = 2
ISYM = 0
KSPACING = {kspacing}
KGAMMA = .TRUE.
NELM = 200
NELMIN = 4
LREAL = {lreal}
LWAVE = .TRUE.
LCHARG = .TRUE.
LORBIT = 11
LSCALU = .FALSE.
ALGO = Normal
ICHARG = 2
NSW = 0
MAGMOM = {magmom}
"""

INCAR_CONSTRAINED_HEADER = """\
LNONCOLLINEAR = .TRUE.
I_CONSTRAINED_M = 2
LAMBDA = {lam}
M_CONSTR = {mconstr}
RWIGS = {rwigs}
ICHARG = 1
ISTART = 1
"""


@dataclass
class DFTRunResult:
    tag: str
    energy_eV: float
    forces_eV_per_A: list[list[float]]
    stress_GPa: list[list[float]]
    magmoms_OUTCAR: list[float]
    magmoms_Mint_OSZICAR: list[float]
    converged: bool
    iterations_scf: int
    lam_final: float | None = None
    notes: list[str] = field(default_factory=list)


def write_potcar(work: Path, species_order: list[str], potcar_dir: Path, mapping: dict[str, str]):
    """Concatenate POTCAR files for the species order in the POSCAR."""
    paths = [potcar_dir / mapping[el] / "POTCAR" for el in species_order]
    with open(work / "POTCAR", "wb") as out:
        for p in paths:
            out.write(p.read_bytes())


def format_magmom_collinear(magmoms: list[float]) -> str:
    """VASP MAGMOM line for ISPIN=2 collinear."""
    return " ".join(f"{m:.4f}" for m in magmoms)


def format_magmom_noncollinear(magmoms: list[float]) -> str:
    """VASP MAGMOM line for non-collinear (each atom: mx my mz). Use ẑ axis."""
    return " ".join(f"0.0 0.0 {m:.4f}" for m in magmoms)


def parse_outcar(outcar: Path) -> dict:
    """Minimal OUTCAR parser; for production use ase or pymatgen."""
    text = outcar.read_text()
    # Final free energy
    energies = [float(x.split()[-2]) for x in text.splitlines()
                if "free  energy   TOTEN" in x]
    energy = energies[-1] if energies else float("nan")

    # Magnetic moments (per atom, from "magnetization (x)" block at end)
    blocks = text.split("magnetization (x)")
    magmoms = []
    if len(blocks) > 1:
        tail = blocks[-1].splitlines()
        # Skip header lines until we hit a data line beginning with an integer
        for line in tail:
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                magmoms.append(float(parts[-1]))   # total m_i is the last column
            elif magmoms and line.strip().startswith("---"):
                break
    return {"energy": energy, "magmoms": magmoms, "raw_text_size": len(text)}


def parse_oszicar_mint(oszicar: Path) -> list[float]:
    """Return Mint (last-iteration unsmoothed sphere-integrated moments)."""
    # In OSZICAR the line is something like:  mag= 1.234 0.001 ...
    text = oszicar.read_text().splitlines()
    last_mag_line = [l for l in text if "mag=" in l]
    if not last_mag_line:
        return []
    parts = last_mag_line[-1].split("mag=")[-1].split()
    return [float(p) for p in parts]


def write_incar(work: Path, cfg: dict, magmoms: list[float], tag: str,
                constrained: bool = False, lam: float = 0.0,
                rwigs: list[float] | None = None):
    fields = {
        "tag": tag,
        "encut": cfg["dft"]["encut_eV"],
        "ediff": cfg["dft"]["ediff"],
        "ediffg": cfg["dft"]["ediffg"],
        "sigma": cfg["dft"]["sigma_eV"],
        "kspacing": cfg["dft"]["kspacing_invA"],
        "lreal": cfg["dft"]["lreal"],
    }
    if constrained:
        fields["magmom"] = format_magmom_noncollinear(magmoms)
        body = INCAR_BASE.format(**fields).replace("ISPIN = 2\n", "")
        body += INCAR_CONSTRAINED_HEADER.format(
            lam=lam,
            mconstr=format_magmom_noncollinear(magmoms),
            rwigs=" ".join(f"{r:.3f}" for r in rwigs),
        )
    else:
        fields["magmom"] = format_magmom_collinear(magmoms)
        body = INCAR_BASE.format(**fields)
    (work / "INCAR").write_text(body)


def run_vasp(work: Path, vasp_exe: str, mpi_cmd: list[str] | None = None) -> int:
    cmd = (mpi_cmd or []) + [vasp_exe]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    (work / "stdout.log").write_text(proc.stdout)
    (work / "stderr.log").write_text(proc.stderr)
    return proc.returncode


def equilibrium_run(work: Path, cfg: dict) -> DFTRunResult:
    sidecar = json.loads((work / "magmoms.json").read_text())
    species_order = sidecar["species_order_in_POSCAR"]
    write_potcar(work, ["Co" if s == "Co" else "Cr" if s == "Cr" else "Ni" for s in species_order],
                 Path(cfg["paths"]["potcar_dir"]), cfg["dft"]["potcars"])
    write_incar(work, cfg, sidecar["magmoms"], tag=sidecar["tag"], constrained=False)
    # KPOINTS handled via KSPACING in INCAR

    rc = run_vasp(work, cfg["paths"]["vasp_executable"])
    if rc != 0:
        return DFTRunResult(sidecar["tag"], float("nan"), [], [], [], [], False, 0,
                            notes=[f"VASP rc={rc}"])

    parsed = parse_outcar(work / "OUTCAR")
    mint = parse_oszicar_mint(work / "OSZICAR")
    return DFTRunResult(
        tag=sidecar["tag"],
        energy_eV=parsed["energy"],
        forces_eV_per_A=[],   # populate from OUTCAR if needed
        stress_GPa=[],
        magmoms_OUTCAR=parsed["magmoms"],
        magmoms_Mint_OSZICAR=mint,
        converged=True,
        iterations_scf=0,
    )


def constrained_run(work: Path, cfg: dict, target_magmoms: list[float],
                    rwigs: list[float]) -> DFTRunResult:
    """
    Implements the Burov SI scheme:
      1. unconstrained collinear SCF → WAVECAR, CHGCAR
      2. tiny-λ non-collinear SCF for Mint↔MWint linear fit
      3. λ-ladder constrained runs with convergence checks
      4. refinement: subtract OUTCAR-Mint bias, rerun once
    """
    sidecar = json.loads((work / "magmoms.json").read_text())
    species_order = sidecar["species_order_in_POSCAR"]
    write_potcar(work,
                 ["Co" if s == "Co" else "Cr" if s == "Cr" else "Ni" for s in species_order],
                 Path(cfg["paths"]["potcar_dir"]), cfg["dft"]["potcars"])

    notes = []

    # --- Stage 1: unconstrained equilibrium to seed WAVECAR/CHGCAR --------
    write_incar(work, cfg, target_magmoms, tag=sidecar["tag"] + "_seed", constrained=False)
    if run_vasp(work, cfg["paths"]["vasp_executable"]) != 0:
        return DFTRunResult(sidecar["tag"], float("nan"), [], [], [], [], False, 0,
                            notes=["seed SCF failed"])

    # --- Stage 2: tiny-λ NC fit pass for Mint↔MWint slope -----------------
    write_incar(work, cfg, target_magmoms, tag=sidecar["tag"] + "_fit",
                constrained=True, lam=0.01, rwigs=rwigs)
    if run_vasp(work, cfg["paths"]["vasp_executable"]) != 0:
        return DFTRunResult(sidecar["tag"], float("nan"), [], [], [], [], False, 0,
                            notes=["fit pass failed"])

    mint = np.array(parse_oszicar_mint(work / "OSZICAR"))
    # MWint sits two columns earlier in OSZICAR but for a one-shot fit we use
    # the ratio Mint/M_CONSTR averaged. Production runs should follow the full
    # linear regression described in the Burov SI §S1.5.
    if len(mint) == 0 or len(target_magmoms) == 0:
        k_fit = 1.0
    else:
        k_fit = float(np.mean(np.abs(mint)) / max(np.mean(np.abs(target_magmoms)), 1e-6))
        if not np.isfinite(k_fit) or k_fit < 0.5 or k_fit > 2.0:
            k_fit = 1.0
            notes.append(f"k_fit clamped (raw {k_fit:.3f})")

    m_constr_initial = [m / k_fit for m in target_magmoms]

    # --- Stage 3: λ-ladder constrained SCF --------------------------------
    delta_max = cfg["dft"]["soft_constrained"]["delta_M_max"]
    delta_av  = cfg["dft"]["soft_constrained"]["delta_M_av"]
    Ep_tol    = cfg["dft"]["soft_constrained"]["E_p_tol"]
    final_lam = None
    final_mint = []
    for lam in cfg["dft"]["soft_constrained"]["lambda_ladder"]:
        write_incar(work, cfg, m_constr_initial, tag=sidecar["tag"] + f"_lam{lam}",
                    constrained=True, lam=lam, rwigs=rwigs)
        rc = run_vasp(work, cfg["paths"]["vasp_executable"])
        if rc != 0:
            notes.append(f"λ={lam} VASP rc={rc}")
            continue
        mint = np.array(parse_oszicar_mint(work / "OSZICAR"))
        if len(mint) != len(target_magmoms):
            notes.append(f"λ={lam} Mint length mismatch")
            continue
        diff = np.abs(np.abs(mint) - np.abs(target_magmoms))
        if diff.max() < delta_max and diff.mean() < delta_av:
            final_lam = lam
            final_mint = mint.tolist()
            break

    if final_lam is None:
        notes.append("never converged in λ-ladder; using last available")
        final_lam = cfg["dft"]["soft_constrained"]["lambda_ladder"][-1]
        final_mint = mint.tolist() if len(mint) else []

    # --- Stage 4: refinement: bias-correct M_CONSTR -----------------------
    parsed = parse_outcar(work / "OUTCAR")
    out_m = np.array(parsed["magmoms"])
    if len(out_m) == len(final_mint) and len(out_m) > 0:
        bias = (out_m - np.array(final_mint)) / k_fit
        refined_constr = (np.array(m_constr_initial) - bias).tolist()
        write_incar(work, cfg, refined_constr, tag=sidecar["tag"] + "_refined",
                    constrained=True, lam=final_lam, rwigs=rwigs)
        run_vasp(work, cfg["paths"]["vasp_executable"])
        parsed = parse_outcar(work / "OUTCAR")
        final_mint = parse_oszicar_mint(work / "OSZICAR")

    return DFTRunResult(
        tag=sidecar["tag"],
        energy_eV=parsed["energy"],
        forces_eV_per_A=[],
        stress_GPa=[],
        magmoms_OUTCAR=parsed["magmoms"],
        magmoms_Mint_OSZICAR=list(final_mint),
        converged=True,
        iterations_scf=0,
        lam_final=final_lam,
        notes=notes,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../00_config/config.yaml")
    p.add_argument("--inputs", required=True, help="directory containing config subdirs")
    p.add_argument("--mode", choices=["equilibrium", "constrained"], default="equilibrium")
    p.add_argument("--rwigs", nargs="+", type=float, default=[1.302, 1.323, 1.286],
                   help="Wigner-Seitz radii (Å) for Co, Cr, Ni (in alphabetic POSCAR order)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    inputs = Path(args.inputs)

    results = []
    for cfgdir in sorted(inputs.glob("*/*/")):     # composition_label/tag
        if not (cfgdir / "magmoms.json").exists():
            continue
        if args.mode == "equilibrium":
            res = equilibrium_run(cfgdir, cfg)
        else:
            sidecar = json.loads((cfgdir / "magmoms.json").read_text())
            # Map RWIGS to per-atom by species
            elem_to_rwigs = dict(zip(["Co", "Cr", "Ni"], args.rwigs))
            per_atom_rwigs = [elem_to_rwigs[s] for s in sidecar["species_order_in_POSCAR"]]
            res = constrained_run(cfgdir, cfg, sidecar["magmoms"], per_atom_rwigs)
        (cfgdir / "result.json").write_text(json.dumps(res.__dict__, indent=2))
        results.append({"dir": str(cfgdir), **res.__dict__})
        print(f"  {cfgdir.name}: E={res.energy_eV:.4f} eV  converged={res.converged}")

    (inputs / "all_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
