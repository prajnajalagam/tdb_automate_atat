"""
03_mmtp/train_and_al.py

Wrapper around the spin-MLIP toolchain (mlp executable + LAMMPS interface).

Pretrain mode:
  python train_and_al.py --pretrain --inputs ../02_dft/out/  --out mtp.mtp

Active-learn mode:
  python train_and_al.py --active-learn --start mtp.mtp \
      --md-template lammps_md.in --max-iters 12

The script:
  1. Builds a spin-MLIP-formatted training set ("train.cfg") from VASP runs.
  2. Calls `mlp train` to fit a level-12 mMTP.
  3. Calls LAMMPS via the spin-MLIP interface with extrapolation control on.
  4. Reads `preselected.cfg`, sends those configs to soft-constrained VASP,
     adds the resulting points to train.cfg, retrains, repeats.

Stop condition: an AL pass that preselects zero configs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import yaml


# ----------------------------- I/O helpers ------------------------------------

def vasp_results_to_mlp_cfg(results_root: Path, cfg_out: Path):
    """
    Walk all `result.json` under results_root and emit a spin-MLIP-format
    training set. Each block:

        BEGIN_CFG
         Size
          N
         SuperCell
          ax ay az
          bx by bz
          cx cy cz
         AtomData:  id type cartes_x  cartes_y  cartes_z  fx fy fz  m
          ...
         Energy
          E
         PlusStress:  xx yy zz yz xz xy
          ...
         Feature  EFS_by  ...
        END_CFG
    """
    type_map = {"Co": 0, "Cr": 1, "Ni": 2}
    blocks = []
    n_used = 0
    for rj in results_root.rglob("result.json"):
        res = json.loads(rj.read_text())
        if not res.get("converged"):
            continue
        cfgdir = rj.parent
        poscar = (cfgdir / "POSCAR").read_text().splitlines()
        # Read lattice + positions from POSCAR (assume scaling = 1)
        scale = float(poscar[1].strip())
        a = np.array([float(x) for x in poscar[2].split()]) * scale
        b = np.array([float(x) for x in poscar[3].split()]) * scale
        c = np.array([float(x) for x in poscar[4].split()]) * scale
        species_line = poscar[5].split()
        counts = [int(x) for x in poscar[6].split()]
        coord_type = poscar[7].strip().lower()
        positions = []
        i = 8
        for sp, cnt in zip(species_line, counts):
            for _ in range(cnt):
                r = np.array([float(x) for x in poscar[i].split()[:3]])
                if coord_type.startswith("d"):
                    r = r[0] * a + r[1] * b + r[2] * c
                positions.append((sp, r))
                i += 1

        magmoms = res.get("magmoms_OUTCAR", []) or [0.0] * len(positions)
        if len(magmoms) != len(positions):
            magmoms = (magmoms + [0.0] * len(positions))[: len(positions)]

        block = ["BEGIN_CFG", " Size", f"  {len(positions)}", " SuperCell"]
        for vec in (a, b, c):
            block.append(f"  {vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}")
        block.append(" AtomData:  id type cartes_x  cartes_y  cartes_z  fx fy fz  m")
        for j, (sp, r) in enumerate(positions, start=1):
            block.append(
                f"  {j} {type_map[sp]} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}  0.0 0.0 0.0  {magmoms[j-1]:.4f}"
            )
        block.append(" Energy")
        block.append(f"  {res['energy_eV']:.8f}")
        # Stress block intentionally omitted unless populated upstream
        block.append("END_CFG")
        blocks.append("\n".join(block))
        n_used += 1

    cfg_out.write_text("\n".join(blocks))
    print(f"[mlp-cfg] wrote {n_used} configurations → {cfg_out}")


def write_mtp_template(path: Path, level: int = 12, n_radial: int = 8,
                       n_magnetic: int = 2, r_cut: float = 5.0):
    """Minimal mMTP template file. spin-MLIP fills in trainable params from random init."""
    body = textwrap.dedent(f"""\
        MTP
        version = 1.1.0
        potential_name = mMTP_CoCrNi
        species_count = 3
        potential_tag = collinear
        radial_basis_type = RBChebyshev
            min_dist = 1.8
            max_dist = {r_cut}
            radial_basis_size = {n_radial}
            magnetic_basis_size = {n_magnetic}
            max_magmom_per_type = 3.0 2.5 1.5
        alpha_moments_count = 0
        alpha_index_basic_count = 0
        alpha_index_times_count = 0
        alpha_scalar_moments = 0
        species_coeffs = 0 0 0
        moment_coeffs = 0
        level = {level}
        """)
    path.write_text(body)


def write_lammps_md_template(path: Path, T_K: float, n_steps: int, dt_fs: float,
                             mtp_path: str, structure_data: str,
                             gamma_low: float, gamma_up: float):
    """LAMMPS input for an NPT MD run with spin-MLIP extrapolation control."""
    body = textwrap.dedent(f"""\
        units           metal
        atom_style      atomic
        boundary        p p p
        read_data       {structure_data}
        pair_style      mlip_spin {mtp_path} \\
                        extrapolation_control on \\
                        threshold_break {gamma_up} \\
                        threshold_save {gamma_low} \\
                        preselected_file preselected.cfg
        pair_coeff      * * Co Cr Ni
        velocity        all create {T_K} 12345 mom yes rot yes
        fix             1 all npt temp {T_K} {T_K} 0.1 iso 0.0 0.0 1.0
        timestep        {dt_fs / 1000.0}
        thermo          200
        run             {n_steps}
        """)
    path.write_text(body)


# ----------------------------- mlp driver -------------------------------------

def mlp_train(mlp_exe: str, train_cfg: Path, init_mtp: Path, out_mtp: Path,
              weights: dict):
    cmd = [
        mlp_exe, "train", str(init_mtp), str(train_cfg),
        f"--save-to={out_mtp}",
        f"--energy-weight={weights['e']}",
        f"--force-weight={weights['f']}",
        f"--stress-weight={weights['s']}",
        "--bfgs-conv-tol=1e-6",
        "--max-iter=2000",
    ]
    subprocess.run(cmd, check=True)


def mlp_select(mlp_exe: str, mtp: Path, preselected: Path, train_cfg: Path,
               selected: Path):
    """Run maxvol selection on preselected.cfg → selected.cfg (≤P new configs)."""
    cmd = [mlp_exe, "select-add", str(mtp), str(train_cfg), str(preselected), str(selected)]
    subprocess.run(cmd, check=True)


# ----------------------------- main loop --------------------------------------

def pretrain(args, cfg):
    train_cfg = Path("train.cfg")
    vasp_results_to_mlp_cfg(Path(args.inputs), train_cfg)
    init = Path("mtp_init.mtp")
    write_mtp_template(init, level=cfg["mmtp"]["level"],
                       n_radial=cfg["mmtp"]["n_radial"],
                       n_magnetic=cfg["mmtp"]["n_magnetic"],
                       r_cut=cfg["mmtp"]["r_cut"])
    out = Path(args.out)
    mlp_train(cfg["paths"]["mlp_executable"], train_cfg, init, out, cfg["mmtp"]["weights"])
    print(f"[pretrain] wrote {out}")


def active_learn(args, cfg):
    mtp = Path(args.start)
    train_cfg = Path("train.cfg")
    al = cfg["mmtp"]["active_learning"]

    for it in range(args.max_iters):
        print(f"\n=== AL iteration {it} ===")
        preselected_total = Path(f"preselected_iter{it:02d}.cfg")
        preselected_total.write_text("")

        # MD across the composition × T grid
        T_lo, T_hi = al["md_T_range_K"]
        for T in range(T_lo, T_hi + 1, al["md_T_step_K"]):
            for structure_data in sorted(Path("md_seeds").glob("*.data")):
                lmp_in = Path("lammps_md.in")
                write_lammps_md_template(
                    lmp_in, T_K=T, n_steps=al["md_steps_per_iter"],
                    dt_fs=al["md_timestep_fs"], mtp_path=str(mtp),
                    structure_data=str(structure_data),
                    gamma_low=al["gamma_low"], gamma_up=al["gamma_up"],
                )
                rc = subprocess.run([cfg["paths"]["lammps_executable"], "-in", str(lmp_in)],
                                    capture_output=True, text=True).returncode
                if rc != 0:
                    print(f"  LAMMPS failed at T={T} {structure_data.name}")
                if Path("preselected.cfg").exists():
                    with open(preselected_total, "a") as out:
                        out.write(Path("preselected.cfg").read_text())
                    Path("preselected.cfg").unlink()

        # Did anything get preselected this iteration?
        if preselected_total.stat().st_size == 0:
            print("[AL] no preselections this iteration → done")
            break

        # Maxvol select
        selected = Path(f"selected_iter{it:02d}.cfg")
        mlp_select(cfg["paths"]["mlp_executable"], mtp, preselected_total, train_cfg, selected)

        # Hand off to the soft-constrained VASP driver
        print(f"[AL] {selected} ready for soft-constrained DFT (call 02_dft driver manually)")
        print(f"     after DFT completes, run:")
        print(f"     python {__file__} --append-dft --new-results <results_dir>")
        print(f"     then re-invoke --active-learn to continue.")
        return  # one round, then user fans out DFT


def append_dft(args, cfg):
    """Append new DFT-calculated configs to train.cfg and retrain."""
    train_cfg = Path("train.cfg")
    tmp = Path("train_new.cfg")
    vasp_results_to_mlp_cfg(Path(args.new_results), tmp)
    # concatenate
    with open(train_cfg, "a") as out:
        out.write("\n")
        out.write(tmp.read_text())
    init = Path(args.start)
    out_mtp = Path(args.out)
    mlp_train(cfg["paths"]["mlp_executable"], train_cfg, init, out_mtp, cfg["mmtp"]["weights"])
    print(f"[append-dft] retrained → {out_mtp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../00_config/config.yaml")
    p.add_argument("--pretrain", action="store_true")
    p.add_argument("--active-learn", action="store_true")
    p.add_argument("--append-dft", action="store_true")
    p.add_argument("--inputs")
    p.add_argument("--start", default="mtp.mtp")
    p.add_argument("--out", default="mtp.mtp")
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--new-results")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.pretrain:
        pretrain(args, cfg)
    elif args.active_learn:
        active_learn(args, cfg)
    elif args.append_dft:
        append_dft(args, cfg)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
