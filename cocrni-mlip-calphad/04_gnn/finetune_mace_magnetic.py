"""
04_gnn/finetune_mace_magnetic.py

Fine-tune a MACE foundation model on the spin-polarized Co-Cr-Ni dataset.
The trick to making MACE magnetism-aware without modifying its architecture:
inject |m_i| as a per-atom node attribute, alongside the usual one-hot Z.

Usage:
    python finetune_mace_magnetic.py --train ../03_mmtp/train.cfg --epochs 200

This reuses the spin-MLIP-format train.cfg from step 3 — no double bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def parse_mlp_cfg(path: Path) -> list[Atoms]:
    """Parse spin-MLIP train.cfg → list of ASE Atoms with energy, magmoms."""
    text = path.read_text().split("BEGIN_CFG")[1:]
    atoms_list = []
    type_map = ["Co", "Cr", "Ni"]
    for block in text:
        lines = block.splitlines()
        n_atoms = int(lines[lines.index(" Size") + 1].strip())
        sc_idx = lines.index(" SuperCell")
        cell = np.array([[float(x) for x in lines[sc_idx + 1 + i].split()] for i in range(3)])
        atom_idx = lines.index(" AtomData:  id type cartes_x  cartes_y  cartes_z  fx fy fz  m")
        species, positions, magmoms = [], [], []
        for j in range(n_atoms):
            parts = lines[atom_idx + 1 + j].split()
            species.append(type_map[int(parts[1])])
            positions.append([float(x) for x in parts[2:5]])
            magmoms.append(float(parts[8]))
        energy_idx = lines.index(" Energy")
        energy = float(lines[energy_idx + 1].strip())

        atoms = Atoms(symbols=species, positions=positions, cell=cell, pbc=True)
        atoms.set_initial_magnetic_moments(magmoms)
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        # Carry |m_i| as an extra array MACE can pick up as a node attribute
        atoms.arrays["abs_magmom"] = np.abs(np.array(magmoms))
        atoms_list.append(atoms)
    return atoms_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../00_config/config.yaml")
    p.add_argument("--train", required=True, help="spin-MLIP train.cfg")
    p.add_argument("--foundation", default="medium")  # MACE-MP small/medium/large
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out", default="mace_cocrni.model")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    atoms_list = parse_mlp_cfg(Path(args.train))
    print(f"[mace] {len(atoms_list)} configs loaded")

    # Save to extxyz so the MACE CLI can ingest it
    from ase.io import write as ase_write
    train_xyz = Path("train_mace.extxyz")
    ase_write(str(train_xyz), atoms_list, format="extxyz")

    # MACE foundation model fine-tuning via CLI (more stable than calling Python API directly)
    cmd = [
        "mace_run_train",
        f"--name={Path(args.out).stem}",
        f"--train_file={train_xyz}",
        f"--valid_fraction=0.1",
        "--model=MACE",
        f"--foundation_model={args.foundation}",
        "--multiheads_finetuning=False",
        "--num_interactions=2",
        "--max_L=2",
        "--correlation=3",
        "--r_max=5.0",
        "--batch_size=4",
        "--valid_batch_size=2",
        f"--max_num_epochs={args.epochs}",
        f"--lr={cfg['gnn']['lr']}",
        f"--energy_weight={cfg['gnn']['loss_weights']['e']}",
        f"--forces_weight={cfg['gnn']['loss_weights']['f']}",
        "--ema",
        "--ema_decay=0.99",
        "--scaling=rms_forces_scaling",
        "--device=cuda" if torch.cuda.is_available() else "--device=cpu",
        "--default_dtype=float32",
        # Inject |m_i| as an additional node feature read from atoms.arrays["abs_magmom"]
        "--atomic_node_features=abs_magmom",
    ]
    print("[mace] $", " ".join(cmd))
    import subprocess
    subprocess.run(cmd, check=True)
    print(f"[mace] fine-tuned → {args.out}")


if __name__ == "__main__":
    main()
