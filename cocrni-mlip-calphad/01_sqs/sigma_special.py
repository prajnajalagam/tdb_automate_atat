"""
01_sqs/sigma_special.py

σ-CoCr-Ni (D8b, tP30, space group P4_2/mnm) needs explicit per-sublattice
occupation. Five inequivalent Wyckoff sites: 2a, 4f, 8i, 8i', 8j with
multiplicities {2, 4, 8, 8, 8}. CALPHAD models typically use a 3-sublattice
contraction; we generate the full 5-sublattice configurations because mMTP
needs to learn local environments, not just thermodynamic occupancies.

For each composition on the grid, we generate:
  - one "Cr-rich on high-coordination sites" reference
  - one "Co-rich on low-coordination sites" reference
  - several randomized DLM realizations
  - a CrCoNi-equiatomic enumeration set (for the matrix MEA)

Run after generate_sqs_dlm.py for the simple phases.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml
from pymatgen.core import Lattice, Structure

# Idealized σ-phase fractional coordinates (D8b, from ICSD reference templates).
# Lattice parameters are placeholders; the DFT driver will relax cell.
SIGMA_LATTICE = (8.80, 8.80, 4.55, 90.0, 90.0, 90.0)
SIGMA_SITES = {
    # site_label: (multiplicity, fractional coords list, coordination)
    "A_2a":  (2, [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)], 12),
    "B_4f":  (4, [(0.3981, 0.3981, 0.0), (0.6019, 0.6019, 0.0),
                  (0.8981, 0.1019, 0.5), (0.1019, 0.8981, 0.5)], 15),
    "C_8i":  (8, _sigma_8i_a(), 14),
    "D_8i2": (8, _sigma_8i_b(), 12),
    "E_8j":  (8, _sigma_8j(),   14),
} if False else {}  # populated below; helpers defined first


def _sigma_8i_a():
    u, v = 0.4632, 0.1316
    return [(u, v, 0), (-u, -v, 0), (-v, u, 0), (v, -u, 0),
            (0.5 + u, 0.5 - v, 0.5), (0.5 - u, 0.5 + v, 0.5),
            (0.5 - v, 0.5 - u, 0.5), (0.5 + v, 0.5 + u, 0.5)]


def _sigma_8i_b():
    u, v = 0.7376, 0.0653
    return [(u, v, 0), (-u, -v, 0), (-v, u, 0), (v, -u, 0),
            (0.5 + u, 0.5 - v, 0.5), (0.5 - u, 0.5 + v, 0.5),
            (0.5 - v, 0.5 - u, 0.5), (0.5 + v, 0.5 + u, 0.5)]


def _sigma_8j():
    u, w = 0.1823, 0.2522
    return [(u, u, w), (-u, -u, w), (-u, u, -w), (u, -u, -w),
            (0.5 + u, 0.5 + u, 0.5 - w), (0.5 - u, 0.5 - u, 0.5 - w),
            (0.5 - u, 0.5 + u, 0.5 + w), (0.5 + u, 0.5 - u, 0.5 + w)]


SIGMA_SITES = {
    "A_2a":  (2, [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)], 12),
    "B_4f":  (4, [(0.3981, 0.3981, 0.0), (0.6019, 0.6019, 0.0),
                  (0.8981, 0.1019, 0.5), (0.1019, 0.8981, 0.5)], 15),
    "C_8i":  (8, _sigma_8i_a(), 14),
    "D_8i2": (8, _sigma_8i_b(), 12),
    "E_8j":  (8, _sigma_8j(),   14),
}

# Standard σ-phase site-preference heuristics in Co-Cr (used as priors only):
# Low-CN sites (A_2a, D_8i2) prefer Co/Ni; high-CN sites (B_4f, C_8i, E_8j) prefer Cr.
PREFERRED = {"A_2a": "Co", "B_4f": "Cr", "C_8i": "Cr", "D_8i2": "Co", "E_8j": "Cr"}


def realize_sigma(comp_counts: dict[str, int], rng: np.random.Generator,
                  mode: str = "biased") -> Structure:
    """
    Place 30 atoms onto the σ sublattices according to `mode`:
      biased: fill preferred element first, fill remainder randomly
      random: fully random placement
    """
    n_sites = sum(mult for mult, _, _ in [(m, c, cn) for m, c, cn in SIGMA_SITES.values()])
    assert n_sites == 30
    assert sum(comp_counts.values()) == 30, f"σ needs exactly 30 atoms; got {sum(comp_counts.values())}"

    species_pool = []
    for el, n in comp_counts.items():
        species_pool.extend([el] * n)

    # Build (label, frac) pairs in deterministic order
    site_records = []
    for label, (_mult, coords, _cn) in SIGMA_SITES.items():
        for c in coords:
            site_records.append((label, c))

    placements = ["?"] * 30
    if mode == "biased":
        remaining = list(species_pool)
        # First pass: place preferred element at each site if available
        for i, (label, _) in enumerate(site_records):
            pref = PREFERRED[label]
            if pref in remaining:
                placements[i] = pref
                remaining.remove(pref)
        # Second pass: fill remaining sites randomly
        empty_idx = [i for i, x in enumerate(placements) if x == "?"]
        rng.shuffle(empty_idx)
        for idx in empty_idx:
            placements[idx] = remaining.pop()
    else:
        order = list(range(30))
        rng.shuffle(order)
        for idx, sp in zip(order, species_pool):
            placements[idx] = sp

    coords = [site_records[i][1] for i in range(30)]
    return Structure(Lattice.from_parameters(*SIGMA_LATTICE), placements, coords)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../00_config/config.yaml")
    p.add_argument("--out", default="out/SIGMA")
    p.add_argument("--seed", type=int, default=20260615)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(args.seed)
    n_realizations = cfg["phases"]["SIGMA"]["dlm_realizations"]

    # σ is Co-Cr dominated; we sweep x_Cr from 0.45 to 0.75 in steps of 0.05,
    # with Ni up to 0.20 substituting for Co preferentially.
    sigma_grid = []
    for n_cr in range(14, 22):      # 14..21 of 30 → x_Cr 0.467..0.700
        for n_ni in range(0, 7):    # 0..6 of 30 Ni
            n_co = 30 - n_cr - n_ni
            if n_co < 4 or n_co > 16:
                continue
            sigma_grid.append({"Co": n_co, "Cr": n_cr, "Ni": n_ni})

    out_root = Path(args.out)
    for counts in sigma_grid:
        x = {k: v / 30 for k, v in counts.items()}
        label = f"Co{int(x['Co']*1000):04d}_Cr{int(x['Cr']*1000):04d}_Ni{int(x['Ni']*1000):04d}"
        config_dir = out_root / label
        config_dir.mkdir(parents=True, exist_ok=True)

        # 1 biased + 1 random reference + N DLM with random placement
        variants = [("biased_FM", realize_sigma(counts, rng, mode="biased"))]
        for k in range(n_realizations):
            variants.append((f"random_dlm_{k:02d}", realize_sigma(counts, rng, mode="random")))

        for tag, struct in variants:
            sub = config_dir / tag
            sub.mkdir(parents=True, exist_ok=True)
            struct.to(filename=str(sub / "POSCAR"), fmt="poscar")
            # Magmoms: ±|m| for DLM tags; all-positive for biased_FM
            from generate_sqs_dlm import ORDERED_MOMENTS
            species = [str(s.specie.symbol) for s in struct]
            abs_m = np.array([ORDERED_MOMENTS[("SIGMA", sp)] for sp in species])
            if tag.startswith("biased"):
                mm = abs_m.tolist()
            else:
                signs = rng.choice([-1, 1], size=30)
                mm = (signs * abs_m).tolist()
            (sub / "magmoms.json").write_text(json.dumps({
                "tag": tag, "magmoms": mm,
                "species_order_in_POSCAR": species,
            }, indent=2))
        print(f"  σ {label}  variants={len(variants)}")


if __name__ == "__main__":
    main()
