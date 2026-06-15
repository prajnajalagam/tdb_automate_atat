"""
01_sqs/generate_sqs_dlm.py

For each (phase, composition) on the grid, run ATAT mcsqs to produce an SQS,
then generate N_DLM collinear ±m_i realizations plus the ordered (FM-aligned)
reference. Output: POSCAR + a MAGMOM line ready for VASP.

Usage:
    python generate_sqs_dlm.py --phase FCC_A1 --config ../00_config/config.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from ase.build import bulk
from ase.io import write
from pymatgen.core import Composition, Lattice, Structure


# -- Element-specific moment magnitudes used to seed VASP MAGMOM --------------
# These are the |m_i| ordered-state targets per element per phase, in μ_B.
# Sign comes from the DLM realization (random ±).
ORDERED_MOMENTS = {
    ("FCC_A1", "Co"): 1.70,
    ("FCC_A1", "Cr"): 0.50,   # small but nonzero for SCF stability
    ("FCC_A1", "Ni"): 0.55,
    ("HCP_A3", "Co"): 1.65,
    ("HCP_A3", "Cr"): 0.50,
    ("HCP_A3", "Ni"): 0.60,
    ("BCC_A2", "Co"): 1.35,
    ("BCC_A2", "Cr"): 0.80,
    ("BCC_A2", "Ni"): 0.50,
    ("SIGMA",  "Co"): 1.20,
    ("SIGMA",  "Cr"): 0.60,
    ("SIGMA",  "Ni"): 0.50,
}


@dataclass
class GridPoint:
    x_Co: float
    x_Cr: float
    x_Ni: float

    @property
    def label(self) -> str:
        return f"Co{int(self.x_Co*1000):04d}_Cr{int(self.x_Cr*1000):04d}_Ni{int(self.x_Ni*1000):04d}"

    def composition(self, n_atoms: int) -> dict[str, int]:
        # Round to nearest integer count, then fix-up to preserve total
        raw = {"Co": self.x_Co * n_atoms,
               "Cr": self.x_Cr * n_atoms,
               "Ni": self.x_Ni * n_atoms}
        counts = {k: int(round(v)) for k, v in raw.items()}
        diff = n_atoms - sum(counts.values())
        if diff != 0:
            # adjust the element with largest fractional remainder
            remainders = {k: raw[k] - counts[k] for k in raw}
            key = max(remainders, key=lambda k: abs(remainders[k]))
            counts[key] += diff
        return counts


def make_ternary_grid(step: float, include_edges_at: float | None = None) -> list[GridPoint]:
    """ΔX grid over the Gibbs triangle; include_edges_at adds finer points on the three binary edges."""
    pts: list[GridPoint] = []
    n = int(round(1.0 / step))
    for i, j in itertools.product(range(n + 1), repeat=2):
        k = n - i - j
        if k < 0:
            continue
        pts.append(GridPoint(i / n, j / n, k / n))

    if include_edges_at is not None and include_edges_at < step:
        m = int(round(1.0 / include_edges_at))
        for i in range(m + 1):
            # Co-Cr edge (x_Ni = 0)
            pts.append(GridPoint(i / m, (m - i) / m, 0.0))
            # Co-Ni edge (x_Cr = 0)
            pts.append(GridPoint(i / m, 0.0, (m - i) / m))
            # Cr-Ni edge (x_Co = 0)
            pts.append(GridPoint(0.0, i / m, (m - i) / m))

    # Deduplicate
    seen = set()
    out = []
    for p in pts:
        key = (round(p.x_Co, 4), round(p.x_Cr, 4), round(p.x_Ni, 4))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def make_parent_lattice(phase: str, a: float = 3.55, ca: float = 1.623) -> Structure:
    """Reference parent lattice for mcsqs. Composition is dummy here; mcsqs handles it."""
    if phase == "FCC_A1":
        atoms = bulk("X", crystalstructure="fcc", a=a, cubic=True)
    elif phase == "HCP_A3":
        atoms = bulk("X", crystalstructure="hcp", a=a, c=a * ca)
    elif phase == "BCC_A2":
        atoms = bulk("X", crystalstructure="bcc", a=a, cubic=True)
    else:
        raise NotImplementedError(f"Use sigma_special.py for {phase}.")
    s = Structure(Lattice(atoms.cell.array), ["X"] * len(atoms), atoms.get_scaled_positions())
    return s


def write_mcsqs_input(work: Path, parent: Structure, comp_counts: dict[str, int], n_atoms: int):
    """Write rndstr.in and sqscell.out for mcsqs."""
    work.mkdir(parents=True, exist_ok=True)
    # rndstr.in: cell + sublattice occupation
    L = parent.lattice
    lines = []
    lines.append(f"{L.a:.6f} {L.b:.6f} {L.c:.6f} {L.alpha:.4f} {L.beta:.4f} {L.gamma:.4f}")
    lines.append("1 0 0")
    lines.append("0 1 0")
    lines.append("0 0 1")
    occ_str = ",".join(f"{el}={comp_counts[el] / n_atoms:.4f}" for el in ("Co", "Cr", "Ni") if comp_counts[el] > 0)
    for site in parent.sites:
        x, y, z = site.frac_coords
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {occ_str}")
    (work / "rndstr.in").write_text("\n".join(lines))

    # sqscell.out: supercell size
    n_primitive = n_atoms // len(parent.sites)
    # Pick the most cubic-looking supercell available
    if n_primitive == 8:
        sc = "2 0 0\n0 2 0\n0 0 2"
    elif n_primitive == 16:
        sc = "2 0 0\n0 2 0\n0 0 4"
    elif n_primitive == 27:
        sc = "3 0 0\n0 3 0\n0 0 3"
    elif n_primitive == 32:
        sc = "2 0 0\n0 2 0\n0 0 8"
    else:
        sc = f"{n_primitive} 0 0\n0 1 0\n0 0 1"
    (work / "sqscell.out").write_text(f"1\n\n{sc}\n")


def run_mcsqs(work: Path, mcsqs_bin: str, walltime_s: int = 600):
    """Run mcsqs in `work`. Caller is responsible for input files."""
    # Get pair/triplet correlations out to 3rd NN
    subprocess.run([mcsqs_bin, "-2=3.0", "-3=2.5", "-wr=1"], cwd=work,
                   timeout=10, check=False)
    # Actual SQS search
    try:
        subprocess.run([mcsqs_bin, "-n", str(_total_atoms(work))], cwd=work,
                       timeout=walltime_s, check=False)
    except subprocess.TimeoutExpired:
        pass  # mcsqs runs forever; we kill it
    if not (work / "bestsqs.out").exists():
        raise RuntimeError(f"mcsqs did not produce bestsqs.out in {work}")


def _total_atoms(work: Path) -> int:
    """Parse total atoms from sqscell.out × rndstr.in basis."""
    text = (work / "sqscell.out").read_text().splitlines()
    sc = np.array([[int(x) for x in line.split()] for line in text[2:5]])
    det = abs(int(round(np.linalg.det(sc))))
    n_basis = sum(1 for line in (work / "rndstr.in").read_text().splitlines()[4:] if line.strip())
    return det * n_basis


def parse_bestsqs(path: Path) -> Structure:
    """Convert ATAT bestsqs.out → pymatgen Structure."""
    lines = path.read_text().splitlines()
    coord_sys = np.array([[float(x) for x in lines[i].split()] for i in (0, 1, 2)])
    sc = np.array([[float(x) for x in lines[i].split()] for i in (3, 4, 5)])
    lattice = sc @ coord_sys
    species, fracs = [], []
    for line in lines[6:]:
        if not line.strip():
            continue
        parts = line.split()
        fracs.append([float(parts[0]), float(parts[1]), float(parts[2])])
        species.append(parts[3])
    fracs = np.array(fracs)
    # ATAT gives Cartesian-in-coord_sys coords; project into supercell fractional
    cart = fracs @ coord_sys
    inv = np.linalg.inv(lattice)
    frac = cart @ inv
    return Structure(Lattice(lattice), species, frac % 1.0)


def make_dlm_magmom_lines(structure: Structure, phase: str,
                          n_realizations: int, rng: np.random.Generator) -> list[tuple[str, list[float]]]:
    """
    Produce DLM realizations: per-atom collinear ±|m| with random sign,
    constrained so the net cell magnetization is small (DLM = paramagnetic).
    Also produce one FM-aligned 'ordered' reference (no sign flips, returned as the first item).
    """
    magmoms_list: list[tuple[str, list[float]]] = []
    species = [str(s.specie.symbol) for s in structure]
    abs_m = np.array([ORDERED_MOMENTS[(phase, sp)] for sp in species])

    # Ordered (FM) reference
    magmoms_list.append(("ordered_FM", abs_m.tolist()))

    for k in range(n_realizations):
        signs = rng.choice([-1, 1], size=len(structure))
        # Enforce |sum(signs * |m|)| / N < 0.05 μ_B to be a 'good' DLM (paramagnetic)
        # by flipping the largest-moment atom if needed
        for _attempt in range(50):
            net = float((signs * abs_m).sum() / len(signs))
            if abs(net) < 0.05:
                break
            idx = int(np.argmax(abs_m * (signs == np.sign(net))))
            signs[idx] *= -1
        magmoms_list.append((f"dlm_{k:02d}", (signs * abs_m).tolist()))
    return magmoms_list


def write_vasp_inputs(out_dir: Path, structure: Structure,
                      tag: str, magmoms: list[float]):
    """Write POSCAR + a magmoms.json sidecar that the DFT driver reads to build INCAR."""
    cfg = out_dir / tag
    cfg.mkdir(parents=True, exist_ok=True)
    write(str(cfg / "POSCAR"), structure.to_ase_atoms(), format="vasp", direct=True, sort=True)
    # Sort POSCAR by species → reorder magmoms accordingly
    species_sorted = sorted(range(len(structure)), key=lambda i: structure[i].specie.symbol)
    m_sorted = [magmoms[i] for i in species_sorted]
    (cfg / "magmoms.json").write_text(json.dumps({
        "tag": tag, "magmoms": m_sorted,
        "species_order_in_POSCAR": [structure[i].specie.symbol for i in species_sorted],
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["FCC_A1", "HCP_A3", "BCC_A2"])
    p.add_argument("--config", default="../00_config/config.yaml")
    p.add_argument("--out", default="out")
    p.add_argument("--seed", type=int, default=20260615)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    phase_cfg = cfg["phases"][args.phase]
    rng = np.random.default_rng(args.seed)

    grid = make_ternary_grid(cfg["composition_grid"]["step_ternary"],
                             cfg["composition_grid"]["step_binary"])
    print(f"[{args.phase}] {len(grid)} grid points")

    parent = make_parent_lattice(args.phase, ca=phase_cfg.get("ca_ratio", 1.623))
    n_atoms = phase_cfg["sqs_size_atoms"]
    out_root = Path(args.out) / args.phase
    out_root.mkdir(parents=True, exist_ok=True)

    for gp in grid:
        counts = gp.composition(n_atoms)
        if min(counts.values()) < 0:
            continue
        # Skip pure unaries here (handled separately for endmember references)
        nonzero = sum(1 for v in counts.values() if v > 0)
        if nonzero == 1 and gp.label not in [
            "Co1000_Cr0000_Ni0000", "Co0000_Cr1000_Ni0000", "Co0000_Cr0000_Ni1000"
        ]:
            continue

        work = out_root / gp.label / "mcsqs"
        write_mcsqs_input(work, parent, counts, n_atoms)
        print(f"  → mcsqs for {gp.label}  {counts}")
        try:
            run_mcsqs(work, cfg["paths"]["mcsqs_executable"], walltime_s=900)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"    SKIP (mcsqs not run): {e}")
            continue

        sqs = parse_bestsqs(work / "bestsqs.out")
        configs = make_dlm_magmom_lines(sqs, args.phase, phase_cfg["dlm_realizations"], rng)
        for tag, mm in configs:
            write_vasp_inputs(out_root / gp.label, sqs, tag, mm)
        print(f"    → wrote {len(configs)} configs for {gp.label}")


if __name__ == "__main__":
    main()
