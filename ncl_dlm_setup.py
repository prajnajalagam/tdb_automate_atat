#!/usr/bin/env python3
"""
Convert an ezvasp-generated collinear DLM input set into a constrained
NONCOLLINEAR VASP calculation, inside an ATAT/sqs2tdb structure directory.

Designed to slot into a jobfile between `runstruct_vasp -nr` and the mpirun:

    runstruct_vasp -nr                        # generate INCAR/POSCAR/POTCAR/KPOINTS
    python3 ncl_dlm_setup.py setup --lambda 10
    mpirun -np $NP vasp_ncl                   # NONCOLLINEAR binary!
    runstruct_vasp -ex                        # write ATAT `energy` file
    python3 ncl_dlm_setup.py fixenergy        # subtract penalty E_p from `energy`

What `setup` does:
  * reads the collinear MAGMOM that ezvasp built from the DLM +/- labels
    (magnitudes are kept as the per-site seed moments),
  * assigns each magnetic site a unit vector; two modes:
      --mode random  (default) per-site orientations optimized so that the
                     net moment ~ 0 and the nearest-neighbor-shell average
                     <e_i . e_j> ~ 0  (noncollinear DLM),
      --mode rotate  keep the collinear +/- partition but rotate the common
                     axis randomly (sanity check: same physics as collinear),
  * rewrites INCAR: 3N-component MAGMOM, M_CONSTR (unit vectors),
    LNONCOLLINEAR/GGA_COMPAT/ISYM/I_CONSTRAINED_M=1/LAMBDA/RWIGS and
    conservative mixing tags; removes ISPIN/NBANDS. Original INCAR is
    backed up to INCAR.collinear.

What `fixenergy` does:
  * reads the last penalty energy E_p from OUTCAR and rewrites the ATAT
    `energy` file as (E - E_p), backing up the raw value to energy.raw.

Only numpy is required. Deterministic: the RNG is seeded from the POSCAR
contents (override with --seed).
"""

import argparse
import hashlib
import os
import re
import shutil
import sys

import numpy as np

# per-species defaults (edit or override on the command line)
DEFAULT_RWIGS = {"Co": 1.302, "Cr": 1.323, "Ni": 1.286}
NCL_TAGS = {          # written/overwritten by `setup`
    "LNONCOLLINEAR": ".TRUE.",
    "GGA_COMPAT": ".FALSE.",
    "LSORBIT": ".FALSE.",
    "ISYM": "-1",
    "I_CONSTRAINED_M": "1",
    "LORBIT": "11",
}
ADD_IF_MISSING = {    # only added when the tag is not already in INCAR
    "ALGO": "Normal",
    "NELM": "200",
    "AMIX": "0.2",
    "BMIX": "0.0001",
    "AMIX_MAG": "0.8",
    "BMIX_MAG": "0.0001",
    "LMAXMIX": "4",
}
REMOVE_TAGS = {"ISPIN", "NBANDS", "MAGMOM", "M_CONSTR"}


# --------------------------------------------------------------- file parsing
def read_poscar(path):
    """Return (cell 3x3, cartesian positions Nx3, counts per species block)."""
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh]
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]]
                     for i in (2, 3, 4)]) * scale
    i = 5
    tok = lines[i].split()
    if not tok[0].lstrip("+-").isdigit():        # VASP5 symbols line
        i += 1
        tok = lines[i].split()
    counts = [int(t) for t in tok]
    n = sum(counts)
    i += 1
    if lines[i].strip().lower().startswith("s"):  # selective dynamics
        i += 1
    cartesian = lines[i].strip().lower().startswith(("c", "k"))
    i += 1
    pos = np.array([[float(x) for x in lines[i + j].split()[:3]]
                    for j in range(n)])
    if not cartesian:
        pos = pos @ cell
    else:
        pos = pos * scale
    return cell, pos, counts


def read_potcar_species(path):
    """Species name per POSCAR block, from POTCAR TITEL lines."""
    species = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "TITEL" in line:
                # e.g. "TITEL  = PAW_PBE Co_pv 02Aug2007" -> "Co"
                name = line.split("=")[1].split()[1]
                species.append(re.split(r"[_\d]", name)[0])
    return species


def parse_incar(path):
    """Return list of (key, value, rawline); non-tag lines get key=None."""
    entries = []
    with open(path) as fh:
        for raw in fh:
            m = re.match(r"\s*([A-Za-z_]+)\s*=\s*(.*?)\s*$", raw)
            if m:
                entries.append((m.group(1).upper(), m.group(2), raw))
            else:
                entries.append((None, None, raw))
    return entries


def expand_magmom(value):
    """Expand a scalar MAGMOM string ('8*1.5 8*-1.5 4*0' or plain list)."""
    out = []
    for tok in value.split():
        if "*" in tok:
            n, v = tok.split("*")
            out.extend([float(v)] * int(n))
        else:
            out.append(float(tok))
    return np.array(out)


# ------------------------------------------------------- orientation assembly
def structure_seed(poscar_path, user_seed):
    if user_seed is not None:
        return user_seed
    with open(poscar_path, "rb") as fh:
        return int(hashlib.sha1(fh.read()).hexdigest()[:8], 16)


def random_unit(rng, n=1):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def nn_pairs(cell, pos, shell_tol=1.05):
    """Nearest-neighbor pairs via minimum image (fine for compact SQS cells)."""
    n = len(pos)
    frac = pos @ np.linalg.inv(cell)
    d = np.full((n, n), np.inf)
    for a in range(n):
        df = frac - frac[a]
        df -= np.round(df)
        d[a] = np.linalg.norm(df @ cell, axis=1)
        d[a, a] = np.inf
    dmin = d.min()
    ii, jj = np.where(d < dmin * shell_tol)
    return [(a, b) for a, b in zip(ii, jj) if a < b], dmin


def optimize_orientations(rng, n_atoms, active, pairs, sweeps=400, tries=24):
    """Unit vectors with ~zero net moment and ~zero NN-shell correlation.

    Greedy per-site resampling of cost = |sum e|^2/N^2 + <e_i.e_j>_NN^2.
    `active` marks magnetic sites; inactive sites keep zero vectors.
    """
    e = np.zeros((n_atoms, 3))
    idx = np.where(active)[0]
    e[idx] = random_unit(rng, len(idx))
    pairs = [(a, b) for a, b in pairs if active[a] and active[b]]

    def cost(ev):
        c1 = np.sum(ev[idx].sum(axis=0) ** 2) / max(len(idx), 1) ** 2
        if pairs:
            c2 = np.mean([ev[a] @ ev[b] for a, b in pairs]) ** 2
        else:
            c2 = 0.0
        return c1 + c2

    best = cost(e)
    for _ in range(sweeps):
        if best < 1e-6:
            break
        for a in rng.permutation(idx):
            trial_vecs = random_unit(rng, tries)
            old = e[a].copy()
            for v in trial_vecs:
                e[a] = v
                c = cost(e)
                if c < best:
                    best, old = c, v.copy()
            e[a] = old
    return e, best


# ------------------------------------------------------------------ commands
def cmd_setup(args):
    d = args.dir
    incar_p = os.path.join(d, "INCAR")
    poscar_p = os.path.join(d, "POSCAR")
    potcar_p = os.path.join(d, "POTCAR")
    for p in (incar_p, poscar_p, potcar_p):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found -- run `runstruct_vasp -nr` first.")

    entries = parse_incar(incar_p)
    tags = {k: v for k, v, _ in entries if k}
    if "LNONCOLLINEAR" in tags and not args.force:
        sys.exit("INCAR already noncollinear; use --force to redo.")

    cell, pos, counts = read_poscar(poscar_p)
    n = len(pos)
    species = read_potcar_species(potcar_p)
    if len(species) != len(counts):
        sys.exit(f"ERROR: {len(species)} POTCAR entries vs "
                 f"{len(counts)} POSCAR blocks.")
    site_species = [s for s, c in zip(species, counts) for _ in range(c)]

    # ---- per-site seed magnitudes: --mag overrides, else collinear MAGMOM
    mag_over = dict(kv.split("=") for kv in args.mag.split(",")) if args.mag \
        else {}
    if "MAGMOM" in tags:
        m_col = expand_magmom(tags["MAGMOM"])
        if len(m_col) != n:
            sys.exit(f"ERROR: MAGMOM has {len(m_col)} values for {n} atoms.")
    elif mag_over:
        m_col = np.zeros(n)
    else:
        sys.exit("ERROR: no scalar MAGMOM in INCAR (did vasp.wrap use "
                 "MAGATOM?) and no --mag given.")
    mags = np.array([abs(float(mag_over.get(s, m_col[i])))
                     for i, s in enumerate(site_species)])
    signs = np.sign(m_col) if "MAGMOM" in tags else np.ones(n)
    active = mags > 1e-6

    # ---- orientations
    rng = np.random.default_rng(structure_seed(poscar_p, args.seed))
    if args.mode == "rotate":
        axis = random_unit(rng)[0]
        e = np.outer(np.where(signs == 0, 0.0, signs), axis)
        e[~active] = 0.0
        corr_note = "collinear +/- partition on random common axis"
    else:
        pairs, dmin = nn_pairs(cell, pos, args.shell_tol)
        e, resid = optimize_orientations(rng, n, active, pairs,
                                         sweeps=args.sweeps)
        corr_note = (f"{len(pairs)} NN pairs (d~{dmin:.2f} A), "
                     f"residual cost {resid:.2e}")
    print(f"  orientations: mode={args.mode}, {corr_note}")

    magmom = np.round(e * mags[:, None], 6)
    mconstr = np.round(e, 6)

    # ---- rebuild INCAR
    shutil.copy2(incar_p, incar_p + ".collinear")
    fmt = lambda arr: "  ".join(f"{x:.4f} {y:.4f} {z:.4f}"
                                for x, y, z in arr)
    rwigs_over = dict(kv.split("=") for kv in args.rwigs.split(",")) \
        if args.rwigs else {}
    rwigs = []
    for s in species:
        r = rwigs_over.get(s, DEFAULT_RWIGS.get(s))
        if r is None:
            sys.exit(f"ERROR: no RWIGS default for '{s}'; pass --rwigs {s}=...")
        rwigs.append(f"{float(r):.3f}")

    new = dict(NCL_TAGS)
    new["LAMBDA"] = str(args.lam)
    new["RWIGS"] = " ".join(rwigs)
    new["MAGMOM"] = fmt(magmom)
    new["M_CONSTR"] = fmt(mconstr)

    out, present = [], set()
    for k, v, raw in entries:
        if k in REMOVE_TAGS:
            continue
        if k in new:
            out.append(f"{k} = {new.pop(k)}\n")
            present.add(k)
        else:
            out.append(raw)
            if k:
                present.add(k)
    out.append("\n# --- added by ncl_dlm_setup.py ---\n")
    for k, v in new.items():
        out.append(f"{k} = {v}\n")
    for k, v in ADD_IF_MISSING.items():
        if k not in present:
            out.append(f"{k} = {v}\n")

    if args.dry_run:
        sys.stdout.writelines(out)
        return
    with open(incar_p, "w") as fh:
        fh.writelines(out)
    print(f"  INCAR rewritten (backup: INCAR.collinear); "
          f"run with vasp_ncl, LAMBDA={args.lam}")


def cmd_fixenergy(args):
    d = args.dir
    outcar_p = os.path.join(d, "OUTCAR")
    energy_p = os.path.join(d, "energy")
    if not os.path.exists(energy_p):
        sys.exit("ERROR: no `energy` file -- run `runstruct_vasp -ex` first.")
    if args.ep is not None:
        ep = args.ep
    else:
        ep, hits = None, []
        with open(outcar_p, errors="ignore") as fh:
            for line in fh:
                m = re.search(r"E_?p\s*=\s*([-+0-9.EeDd]+)", line)
                if m:
                    hits.append(float(m.group(1).replace("D", "E")))
        if hits:
            ep = hits[-1]
    if ep is None:
        print("  WARNING: no E_p found in OUTCAR (unconstrained run?); "
              "`energy` left unchanged.")
        return
    with open(energy_p) as fh:
        e_raw = float(fh.read().split()[0])
    shutil.copy2(energy_p, energy_p + ".raw")
    with open(energy_p, "w") as fh:
        fh.write(f"{e_raw - ep:.8f}\n")
    print(f"  energy: {e_raw:.6f} - E_p({ep:.6f}) = {e_raw - ep:.6f} "
          f"(raw kept in energy.raw)")
    if abs(ep) > args.ep_warn:
        print(f"  WARNING: |E_p| = {abs(ep):.4f} eV > {args.ep_warn} eV -- "
              f"constraint poorly satisfied; consider raising LAMBDA.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("setup", help="rewrite INCAR for constrained "
                                     "noncollinear DLM")
    s.add_argument("--dir", default=".", help="structure directory")
    s.add_argument("--mode", choices=["random", "rotate"], default="random")
    s.add_argument("--lambda", dest="lam", type=float, default=10)
    s.add_argument("--rwigs", default="", help="override, e.g. Co=1.30,Cr=1.32")
    s.add_argument("--mag", default="", help="seed moment override per "
                                             "species, e.g. Ni=0.6")
    s.add_argument("--seed", type=int, default=None,
                   help="RNG seed (default: hash of POSCAR)")
    s.add_argument("--shell-tol", type=float, default=1.05)
    s.add_argument("--sweeps", type=int, default=400)
    s.add_argument("--force", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_setup)

    f = sub.add_parser("fixenergy", help="subtract penalty E_p from the "
                                         "ATAT energy file")
    f.add_argument("--dir", default=".")
    f.add_argument("--ep", type=float, default=None,
                   help="penalty energy override (eV)")
    f.add_argument("--ep-warn", type=float, default=0.005,
                   help="warn if |E_p| exceeds this (eV)")
    f.set_defaults(func=cmd_fixenergy)

    args = ap.parse_args()
    if not args.cmd:
        ap.error("choose a command: setup | fixenergy")
    args.func(args)


if __name__ == "__main__":
    main()
