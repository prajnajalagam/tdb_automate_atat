#!/usr/bin/env python3
"""
POTCAR ENMAX extraction and ENCUT / KPPRA convergence grids.

ENMAX is the recommended plane-wave cutoff (eV) printed in every VASP
POTCAR ("ENMAX  =  267.882; ENMIN ..."). For a multi-element calculation
VASP requires ENCUT >= max(ENMAX) over all species, so the convergence
sweep is anchored on the *largest* ENMAX among the POTCARs involved.

Sweep definitions (from the design spec)
-----------------------------------------
ENCUT sweep : 1.00 x MAX_ENMAX .. 1.25 x MAX_ENMAX, 5 points inclusive.
KPPRA sweep : 4000 .. 10000, step 1000 (7 points).
KPPRA sweep is run at a fixed ENCUT of 1.125 x MAX_ENMAX (mid-range, so
the k-point convergence isn't contaminated by an under-converged basis).
The converged KPPRA is then frozen while the ENCUT sweep is run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

# "ENMAX  =  267.882; ENMIN  =  200.911 eV"
_ENMAX_RE = re.compile(r"ENMAX\s*=\s*([0-9]+\.?[0-9]*)")

# Fractions of MAX_ENMAX spanned by the ENCUT sweep.
ENCUT_LOW_FACTOR = 1.00
ENCUT_HIGH_FACTOR = 1.25
ENCUT_N_POINTS = 5

# ENCUT factor used while the KPPRA sweep runs.
KPPRA_PROBE_ENCUT_FACTOR = 1.125

# KPPRA sweep.
KPPRA_MIN = 4000
KPPRA_MAX = 10000
KPPRA_STEP = 1000


def parse_enmax(potcar_path: Path) -> List[float]:
    """Return the list of ENMAX values found in a (possibly concatenated)
    POTCAR. A multi-element POTCAR has one ENMAX block per species, in the
    same order the species appear in POSCAR/str.out."""
    text = Path(potcar_path).read_text(errors="ignore")
    return [float(m) for m in _ENMAX_RE.findall(text)]


def max_enmax(potcar_paths: List[Path]) -> float:
    """Largest ENMAX across one or more POTCAR files. Raises if none found
    so a silent 0.0 cutoff can never slip through."""
    vals: List[float] = []
    for p in potcar_paths:
        p = Path(p)
        if p.is_file():
            vals.extend(parse_enmax(p))
    if not vals:
        raise ValueError(
            f"No ENMAX values found in any of: "
            f"{[str(p) for p in potcar_paths]}")
    return max(vals)


def encut_grid(max_enmax_ev: float,
               low_factor: float = ENCUT_LOW_FACTOR,
               high_factor: float = ENCUT_HIGH_FACTOR,
               n_points: int = ENCUT_N_POINTS) -> List[int]:
    """ENCUT sweep values (eV, rounded to the nearest integer), inclusive of
    both endpoints. n_points evenly spaced between low_factor*MAX and
    high_factor*MAX."""
    if n_points < 2:
        raise ValueError("ENCUT sweep needs at least 2 points")
    lo = low_factor * max_enmax_ev
    hi = high_factor * max_enmax_ev
    step = (hi - lo) / (n_points - 1)
    vals = [int(round(lo + i * step)) for i in range(n_points)]
    # De-dup while preserving order (possible if MAX is tiny).
    seen: set = set()
    out: List[int] = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def kppra_probe_encut(max_enmax_ev: float,
                      factor: float = KPPRA_PROBE_ENCUT_FACTOR) -> int:
    """ENCUT (eV) at which the KPPRA sweep is performed."""
    return int(round(factor * max_enmax_ev))


def kppra_grid(kmin: int = KPPRA_MIN,
               kmax: int = KPPRA_MAX,
               step: int = KPPRA_STEP) -> List[int]:
    """KPPRA sweep values, inclusive of kmax when it lands on the grid."""
    vals = list(range(kmin, kmax + 1, step))
    if vals and vals[-1] != kmax:
        vals.append(kmax)
    return vals


def find_potcars(work_dir: Path) -> List[Path]:
    """Locate POTCAR files relevant to a phase working directory.

    runstruct_vasp builds POTCAR from per-element pseudopotentials named in
    vasp.wrap, so a POTCAR may not exist until a run has happened. We look,
    in priority order, for:
      1. an already-assembled POTCAR in work_dir,
      2. POTCAR files inside any first-level structure subdir,
      3. per-element POTCARs under a 'potcars'/'pot' helper dir.
    Returns a de-duplicated list; caller takes max_enmax over it.
    """
    work_dir = Path(work_dir)
    found: List[Path] = []
    direct = work_dir / "POTCAR"
    if direct.is_file():
        found.append(direct)
    for sub in sorted(work_dir.iterdir()) if work_dir.is_dir() else []:
        if sub.is_dir():
            cand = sub / "POTCAR"
            if cand.is_file():
                found.append(cand)
    for helper in ("potcars", "pot", "POTCARS"):
        hp = work_dir / helper
        if hp.is_dir():
            found.extend(sorted(hp.rglob("POTCAR")))
    # De-dup by resolved path.
    seen: set = set()
    out: List[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def element_potcar_map(pot_root: Path, elements: List[str]) -> Dict[str, Path]:
    """Map element symbol -> POTCAR path under a VASP pseudopotential tree.

    Tries '<pot_root>/<EL>/POTCAR' and '<pot_root>/<EL>_*/POTCAR' (the latter
    picks the first match, e.g. Co_pv). Missing elements are simply absent
    from the returned dict so the caller can warn precisely.
    """
    pot_root = Path(pot_root)
    out: Dict[str, Path] = {}
    for el in elements:
        exact = pot_root / el / "POTCAR"
        if exact.is_file():
            out[el] = exact
            continue
        matches = sorted(pot_root.glob(f"{el}_*/POTCAR"))
        if matches:
            out[el] = matches[0]
    return out
