#!/usr/bin/env python3
"""
Adaptive refinement: put compute where the FIT says it is needed.

Two independent refiners (user spec 2026-07-22):

1. ENERGY mesh refinement from `fit_energy.out`
   ------------------------------------------------
   sqs2tdb -fit writes fit_energy.out with, per SQS, the composition,
   the first-principles energy and the value the CALPHAD fit
   reproduces for the .TDB. Where |E_dft - E_fit| peaks, the fit is
   under-supported: generate the next SQS mesh level and KEEP only the
   new compositions bracketing the worst point (one on each side);
   every other freshly generated dir is marked `refine_skip` so the
   pipeline does not burn VASP time on compositions the fit already
   nails. Rerunning the upstream job then computes exactly the two new
   points (all previously finished dirs fast-forward).

   CLI:  python3 refine.py energy --phase-root <WORK/PHASE_small> \
             --fit-energy <path/to/fit_energy.out> --level 3

2. VIBRATIONAL entropy (svib_ht) adaptivity  (--phonon-scope adaptive)
   ------------------------------------------------
   The published workflow computes phonons at the ENDMEMBERS only and
   lets sqs2tdb -fit interpolate svib linearly in composition. The
   adaptive scope tests that assumption instead of trusting it:

   a. run phonons at lev=1 (x=0.5) in addition to the endmembers;
   b. LINEAR TEST: if |svib(0.5) - mean(endmembers)| <= tol, linearity
      holds — record it and stop (no further phonon spend);
   c. refuted -> run phonons on ONE lev=2 SQS, on the side where the
      phase is more likely stable (lower mixing energy per atom);
      fit a least-squares QUADRATIC to all 4 points. The lev=1
      baseline is the exact quadratic through {0, 0.5, 1}; its
      prediction error at the new point is the "lev=1 deviation".
   d. if the 4-point quadratic's RMSE exceeds that lev=1 deviation,
      the chosen side did not help — compute the OTHER lev=2 side and
      keep whichever 4-point fit has the lower RMSE; else keep and
      finish. Decisions + coefficients land in svib_adaptive.json.

   svib_ht values are normalized PER ATOM before comparison (fitfc
   writes per-cell values; endmember cells and lev>=1 supercells have
   very different atom counts).

All ATAT interaction goes through the existing sqsgen/phonon modules —
this module only decides WHERE to spend, per CLAUDE.md's rule of using
prewritten ATAT machinery for the actual work.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Composition tokens in a decorated calc-dir name: site_El=frac
_COMP_TOKEN_RE = re.compile(r"[a-z]+_([A-Z][a-z]?)[+-]?\d*=([0-9.]+)")

REFINE_SKIP = "refine_skip"      # marker: generated but NOT selected
REFINE_PICK = "refine_pick"      # marker: selected by the refiner


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def composition_fraction(dirname: str, element: str) -> Optional[float]:
    """Fraction of `element` parsed from a decorated dir name (site
    multiplicities ignored — adequate for single-sublattice phases,
    which is where composition-mesh refinement applies)."""
    fr: Dict[str, float] = {}
    for el, v in _COMP_TOKEN_RE.findall(dirname):
        fr[el] = fr.get(el, 0.0) + float(v)
    tot = sum(fr.values())
    if tot <= 0:
        return None
    return fr.get(element, 0.0) / tot


def _count_atoms(str_out: Path) -> Optional[int]:
    try:
        from strfile import read_structure
        return len(read_structure(str_out).atoms) or None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# 1. energy-mesh refinement from fit_energy.out
# ---------------------------------------------------------------------------

def parse_fit_energy(path: Path) -> List[Tuple[float, float, float]]:
    """Rows of fit_energy.out as (x, e_dft, e_fit).

    Format tolerance: any line of >= 3 floats counts; the FIRST column
    is the composition coordinate, the last two are the first-
    principles energy and the fitted value (sqs2tdb -fit layout).
    """
    rows: List[Tuple[float, float, float]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 3:
            rows.append((vals[0], vals[-2], vals[-1]))
    return rows


def worst_fit_point(rows: Sequence[Tuple[float, float, float]]
                    ) -> Tuple[float, float]:
    """(x*, |E_dft - E_fit|) at the composition where the fit is worst."""
    if not rows:
        raise ValueError("fit_energy.out contained no data rows")
    x, d, f = max(rows, key=lambda r: abs(r[1] - r[2]))
    return x, abs(d - f)


def refinement_targets(xs: Sequence[float], x_star: float
                       ) -> List[float]:
    """Target compositions bracketing x*: the midpoints between x* and
    its nearest already-sampled neighbour on EACH side (user spec:
    'sample on either side of where the error is the highest')."""
    lo = [x for x in xs if x < x_star]
    hi = [x for x in xs if x > x_star]
    targets: List[float] = []
    if lo:
        targets.append((max(lo) + x_star) / 2.0)
    if hi:
        targets.append((x_star + min(hi)) / 2.0)
    return targets


def select_new_dirs(phase_root: Path,
                    targets: Sequence[float],
                    element: str) -> Dict[str, Optional[str]]:
    """Among freshly generated dirs (str.out but NO energy yet), pick
    the nearest composition to each target; mark every other new dir
    `refine_skip` so the pipeline ignores it. Existing computed dirs
    are never touched. Returns {target: chosen dirname or None}."""
    phase_root = Path(phase_root)
    fresh: List[Tuple[float, Path]] = []
    for d in sorted(phase_root.iterdir()):
        if not d.is_dir() or not (d / "str.out").is_file():
            continue
        if (d / "energy").is_file():          # already computed: keep
            continue
        x = composition_fraction(d.name, element)
        if x is not None:
            fresh.append((x, d))

    chosen: Dict[str, Optional[str]] = {}
    picked: set = set()
    for t in targets:
        best = min(((abs(x - t), x, d) for x, d in fresh
                    if d not in picked), default=None)
        if best is None:
            chosen[f"{t:.4f}"] = None
            continue
        _dist, _x, d = best
        picked.add(d)
        chosen[f"{t:.4f}"] = d.name
        (d / REFINE_PICK).write_text(f"target x={t:.4f}\n")
        skip = d / REFINE_SKIP
        if skip.is_file():
            skip.unlink()
    for x, d in fresh:
        if d not in picked and not (d / REFINE_PICK).is_file():
            (d / REFINE_SKIP).write_text(
                "generated for refinement but not selected — the fit "
                "does not need this composition (yet)\n")
    return chosen


def refine_energy_mesh(phase_root: Path,
                       fit_energy: Path,
                       level: int,
                       element: str,
                       min_err_ev: float = 0.001,
                       env_bin: Optional[str] = None) -> Dict:
    """End-to-end energy refinement: parse -> locate worst point ->
    generate next mesh level -> keep only the bracketing compositions.
    Returns (and writes to <phase_root>/refinement_plan.json) the plan.
    """
    import sqsgen
    phase_root = Path(phase_root)
    rows = parse_fit_energy(fit_energy)
    x_star, err = worst_fit_point(rows)
    plan: Dict = {"fit_energy": str(fit_energy),
                  "x_star": x_star, "max_err_ev": err,
                  "min_err_ev": min_err_ev, "level": level,
                  "element": element}
    if err < min_err_ev:
        plan["action"] = "none — fit already within min_err_ev"
        (phase_root / "refinement_plan.json").write_text(
            json.dumps(plan, indent=2))
        return plan

    targets = refinement_targets(sorted(r[0] for r in rows), x_star)
    plan["targets"] = targets
    # sqs2tdb -cp is cumulative and skips existing dirs, so this only
    # ADDS the new level's structures.
    sqsgen.generate_phase_sqs(phase_root.parent, phase_root.name,
                              elements=None if
                              (phase_root.parent / "species.in").is_file()
                              else [element],
                              level=level, use_small=False,
                              env_bin=env_bin)
    plan["chosen"] = select_new_dirs(phase_root, targets, element)
    plan["action"] = ("rerun the upstream job — finished dirs fast-"
                      "forward; only the chosen dirs compute")
    (phase_root / "refinement_plan.json").write_text(
        json.dumps(plan, indent=2))
    return plan


# ---------------------------------------------------------------------------
# 2. adaptive vibrational-entropy sampling
# ---------------------------------------------------------------------------

def svib_per_atom(sqs_dir: Path) -> Optional[float]:
    """svib_ht normalized per atom (fitfc writes per-cell values)."""
    f = Path(sqs_dir) / "svib_ht"
    if not f.is_file():
        return None
    try:
        val = float(f.read_text().split()[0])
    except (ValueError, IndexError, OSError):
        return None
    n = _count_atoms(Path(sqs_dir) / "str.out")
    return val / n if n else None


def _quad_lstsq(pts: Sequence[Tuple[float, float]]
                ) -> Tuple[List[float], float]:
    """Least-squares quadratic a+bx+cx^2 through pts -> (coeffs, rmse)."""
    import numpy as np
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    A = np.vstack([np.ones_like(xs), xs, xs * xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = A @ coef - ys
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return [float(c) for c in coef], rmse


def _mixing_energy_per_atom(sqs_dir: Path, x: float,
                            e0: float, e1: float) -> Optional[float]:
    """E_mix(x) = E/atom - [(1-x)E0 + xE1] (per-atom endmember refs)."""
    e = Path(sqs_dir) / "energy"
    n = _count_atoms(Path(sqs_dir) / "str.out")
    try:
        val = float(e.read_text().split()[0])
    except (ValueError, IndexError, OSError):
        return None
    if not n:
        return None
    return val / n - ((1.0 - x) * e0 + x * e1)


def adaptive_svib_phase(phase_root: Path,
                        element: str,
                        run_phonon: Callable[[Path], None],
                        tol: float = 0.1,
                        log=print) -> Dict:
    """Decision tree for --phonon-scope adaptive (user spec 2026-07-22).

    Requires endmember + lev=1 phonons to already exist (the adaptive
    scope runs them during normal phase processing). `run_phonon(dir)`
    is invoked only for the lev=2 side(s) this tree decides to buy.
    tol is in the same per-atom units as svib_ht/atom (k_B/atom).
    """
    phase_root = Path(phase_root)
    dirs: Dict[float, Path] = {}
    for d in sorted(phase_root.iterdir()):
        if d.is_dir() and (d / "str.out").is_file():
            x = composition_fraction(d.name, element)
            if x is not None:
                dirs[round(x, 4)] = d

    out: Dict = {"element": element, "tol_per_atom": tol}

    def _svib(x: float) -> Optional[float]:
        return svib_per_atom(dirs[x]) if x in dirs else None

    s0, s1, smid = _svib(0.0), _svib(1.0), _svib(0.5)
    if s0 is None or s1 is None or smid is None:
        out["decision"] = ("insufficient data — need svib_ht at both "
                           "endmembers and lev=1 (x=0.5)")
        out["have"] = {str(x): (_svib(x) is not None) for x in dirs}
        return out

    linear_pred = 0.5 * (s0 + s1)
    dev = smid - linear_pred
    out["endmembers_per_atom"] = [s0, s1]
    out["lev1_per_atom"] = smid
    out["linear_deviation"] = dev
    if abs(dev) <= tol:
        out["decision"] = ("linear interpolation HOLDS (|dev| <= tol); "
                           "svib fitted from endmembers + lev=1, no "
                           "lev=2 phonons purchased")
        out["model"] = {"kind": "linear", "svib0": s0, "svib1": s1}
        _write(phase_root, out)
        return out

    log(f"    [svib-adaptive] linearity REFUTED: dev "
        f"{dev:+.4f}/atom > tol {tol} — buying a lev=2 phonon point")

    # side where the phase is more likely stable = lower mixing energy
    sides = [x for x in (0.25, 0.75) if x in dirs]
    if not sides:
        out["decision"] = "refuted, but no lev=2 SQS dirs exist"
        _write(phase_root, out)
        return out
    e0pa = _energy_pa(dirs.get(0.0))
    e1pa = _energy_pa(dirs.get(1.0))
    if e0pa is not None and e1pa is not None and len(sides) == 2:
        emix = {x: _mixing_energy_per_atom(dirs[x], x, e0pa, e1pa)
                for x in sides}
        out["mixing_energy_per_atom"] = {str(x): emix[x] for x in sides}
        sides.sort(key=lambda x: (emix[x] if emix[x] is not None
                                  else float("inf")))

    base = [(0.0, s0), (0.5, smid), (1.0, s1)]
    quad3, _ = _quad_lstsq(base)          # exact through 3 points

    def _eval(coef, x):
        return coef[0] + coef[1] * x + coef[2] * x * x

    tried: List[Dict] = []
    best = None
    for i, x_side in enumerate(sides):
        if svib_per_atom(dirs[x_side]) is None:
            log(f"    [svib-adaptive] phonons at x={x_side} "
                f"({dirs[x_side].name})")
            run_phonon(dirs[x_side])
        s_new = svib_per_atom(dirs[x_side])
        if s_new is None:
            tried.append({"x": x_side, "error": "phonons failed / "
                          "no svib_ht — see the dir's fitfc logs"})
            continue
        lev1_dev = abs(_eval(quad3, x_side) - s_new)
        coef4, rmse4 = _quad_lstsq(base + [(x_side, s_new)])
        rec = {"x": x_side, "svib_per_atom": s_new,
               "lev1_prediction_error": lev1_dev,
               "quad4_coeffs": coef4, "quad4_rmse": rmse4}
        tried.append(rec)
        if best is None or rmse4 < best["quad4_rmse"]:
            best = rec
        # Keep-or-try-other-side rule. NOTE on the spec ("if RMSE of
        # the new fit is higher than lev=1, do the other lev=2"): a
        # least-squares 4-point quadratic ALWAYS has lower RMSE than
        # the lev=1 quadratic evaluated on the same 4 points, so the
        # comparison is operationalized as: the new fit must describe
        # the data to within the SAME tolerance demanded of linearity
        # (tol). lev1_prediction_error is recorded as a diagnostic.
        if rmse4 <= tol:
            break                      # quadratic model holds: finish
        log(f"    [svib-adaptive] quadratic RMSE {rmse4:.4f} > tol "
            f"{tol} at x={x_side} — trying the other lev=2 side")

    out["tried"] = tried
    if best is None:
        out["decision"] = "refuted, but no lev=2 phonon run succeeded"
    else:
        out["decision"] = (f"QUADRATIC svib fit kept (x={best['x']} "
                           f"included, RMSE {best['quad4_rmse']:.4f})")
        out["model"] = {"kind": "quadratic",
                        "coeffs": best["quad4_coeffs"],
                        "rmse": best["quad4_rmse"]}
    _write(phase_root, out)
    return out


def _energy_pa(d: Optional[Path]) -> Optional[float]:
    if d is None:
        return None
    try:
        val = float((d / "energy").read_text().split()[0])
    except (ValueError, IndexError, OSError):
        return None
    n = _count_atoms(d / "str.out")
    return val / n if n else None


def _write(phase_root: Path, out: Dict) -> None:
    (Path(phase_root) / "svib_adaptive.json").write_text(
        json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("energy", help="mesh refinement from fit_energy.out")
    e.add_argument("--phase-root", type=Path, required=True,
                   help="e.g. <WORK_ROOT>/FCC_A1_small")
    e.add_argument("--fit-energy", type=Path, required=True)
    e.add_argument("--level", type=int, required=True,
                   help="next sqs2tdb mesh level to generate (e.g. 3)")
    e.add_argument("--element", required=True,
                   help="element whose fraction is the x axis (the "
                        "same convention as fit_energy.out col 1)")
    e.add_argument("--min-err-ev", type=float, default=0.001,
                   help="skip refinement when max |E_dft-E_fit| is "
                        "already below this (eV/atom-scale, default "
                        "1 meV)")
    args = ap.parse_args(argv)

    if args.cmd == "energy":
        plan = refine_energy_mesh(args.phase_root, args.fit_energy,
                                  args.level, args.element,
                                  min_err_ev=args.min_err_ev)
        print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
