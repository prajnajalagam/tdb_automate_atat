#!/usr/bin/env python3
"""
STEP 1 (multicomponent): interactive endmember selection for an N-element
alloy (N >= 2).

Generalizes the binary select_endmembers.py:

  * Single-sublattice phases (FCC_A1, BCC_A2, HCP_A3) need ONE endmember
    per pure element — N corners instead of 2.
  * SIGMA_D8B (multi-sublattice) needs one endmember per unique site-
    occupation corner. For N elements over 3 sublattices that's up to
    N**3 corners (27 for ternary). Same per-corner interactive selection
    pattern as the binary fix; --auto-sigma suppresses prompts.

Output: system.yaml consumed by sqs2tdb_pipeline_mc.py (Stage 1/2).

Usage
-----
    python3 select_endmembers_mc.py \
        --elements Co,Cr,Ni \
        --data-roots /path/to/CoCrNi_data,/path/to/CoCr_data,... \
        --out system.yaml

For non-interactive (PBS) runs add --auto-sigma to skip per-corner prompts;
single-sublattice prompts still need a terminal because the user chooses
the preferred DFT run per element. To run those non-interactively too,
just pipe an empty stdin (every prompt defaults to index 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import subsystems as sub


# ────────────────────────────────────────────────────────────────────
#  Interactive selectors
# ────────────────────────────────────────────────────────────────────

def _prompt_index(n: int, prompt: str) -> int:
    """Read an index in [0, n) from stdin; Enter or EOF accepts 0."""
    while True:
        try:
            s = input(prompt).strip()
        except EOFError:
            return 0
        if not s:
            return 0
        try:
            i = int(s)
            if 0 <= i < n:
                return i
            print(f"    ERROR: index out of range [0, {n - 1}]")
        except ValueError:
            print("    ERROR: enter an integer (or Enter for default)")


def _candidate_sort_key(c: sub.SQSCandidate):
    """Recommended sort: has-svib first, then lowest energy, then path length."""
    return (
        0 if c.has_svib else 1,
        c.energy if c.energy is not None else float("inf"),
        len(str(c.path)),
    )


def select_single_sublattice(
    phase: str,
    candidates: List[sub.SQSCandidate],
    elements: List[str],
) -> Dict[str, str]:
    """
    Pick one endmember per element for a single-sublattice phase.
    Returns {El: path_str}; elements with no candidate are omitted.
    """
    elements = [e.upper() for e in elements]
    by_element: Dict[str, List[sub.SQSCandidate]] = {e: [] for e in elements}
    for c in candidates:
        el = sub.pure_endmember_element(c.occupation, c.phase, elements)
        if el is not None:
            by_element[el].append(c)

    selected: Dict[str, str] = {}
    for el in elements:
        cands = sorted(by_element.get(el, []), key=_candidate_sort_key)
        if not cands:
            print(f"  [{phase}/{el}] no pure-{el} endmember found — skipping")
            continue
        print(f"\n  [{phase}] Pure-{el} endmember  "
              f"({len(cands)} candidate{'s' if len(cands) != 1 else ''}):")
        for i, c in enumerate(cands):
            e_str = f"{c.energy:+.6f}" if c.energy is not None else "N/A"
            sv = "YES" if c.has_svib else "NO"
            tag = "  (default)" if i == 0 else ""
            print(f"    [{i}] E={e_str}  svib={sv}{tag}")
            print(f"        {c.path}")
        if len(cands) == 1:
            selected[el] = str(cands[0].path)
        else:
            i = _prompt_index(
                len(cands),
                f"    Select {el} endmember index "
                f"[0-{len(cands) - 1}] (default 0): ",
            )
            selected[el] = str(cands[i].path)
    return selected


def select_sigma(
    candidates: List[sub.SQSCandidate],
    elements: List[str],
    auto_sigma: bool = False,
) -> List[str]:
    """
    Per-corner SIGMA selection. Returns the list of selected paths
    (one per unique corner occupation).
    """
    elements = [e.upper() for e in elements]
    by_corner: Dict[Tuple[Tuple[str, str], ...], List[sub.SQSCandidate]] = {}
    for c in candidates:
        if c.phase != "SIGMA_D8B":
            continue
        key = sub.sigma_corner_key(c.occupation, elements)
        if key is None:
            continue
        by_corner.setdefault(key, []).append(c)

    if not by_corner:
        return []

    n_expected = len(elements) ** len(sub.PHASE_SUBLATTICES["SIGMA_D8B"])
    print(f"\n{'=' * 60}")
    print(f"  Phase: SIGMA_D8B  "
          f"({len(by_corner)} of {n_expected} possible corners found)")
    print(f"{'=' * 60}")
    if not auto_sigma:
        print("  Multi-candidate corners prompt; <Enter> = recommended default.")
        print("  Pass --auto-sigma to skip prompts.")

    selected: List[str] = []
    for key in sorted(by_corner.keys()):
        cands = sorted(by_corner[key], key=_candidate_sort_key)
        cfg_str = ",".join(f"{s}_{el}=1" for s, el in key)

        if len(cands) == 1 or auto_sigma:
            c = cands[0]
            e_str = f"{c.energy:+.6f}" if c.energy is not None else "N/A"
            sv = "YES" if c.has_svib else "NO"
            print(f"\n  {cfg_str}")
            print(f"    E={e_str}  svib={sv}")
            if len(cands) > 1:
                print(f"    auto-picked from {len(cands)} candidates "
                      f"(omit --auto-sigma to choose)")
            print(f"    -> {c.path}")
            selected.append(str(c.path))
            continue

        print(f"\n  {cfg_str}   ({len(cands)} candidates)")
        for i, c in enumerate(cands):
            e_str = f"{c.energy:+.6f}" if c.energy is not None else "N/A"
            sv = "YES" if c.has_svib else "NO"
            tag = "  (default)" if i == 0 else ""
            print(f"    [{i}] E={e_str}  svib={sv}{tag}")
            print(f"        {c.path}")
        i = _prompt_index(
            len(cands),
            f"    Select index [0-{len(cands) - 1}] (default 0): ",
        )
        selected.append(str(cands[i].path))
    return selected


# ────────────────────────────────────────────────────────────────────
#  system.yaml writer
# ────────────────────────────────────────────────────────────────────

def _yaml_inline_list(items: List[str]) -> str:
    return "[" + ", ".join(items) + "]"


def write_system_yaml(
    out_path: Path,
    elements: List[str],
    selections_single: Dict[str, Dict[str, str]],
    sigma_paths: List[str],
) -> None:
    """
    Emit the multicomponent data-model YAML described in DESIGN.md §4.

    Hand-written (rather than using PyYAML) for readable, deterministic
    inline lists and to avoid an extra dependency at Stage 1.
    """
    els = list(elements)
    species_inline = _yaml_inline_list(els)
    bin_t = sub.enumerate_subsystems(els, 2, 2).get(2, [])
    ter_t = sub.enumerate_subsystems(els, 3, 3).get(3, [])
    quat_t = sub.enumerate_subsystems(els, 4, 4).get(4, [])

    with out_path.open("w") as f:
        f.write("# system.yaml — multicomponent endmember selection\n")
        f.write("# Generated by select_endmembers_mc.py\n\n")
        f.write(f"system: {'-'.join(els)}\n")
        f.write(f"elements: {species_inline}\n\n")
        f.write("phases:\n")
        for ph in ("FCC_A1", "BCC_A2", "HCP_A3"):
            sel = selections_single.get(ph, {})
            if not sel:
                continue
            site = sub.SITE_FOR_PHASE[ph]
            mult = sub.PHASE_MULT[ph][site]
            f.write(f"  {ph}:\n")
            f.write("    sublattices:\n")
            f.write(f"      - {{site: {site}, mult: {mult}, "
                    f"species: {species_inline}}}\n")
            f.write("    endmembers:\n")
            for el in els:
                if el in sel:
                    f.write(f"      {el}: {sel[el]}\n")
            f.write("\n")
        if sigma_paths:
            f.write("  SIGMA_D8B:\n")
            f.write("    sublattices:\n")
            for site in sub.PHASE_SUBLATTICES["SIGMA_D8B"]:
                m = sub.PHASE_MULT["SIGMA_D8B"][site]
                f.write(f"      - {{site: {site}, mult: {m}, "
                        f"species: {species_inline}}}\n")
            f.write("    endmembers:\n")
            f.write("      ALL:\n")
            for p in sigma_paths:
                f.write(f"        - {p}\n")
            f.write("\n")
        f.write("subsystems:\n")
        if bin_t:
            f.write("  binary:\n")
            for b in bin_t:
                f.write(f"    - {'-'.join(b)}\n")
        if ter_t:
            f.write("  ternary:\n")
            for t in ter_t:
                f.write(f"    - {'-'.join(t)}\n")
        if quat_t:
            f.write("  quaternary:\n")
            for q in quat_t:
                f.write(f"    - {'-'.join(q)}\n")
        f.write("\n")


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multicomponent endmember selection (N >= 2 elements)")
    ap.add_argument("--elements", required=True,
                    help="Comma-separated element list, e.g. Co,Cr,Ni")
    ap.add_argument("--data-roots", required=True,
                    help="Comma-separated root directories to scan")
    ap.add_argument("--scan-depth", type=int, default=6,
                    help="Max os.walk depth (default 6)")
    ap.add_argument("--out", default="system.yaml",
                    help="Output YAML path (default: ./system.yaml)")
    ap.add_argument("--phases", default=None,
                    help="Comma-separated phases (default: all four)")
    ap.add_argument("--auto-sigma", action="store_true",
                    help="Auto-pick SIGMA candidates per corner without "
                         "prompting (matches the binary --auto-sigma flag)")
    args = ap.parse_args()

    elements = sub.normalize_elements(args.elements.split(","))
    if len(elements) < 2:
        print("ERROR: need at least 2 elements", file=sys.stderr)
        return 2
    roots = [Path(r.strip()).resolve() for r in args.data_roots.split(",")]
    phases = ([p.strip() for p in args.phases.split(",")]
              if args.phases
              else ["FCC_A1", "BCC_A2", "HCP_A3", "SIGMA_D8B"])

    print(f"\n{'=' * 60}")
    print("  Multicomponent endmember selection")
    print(f"  System  : {'-'.join(elements)}")
    print(f"  Phases  : {', '.join(phases)}")
    print(f"  Roots   : {len(roots)}")
    for r in roots:
        print(f"            {r}")
    print(f"{'=' * 60}\n")

    all_cands = sub.scan_sqs(
        roots=roots, elements=elements, phases=phases,
        scan_depth=args.scan_depth, require_files=True, verbose=True,
    )
    lev0 = [c for c in all_cands if c.level == 0]
    print(f"\n  Found {len(lev0)} lev=0 candidates "
          f"(out of {len(all_cands)} total SQS dirs)\n")

    selections_single: Dict[str, Dict[str, str]] = {}
    for ph in ("FCC_A1", "BCC_A2", "HCP_A3"):
        if ph not in phases:
            continue
        ph_cands = [c for c in lev0 if c.phase == ph]
        if not ph_cands:
            print(f"\n  [{ph}] no candidates found — skipping")
            continue
        print(f"\n{'=' * 60}")
        print(f"  Phase: {ph}   ({len(ph_cands)} candidates)")
        print(f"{'=' * 60}")
        sel = select_single_sublattice(ph, ph_cands, elements)
        if sel:
            selections_single[ph] = sel

    sigma_paths: List[str] = []
    if "SIGMA_D8B" in phases:
        sigma_cands = [c for c in lev0 if c.phase == "SIGMA_D8B"]
        if sigma_cands:
            sigma_paths = select_sigma(sigma_cands, elements, args.auto_sigma)
        else:
            print("\n  [SIGMA_D8B] no candidates found — skipping")

    if not selections_single and not sigma_paths:
        print("\nERROR: nothing selected — check --data-roots and --phases",
              file=sys.stderr)
        return 1

    out = Path(args.out).resolve()
    write_system_yaml(out, elements, selections_single, sigma_paths)
    print(f"\n{'=' * 60}")
    print(f"  Wrote: {out}")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
