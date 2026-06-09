#!/usr/bin/env python3
"""
Standalone diagnostic for a reference TDB.

Run this on a node where pycalphad imports successfully (your Stage-3
venv) when score_tdb_combinations.py's equilibrium() call fails. The
output tells you:

  - Whether the TDB parses at all
  - The exact element names it contains (case included)
  - The exact phase names + each phase's sublattice model
  - Whether a minimal scalar (T, X) equilibrium succeeds — for each
    phase individually, then for the requested set together
  - Whether a small range-based equilibrium (the form Stage 3 uses)
    succeeds

NOTE: equilibrium() is called without output="NP" anywhere in this
script. pycalphad 0.11.x has a regression in its NP-only property
path (TypeError: only 0-dimensional arrays can be converted to Python
scalars). The default Dataset already contains NP, GM, MU, X, Y, and
Phase — which is everything we need. score_tdb_combinations.py was
patched the same way in commit 848c36f.

This narrows down whether the failure is:
  (a) phase-name case / spelling mismatch (FCC vs FCC_A1 vs FCC_A1#1)
  (b) a missing component
  (c) one specific phase's model that pycalphad can't construct
  (d) endpoint instability at X=0/X=1 (vs an interior sample)
  (e) something deeper in the TDB parameters

Usage
-----
  python3 diagnose_ref_tdb.py \\
      --ref-tdb /path/to/AlCoCrNi.TDB \\
      --comp-element CO \\
      --extra-comps CR

Try also:
  --T 800 --X 0.5         # other sample conditions
  --phases FCC_A1,BCC_A2  # force a specific phase list to test
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

try:
    from pycalphad import Database, equilibrium, variables as v
except ImportError:
    sys.exit("ERROR: pycalphad not importable. Activate the Stage-3 venv first.")


def _describe_phase(db, ph: str) -> str:
    """Return a one-line 'mult(species)/mult(species)/...' summary."""
    try:
        ph_obj = db.phases[ph]
        sl = ph_obj.sublattices
        cons = ph_obj.constituents
        parts = []
        for mult, species in zip(sl, cons):
            sp = ",".join(sorted(str(s) for s in species))
            parts.append(f"{mult}({sp})")
        return " / ".join(parts)
    except Exception as exc:
        return f"<could not describe: {exc}>"


def _print_eq_output(eq) -> None:
    """Dump the structure of a small equilibrium result for inspection."""
    print(f"      coords      : {dict(eq.NP.coords)}")
    print(f"      NP shape    : {eq.NP.values.shape}")
    try:
        np_squeezed = eq.NP.values.squeeze()
        print(f"      NP values   : {np_squeezed}")
    except Exception:
        print(f"      NP values   : <complex shape, skipped>")
    if hasattr(eq, "Phase"):
        try:
            ph_squeezed = eq.Phase.values.squeeze()
            print(f"      Phase tags  : {ph_squeezed}")
        except Exception:
            print(f"      Phase tags  : <complex shape, skipped>")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reference TDB diagnostic")
    ap.add_argument("--ref-tdb", required=True,
                    help="Path to the reference .tdb / .TDB file")
    ap.add_argument("--comp-element", required=True,
                    help="Element used for v.X axis (upper case, e.g. CO)")
    ap.add_argument("--extra-comps", default="",
                    help="Comma-separated other comps, e.g. CR")
    ap.add_argument("--T", type=float, default=800.0,
                    help="Sample temperature for scalar test (K)")
    ap.add_argument("--X", type=float, default=0.5,
                    help="Sample composition for scalar test")
    ap.add_argument("--P", type=float, default=101325.0,
                    help="Pressure (Pa)")
    ap.add_argument("--phases", default=None,
                    help="Comma-separated phase list (default: auto from TDB)")
    args = ap.parse_args()

    ref_tdb = Path(args.ref_tdb).resolve()
    if not ref_tdb.is_file():
        print(f"ERROR: file not found: {ref_tdb}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"  Reference TDB diagnostic")
    print(f"  File         : {ref_tdb}")
    print(f"  Sample T     : {args.T} K")
    print(f"  Sample X     : x({args.comp_element}) = {args.X}")
    print(f"  Pressure     : {args.P} Pa")
    print(f"{'=' * 60}\n")

    # ── [1] Load Database ──────────────────────────────────────
    print("[1] Loading database...")
    try:
        db = Database(str(ref_tdb))
        print("    OK")
    except Exception as exc:
        print(f"    FAILED: {exc}")
        traceback.print_exc()
        return 1

    # ── [2] Elements ────────────────────────────────────────────
    elements = sorted(db.elements)
    print(f"\n[2] Elements in TDB ({len(elements)}): {elements}")
    comp_el = args.comp_element.upper()
    extra = [e.strip().upper() for e in args.extra_comps.split(",") if e.strip()]
    requested = [comp_el] + extra
    missing = [c for c in requested if c not in elements]
    if missing:
        print(f"    *** REQUESTED elements not in TDB: {missing}")
        print(f"        Available: {elements}")
        print(f"    Hint: maybe the TDB stores them in different case "
              f"or as full names (e.g. 'COBALT'). Match exactly.")
        return 1
    if "VA" not in elements:
        print("    NOTE: 'VA' not in elements; pycalphad may still accept "
              "it as a placeholder.")
    comps = sorted({comp_el, *extra, "VA"})
    print(f"    Using comps : {comps}")

    # ── [3] Phases + sublattice models ──────────────────────────
    phases_all = sorted(db.phases.keys())
    print(f"\n[3] Phases in TDB ({len(phases_all)} total):")
    for ph in phases_all:
        print(f"    {ph:30s}  {_describe_phase(db, ph)}")

    # ── [4] Resolve test phase list ─────────────────────────────
    if args.phases:
        test_phases = [p.strip() for p in args.phases.split(",")]
    else:
        # Try every common SGTE spelling we know about, then fall back
        # to whatever the TDB advertises that matches.
        candidates = [
            "FCC_A1", "FCC_A1#1", "FCC",
            "BCC_A2", "BCC_A2#1", "BCC",
            "HCP_A3", "HCP_A3#1", "HCP",
            "SIGMA_D8B", "SIGMA",
            "LIQUID",
        ]
        test_phases = [p for p in candidates if p in phases_all]
    print(f"\n[4] Phases to test : {test_phases}")
    in_db = [p for p in test_phases if p in phases_all]
    not_in_db = [p for p in test_phases if p not in phases_all]
    if not_in_db:
        print(f"    *** Not in TDB (will be skipped): {not_in_db}")
        print(f"        Available phases listed in [3] above.")
    test_phases = in_db
    if not test_phases:
        print("    No matching phases — pass --phases <name> explicitly.")
        return 1

    # ── [5] Scalar (T, X) equilibrium per phase ─────────────────
    print("\n[5] Scalar equilibrium at sample (T, X):")
    conds_scalar = {
        v.X(comp_el): args.X,
        v.T: args.T,
        v.P: args.P,
    }
    print(f"    conds = {{v.X('{comp_el}'): {args.X}, "
          f"v.T: {args.T}, v.P: {args.P}}}")

    succeeded_singly = []
    for ph in test_phases:
        print(f"\n  → Phase: {ph}")
        try:
            eq = equilibrium(db, comps, [ph], conds_scalar)
            print("    SUCCESS")
            _print_eq_output(eq)
            succeeded_singly.append(ph)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            traceback.print_exc(limit=3)

    print(f"\n  Single-phase OK : {succeeded_singly}")
    print(f"  Single-phase BAD: "
          f"{[p for p in test_phases if p not in succeeded_singly]}")

    # ── [6] All test phases at once ─────────────────────────────
    print(f"\n[6] Combined equilibrium: phases={test_phases}")
    try:
        eq = equilibrium(db, comps, test_phases, conds_scalar)
        print("    SUCCESS")
        _print_eq_output(eq)
    except Exception as exc:
        print(f"    FAILED: {exc}")
        traceback.print_exc(limit=5)

    # ── [7] Small range eq (what Stage 3 actually uses) ─────────
    print(f"\n[7] Range equilibrium (x in [1e-4, 1-1e-4], T spanning 50K):")
    conds_range = {
        # Avoid the X=0/X=1 endpoints where pycalphad's internal solver
        # sometimes hits singularities even when the rest of the range
        # is fine. This is the most common cause of the "ragged array"
        # error in score_tdb_combinations.py's current conds.
        v.X(comp_el): (1e-4, 1 - 1e-4, 0.05),
        v.T: (args.T, args.T + 50, 50),
        v.P: args.P,
    }
    print(f"    conds = {conds_range}")
    try:
        eq = equilibrium(db, comps, succeeded_singly or test_phases,
                         conds_range)
        print("    SUCCESS")
        print(f"    NP shape: {eq.NP.values.shape}")
    except Exception as exc:
        print(f"    FAILED: {exc}")
        traceback.print_exc(limit=5)

    # ── [8] Same call with X starting at 0 (the offending form) ─
    print(f"\n[8] Range equilibrium starting at X=0 (Stage-3 form):")
    conds_zero = {
        v.X(comp_el): (0.0, 1.0, 0.05),
        v.T: (args.T, args.T + 50, 50),
        v.P: args.P,
    }
    print(f"    conds = {conds_zero}")
    try:
        eq = equilibrium(db, comps, succeeded_singly or test_phases,
                         conds_zero)
        print("    SUCCESS — Stage 3's form works after all; failure must")
        print("    have been phase-/X-grid-specific.")
    except Exception as exc:
        print(f"    FAILED: {exc}")
        print("    If this fails but [7] above SUCCEEDED, the endpoint")
        print("    instability is the bug — Stage 3 needs to use a small")
        print("    epsilon offset on X (1e-4..1-1e-4) instead of (0..1).")
        traceback.print_exc(limit=3)

    print(f"\n{'=' * 60}")
    print("  Diagnostic complete.")
    print("  Interpret:")
    print("    - Mismatched phase name?  Use the spelling from [3] in")
    print("      --eq-phases when re-running score_tdb_combinations.py.")
    print("    - One phase always fails? Drop it from --eq-phases.")
    print("    - [7] passes but [8] fails? It's the X=0 endpoint —")
    print("      we'll patch score_tdb_combinations.py to use 1e-4 offset.")
    print("    - Everything fails?  Likely a numpy/pycalphad version")
    print("      mismatch or a broken parameter in the TDB itself.")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
