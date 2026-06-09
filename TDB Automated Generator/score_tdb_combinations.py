#!/usr/bin/env python3
"""
STAGE 3: Cross-phase TDB combination and equilibrium match scoring.

Reads the tdb_manifest.json produced by sqs2tdb_pipeline.py (Stages 1-2),
enumerates all cross-phase combinations of surviving per-phase TDB files,
combines them via `sqs2tdb -tdb`, and scores each combined TDB against a
reference database using pycalphad phase-fraction comparison.

This is designed to run as a SEPARATE step after the fitting pipeline
completes, potentially on a different node with pycalphad installed.

Usage:
  python3 score_tdb_combinations.py \
    --manifest /path/to/tdb_manifest.json \
    --ref-tdb /path/to/reference.tdb \
    --phases FCC_A1,HCP_A3,BCC_A2 \
    --comp-element CO \
    --T-range 500,1200,50 \
    --X-grid 0.005 \
    --n-workers 8
"""

import argparse
import heapq
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr
    from pycalphad import Database, equilibrium, variables as v
except ImportError:
    sys.exit(
        "ERROR: pycalphad and xarray required.\n"
        "  pip install pycalphad xarray"
    )


# ====================================================================
# Scoring logic (from your match-score code, cleaned up)
# ====================================================================

# Phase-name aliasing.
# Different databases label the same physical phase with different names —
# e.g. our sqs2tdb-fitted TDBs write the topologically-close-packed sigma
# phase as SIGMA_D8B, while several public reference TDBs (incl. SGTE-
# style AlCoCrNi) write SIGMA_SGTE for the same phase. To score one
# against the other, we treat all members of an alias group as equivalent:
# the user's requested phase list is the "label" used for the comparison
# array, but for each database we look up whichever group member actually
# exists in that database and pass THAT name to pycalphad. Result tags
# are then mapped back to the requested label before NP arrays are
# concatenated, so test- and ref-side NP arrays share the same `phase`
# dimension and L1 / boundary metrics make sense.
#
# To add more aliases at runtime, pass --phase-aliases on the CLI; the
# built-in default list below is appended to.
DEFAULT_PHASE_ALIAS_GROUPS: List[List[str]] = [
    ["SIGMA", "SIGMA_D8B", "SIGMA_SGTE"],
]


def parse_alias_arg(arg: Optional[str]) -> List[List[str]]:
    """
    Parse a --phase-aliases CLI string into a list of alias groups.

    Format: semicolon-separated groups, each group a comma-separated
    list of phase names.

      "SIGMA_D8B,SIGMA_SGTE;FCC#1,FCC_A1"
        ->  [["SIGMA_D8B", "SIGMA_SGTE"], ["FCC#1", "FCC_A1"]]
    """
    if not arg:
        return []
    out: List[List[str]] = []
    for chunk in arg.split(";"):
        members = [m.strip() for m in chunk.split(",") if m.strip()]
        if members:
            out.append(members)
    return out


def merge_alias_groups(*group_lists: List[List[str]]) -> List[List[str]]:
    """Union alias groups across multiple sources, deduplicating."""
    seen: List[set] = []
    for groups in group_lists:
        for g in groups:
            gs = set(g)
            # Merge into any existing group that overlaps.
            merged = gs
            keep: List[set] = []
            for existing in seen:
                if existing & merged:
                    merged = merged | existing
                else:
                    keep.append(existing)
            keep.append(merged)
            seen = keep
    return [sorted(g) for g in seen]


def resolve_phase_for_db(
    requested: str, db_phase_names: set, alias_groups: List[List[str]]
) -> Optional[str]:
    """
    Given a phase the user asked for and the actual phase names in a
    particular database, return the database's name for that phase
    (preferring an exact match, then any alias-group member that exists).
    None if no member of the alias group is in the database.
    """
    if requested in db_phase_names:
        return requested
    for g in alias_groups:
        if requested in g:
            for member in g:
                if member != requested and member in db_phase_names:
                    return member
            return None
    return None


def build_phase_fraction_array(
    eq_result,
    requested_phases: List[str],
    P: float,
    actual_per_request: Optional[Dict[str, str]] = None,
):
    """
    Convert a pycalphad equilibrium result to a (phase, T, X) NP array
    keyed by the user's REQUESTED phase names.

    `actual_per_request` maps each requested name to whatever name
    pycalphad actually tagged it with (per-database resolution; see
    `resolve_phase_for_db`). For phases not present in this database,
    NP contributes a zero column so the output shape is invariant in
    `requested_phases`.
    """
    actual_per_request = actual_per_request or {}
    NP = eq_result.NP.sel(P=P)

    per_phase = []
    zero_template = None
    for req in requested_phases:
        actual = actual_per_request.get(req, req)
        # The mask picks rows where pycalphad tagged this exact name.
        ph_np = NP.where(eq_result.Phase == actual).fillna(0.0)
        per_phase.append(ph_np)
        if zero_template is None:
            zero_template = ph_np * 0.0

    NP_phase = xr.concat(
        per_phase,
        dim=xr.DataArray(requested_phases, dims="phase", name="phase"))

    if "vertex" in NP_phase.dims:
        NP_phase = NP_phase.sum("vertex")

    s = NP_phase.sum("phase")
    NP_phase = xr.where(s > 0, NP_phase / s, NP_phase)
    return NP_phase


def boundary_indicator(stable_bool, x_dim: str):
    """True where the stable phase set changes between adjacent x points."""
    shifted = stable_bool.shift({x_dim: -1})
    change = (stable_bool != shifted).any("phase").fillna(False)
    return change


def boundary_misplacement_penalty(
    NP_test, NP_ref, stable_tol: float = 1e-6, x_dim: str = None
):
    """
    Per-temperature penalty in [0, 1] for shifted phase boundaries.
    Uses symmetric mean nearest-boundary distance normalized by grid size.
    """
    if x_dim is None:
        candidates = [d for d in NP_test.dims if d.startswith("X_")]
        if len(candidates) != 1:
            raise ValueError(f"Cannot infer composition dim: {candidates}")
        x_dim = candidates[0]

    b_test = boundary_indicator(NP_test > stable_tol, x_dim)
    b_ref = boundary_indicator(NP_ref > stable_tol, x_dim)

    other_dims = [d for d in b_ref.dims if d != x_dim]
    bT = b_test.transpose(*other_dims, x_dim).values
    bR = b_ref.transpose(*other_dims, x_dim).values
    nX = bR.shape[-1]

    penalty = np.zeros(bR.shape[:-1], dtype=float)
    it = np.nditer(penalty, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        ti = np.flatnonzero(bT[idx])
        ri = np.flatnonzero(bR[idx])

        if ri.size == 0 and ti.size == 0:
            it[0] = 0.0
        elif ri.size == 0 or ti.size == 0:
            it[0] = 1.0
        else:
            d_rt = np.min(np.abs(ri[:, None] - ti[None, :]), axis=1)
            d_tr = np.min(np.abs(ti[:, None] - ri[None, :]), axis=1)
            it[0] = 0.5 * (d_rt.mean() + d_tr.mean()) / max(nX - 1, 1)
        it.iternext()

    return xr.DataArray(
        penalty,
        coords={d: b_ref.coords[d] for d in other_dims},
        dims=other_dims, name="boundary_penalty")


def enumerate_phase_pairs(phases: List[str]) -> List[Tuple[str, ...]]:
    """All canonical (sorted) 2-phase combinations of `phases`."""
    return [tuple(sorted([phases[i], phases[j]]))
            for i in range(len(phases))
            for j in range(i + 1, len(phases))]


def phase_set_label(phase_set: Tuple[str, ...]) -> str:
    """Human-readable key for a phase set, e.g. ('BCC_A2','FCC_A1') -> 'BCC_A2+FCC_A1'."""
    return "+".join(phase_set)


def _score_one_phase_set(
    test_db,
    phase_set: Tuple[str, ...],
    NP_ref,
    comps: List[str],
    conds: dict,
    P: float,
    stable_tol: float,
    boundary_weight: float,
    boundary_power: float,
    alias_groups: List[List[str]],
) -> dict:
    """
    Score one phase set (full or pair) for a single test DB.
    Returns a dict with base_score / boundary_penalty / final_score / error.
    """
    test_phase_names = set(test_db.phases.keys())
    actual_per_request = {
        req: resolve_phase_for_db(req, test_phase_names, alias_groups)
        for req in phase_set
    }
    active_actuals = [a for a in actual_per_request.values() if a]
    if not active_actuals:
        return {"error": f"None of {list(phase_set)} present in test TDB",
                "base_score": 0.0, "boundary_penalty": 1.0,
                "final_score": 0.0}
    try:
        # See full-file comment on output="NP" — broken in pycalphad 0.11.x;
        # NP comes back in the default Dataset.
        test_eq = equilibrium(test_db, comps, active_actuals, conds)
        NP_test = build_phase_fraction_array(
            test_eq, list(phase_set), P,
            actual_per_request={r: a for r, a in actual_per_request.items() if a},
        )
        l1 = np.abs(NP_test - NP_ref).sum("phase")
        base = (1.0 - 0.5 * l1).clip(min=0.0, max=1.0)
        bp = boundary_misplacement_penalty(NP_test, NP_ref, stable_tol)
        final = (base - boundary_weight * (bp ** boundary_power)
                 ).clip(min=0.0, max=1.0)
        return {
            "base_score": float(base.mean()),
            "boundary_penalty": float(bp.mean()),
            "final_score": float(final.mean()),
            "error": None,
        }
    except Exception as exc:
        return {"error": str(exc),
                "base_score": 0.0, "boundary_penalty": 1.0,
                "final_score": 0.0}


def score_tdb(
    test_tdb_path: str,
    NP_refs: Dict[Tuple[str, ...], object],
    comps: List[str],
    conds: dict,
    P: float = 101325,
    stable_tol: float = 1e-6,
    boundary_weight: float = 0.25,
    boundary_power: float = 1.0,
    alias_groups: Optional[List[List[str]]] = None,
    pair_aggregate: str = "mean",
) -> dict:
    """
    Score a test TDB against multiple precomputed reference phase-fraction
    arrays — one per phase set the caller wants evaluated (full, pairs,
    or both per --scoring-mode).

    For each phase set we run a fresh equilibrium restricted to those
    phases (alias-resolved to the names this test DB actually uses),
    compute L1 + boundary-penalty against the matching reference array,
    and aggregate the per-set final_scores via `pair_aggregate`:
      "mean" -> arithmetic mean of valid per-set final_scores
      "min"  -> worst valid per-set final_score (rewards "no weak link")
    The aggregated final_score is what the top-N tracker / ranking uses;
    the per-set breakdown is returned in `per_set_scores` so the JSON
    record retains every pair's score.
    """
    try:
        test_db = Database(test_tdb_path)
    except Exception as exc:
        return {"error": f"Cannot load TDB: {exc}", "final_score": 0.0,
                "per_set_scores": {}}

    groups = alias_groups or DEFAULT_PHASE_ALIAS_GROUPS

    per_set_scores: Dict[str, dict] = {}
    for phase_set, NP_ref in NP_refs.items():
        per_set_scores[phase_set_label(phase_set)] = _score_one_phase_set(
            test_db, phase_set, NP_ref, comps, conds, P,
            stable_tol, boundary_weight, boundary_power, groups,
        )

    # Aggregate over valid per-set results.
    valid = [s for s in per_set_scores.values() if s["error"] is None]
    if not valid:
        return {"error": "all phase-set scorings failed",
                "base_score": 0.0, "boundary_penalty": 1.0, "final_score": 0.0,
                "per_set_scores": per_set_scores}

    finals = [s["final_score"] for s in valid]
    bases = [s["base_score"] for s in valid]
    bps = [s["boundary_penalty"] for s in valid]
    if pair_aggregate == "min":
        agg = {"final_score": min(finals),
               "base_score": min(bases),
               "boundary_penalty": max(bps)}
    else:  # mean (default)
        n = len(valid)
        agg = {"final_score": sum(finals) / n,
               "base_score": sum(bases) / n,
               "boundary_penalty": sum(bps) / n}
    agg["per_set_scores"] = per_set_scores
    agg["error"] = None
    return agg


# ====================================================================
# TDB combination via sqs2tdb -tdb
# ====================================================================

BASE_ENV = os.environ.copy()


def combine_tdbs(
    phase_tdb_paths: Dict[str, str],
    combo_dir: Path,
    el1: str, el2: str,
) -> Optional[Path]:
    """
    Mimic `sqs2tdb -tdb` directory layout:
      combo_dir/
        PHASE1/PHASE1.tdb  (symlink or copy)
        PHASE2/PHASE2.tdb
        ...
    Then run `sqs2tdb -tdb` from combo_dir.
    Returns path to combined TDB or None on failure.
    """
    combo_dir.mkdir(parents=True, exist_ok=True)

    for phase, tdb_path in phase_tdb_paths.items():
        src = Path(tdb_path)
        dest_dir = combo_dir / phase
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / f"{phase}.tdb"
        if not dest.exists():
            shutil.copy2(src, dest)

    # Run sqs2tdb -tdb
    log = combo_dir / "sqs2tdb_tdb.log"
    try:
        with open(log, "w") as f:
            proc = subprocess.run(
                ["sqs2tdb", "-tdb"], cwd=str(combo_dir), env=BASE_ENV,
                stdout=f, stderr=subprocess.STDOUT, text=True, timeout=300)
        if proc.returncode != 0:
            return None
    except Exception:
        return None

    # Find the combined TDB (named ELEM1_ELEM2.tdb)
    expected = combo_dir / f"{el1.upper()}_{el2.upper()}.tdb"
    if expected.is_file():
        return expected

    # Fallback: try reversed element order
    expected2 = combo_dir / f"{el2.upper()}_{el1.upper()}.tdb"
    if expected2.is_file():
        return expected2

    # Last resort: find any .tdb that isn't a per-phase one
    per_phase_names = {f"{ph}.tdb" for ph in phase_tdb_paths}
    for f in combo_dir.rglob("*.tdb"):
        if f.name not in per_phase_names and f.parent == combo_dir:
            return f

    return None


# ====================================================================
# Single combo evaluation
# ====================================================================

@dataclass
class ComboResult:
    combo_id: int
    phase_tdbs: Dict[str, str]
    combined_tdb: Optional[str]
    base_score: float
    boundary_penalty: float
    final_score: float
    error: Optional[str]
    # Per-phase-set breakdown when scoring-mode != 'full'.
    # Keys are "PhaseA+PhaseB" (or the full set's label); values are
    # {base_score, boundary_penalty, final_score, error} dicts. Empty
    # for the legacy full-only mode.
    per_set_scores: Optional[Dict[str, dict]] = None


class _TopNTracker:
    """
    Streaming top-N tracker for combo directories.

    For each completed combo we either:
      - keep the combo's directory on disk if it's among the top-N
        by final_score so far (initially everything fits while we're
        below N; afterwards a new combo only replaces the current
        worst-kept if it strictly beats that worst_score), or
      - delete the directory immediately to bound disk usage.

    Failed combos (error not None, or no combined_tdb) are always
    deleted — the JSON results record retains the per-combo error
    message either way, so nothing is lost beyond the scratch dir.
    """

    def __init__(self, n: int, enabled: bool = True):
        self.n = max(1, n)
        self.enabled = enabled
        # min-heap of (score, combo_id, combo_dir) — heap[0] is the worst kept
        self._heap: List[Tuple[float, int, Path]] = []
        self.kept_count = 0
        self.evicted_count = 0
        self.failed_deleted = 0

    def consider(self, r: "ComboResult", work_root: Path) -> None:
        if not self.enabled:
            return
        combo_dir = work_root / f"combo_{r.combo_id:06d}"

        # Failures: always delete, never count toward top-N.
        if r.error is not None or r.combined_tdb is None:
            shutil.rmtree(combo_dir, ignore_errors=True)
            self.failed_deleted += 1
            return

        if len(self._heap) < self.n:
            heapq.heappush(self._heap,
                           (r.final_score, r.combo_id, combo_dir))
            self.kept_count += 1
            return

        worst_score = self._heap[0][0]
        if r.final_score > worst_score:
            # Evict the current worst kept; promote this one in.
            ev_score, ev_id, ev_dir = heapq.heappop(self._heap)
            shutil.rmtree(ev_dir, ignore_errors=True)
            self.evicted_count += 1
            heapq.heappush(self._heap,
                           (r.final_score, r.combo_id, combo_dir))
        else:
            shutil.rmtree(combo_dir, ignore_errors=True)
            self.evicted_count += 1

    def summary(self) -> str:
        if not self.enabled:
            return "cleanup disabled (--no-cleanup-losers)"
        lines = [
            f"kept {len(self._heap)} top-N combo dir(s); "
            f"deleted {self.evicted_count} loser dir(s), "
            f"{self.failed_deleted} failed dir(s)"
        ]
        for score, cid, cdir in sorted(self._heap, reverse=True):
            lines.append(f"    combo_{cid:06d}  score={score:.4f}  -> {cdir}")
        return "\n  ".join(lines)


def evaluate_combo(
    combo_id: int,
    phase_tdb_paths: Dict[str, str],
    work_root: Path,
    NP_refs: Dict[Tuple[str, ...], object],
    comps: List[str],
    conds: dict,
    el1: str, el2: str,
    P: float,
    stable_tol: float,
    boundary_weight: float,
    boundary_power: float,
    alias_groups: Optional[List[List[str]]] = None,
    pair_aggregate: str = "mean",
) -> ComboResult:
    """Combine per-phase TDBs, score against the precomputed references.

    NP_refs is {phase_set_tuple: NP_ref_array} — one entry per phase set
    the caller wants scored (full and/or pairs, per --scoring-mode).
    """

    combo_dir = work_root / f"combo_{combo_id:06d}"

    try:
        combined = combine_tdbs(phase_tdb_paths, combo_dir, el1, el2)
        if combined is None:
            return ComboResult(
                combo_id=combo_id, phase_tdbs=phase_tdb_paths,
                combined_tdb=None, base_score=0.0, boundary_penalty=1.0,
                final_score=0.0, error="sqs2tdb -tdb failed",
                per_set_scores=None)

        result = score_tdb(
            str(combined), NP_refs, comps, conds,
            P=P, stable_tol=stable_tol,
            boundary_weight=boundary_weight,
            boundary_power=boundary_power,
            alias_groups=alias_groups,
            pair_aggregate=pair_aggregate)

        return ComboResult(
            combo_id=combo_id,
            phase_tdbs=phase_tdb_paths,
            combined_tdb=str(combined),
            base_score=result.get("base_score", 0.0),
            boundary_penalty=result.get("boundary_penalty", 1.0),
            final_score=result.get("final_score", 0.0),
            error=result.get("error"),
            per_set_scores=result.get("per_set_scores"))

    except Exception as exc:
        return ComboResult(
            combo_id=combo_id, phase_tdbs=phase_tdb_paths,
            combined_tdb=None, base_score=0.0, boundary_penalty=1.0,
            final_score=0.0, error=str(exc),
            per_set_scores=None)


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Stage 3: combine per-phase TDBs and score against reference")
    ap.add_argument("--manifest", required=True,
                    help="tdb_manifest.json from sqs2tdb_pipeline.py")
    ap.add_argument("--ref-tdb", required=True,
                    help="Reference TDB database path")
    ap.add_argument("--comp-element", required=True,
                    help="Composition element for X axis (e.g., CO)")
    ap.add_argument("--eq-phases", default=None,
                    help="Phases for equilibrium calc (comma-sep, default: from manifest)")
    ap.add_argument("--T-range", default="500,1200,50",
                    help="T_min,T_max,T_step (K)")
    ap.add_argument("--X-grid", type=float, default=0.005)
    ap.add_argument("--P", type=float, default=101325)
    ap.add_argument("--stable-tol", type=float, default=1e-6)
    ap.add_argument("--boundary-weight", type=float, default=0.25)
    ap.add_argument("--boundary-power", type=float, default=1.0)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--max-combos", type=int, default=0,
                    help="Cap on number of combinations (0 = unlimited)")
    ap.add_argument("--workdir", default=None,
                    help="Working directory (default: next to manifest)")
    ap.add_argument("--keep-top", type=int, default=1,
                    help="Preserve only the top-N combo directories by "
                         "final_score; loser dirs are deleted as soon as "
                         "they finish. This bounds disk usage to N combos "
                         "regardless of how many combinations were evaluated, "
                         "so you can leave --max-combos at 0 (unlimited). "
                         "Default 1 (keep only the best).")
    ap.add_argument("--no-cleanup-losers", action="store_true",
                    help="Disable the running-best cleanup. Every combo dir "
                         "stays on disk forever; use only when debugging "
                         "with a small --max-combos.")
    ap.add_argument("--phase-aliases", default=None,
                    help="Additional phase-alias groups, beyond the built-in "
                         "default ([SIGMA, SIGMA_D8B, SIGMA_SGTE]). Format: "
                         "semicolon-separated groups, each a comma-separated "
                         "list of equivalent names. Example: "
                         "'FCC#1,FCC_A1;BCC#1,BCC_A2'. Phases inside a group "
                         "are treated as the same column when scoring test "
                         "vs reference TDBs that label them differently.")
    ap.add_argument("--scoring-mode", default="pairs",
                    choices=["full", "pairs", "both"],
                    help="full = one equilibrium per combo with all "
                         "--eq-phases together (legacy). pairs (default) = "
                         "C(N,2) equilibria per combo, one per phase pair; "
                         "exposes each phase's free-energy description "
                         "wherever it competes with every other phase, "
                         "instead of only when it is the global minimum. "
                         "both = full + all pairs.")
    ap.add_argument("--pair-aggregate", default="mean",
                    choices=["mean", "min"],
                    help="How to combine per-phase-set final_scores into "
                         "the combo score that the top-N tracker uses. "
                         "'mean' (default) is smooth; 'min' is the worst-"
                         "pair score (rewards combos with no weak link).")
    args = ap.parse_args()

    # Merge user-supplied alias groups (if any) with the built-in defaults.
    alias_groups = merge_alias_groups(
        DEFAULT_PHASE_ALIAS_GROUPS,
        parse_alias_arg(args.phase_aliases),
    )

    # ── Load manifest ────────────────────────────────────────────
    with open(args.manifest) as f:
        manifest = json.load(f)

    binary = manifest["binary"]
    el1, el2 = binary.split("-")
    phase_tdbs: Dict[str, List[str]] = manifest["phases"]

    # ── Setup ────────────────────────────────────────────────────
    T_min, T_max, T_step = [float(x) for x in args.T_range.split(",")]
    comp_el = args.comp_element.upper()
    comps = sorted({el1.upper(), el2.upper(), "VA"})

    eq_phases = (
        [p.strip() for p in args.eq_phases.split(",")]
        if args.eq_phases else list(phase_tdbs.keys()))

    conds = {
        v.X(comp_el): (0, 1, args.X_grid),
        v.T: (T_min, T_max, T_step),
        v.P: args.P,
    }

    work_root = Path(args.workdir) if args.workdir else (
        Path(manifest["workdir"]) / "stage3_scoring")
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Stage 3: TDB Combination & Scoring — {binary}")
    print(f"{'='*70}")
    print(f"  Reference TDB : {args.ref_tdb}")
    print(f"  Comp element  : {comp_el}")
    print(f"  T range       : {T_min}–{T_max} K, step {T_step}")
    print(f"  X grid        : {args.X_grid}")
    print(f"  Eq phases     : {', '.join(eq_phases)}")
    if alias_groups:
        print(f"  Phase aliases : "
              + "; ".join("[" + ",".join(g) + "]" for g in alias_groups))
    print(f"  Work root     : {work_root}")

    # ── Enumerate combinations ───────────────────────────────────
    # For each phase, get list of TDB paths; if a phase has no survivors,
    # skip it (the combined TDB simply won't include that phase).
    phase_lists = {}
    for ph in eq_phases:
        tdbs = phase_tdbs.get(ph, [])
        if tdbs:
            phase_lists[ph] = tdbs
            print(f"  {ph:12s}: {len(tdbs)} TDB candidates")
        else:
            print(f"  {ph:12s}: no survivors — omitted from combinations")

    if not phase_lists:
        print("\n  No TDB candidates found. Nothing to score.\n")
        return

    # Cartesian product of per-phase TDB choices
    phases_ordered = sorted(phase_lists.keys())
    lists_ordered = [phase_lists[ph] for ph in phases_ordered]
    all_combos = list(itertools.product(*lists_ordered))

    if args.max_combos > 0 and len(all_combos) > args.max_combos:
        print(f"\n  WARNING: {len(all_combos)} combos exceeds --max-combos "
              f"{args.max_combos}. Sampling randomly.")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_combos), size=args.max_combos, replace=False)
        all_combos = [all_combos[i] for i in sorted(indices)]

    print(f"\n  Total combinations to evaluate: {len(all_combos)}")
    print(f"  Workers       : {args.n_workers}")
    if args.no_cleanup_losers:
        print(f"  Cleanup       : disabled (every combo dir kept on disk)")
    else:
        print(f"  Cleanup       : keep top-{args.keep_top}, "
              f"delete losers + failures as they finish")
    print(f"{'='*70}\n")

    # ── Reference equilibrium (computed ONCE, reused for every combo) ─
    # The reference is identical across all combinations, so computing it
    # per-combo wastes one full pycalphad equilibrium call per candidate.
    print("  Computing reference equilibrium (once)...")
    ref_db = Database(args.ref_tdb)

    # Per-phase, resolve which actual name the reference DB uses for
    # whatever we asked for. Phases the user requested that are
    # genuinely absent (in name AND in alias group) get warned about
    # and dropped from the ref-side equilibrium call; they will appear
    # as zero columns in NP_ref via build_phase_fraction_array.
    ref_phase_names = set(ref_db.phases.keys())
    ref_actual_per_request: Dict[str, str] = {}
    ref_missing: List[str] = []
    for req in eq_phases:
        actual = resolve_phase_for_db(req, ref_phase_names, alias_groups)
        if actual is None:
            ref_missing.append(req)
        else:
            ref_actual_per_request[req] = actual
    if ref_missing:
        print(f"  Reference TDB has no match for: {ref_missing}")
        print(f"    -> those phases will contribute 0 on the reference side")
    if any(req != act for req, act in ref_actual_per_request.items()):
        renames = ", ".join(f"{req}→{act}"
                            for req, act in ref_actual_per_request.items()
                            if req != act)
        print(f"  Reference alias resolution: {renames}")

    def _try_ref_eq(phase_set: List[str]):
        try:
            # Default-output form, not output="NP" — see test_eq comment
            # in score_tdb above; the NP-only path is broken in
            # pycalphad 0.11.x.
            actuals = [ref_actual_per_request[p] for p in phase_set
                       if p in ref_actual_per_request]
            if not actuals:
                return RuntimeError(
                    f"No requested phases present in reference TDB "
                    f"(after alias resolution): {phase_set}")
            eq = equilibrium(ref_db, comps, actuals, conds)
            return build_phase_fraction_array(
                eq, phase_set, args.P,
                actual_per_request={p: ref_actual_per_request[p]
                                    for p in phase_set
                                    if p in ref_actual_per_request},
            )
        except Exception as exc:
            return exc

    # ── Build the list of phase sets to score per --scoring-mode ───
    # NP_refs maps each phase-set tuple to its precomputed reference NP
    # array. A phase set is either the full eq_phases (legacy "full")
    # or one of the C(N,2) phase pairs (default "pairs"); "both" does
    # both. score_tdb computes one test-side equilibrium per phase set
    # per combo and aggregates per --pair-aggregate.
    requested_sets: List[Tuple[str, ...]] = []
    if args.scoring_mode in ("full", "both"):
        requested_sets.append(tuple(sorted(eq_phases)))
    if args.scoring_mode in ("pairs", "both"):
        requested_sets.extend(enumerate_phase_pairs(eq_phases))
    # de-dupe preserving order (so "both" with N=2 doesn't ask the same
    # equilibrium twice — full and pair are identical there)
    seen: set = set()
    requested_sets = [s for s in requested_sets if not (s in seen or seen.add(s))]
    print(f"  Scoring mode  : {args.scoring_mode} "
          f"({len(requested_sets)} phase set(s) per combo)")
    print(f"  Aggregate     : {args.pair_aggregate}")
    for s in requested_sets:
        print(f"    -> {phase_set_label(s)}")

    NP_refs: Dict[Tuple[str, ...], object] = {}
    failed_sets: List[Tuple[Tuple[str, ...], Exception]] = []
    for phase_set in requested_sets:
        result = _try_ref_eq(list(phase_set))
        if isinstance(result, Exception):
            print(f"  Ref eq for {phase_set_label(phase_set)} failed: {result}")
            failed_sets.append((phase_set, result))
            continue
        NP_refs[phase_set] = result

    if not NP_refs:
        # Same resilient fallback as before, but only triggered when EVERY
        # requested phase set failed: drop phases one at a time and retry
        # the full set.
        print("  Every requested phase-set ref equilibrium failed; "
              "retrying full set with reduced subsets...")
        working_set: Optional[List[str]] = None
        for n_keep in range(len(eq_phases) - 1, 0, -1):
            for subset in itertools.combinations(eq_phases, n_keep):
                cand = list(subset)
                result = _try_ref_eq(cand)
                if not isinstance(result, Exception):
                    NP_refs[tuple(sorted(cand))] = result
                    working_set = cand
                    dropped = sorted(set(eq_phases) - set(cand))
                    print(f"  Succeeded with phases: {cand}")
                    print(f"  Dropped (incompatible with ref TDB): {dropped}")
                    break
            if working_set is not None:
                break
        if not NP_refs:
            first_err = failed_sets[0][1] if failed_sets else "unknown"
            sys.exit(
                f"  ERROR: reference equilibrium failed for every subset of "
                f"{eq_phases}.\n"
                f"  First error was: {first_err}\n"
                f"  Check the reference TDB and the --eq-phases / "
                f"--comp-element settings."
            )
        # Restrict eq_phases to the working set going forward.
        eq_phases = working_set or eq_phases
    print(f"  Reference equilibria ready: {len(NP_refs)} phase set(s).\n")

    # ── Run scoring ──────────────────────────────────────────────
    results: List[ComboResult] = []
    t0 = time.time()
    done = 0
    ok = 0

    # Stage 3 is embarrassingly parallel: each combo is an independent
    # pycalphad equilibrium + sqs2tdb -tdb pair, and ProcessPoolExecutor
    # gives every worker its own interpreter (so per-process pycalphad
    # state is fresh — the only concern with Cython state was threads,
    # which we don't use). Use --n-workers as high as your node allows;
    # set 1 only as a fallback if a worker pool hangs on your build.

    # Streaming top-N cleanup keeps disk bounded: only the top
    # --keep-top combo directories survive; everything else is deleted
    # as it finishes. The scoring_results.json record below still has
    # every combo's score and error, so no information is lost — only
    # the per-combo scratch dirs.
    tracker = _TopNTracker(args.keep_top,
                           enabled=not args.no_cleanup_losers)

    def _record_and_cleanup(r: ComboResult) -> None:
        nonlocal done, ok
        results.append(r)
        done += 1
        if r.error is None:
            ok += 1
        tracker.consider(r, work_root)
        if done % max(1, len(all_combos) // 20) == 0 or done == len(all_combos):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            best = max((r2.final_score for r2 in results), default=0)
            print(f"  {done}/{len(all_combos)}  ok={ok}  "
                  f"{rate:.2f}/s  best={best:.4f}")

    if args.n_workers <= 1:
        # Sequential
        for cid, combo in enumerate(all_combos):
            ptdbs = dict(zip(phases_ordered, combo))
            r = evaluate_combo(
                cid, ptdbs, work_root, NP_refs,
                comps, conds, el1, el2,
                args.P, args.stable_tol,
                args.boundary_weight, args.boundary_power,
                alias_groups=alias_groups,
                pair_aggregate=args.pair_aggregate)
            _record_and_cleanup(r)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futs = {}
            for cid, combo in enumerate(all_combos):
                ptdbs = dict(zip(phases_ordered, combo))
                fut = pool.submit(
                    evaluate_combo,
                    cid, ptdbs, work_root, NP_refs,
                    comps, conds, el1, el2,
                    args.P, args.stable_tol,
                    args.boundary_weight, args.boundary_power,
                    alias_groups, args.pair_aggregate)
                futs[fut] = cid

            for fut in as_completed(futs):
                r = fut.result()
                _record_and_cleanup(r)

    # ── Sort and report ──────────────────────────────────────────
    results.sort(key=lambda r: -r.final_score)

    print(f"\n{'='*70}")
    print(f"  SCORING COMPLETE")
    print(f"{'='*70}")
    print(f"  Scored: {ok}/{len(all_combos)} successful")
    print(f"  Time:   {time.time() - t0:.0f}s\n")

    print(f"  Top 10 combinations:")
    for i, r in enumerate(results[:10]):
        print(f"    #{i+1}  score={r.final_score:.4f}  "
              f"base={r.base_score:.4f}  "
              f"bndry_pen={r.boundary_penalty:.4f}")
        for ph, tdb in r.phase_tdbs.items():
            print(f"         {ph}: .../{Path(tdb).parent.parent.name}/{Path(tdb).parent.name}/{Path(tdb).name}")
        if r.combined_tdb:
            print(f"         combined: {r.combined_tdb}")
        if r.per_set_scores:
            print(f"         per-phase-set scores:")
            for label, s in sorted(r.per_set_scores.items()):
                if s.get("error"):
                    print(f"             {label:30s}  FAILED: {s['error']}")
                else:
                    print(f"             {label:30s}  final={s['final_score']:.4f}"
                          f"  base={s['base_score']:.4f}"
                          f"  bndry_pen={s['boundary_penalty']:.4f}")
        print()

    # ── Save results ─────────────────────────────────────────────
    out_file = work_root / "scoring_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)

    # Save best TDB path for convenience. The top-N tracker preserves
    # the best combo's directory, so results[0].combined_tdb is still
    # on disk and shutil.copy2 succeeds.
    if results and results[0].combined_tdb \
            and Path(results[0].combined_tdb).is_file():
        best_tdb = Path(results[0].combined_tdb)
        best_link = work_root / f"BEST_{el1}_{el2}.tdb"
        if best_link.exists():
            best_link.unlink()
        shutil.copy2(best_tdb, best_link)
        print(f"  Best TDB: {best_link}")

    print(f"  Full results: {out_file}")
    print(f"  Disk: {tracker.summary()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()