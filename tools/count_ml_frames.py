#!/usr/bin/env python3
"""
Count ML-training frames (well-converged ionic steps) in VASP OUTCARs
under a directory tree, grouped by chemical system.

Purpose: inventory how many usable training frames exist for
machine-learned interatomic potentials. One FRAME = one ionic step
whose electronic loop converged — VASP marks each ionic step with
either

    "aborting loop because EDIFF is reached"        -> converged (frame)
    "aborting loop EDIFF was not reached (unconverged)" -> rejected

Unconverged steps carry noisy forces and must not enter a training set,
so only the first kind is counted. For charge/spin-equilibrating
potentials the magnetization labels matter too, so each file's ISPIN is
recorded and frame counts are split into spin-polarized vs
non-spin-polarized (legacy runs from before the 2026-07 spin fix are
ISPIN=1 — those frames have NO magnetization labels).

Scans OUTCAR.relax and OUTCAR.relax.gz by default (the ezvasp relax
trajectories); --include-plain-outcar adds bare OUTCAR / OUTCAR.gz
(e.g. the 00/ subdirectories of inflection-detection runs and phonon
force runs — the latter contribute single-point frames).

System classification comes from the VRHFIN lines inside each OUTCAR
(bare element symbols, so Cr_pv POTCARs still read as Cr). Pure-element
files are reported as their own buckets and NOT folded into a binary
automatically — a pure-Cr frame is shared training data for both Ni-Cr
and Co-Cr, so the rollup lists it separately and prints both totals.

Usage (front-end is fine — this is I/O only; use nohup for huge trees):
    python3 count_ml_frames.py /nobackupp27/pjalagam \\
        --csv frames_inventory.csv --json frames_summary.json
    python3 count_ml_frames.py <root> --systems CR-NI,CO-CR
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

CONVERGED = "aborting loop because EDIFF is reached"
UNCONVERGED = "aborting loop EDIFF was not reached"
FOOTER = "General timing and accounting"
REACHED = "reached required accuracy"
_VRHFIN = re.compile(r"VRHFIN\s*=\s*([A-Za-z][A-Za-z]?)\s*:")
_NIONS = re.compile(r"NIONS\s*=\s*(\d+)")
_ISPIN = re.compile(r"ISPIN\s*=\s*(\d)")

DEFAULT_NAMES = ("OUTCAR.relax", "OUTCAR.relax.gz")
PLAIN_NAMES = ("OUTCAR", "OUTCAR.gz")


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return path.open(errors="ignore")


def scan_outcar(path: Path) -> Optional[Dict]:
    """Stream one OUTCAR; return its frame inventory record.

    Line-by-line so multi-hundred-MB trajectories don't blow memory.
    Returns None only if the file cannot be opened at all.
    """
    elements: List[str] = []
    natoms = 0
    ispin = 1
    frames = 0
    rejected = 0
    complete = False
    reached_acc = False
    try:
        with _open_text(path) as fh:
            for line in fh:
                if CONVERGED in line:
                    frames += 1
                elif UNCONVERGED in line:
                    rejected += 1
                elif "VRHFIN" in line:
                    m = _VRHFIN.search(line)
                    if m and m.group(1).capitalize() not in elements:
                        elements.append(m.group(1).capitalize())
                elif "NIONS" in line and not natoms:
                    m = _NIONS.search(line)
                    if m:
                        natoms = int(m.group(1))
                elif "ISPIN" in line:
                    m = _ISPIN.search(line)
                    if m:
                        ispin = int(m.group(1))
                elif FOOTER in line:
                    complete = True
                elif REACHED in line:
                    reached_acc = True
    except OSError as exc:
        print(f"  WARN unreadable: {path} ({exc})", file=sys.stderr)
        return None

    system = "-".join(sorted(e.upper() for e in elements)) or "UNKNOWN"
    return {
        "path": str(path),
        "system": system,
        "natoms": natoms,
        "ispin": ispin,
        "frames": frames,                 # SCF-converged ionic steps
        "rejected_steps": rejected,       # unconverged SCF -> not frames
        "envs": frames * natoms,          # atomic environments
        "run_complete": complete,         # timing footer present
        "ionic_converged": reached_acc,   # 'reached required accuracy'
    }


def rollup(records: List[Dict]) -> Dict[str, Dict]:
    by_sys: Dict[str, Dict] = defaultdict(
        lambda: {"files": 0, "frames": 0, "frames_spin": 0,
                 "frames_nospin": 0, "rejected_steps": 0, "envs": 0})
    for r in records:
        b = by_sys[r["system"]]
        b["files"] += 1
        b["frames"] += r["frames"]
        b["rejected_steps"] += r["rejected_steps"]
        b["envs"] += r["envs"]
        if r["ispin"] == 2:
            b["frames_spin"] += r["frames"]
        else:
            b["frames_nospin"] += r["frames"]
    return dict(by_sys)


def print_report(by_sys: Dict[str, Dict], records: List[Dict]) -> None:
    hdr = (f"{'system':<12}{'files':>7}{'frames':>9}{'spin':>9}"
           f"{'no-spin':>9}{'rej.SCF':>9}{'atom-envs':>12}")
    print(hdr)
    print("-" * len(hdr))
    for sysname in sorted(by_sys):
        b = by_sys[sysname]
        print(f"{sysname:<12}{b['files']:>7}{b['frames']:>9}"
              f"{b['frames_spin']:>9}{b['frames_nospin']:>9}"
              f"{b['rejected_steps']:>9}{b['envs']:>12}")
    print("-" * len(hdr))

    # Requested binary rollups. Pure-element frames are SHARED between
    # systems, so they are added as a separate, clearly-labeled line
    # rather than silently folded in.
    for label, mixed, pures in (("a) Ni-Cr", "CR-NI", ("CR", "NI")),
                                ("b) Co-Cr", "CO-CR", ("CO", "CR"))):
        m = by_sys.get(mixed, None)
        p_frames = sum(by_sys.get(p, {"frames": 0})["frames"]
                       for p in pures)
        p_spin = sum(by_sys.get(p, {"frames_spin": 0})["frames_spin"]
                     for p in pures)
        mf = m["frames"] if m else 0
        ms = m["frames_spin"] if m else 0
        print(f"{label}: {mf} mixed-composition frames "
              f"({ms} with spin labels) "
              f"+ {p_frames} pure-element frames ({p_spin} with spin, "
              f"shared with other systems) "
              f"= {mf + p_frames} max usable")
    nospin_total = sum(b["frames_nospin"] for b in by_sys.values())
    if nospin_total:
        print(f"\nNOTE: {nospin_total} frames are ISPIN=1 (non-spin-"
              f"polarized, mostly pre-2026-07 runs) — they have NO "
              f"magnetization labels. For a charge/spin-equilibrating "
              f"potential, treat them as energy/force-only data or "
              f"recompute; the CSV marks each file's ispin.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("roots", nargs="+",
                    help="Tree(s) to scan, e.g. /nobackupp27/pjalagam")
    ap.add_argument("--include-plain-outcar", action="store_true",
                    help="Also scan bare OUTCAR/OUTCAR.gz (infdet 00/ "
                         "subdirs, phonon force runs -> single-point "
                         "frames).")
    ap.add_argument("--systems", default=None,
                    help="Comma list to keep, e.g. CR-NI,CO-CR,CO,CR,NI "
                         "(element symbols sorted alphabetically, "
                         "hyphen-joined). Default: report everything.")
    ap.add_argument("--csv", default=None,
                    help="Write the per-file inventory (path, system, "
                         "natoms, ispin, frames, ...) — the harvest "
                         "list for actually extracting training data.")
    ap.add_argument("--json", default=None,
                    help="Write the per-system summary as JSON.")
    args = ap.parse_args(argv)

    names = DEFAULT_NAMES + (PLAIN_NAMES if args.include_plain_outcar
                             else ())
    t0 = time.time()
    records: List[Dict] = []
    seen = set()
    for root in args.roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            print(f"WARN: {root} is not a directory — skipped",
                  file=sys.stderr)
            continue
        for name in names:
            for path in root.rglob(name):
                real = path.resolve()
                if real in seen:          # symlinked duplicates
                    continue
                seen.add(real)
                rec = scan_outcar(path)
                if rec:
                    records.append(rec)

    if args.systems:
        keep = {s.strip().upper() for s in args.systems.split(",")}
        records = [r for r in records if r["system"] in keep]

    by_sys = rollup(records)
    print(f"scanned {len(records)} OUTCAR file(s) in "
          f"{time.time() - t0:.1f}s\n")
    print_report(by_sys, records)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys())
                               if records else ["path"])
            w.writeheader()
            w.writerows(records)
        print(f"\nper-file inventory: {args.csv}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"by_system": by_sys, "n_files": len(records)}, indent=2))
        print(f"summary JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
