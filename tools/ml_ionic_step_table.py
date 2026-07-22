#!/usr/bin/env python3
"""
Per-run ionic/electronic convergence table for ML-potential curation.

Companion to count_ml_frames.py (which answers "how many usable frames
in total"); THIS tool answers "what does each run's convergence history
look like", one row per OUTCAR, in the layout requested for the
charge/spin-equilibrating ML-potential work:

    X_Co | X_Cr | X_Ni | Phase | ion_step_1 | ion_step_2 | ... | comments

VASP terminology used throughout (and worth being precise about):
  * OUTER loop  = the IONIC loop: one geometry update per step; these
    are the "N F= ..." lines in OSZICAR and "Iteration N(...)" in
    OUTCAR. Columns ion_step_i correspond to outer step i.
  * INNER loop  = the ELECTRONIC SCF loop inside each ionic step; the
    DAV/RMM lines, capped by NELM. The NUMBER IN EACH CELL is how many
    inner (electronic) iterations ionic step i needed. A step that
    ends with "aborting loop because EDIFF is reached" is
    SCF-converged; one that hits NELM without that marker carries
    noisy forces/magnetizations and must not enter a training set.

Everything is read from OUTCAR(.gz) alone — it carries ISPIN, NELM,
the per-iteration markers, the POTCAR species (VRHFIN, so Cr_pv still
reads as Cr) and the ion counts, so the table does not depend on
directory naming. ISPIN=2 rows ONLY by default (the ML target needs
magnetization labels; pre-2026-07 spin-fix runs are ISPIN=1 and are
skipped — count them with count_ml_frames.py if ever needed).

Comment column flags, per run:
  * whether every ionic step was SCF-converged (and which were not)
  * whether NELM was ever hit (and at which ionic steps)
  * termination: "clean exit" = the 'General timing and accounting'
    footer exists; "TRUNCATED" = no footer (walltime kill / crash);
    plus "ionic minimisation converged" when VASP printed
    'reached required accuracy'.

NAS usage (front end is fine — pure file I/O; nohup for huge trees):
    python3 ml_ionic_step_table.py /nobackupp27/pjalagam \
        --csv ionic_steps.csv --md ionic_steps.md
    python3 ml_ionic_step_table.py <root> --systems CO-CR,CR-NI \
        --include-plain-outcar     # adds bare OUTCARs (infdet 00/01,
                                   # phonon statics = 1-step rows)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CONVERGED = "aborting loop because EDIFF is reached"
UNCONVERGED = "aborting loop EDIFF was not reached"
FOOTER = "General timing and accounting"
REACHED = "reached required accuracy"

_VRHFIN = re.compile(r"VRHFIN\s*=\s*([A-Za-z][A-Za-z]?)\s*:")
_ISPIN = re.compile(r"ISPIN\s*=\s*(\d)")
_NELM = re.compile(r"NELM\s*=\s*(\d+)")
_IONS_PER_TYPE = re.compile(r"ions per type\s*=\s*((?:\d+\s*)+)")
# "--------- Iteration      3(  42) ---------" -> ionic step 3, SCF it. 42
_ITER = re.compile(r"Iteration\s+(\d+)\(\s*(\d+)\)")

DEFAULT_NAMES = ("OUTCAR.relax", "OUTCAR.relax.gz")
PLAIN_NAMES = ("OUTCAR", "OUTCAR.gz")

# Phase token = a path component like FCC_A1_small / SIGMA_D8B; the
# generated trees always place SQS dirs under <PHASE>[_small]/.
_PHASE_PART = re.compile(r"^([A-Z]{2,6}_[A-Z]\d?[A-Z0-9]*?)(?:_small)?$")


@dataclass
class RunRow:
    path: str
    system: str                       # e.g. CO-CR, CR (pure)
    frac: Dict[str, float]            # element -> atomic fraction
    phase: str
    ispin: Optional[int]
    nelm: Optional[int]
    inner_counts: List[int] = field(default_factory=list)  # per ionic step
    scf_converged: List[bool] = field(default_factory=list)
    clean_exit: bool = False
    ionic_converged: bool = False     # 'reached required accuracy'

    @property
    def frames(self) -> int:
        """Well-converged ionic steps (usable ML frames) in this run."""
        return sum(self.scf_converged)

    def comments(self) -> str:
        bits: List[str] = []
        n = len(self.inner_counts)
        bad = [i + 1 for i, ok in enumerate(self.scf_converged) if not ok]
        if n and not bad:
            bits.append(f"all {n} ionic steps SCF-converged")
        elif bad:
            bits.append(f"steps {','.join(map(str, bad))} NOT SCF-converged")
        if self.nelm:
            hit = [i + 1 for i, (c, ok) in
                   enumerate(zip(self.inner_counts, self.scf_converged))
                   if c >= self.nelm and not ok]
            if hit:
                bits.append(f"NELM({self.nelm}) exceeded at steps "
                            f"{','.join(map(str, hit))}")
        bits.append("ionic minimisation converged" if self.ionic_converged
                    else "ionic minimisation NOT converged (NSW/inflection/"
                         "static or stopped early)")
        bits.append("clean exit" if self.clean_exit
                    else "TRUNCATED — no timing footer (killed mid-run?)")
        return "; ".join(bits)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def parse_outcar(path: Path) -> RunRow:
    """Stream one OUTCAR(.gz) into a RunRow (never loads it whole)."""
    elements: List[str] = []
    ions: List[int] = []
    ispin = nelm = None
    inner: Dict[int, int] = {}        # ionic step -> max SCF iteration
    scf_ok: Dict[int, bool] = {}
    cur = 0
    clean = reached = False

    with _open_text(path) as fh:
        for line in fh:
            m = _ITER.search(line)
            if m:
                cur = int(m.group(1))
                inner[cur] = max(inner.get(cur, 0), int(m.group(2)))
                scf_ok.setdefault(cur, False)
                continue
            if CONVERGED in line:
                scf_ok[cur] = True
            elif UNCONVERGED in line:
                scf_ok[cur] = False
            elif REACHED in line:
                reached = True
            elif FOOTER in line:
                clean = True
            elif ispin is None and "ISPIN" in line:
                mm = _ISPIN.search(line)
                if mm:
                    ispin = int(mm.group(1))
            elif nelm is None and "NELM" in line:
                mm = _NELM.search(line)
                if mm:
                    nelm = int(mm.group(1))
            elif "VRHFIN" in line:
                mm = _VRHFIN.search(line)
                if mm:
                    elements.append(mm.group(1).capitalize())
            elif not ions and "ions per type" in line:
                mm = _IONS_PER_TYPE.search(line)
                if mm:
                    ions = [int(x) for x in mm.group(1).split()]

    total = sum(ions) if ions else 0
    frac: Dict[str, float] = {}
    if total and len(ions) == len(elements):
        for el, n in zip(elements, ions):
            frac[el] = frac.get(el, 0.0) + n / total
    system = "-".join(sorted({e.upper() for e in elements})) or "?"

    phase = "?"
    for part in path.parts:
        m = _PHASE_PART.match(part)
        if m:
            phase = m.group(1)
    steps = sorted(inner)
    return RunRow(
        path=str(path), system=system, frac=frac, phase=phase,
        ispin=ispin, nelm=nelm,
        inner_counts=[inner[s] for s in steps],
        scf_converged=[scf_ok.get(s, False) for s in steps],
        clean_exit=clean, ionic_converged=reached)


def scan(root: Path, names, systems: Optional[set],
         spin_only: bool = True) -> List[RunRow]:
    rows: List[RunRow] = []
    t0 = time.time()
    seen = 0
    for name in names:
        for p in sorted(root.rglob(name)):
            seen += 1
            try:
                row = parse_outcar(p)
            except OSError as exc:
                print(f"  WARN unreadable {p}: {exc}", file=sys.stderr)
                continue
            if spin_only and row.ispin != 2:
                continue
            if systems and row.system not in systems:
                continue
            rows.append(row)
            if seen % 200 == 0:
                print(f"  ... {seen} files scanned "
                      f"({time.time() - t0:.0f}s)", file=sys.stderr)
    return rows


ELS = ("Co", "Cr", "Ni")              # fixed leading columns, per request


def write_csv(rows: List[RunRow], out: Path) -> None:
    width = max((len(r.inner_counts) for r in rows), default=0)
    head = ([f"X_{e}" for e in ELS]
            + ["phase", "system", "ISPIN", "NELM", "n_ionic_steps",
               "converged_frames"]
            + [f"ion_step_{i}" for i in range(1, width + 1)]
            + ["comments", "path"])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for r in rows:
            pad = [""] * (width - len(r.inner_counts))
            w.writerow([f"{r.frac.get(e, 0.0):.4f}" for e in ELS]
                       + [r.phase, r.system, r.ispin, r.nelm,
                          len(r.inner_counts), r.frames]
                       + r.inner_counts + pad
                       + [r.comments(), r.path])


def write_md(rows: List[RunRow], out: Path, max_steps: int = 12) -> None:
    """Colleague-friendly markdown table; long runs elide middle steps."""
    head = [f"X_{e}" for e in ELS] + ["Phase"] \
        + [f"Ion {i}" for i in range(1, max_steps + 1)] + ["…", "Comments"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for r in rows:
        cells = [f"{r.frac.get(e, 0.0):.2f}" for e in ELS] + [r.phase]
        cnt = [str(c) if ok else f"{c}*"
               for c, ok in zip(r.inner_counts, r.scf_converged)]
        cells += cnt[:max_steps] + [""] * (max_steps - len(cnt))
        cells.append(f"+{len(cnt) - max_steps} more" if len(cnt) > max_steps
                     else "")
        cells.append(r.comments())
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("`*` = ionic step whose electronic (inner SCF) loop did "
                 "NOT reach EDIFF — exclude from training.")
    out.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-run ionic-step SCF table from OUTCARs "
                    "(ISPIN=2 only by default).")
    ap.add_argument("root", type=Path)
    ap.add_argument("--csv", type=Path, default=Path("ionic_steps.csv"))
    ap.add_argument("--md", type=Path, default=None,
                    help="also write a markdown table (elided view)")
    ap.add_argument("--systems", default=None,
                    help="comma list, e.g. CO-CR,CR-NI,CR (default: all)")
    ap.add_argument("--include-plain-outcar", action="store_true",
                    help="also scan bare OUTCAR(.gz) — infdet 00/01 "
                         "subruns and phonon statics (1-step rows)")
    ap.add_argument("--all-spins", action="store_true",
                    help="include ISPIN=1 runs too (default: ISPIN=2 "
                         "only, per the ML-potential requirement)")
    args = ap.parse_args(argv)

    names = list(DEFAULT_NAMES)
    if args.include_plain_outcar:
        names += list(PLAIN_NAMES)
    systems = ({s.strip().upper() for s in args.systems.split(",")}
               if args.systems else None)

    rows = scan(args.root, names, systems, spin_only=not args.all_spins)
    rows.sort(key=lambda r: (r.system, r.phase,
                             tuple(-r.frac.get(e, 0.0) for e in ELS)))
    write_csv(rows, args.csv)
    if args.md:
        write_md(rows, args.md)

    # Rollup answering the colleague's (a)/(b) directly.
    per_sys: Dict[str, int] = {}
    for r in rows:
        per_sys[r.system] = per_sys.get(r.system, 0) + r.frames
    print(f"{len(rows)} runs tabulated -> {args.csv}"
          + (f" and {args.md}" if args.md else ""))
    print("Well-converged ionic steps (ISPIN=2 frames) by system:")
    for s in sorted(per_sys):
        print(f"  {s:>8}: {per_sys[s]}")
    print("(pure-element buckets are shared training data between "
          "binaries — count them toward both.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
