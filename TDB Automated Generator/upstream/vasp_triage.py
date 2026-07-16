#!/usr/bin/env python3
"""
VASP error triage: scan OUTCAR.relax / OUTCAR.static / OUTCAR and the
VASP stdout logs under a work tree, match them against a catalog of the
most common VASP failure signatures, and report a categorized diagnosis
per calculation directory plus a corpus-wide summary.

The signature catalog follows the de-facto community standard — the
error strings handled by Materials Project's custodian VaspErrorHandler
— plus completion/truncation checks that custodian does via job
management (walltime kills, OOM, unconverged SCF).

IMPORTANT SUBTLETY: many VASP error messages (EDDDAV, ZBRENT, BRMIX,
ZPOTRF, ...) are printed to STDOUT, not OUTCAR. ATAT's runstruct_vasp
captures stdout to vasp.out / out.log / runstruct.log depending on the
wrapper. This tool therefore scans BOTH OUTCAR-family files and
stdout-log candidates in each directory, and the catalog records which
stream each signature usually appears in.

Usage
-----
    python3 vasp_triage.py /nobackup/pjalagam/CoCr_upstream
    python3 vasp_triage.py <root> --json triage.json
    python3 vasp_triage.py <root> --only-problems      # hide clean runs
    python3 vasp_triage.py <root> --category electronic_scf
    python3 vasp_triage.py <root> --fixes              # print remediation
                                                       # playbook per hit

Exit code: 0 (reporting tool). Pass --strict to exit 1 when any
category other than 'clean' was found.

See VASP_TROUBLESHOOTING.md next to this script for the full
remediation workflows per category.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────
#  Signature catalog
#  (id, category, regex, stream, one-line fix)
#  stream: "stdout" | "outcar" | "both" — where the message usually
#  lands. We scan everything regardless; the hint helps interpret a
#  hit found in an unexpected place (e.g. truncated OUTCAR).
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Signature:
    error_id: str
    category: str
    pattern: str
    stream: str
    fix: str

    def regex(self) -> re.Pattern:
        return re.compile(self.pattern)


CATALOG: List[Signature] = [
    # ── input/chain sanity (diagnose FIRST — later errors are often
    #    consequences of these) ────────────────────────────────────────
    Signature("empty_poscar", "cell_basis",
              r"POSCAR found :\s+0 types and\s+0 ions",
              "both",
              "The static step inherited an EMPTY CONTCAR — the relax "
              "substep crashed before writing geometry. Fix the relax-"
              "step error (often MPI decomposition: NCORE/KPAR too "
              "large for a small cell, or rank oversubscription); any "
              "error VASP prints after this line is a red herring."),
    Signature("sick_job", "cell_basis",
              r"REFUSE TO CONTINUE WITH THIS SICK JOB",
              "both",
              "VASP input-validation abort. Read the reason printed "
              "just above; if the same output shows 'POSCAR found : 0 "
              "types', the real cause is an earlier crashed step, not "
              "the tag it names."),
    # ── electronic SCF / diagonalization ────────────────────────────
    Signature("edddav", "electronic_scf",
              r"Error EDDDAV: Call to ZHEGV failed",
              "stdout",
              "Delete CHGCAR+WAVECAR, set ALGO=All (or Normal); if it "
              "persists reduce NSIM or switch ALGO=Damped."),
    Signature("eddrmm", "electronic_scf",
              r"WARNING in EDD?RMM: call to ZHEGV failed",
              "stdout",
              "ALGO=Normal, delete CHGCAR+WAVECAR; for ISIF=3 runs also "
              "reduce POTIM (0.25)."),
    Signature("zheev", "electronic_scf",
              r"ERROR EDDIAG: Call to routine ZHEEV failed",
              "stdout",
              "Switch ALGO to Exact/All; rebuild wavefunctions from "
              "scratch (delete WAVECAR)."),
    Signature("grad_not_orth", "electronic_scf",
              r"EDWAV: internal error, the gradient is not orthogonal",
              "stdout",
              "Delete WAVECAR, set ISMEAR>=0 for insulators-in-doubt, "
              "ALGO=All."),
    Signature("subspacematrix", "electronic_scf",
              r"WARNING: Sub-Space-Matrix is not hermitian",
              "both",
              "Set LREAL=.FALSE. (in vasp.wrap: LREAL = .FALSE.); if it "
              "persists, PREC=Accurate."),
    Signature("brmix", "electronic_scf",
              r"BRMIX: very serious problems",
              "stdout",
              "Often a symmetry/k-mesh mismatch or charge sloshing: set "
              "ISYM=0, or ADDGRID=.TRUE.; metallic cells: increase AMIX/"
              "BMIX mixing damping (AMIX=0.1, BMIX=3.0). If the geometry "
              "went bad first, fix the relaxation (see ionic_relax)."),
    Signature("nicht_konv", "electronic_scf",
              r"ERROR: SBESSELITER : nicht konvergent",
              "stdout",
              "Increase LMAXMIX or switch PREC; usually PAW projector "
              "issue at extreme geometry — check the structure isn't "
              "collapsing."),
    Signature("not_converged_scf", "electronic_scf",
              r"aborting loop EDIFF was not reached \(unconverged\)",
              "outcar",
              "SCF hit NELM without converging: raise NELM (120+), "
              "ALGO=All, add AMIX=0.1/BMIX=3.0 for metals, or start "
              "from a better CHGCAR."),

    # ── ionic relaxation ─────────────────────────────────────────────
    Signature("zbrent", "ionic_relax",
              r"ZBRENT: fatal (?:error|internal) in",
              "stdout",
              "Ionic line-search bracketing failed: tighten EDIFF "
              "(1E-6), switch IBRION=1 after the first few steps, or "
              "restart from CONTCAR. ATAT flow: rerun; robustrelax "
              "(--relax-method normal) tolerates these crashes."),
    Signature("brions", "ionic_relax",
              r"BRIONS problems: POTIM should be increased",
              "stdout",
              "Increase POTIM slightly (e.g. 0.5) or switch IBRION=2 "
              "with default POTIM."),
    Signature("positive_energy", "ionic_relax",
              r"free  energy   TOTEN\s*=\s*\+?[1-9][0-9]*\.",
              "outcar",
              "Positive total energy — structure is unphysical "
              "(overlapping atoms / exploded cell). Regenerate the "
              "geometry; check str.out and the relaxation trajectory."),

    # ── symmetry ─────────────────────────────────────────────────────
    Signature("inv_rot_mat", "symmetry",
              r"rotation matrix was not found \(increase SYMPREC\)",
              "stdout",
              "Raise SYMPREC (1e-4) or set ISYM=0. SQS cells have no "
              "real symmetry — ISYM=0 is the honest setting."),
    Signature("rot_matrix", "symmetry",
              r"Found some non-integer element in rotation matrix|SGRCON",
              "stdout",
              "ISYM=0, or regenerate the cell so lattice vectors are "
              "exactly representable."),
    Signature("pricel", "symmetry",
              r"internal error in subroutine PRICEL",
              "stdout",
              "VASP found a smaller primitive cell inconsistently: set "
              "ISYM=0 and SYMPREC=1e-8 for supercells/SQS."),
    Signature("posmap", "symmetry",
              r"POSMAP internal error: symmetry equivalent atom not found",
              "stdout",
              "Raise SYMPREC to 1e-6..1e-4 or ISYM=0 (frequent for "
              "near-degenerate SQS positions)."),
    Signature("point_group", "symmetry",
              r"Error: point group operation missing",
              "stdout",
              "ISYM=0; usually harmless symmetry-detection noise in "
              "supercells."),
    Signature("rhosyg", "symmetry",
              r"RHOSYG internal error",
              "stdout",
              "SYMPREC=1e-4 or ISYM=0."),
    Signature("symprec_noise", "symmetry",
              r"determination of the symmetry of your systems shows a strong",
              "stdout",
              "Structure noise near symmetry threshold: tighten the "
              "geometry or set SYMPREC explicitly."),

    # ── k-points / tetrahedron ──────────────────────────────────────
    # NB: a bare "BZINTS: Fermi energy: ..." line is ROUTINE per-step
    # output under ISMEAR=-5, not an error (false-positived on the
    # 2026-07-16 smoke run) — only match actual failure text.
    Signature("tet", "kpoints_tet",
              r"Tetrahedron method fails|Fatal error detecting k-mesh"
              r"|Fatal error: unable to match k-point"
              r"|BZINTS: Fermi energy not converged"
              r"|WARNING: DENTET",
              "stdout",
              "Use ISMEAR=1 (metals, SIGMA=0.2) or 0 instead of -5, or "
              "densify the k-mesh (higher KPPRA). ISMEAR=-5 requires "
              ">=4 irreducible k-points."),
    Signature("tetirr", "kpoints_tet",
              r"Routine TETIRR needs special values",
              "stdout",
              "Change the k-mesh (KPPRA bump) or move off ISMEAR=-5."),
    Signature("dentet", "kpoints_tet",
              r"DENTET",
              "stdout",
              "Same family as tet: switch ISMEAR or densify the mesh."),
    Signature("incorrect_shift", "kpoints_tet",
              r"Could not get correct shifts",
              "stdout",
              "Use a Gamma-centred mesh (KGAMMA=.TRUE. / gamma flag in "
              "vasp.wrap)."),

    # ── numerics / LAPACK ───────────────────────────────────────────
    Signature("zpotrf", "numerics_lapack",
              r"LAPACK: Routine ZPOTRF failed",
              "stdout",
              "Usually the cell collapsed mid-relax (atoms too close): "
              "reduce POTIM (0.25), restart from last good CONTCAR, or "
              "pre-relax with ISIF=2 before ISIF=3. Tiny cells: also "
              "seen when NBANDS too small after cell shrink."),
    Signature("pssyevx", "numerics_lapack",
              r"ERROR in subspace rotation PSSYEVX",
              "stdout",
              "Set ALGO=Normal; if persists, NPAR/NCORE=1 for this run."),
    Signature("real_optlay", "numerics_lapack",
              r"REAL_OPTLAY: internal error|REAL_OPT: internal ERROR",
              "stdout",
              "Set LREAL=.FALSE. (accurate projection) and rerun."),
    Signature("rspher", "numerics_lapack",
              r"ERROR RSPHER",
              "stdout",
              "LREAL=.FALSE.; if during relax, geometry likely broke — "
              "restart from a sane structure."),

    # ── cell / basis sanity ─────────────────────────────────────────
    Signature("triple_product", "cell_basis",
              r"ERROR: the triple product of the basis vectors",
              "stdout",
              "Left-handed or near-singular lattice: fix the cell in "
              "str.out/POSCAR (reorder vectors); check the SQS "
              "generation step."),
    Signature("amin", "cell_basis",
              r"One of the lattice vectors is very long \(>50 A\), but AMIN",
              "stdout",
              "Set AMIN=0.01 for very elongated cells."),
    Signature("aliasing", "cell_basis",
              r"WARNING: small aliasing \(wrap around\) errors must be expected",
              "stdout",
              "Cosmetic unless energies drift: raise ENCUT or set "
              "PREC=Accurate to enlarge FFT grids."),
    Signature("grid_insufficient", "cell_basis",
              r"your FFT grids \(NGX,NGY,NGZ\) are not sufficient",
              "both",
              "PREC=Accurate (or set NGX/NGY/NGZ manually)."),

    # ── parallelization / machine ───────────────────────────────────
    Signature("elf_kpar", "parallel_machine",
              r"ELF: KPAR>1 not implemented",
              "stdout",
              "Set KPAR=1 for ELF-producing runs."),
    Signature("oom", "parallel_machine",
              r"oom-kill|Out Of Memory|OOM Killed|Killed\b.*vasp|"
              r"insufficient virtual memory",
              "stdout",
              "Reduce cores per node (undersubscribe), lower KPAR/NCORE, "
              "or split the k-mesh. On PBS: raise the node memory class."),
    Signature("mpi_abort", "parallel_machine",
              r"MPI_ABORT|BAD TERMINATION|APPLICATION TERMINATED",
              "stdout",
              "Generic MPI crash — read the lines just above the abort "
              "for the real error; often a wrapped LAPACK/SCF failure or "
              "an OOM."),
    Signature("bad_vasp_launch", "parallel_machine",
              r"Problem running vasp comm?and|unable to open OSZICAR",
              "stdout",
              "The VASP binary never ran (bare MPI binary without "
              "launcher, wrong path, or module env missing). In this "
              "pipeline: set --cmd-prefix 'mpiexec -n <N>' and load the "
              "mpi modules in the PBS script."),
]

# ── run-completion checks (not error greps) ─────────────────────────
FOOTER_RE = re.compile(r"General timing and accounting informations")
REACHED_ACC_RE = re.compile(r"reached required accuracy")

CATEGORY_ORDER = [
    "electronic_scf", "ionic_relax", "symmetry", "kpoints_tet",
    "numerics_lapack", "cell_basis", "parallel_machine",
    "incomplete_run", "clean",
]

OUTCAR_NAMES = ("OUTCAR.relax", "OUTCAR.static", "OUTCAR")
# runstruct_vasp gzips OUTCARs after extraction; scan those too.
OUTCAR_GZ_NAMES = tuple(n + ".gz" for n in OUTCAR_NAMES)


def _outcar_base(name: str) -> str:
    """OUTCAR.relax.gz -> OUTCAR.relax (identity for plain names)."""
    return name[:-3] if name.endswith(".gz") else name


def _open_text(path):
    """Text handle that transparently decompresses .gz files."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return path.open(errors="ignore")


STDOUT_NAMES = ("vasp.out", "out.log", "vasp.log", "runstruct.log",
                # ezvasp DOSTATIC two-step suffixed stdout captures
                "vasp.out.relax", "vasp.out.static",
                "stdout", "vasp.err")


# ────────────────────────────────────────────────────────────────────
#  Scan
# ────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    error_id: str
    category: str
    file: str
    line_no: int
    line: str
    fix: str


@dataclass
class DirReport:
    directory: str
    findings: List[Finding] = field(default_factory=list)
    outcar_seen: bool = False
    outcar_complete: Optional[bool] = None     # footer present?
    relax_converged: Optional[bool] = None     # 'reached required accuracy'
    categories: List[str] = field(default_factory=list)


def _scan_file(path: Path, max_bytes: int = 50_000_000) -> List[Finding]:
    """Grep one file against the catalog. Line-by-line so huge OUTCARs
    don't blow memory; bail politely past max_bytes."""
    out: List[Finding] = []
    compiled = [(s, s.regex()) for s in CATALOG]
    try:
        read = 0
        with _open_text(path) as fh:
            for i, line in enumerate(fh, 1):
                read += len(line)
                if read > max_bytes:
                    break
                for sig, rx in compiled:
                    if rx.search(line):
                        out.append(Finding(
                            error_id=sig.error_id,
                            category=sig.category,
                            file=str(path),
                            line_no=i,
                            line=line.strip()[:160],
                            fix=sig.fix,
                        ))
    except OSError:
        pass
    return out


def _completion_state(outcar: Path) -> Tuple[bool, bool]:
    """(footer_present, reached_required_accuracy) for one OUTCAR."""
    footer = reached = False
    try:
        with _open_text(outcar) as fh:
            for line in fh:
                if not footer and FOOTER_RE.search(line):
                    footer = True
                if not reached and REACHED_ACC_RE.search(line):
                    reached = True
                if footer and reached:
                    break
    except OSError:
        pass
    return footer, reached


def scan_tree(root: Path,
              extra_stdout_globs: Optional[List[str]] = None) -> List[DirReport]:
    """Walk root; every directory containing an OUTCAR-family file or a
    stdout log becomes one DirReport."""
    by_dir: Dict[Path, DirReport] = {}

    def report_for(d: Path) -> DirReport:
        if d not in by_dir:
            by_dir[d] = DirReport(directory=str(d))
        return by_dir[d]

    candidates: List[Path] = []
    for name in OUTCAR_NAMES + OUTCAR_GZ_NAMES + STDOUT_NAMES:
        candidates.extend(root.rglob(name))
    for pattern in (extra_stdout_globs or []):
        candidates.extend(root.rglob(pattern))

    for path in sorted(set(candidates)):
        if not path.is_file():
            continue
        rep = report_for(path.parent)
        rep.findings.extend(_scan_file(path))
        base = _outcar_base(path.name)
        if base in OUTCAR_NAMES:
            rep.outcar_seen = True
            footer, reached = _completion_state(path)
            # A dir may hold several OUTCARs; "complete" if ANY has the
            # footer (the .static usually finishes even when .relax was
            # interrupted — per-file detail stays in findings).
            rep.outcar_complete = bool(rep.outcar_complete) or footer
            if base == "OUTCAR.relax" or base == "OUTCAR":
                rep.relax_converged = bool(rep.relax_converged) or reached

    # classify
    for rep in by_dir.values():
        cats = sorted({f.category for f in rep.findings})
        if rep.outcar_seen and rep.outcar_complete is False:
            cats.append("incomplete_run")
            rep.findings.append(Finding(
                error_id="truncated_outcar",
                category="incomplete_run",
                file=rep.directory,
                line_no=0,
                line="OUTCAR has no 'General timing' footer — job was "
                     "killed (walltime/OOM/crash) before finishing.",
                fix="Check the PBS job log for walltime/OOM; resubmit "
                    "with more walltime or restart from CONTCAR.",
            ))
        rep.categories = cats if cats else ["clean"]
    return sorted(by_dir.values(), key=lambda r: r.directory)


# ────────────────────────────────────────────────────────────────────
#  Reporting
# ────────────────────────────────────────────────────────────────────

def print_report(reports: List[DirReport],
                 only_problems: bool = False,
                 category_filter: Optional[str] = None,
                 show_fixes: bool = False) -> Counter:
    cat_counts: Counter = Counter()
    err_counts: Counter = Counter()

    for rep in reports:
        for c in rep.categories:
            cat_counts[c] += 1
        for f in rep.findings:
            err_counts[f.error_id] += 1

    shown = 0
    for rep in reports:
        if only_problems and rep.categories == ["clean"]:
            continue
        if category_filter and category_filter not in rep.categories:
            continue
        shown += 1
        flag = "OK  " if rep.categories == ["clean"] else "FAIL"
        print(f"\n[{flag}] {rep.directory}")
        print(f"       categories: {', '.join(rep.categories)}"
              + (f" | relax converged: {rep.relax_converged}"
                 if rep.relax_converged is not None else ""))
        seen_ids = set()
        for f in rep.findings:
            if f.error_id in seen_ids:
                continue                      # one line per error id per dir
            seen_ids.add(f.error_id)
            loc = f"{Path(f.file).name}:{f.line_no}" if f.line_no else "-"
            print(f"       [{f.error_id:>16s}] {loc}  {f.line[:100]}")
            if show_fixes:
                print(f"       {' ' * 18}fix: {f.fix}")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY  ({len(reports)} calculation dirs scanned, "
          f"{shown} shown)")
    print(f"{'=' * 70}")
    print("  by category:")
    for cat in CATEGORY_ORDER:
        if cat_counts.get(cat):
            print(f"    {cat:20s} {cat_counts[cat]:5d} dir(s)")
    if err_counts:
        print("  by error id (finding count, not dirs):")
        for eid, n in err_counts.most_common():
            print(f"    {eid:20s} {n:5d}")
    print(f"{'=' * 70}\n")
    return cat_counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Categorize VASP errors under a work tree "
                    "(OUTCAR.relax/.static/OUTCAR + stdout logs).")
    ap.add_argument("root", help="Directory tree to scan "
                                 "(e.g. the upstream WORK_ROOT)")
    ap.add_argument("--json", default=None,
                    help="Write the full machine-readable report here")
    ap.add_argument("--only-problems", action="store_true",
                    help="Hide clean directories")
    ap.add_argument("--category", default=None,
                    help="Show only dirs hitting this category "
                         f"(one of: {', '.join(CATEGORY_ORDER)})")
    ap.add_argument("--fixes", action="store_true",
                    help="Print the recommended fix under every finding")
    ap.add_argument("--extra-logs", default=None,
                    help="Comma-separated extra stdout log globs to scan, "
                         "e.g. 'slurm-*.out,job.*.log'")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 if any non-clean directory was found")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    extra = ([g.strip() for g in args.extra_logs.split(",") if g.strip()]
             if args.extra_logs else None)
    reports = scan_tree(root, extra_stdout_globs=extra)
    if not reports:
        print(f"No OUTCAR/stdout files found under {root}. "
              f"(Looked for {', '.join(OUTCAR_NAMES + STDOUT_NAMES)})")
        return 0

    cat_counts = print_report(
        reports, only_problems=args.only_problems,
        category_filter=args.category, show_fixes=args.fixes)

    if args.json:
        payload = {
            "root": str(root),
            "n_dirs": len(reports),
            "category_counts": dict(cat_counts),
            "reports": [asdict(r) for r in reports],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"JSON report: {args.json}")

    if args.strict and any(r.categories != ["clean"] for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
