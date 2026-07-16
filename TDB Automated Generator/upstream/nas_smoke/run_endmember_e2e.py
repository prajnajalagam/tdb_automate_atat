#!/usr/bin/env python3
"""
Mini end-to-end NAS test: the REAL upstream pipeline, restricted to the
two FCC endmembers of one binary (default Co-Cr), phonons included.

Where run_smoke.py checks the five VASP call paths in isolation, this
drives the actual production entry point —

    run_upstream.py --phases FCC_A1 --sqs-level 0

— so the whole chain is exercised end-to-end at the smallest possible
scope: sqs2tdb -cp generation, the ENCUT/KPPRA convergence sweeps, the
infdet relaxation (robustrelax_vasp -id -c 0.05), result validation,
checkrelax, energy/energy_end bookkeeping, and the full fitfc phonon
workflow (fvasp.wrap force runs, svib_ht promotion or unstable-mode
disposition). The lev=0 endmembers are 1-atom cells — precisely the
cells that died in the 2026-07-14 run — so this doubles as the
regression test for every fix that came out of that diagnosis.

After the run, a verification pass grades each endmember directory
against an explicit checklist and writes e2e_report.{txt,json}:

  [hard criteria — any failure fails the suite]
  - str_relax.out valid (parseable non-singular cell, real atoms)
  - energy present and parseable (energy_end adoption counts)
  - spin wiring: vasp.wrap carries ISPIN=2 AND an explicit MAGMOM
  - infdet actually invoked with its strain cutoff (-id ... -c 0.05)
  - checkrelax.out written (value reported)
  [phonon criteria — svib_ht OR a documented unstable disposition]
  - fvasp.wrap present; vol_0/p* force runs attempted
  - svib_ht promoted to the top level, OR unstable_modes.log explains
    why the SQS is energy-only (both are correct machinery behavior;
    FCC Cr may legitimately be dynamically unstable)

Cost: two 1-atom endmembers + one ~32-atom perturbation supercell each;
one devel-queue node, well under the 2 h devel walltime.

Usage (see submit_endmember_e2e.pbs for the PBS wrapper):
  python3 run_endmember_e2e.py \
      --potcars $PP/Co/POTCAR,$PP/Cr/POTCAR \
      --cmd-prefix "mpiexec -n 32" --env-bin $HOME/bin
  python3 run_endmember_e2e.py --verify-only <WORK_ROOT>   # re-grade only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _locate_upstream() -> Path:
    """Find the upstream package (runner.py, vaspwrap.py, run_upstream.py).

    Intended layout: nas_smoke/ lives INSIDE the upstream/ directory, so
    the parent is the package. But this suite gets copied around on NAS,
    so also accept: $UPSTREAM_DIR, the script's own directory (flat
    copy), and the cwd — with a copy-paste-able error when none work.
    """
    import os
    cands = []
    if os.environ.get("UPSTREAM_DIR"):
        cands.append(Path(os.environ["UPSTREAM_DIR"]).expanduser())
    here = Path(__file__).resolve().parent
    cands += [here.parent, here, Path.cwd()]
    for c in cands:
        if (c / "runner.py").is_file() and (c / "vaspwrap.py").is_file():
            return c.resolve()
    sys.exit(
        "ERROR: cannot find the upstream package (runner.py / "
        "vaspwrap.py).\n"
        "Searched: " + ", ".join(str(c) for c in cands) + "\n"
        "nas_smoke/ must live INSIDE 'TDB Automated Generator/upstream/'."
        "\nCopy the WHOLE upstream directory to NAS (the e2e test also "
        "needs its run_upstream.py), e.g.:\n"
        "    scp -r 'TDB Automated Generator/upstream' pfe:~/upstream\n"
        "    cd ~/upstream/nas_smoke && qsub submit_smoke.pbs\n"
        "or point at an existing copy:  export UPSTREAM_DIR=~/upstream")


UPSTREAM = _locate_upstream()
sys.path.insert(0, str(UPSTREAM))

from strfile import validate_structure_file          # noqa: E402
from sqsgen import DECORATED_SQS_RE                   # noqa: E402


# ---------------------------------------------------------------------------
# Verification (pure functions — unit-tested without VASP)
# ---------------------------------------------------------------------------

def _parseable_float(p: Path) -> Optional[float]:
    try:
        return float(p.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _grep(p: Path, needle: str) -> bool:
    try:
        return needle in p.read_text(errors="replace")
    except OSError:
        return False


def verify_endmember_dir(d: Path) -> Dict:
    """Grade one lev=0 endmember directory against the checklist."""
    checks: Dict[str, Dict] = {}

    def check(name: str, passed: bool, detail: str, hard: bool = True):
        checks[name] = {"pass": bool(passed), "detail": detail, "hard": hard}

    ok, msg = validate_structure_file(d / "str_relax.out")
    check("str_relax_valid", ok, msg)

    e = _parseable_float(d / "energy")
    e_end = _parseable_float(d / "energy_end")
    check("energy_present", e is not None,
          f"energy={e}" + (f" (energy_end={e_end})" if e_end is not None
                           else ""))

    wrap = d / "vasp.wrap"
    has_ispin = _grep(wrap, "ISPIN = 2")
    has_magmom = _grep(wrap, "MAGMOM =")
    check("wrap_spin_magmom", has_ispin and has_magmom,
          "vasp.wrap has ISPIN=2 + explicit MAGMOM" if has_ispin and has_magmom
          else "vasp.wrap missing" if not wrap.is_file()
          else f"vasp.wrap lacks {'MAGMOM' if has_ispin else 'ISPIN=2'}"
               " — wrap predates the 216f799 spin fix; regenerate")

    # ezvasp must have FORWARDED the tags into the actual INCAR.
    incars = [p for p in (d / "INCAR.relax", d / "INCAR.static", d / "INCAR")
              if p.is_file()]
    if incars:
        fwd = any(_grep(p, "MAGMOM") for p in incars)
        check("incar_magmom_forwarded", fwd,
              f"MAGMOM in {[p.name for p in incars if _grep(p, 'MAGMOM')]}"
              if fwd else f"no MAGMOM in any of "
                          f"{[p.name for p in incars]}", hard=False)

    log = d / "robustrelax_infdet.log"
    has_id = _grep(log, "-id")
    has_c = _grep(log, "-c 0.05")
    check("infdet_with_cutoff", has_id and has_c,
          "robustrelax_vasp -id -c 0.05 invoked" if has_id and has_c
          else "robustrelax_infdet.log missing" if not log.is_file()
          else "invoked WITHOUT the -c 0.05 strain cutoff (pre-216f799 "
               "run) — infdet does not engage without it")

    cr = _parseable_float(d / "checkrelax.out")
    check("checkrelax_recorded", cr is not None,
          f"lattice drift = {cr}" if cr is not None
          else "checkrelax.out missing/unparseable")

    # ── phonon machinery (soft: unstable disposition is also a PASS) ──
    check("fvasp_wrap_written", (d / "fvasp.wrap").is_file(),
          "separate frozen wrap for force runs", hard=False)
    pert = sorted((d / "vol_0").glob("p*")) if (d / "vol_0").is_dir() else []
    forces = sum(1 for p in pert if (p / "force.out").is_file())
    check("force_runs", bool(pert),
          f"{forces}/{len(pert)} perturbation dirs have force.out",
          hard=False)
    svib = _parseable_float(d / "svib_ht")
    unstable = (d / "unstable_modes.log").is_file()
    check("svib_or_disposition", svib is not None or unstable,
          f"svib_ht = {svib}" if svib is not None
          else ("unstable_modes.log present — energy-only by policy "
                "(legitimate for dynamically unstable lattices)"
                if unstable else
                "NEITHER svib_ht NOR unstable_modes.log — phonon "
                "stage silently produced nothing"), hard=False)

    hard_ok = all(c["pass"] for c in checks.values() if c["hard"])
    soft_ok = all(c["pass"] for c in checks.values() if not c["hard"])
    return {"dir": str(d), "checks": checks,
            "hard_pass": hard_ok, "all_pass": hard_ok and soft_ok}


def find_endmember_dirs(work_root: Path) -> List[Path]:
    """lev=0 SQS dirs under the FCC phase tree (FCC_A1 or FCC_A1_small)."""
    out: List[Path] = []
    for phase_dir in ("FCC_A1_small", "FCC_A1"):
        root = work_root / phase_dir
        if root.is_dir():
            out += [d for d in sorted(root.iterdir())
                    if d.is_dir() and "lev=0" in d.name
                    and DECORATED_SQS_RE.search(d.name)]
    return out


def verify_tree(work_root: Path) -> Tuple[List[Dict], bool]:
    dirs = find_endmember_dirs(work_root)
    results = [verify_endmember_dir(d) for d in dirs]
    ok = bool(results) and len(results) >= 2 \
        and all(r["hard_pass"] for r in results)
    return results, ok


def write_report(work_root: Path, results: List[Dict], ok: bool,
                 meta: Dict) -> None:
    (work_root / "e2e_report.json").write_text(json.dumps(
        {"meta": meta, "suite_pass": ok, "endmembers": results},
        indent=2, default=str))
    lines = ["=" * 70,
             f"FCC endmember end-to-end test — {meta.get('timestamp')}",
             f"work root: {work_root}",
             f"SUITE: {'PASS' if ok else 'FAIL'} "
             f"({len(results)} endmember dirs found; need >= 2 with all "
             f"hard criteria green)",
             "=" * 70]
    for r in results:
        flag = "PASS" if r["hard_pass"] else "FAIL"
        soft = "" if r["all_pass"] else "  (soft findings below)"
        lines.append(f"[{flag}] {Path(r['dir']).name}{soft}")
        for name, c in r["checks"].items():
            mark = "ok " if c["pass"] else ("XX " if c["hard"] else "!! ")
            lines.append(f"   {mark}{name:<24} {c['detail']}")
    lines.append("-" * 70)
    lines.append("Hard failures block the suite; '!!' items are soft "
                 "(phonon disposition, INCAR forwarding) — read the detail.")
    lines.append("Debug bundle: e2e_report.json + upstream_live.log + "
                 "the per-step logs in each endmember dir.")
    (work_root / "e2e_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--element1", default="Co")
    ap.add_argument("--element2", default="Cr")
    ap.add_argument("--potcars", default=None,
                    help="Comma-separated POTCAR paths (required unless "
                         "--verify-only).")
    ap.add_argument("--template-root", default=None,
                    help="Dir with the *_small template systems.")
    ap.add_argument("--work-root", default=None,
                    help="Default: ./e2e_fcc_endmembers under the cwd.")
    ap.add_argument("--cmd-prefix", default="mpiexec -n 32",
                    help="VASP launcher. 32 ranks fits the production "
                         "NCORE=8*KPAR=4 decomposition of the ~32-atom "
                         "perturbation supercells exactly; the 1-atom "
                         "endmembers get the adaptive NCORE=1/KPAR=1.")
    ap.add_argument("--env-bin", default=None)
    ap.add_argument("--tol-ev", default="0.0001",
                    help="Successive-step convergence tolerance in "
                         "eV/atom (default 0.1 meV/atom).")
    ap.add_argument("--fitfc-on-unstable", default="mark",
                    choices=("mark", "escalate", "force"))
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--verify-only", metavar="WORK_ROOT", default=None,
                    help="Skip the run; just re-grade an existing tree.")
    args = ap.parse_args(argv)

    if args.verify_only:
        work_root = Path(args.verify_only).resolve()
    else:
        work_root = Path(args.work_root
                         or Path.cwd() / "e2e_fcc_endmembers").resolve()
        if not args.potcars:
            ap.error("--potcars is required to run (or use --verify-only)")
        cmd = [sys.executable, "-u", str(UPSTREAM / "run_upstream.py"),
               "--element1", args.element1,
               "--element2", args.element2,
               "--work-root", str(work_root),
               "--potcars", args.potcars,
               "--phases", "FCC_A1",          # single phase...
               "--sqs-level", "0",            # ...endmembers only
               "--relax-method", "infdet",    # the production default
               "--tol-ev", args.tol_ev,
               "--fitfc-on-unstable", args.fitfc_on_unstable,
               "--cmd-prefix", args.cmd_prefix,
               "--timeout", str(args.timeout)]
        if args.template_root:
            cmd += ["--template-root", args.template_root]
        if args.env_bin:
            cmd += ["--env-bin", args.env_bin]
        print("launching:", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        print(f"run_upstream.py exited rc={rc} — grading outputs "
              f"regardless (partial trees are still informative)",
              flush=True)

    meta = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "binary": f"{args.element1}-{args.element2}",
            "cmd_prefix": args.cmd_prefix,
            "mode": "verify-only" if args.verify_only else "run+verify"}
    results, ok = verify_tree(work_root)
    if not results:
        print(f"ERROR: no lev=0 endmember dirs found under {work_root} "
              f"(looked in FCC_A1_small/ and FCC_A1/) — generation "
              f"failed; see {work_root}/upstream_live.log and "
              f"sqs2tdb_cp_*.log")
        return 2
    write_report(work_root, results, ok, meta)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
