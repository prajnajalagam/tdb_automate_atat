#!/usr/bin/env python3
"""
NAS VASP smoke suite — verify each distinct VASP call path the upstream
pipeline uses, on a tiny 2-atom cell, with minimal cores and aggressive
early termination. Run this BEFORE committing a full upstream job.

The five tests, each exercising a different launch/dependency chain
(the exact chains production uses — the driver imports the pipeline's
own modules, it does not re-implement them):

  T1_static      runstruct_vasp in a prepared dir with a static
                 vasp.wrap — the converge.py ENCUT/KPPRA sweep path
                 (converge.run_static_point). This is the call that
                 historically failed with "unable to open OSZICAR".
  T2_runstruct   runstruct_vasp relaxation (NSW-capped) — the
                 --relax-method runstruct path. Checks str_relax.out,
                 energy, force.out extraction.
  T3_robustrelax robustrelax_vasp -mk, then robustrelax_vasp -id -c 0.05
                 with the launcher trailing — the --relax-method infdet
                 path (the production default), early-stopped via the
                 'stop' sentinel as soon as str_relax.out appears.
  T4_fitfc_wrap  runstruct_vasp -lu -w vaspf.wrap on a frozen, displaced
                 cell — the fitfc force-run convention (separate wrap
                 file, NSW=0, forces extracted to force.out).
  T5_pollmach    pollmach runstruct_vasp over two wait-marked subdirs —
                 the dispatcher every production stage goes through
                 (wait consumption, per-dir energy, stoppoll shutdown).

Outputs (in --workdir):
  smoke_report.txt   human summary: PASS/FAIL, elapsed, expected vs
                     found files, tail of the decisive log per test.
  smoke_report.json  same, machine-readable (paste this back for debug).
  triage.json        vasp_triage.py scan of the whole smoke tree
                     (error-signature categories + suggested fixes).
  plan.json          the exact argv per test (also written by --dry-run).

Resource use: one node, CMD_PREFIX ranks only (default mpiexec -n 8),
2-atom cell, NELM<=25, NSW<=5, per-test --timeout (default 1200 s).
Every test also stops early through its sentinel once its expected
outputs exist, so a healthy suite finishes in a few minutes.

Example (on the node, after modules + venv):
  python3 run_smoke.py --element Co \
      --potcar /home1/zwu6/vasp/POTPAW_PBE.64/Co/POTCAR \
      --cmd-prefix "mpiexec -n 8" --env-bin $HOME/bin
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Import the real pipeline modules — the point is to test THEIR call
# chains, not copies of them.

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

import runner                                    # noqa: E402
from vaspwrap import build_vasp_wrap             # noqa: E402

# ---------------------------------------------------------------------------
# Tiny test structure: 2-atom cubic cell (physics irrelevant — plumbing only)
# ---------------------------------------------------------------------------

def str_out_text(element: str, displaced: bool = False) -> str:
    z2 = "0.52" if displaced else "0.5"
    return (
        "2.8 2.8 2.8 90 90 90\n"
        "1 0 0\n0 1 0\n0 0 1\n"
        f"0 0 0 {element}\n"
        f"0.5 0.5 {z2} {element}\n"
    )


# Tiny-run INCAR overrides: fast algo, few steps, no parallel-decomposition
# surprises on 8 ranks. Spin OFF on purpose — this suite tests plumbing,
# not physics (production auto-enables ISPIN=2; noted in the report).
_SMOKE_EXTRA = {"NELM": 25, "ALGO": "Fast", "PREC": "Normal",
                "ISMEAR": 1, "SIGMA": 0.2, "NCORE": 1, "KPAR": 1}
_SMOKE_ENCUT = 300
_SMOKE_KPPRA = 1000


def _wrap(mode: str, extra: Optional[Dict] = None) -> str:
    e = dict(_SMOKE_EXTRA)
    if extra:
        e.update(extra)
    return build_vasp_wrap(mode, encut=_SMOKE_ENCUT, kppra=_SMOKE_KPPRA,
                           spin=False, extra=e)


# ---------------------------------------------------------------------------
# Case construction (pure file-system; used by --dry-run and by tests)
# ---------------------------------------------------------------------------

def build_cases(workdir: Path, element: str,
                potcar: Optional[Path], cmd_prefix: str) -> List[Dict]:
    """Create every test directory + wrap and return the execution plan:
    [{test, dir, argv, wrap_file, expect: [files...]}, ...]."""
    launch = runner.split_prefix(cmd_prefix)
    cases: List[Dict] = []

    def new_case(name: str, wrap_mode: str, wrap_name: str = "vasp.wrap",
                 displaced: bool = False, wrap_extra: Optional[Dict] = None):
        d = workdir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "str.out").write_text(str_out_text(element, displaced))
        (d / wrap_name).write_text(_wrap(wrap_mode, wrap_extra))
        if potcar is not None:
            shutil.copy2(potcar, d / "POTCAR")
        return d

    # T1: converge.py static point (bare runstruct_vasp + launcher).
    d1 = new_case("T1_static", "static")
    cases.append({
        "test": "T1_static", "dir": str(d1),
        "argv": ["runstruct_vasp"] + launch,
        "expect": ["energy", "str_relax.out"],
        "log": "runstruct.log",
        "covers": "converge.run_static_point launch chain "
                  "(ENCUT/KPPRA sweep points)"})

    # T2: runstruct relaxation, NSW-capped so it terminates quickly.
    d2 = new_case("T2_runstruct", "relax",
                  wrap_extra={"NSW": 5, "EDIFFG": -0.3})
    cases.append({
        "test": "T2_runstruct", "dir": str(d2),
        "argv": ["runstruct_vasp"] + launch,
        "expect": ["str_relax.out", "energy", "force.out"],
        "log": "runstruct.log",
        "covers": "--relax-method runstruct relaxation + extraction"})

    # T3: robustrelax two-step (-mk, then run; sentinel-stopped).
    d3 = new_case("T3_robustrelax", "relax",
                  wrap_extra={"NSW": 5, "EDIFFG": -0.3})
    cases.append({
        "test": "T3_robustrelax", "dir": str(d3),
        # The production default: inflection detection with its
        # REQUIRED strain cutoff (reference job: -id -c 0.05).
        "argv": ["robustrelax_vasp", "-id", "-c", "0.05"] + launch,
        "pre_argv": ["robustrelax_vasp", "-mk"],
        "expect": ["str_relax.out"],
        "log": "robustrelax.log",
        "covers": "--relax-method infdet chain (default: -mk prep + "
                  "robustrelax_vasp -id -c 0.05, stop sentinel "
                  "early-exit)"})

    # T4: fitfc force-run convention — frozen wrap under a SEPARATE
    # file name, selected with runstruct_vasp -w vaspf.wrap.
    d4 = new_case("T4_fitfc_wrap", "phonon", wrap_name="vaspf.wrap",
                  displaced=True)
    cases.append({
        "test": "T4_fitfc_wrap", "dir": str(d4),
        "argv": ["runstruct_vasp", "-lu", "-w", "vaspf.wrap"] + launch,
        "expect": ["force.out", "str_relax.out"],
        "log": "runstruct.log",
        "covers": "fitfc perturbation force runs (frozen geometry, "
                  "forces -> force.out, -w wrap selection)"})

    # T5: pollmach dispatch over two wait-marked subdirs; the parent
    # holds the shared vasp.wrap (walk-up search), as in production.
    d5 = workdir / "T5_pollmach"
    d5.mkdir(parents=True, exist_ok=True)
    (d5 / "vasp.wrap").write_text(_wrap("static"))
    for i in (1, 2):
        sub = d5 / f"p_{i}"
        sub.mkdir(exist_ok=True)
        (sub / "str.out").write_text(str_out_text(element, displaced=i == 2))
        (sub / "wait").write_text("")
        if potcar is not None:
            shutil.copy2(potcar, sub / "POTCAR")
    cases.append({
        "test": "T5_pollmach", "dir": str(d5),
        "argv": ["pollmach", "runstruct_vasp"] + launch,
        "expect": ["p_1/energy", "p_2/energy"],
        "log": "pollmach.log",
        "covers": "pollmach dispatcher (wait consumption, per-dir "
                  "runs, stoppoll shutdown) — every production stage "
                  "routes through this"})

    (workdir / "plan.json").write_text(json.dumps(cases, indent=2))
    return cases


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(cmd_prefix: str, potcar: Optional[Path],
              env_bin: Optional[str]) -> Dict:
    """Cheap environment checks that fail faster (and clearer) than VASP."""
    import os
    path = (f"{env_bin}:" if env_bin else "") + os.environ.get("PATH", "")

    def which(prog):
        return shutil.which(prog, path=path)

    binaries = {b: which(b) for b in
                ("runstruct_vasp", "robustrelax_vasp", "pollmach",
                 "fitfc", "sqs2tdb", "ezvasp")}
    launcher_tok = runner.split_prefix(cmd_prefix)
    checks = {
        "binaries": binaries,
        "missing_binaries": [b for b, p in binaries.items() if p is None],
        "launcher": launcher_tok,
        "launcher_found": bool(launcher_tok) and
                          which(launcher_tok[0]) is not None,
        "ezvasp_rc": str(Path.home() / ".ezvasp.rc"),
        "ezvasp_rc_exists": (Path.home() / ".ezvasp.rc").is_file(),
        "potcar": str(potcar) if potcar else None,
        "potcar_exists": bool(potcar and Path(potcar).is_file()),
    }
    # Without --potcar, runstruct relies on USEPOT + the POTDIR in
    # ~/.ezvasp.rc — surface that dependency explicitly.
    checks["potcar_mechanism"] = (
        "explicit POTCAR copied into each case dir" if checks["potcar_exists"]
        else "ezvasp USEPOT + POTDIR from ~/.ezvasp.rc (no --potcar given)")
    return checks


# ---------------------------------------------------------------------------
# Execution + reporting
# ---------------------------------------------------------------------------

def _tail(path: Path, n: int = 30) -> List[str]:
    if not path.is_file():
        return [f"<no {path.name}>"]
    return path.read_text(errors="replace").splitlines()[-n:]


def run_case(case: Dict, env_bin: Optional[str], timeout: int) -> Dict:
    d = Path(case["dir"])
    log = d / case["log"]
    t0 = time.time()

    if case.get("pre_argv"):
        runner.run_logged(case["pre_argv"], cwd=d,
                          log=d / "prep_mk.log",
                          env_bin=env_bin, timeout=600, check=False)

    def outputs_present(_cwd=None) -> bool:
        return all((d / f).is_file() for f in case["expect"])

    if case["argv"][0] == "pollmach":
        # Dispatcher test: sentinel-based early stop the moment both
        # subdirs have produced energies.
        rc = runner.run_polled(case["argv"], cwd=d, log=log,
                               done_when=outputs_present,
                               stop_sentinel="stoppoll",
                               env_bin=env_bin,
                               poll_interval=10.0, timeout=timeout)
    elif case["argv"][0] == "robustrelax_vasp":
        # robustrelax loops until told to stop: poll for str_relax.out,
        # then drop its 'stop' sentinel (early termination).
        rc = runner.run_polled(case["argv"], cwd=d, log=log,
                               done_when=outputs_present,
                               stop_sentinel="stop",
                               env_bin=env_bin,
                               poll_interval=10.0, timeout=timeout)
    else:
        rc = runner.run_logged(case["argv"], cwd=d, log=log,
                               env_bin=env_bin, timeout=timeout,
                               check=False)

    elapsed = round(time.time() - t0, 1)
    found = [f for f in case["expect"] if (d / f).is_file()]
    missing = [f for f in case["expect"] if f not in found]
    energy = None
    for cand in ("energy", "p_1/energy"):
        ef = d / cand
        if ef.is_file():
            try:
                energy = float(ef.read_text().split()[0])
            except (ValueError, IndexError):
                energy = f"UNPARSEABLE: {ef.read_text()[:40]!r}"
            break

    result = {
        **{k: case[k] for k in ("test", "dir", "argv", "expect", "covers")},
        "rc": rc,
        "elapsed_s": elapsed,
        "found": found,
        "missing": missing,
        "energy_eV": energy,
        "status": "PASS" if not missing else "FAIL",
        "log_tail": _tail(log),
    }
    # On failure also grab the VASP stdout the wrappers captured.
    if missing:
        for extra_log in ("vasp.out", "out.log", "OSZICAR"):
            p = d / extra_log
            if p.is_file():
                result[f"tail_{extra_log}"] = _tail(p, 15)
    return result


def run_triage(workdir: Path, env_bin: Optional[str]) -> Optional[Dict]:
    """Run the repo's vasp_triage.py over the smoke tree (best effort)."""
    triage_py = UPSTREAM / "vasp_triage.py"
    out = workdir / "triage.json"
    if not triage_py.is_file():
        return None
    try:
        subprocess.run(
            [sys.executable, str(triage_py), str(workdir),
             "--json", str(out), "--only-problems", "--fixes"],
            capture_output=True, text=True, timeout=300)
        if out.is_file():
            return json.loads(out.read_text())
    except Exception:                                   # noqa: BLE001
        pass
    return None


def write_reports(workdir: Path, pre: Dict, results: List[Dict],
                  triage: Optional[Dict], meta: Dict) -> None:
    report = {"meta": meta, "preflight": pre, "results": results,
              "triage": triage}
    (workdir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, default=str))

    lines = [
        "=" * 70,
        f"NAS VASP smoke suite — {meta['timestamp']}",
        f"workdir: {workdir}",
        f"launcher: {meta['cmd_prefix'] or '<bare — WILL fail MPI builds>'}",
        f"POTCAR: {pre['potcar_mechanism']}",
        "NOTE: spin (ISPIN=2) intentionally OFF here — plumbing test only;",
        "      production auto-enables it for magnetic elements.",
        "=" * 70,
    ]
    if pre["missing_binaries"]:
        lines.append(f"PREFLIGHT FAIL — missing on PATH: "
                     f"{pre['missing_binaries']} "
                     f"(env_bin={meta['env_bin']}); VASP tests not run.")
    for r in results:
        lines.append(f"[{r['test']}] {r['status']}  rc={r['rc']}  "
                     f"{r['elapsed_s']}s  energy={r['energy_eV']}")
        lines.append(f"    covers : {r['covers']}")
        lines.append(f"    argv   : {' '.join(r['argv'])}")
        if r["missing"]:
            lines.append(f"    MISSING: {r['missing']}")
            lines.append("    --- log tail ---")
            lines.extend(f"    | {ln}" for ln in r["log_tail"][-15:])
    npass = sum(1 for r in results if r["status"] == "PASS")
    lines.append("-" * 70)
    lines.append(f"SUMMARY: {npass}/{len(results)} passed")
    if triage and triage.get("problems"):
        lines.append(f"TRIAGE: see triage.json "
                     f"({len(triage['problems'])} flagged dirs)")
    lines.append("If anything failed: send smoke_report.json (it has the "
                 "log tails and triage) — that is the debugging record.")
    (workdir / "smoke_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--element", default="Co")
    ap.add_argument("--potcar", default=None,
                    help="POTCAR file for --element. If omitted, relies "
                         "on ezvasp USEPOT + POTDIR in ~/.ezvasp.rc.")
    ap.add_argument("--cmd-prefix", default="mpiexec -n 8",
                    help="VASP launcher (trailing args to ATAT tools). "
                         "Keep the rank count SMALL — this is plumbing.")
    ap.add_argument("--env-bin", default=None, help="ATAT bin dir for PATH.")
    ap.add_argument("--workdir", default="vasp_smoke")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="Per-test wall clock cap in seconds; tests also "
                         "self-terminate via sentinels once outputs exist.")
    ap.add_argument("--only", default=None,
                    help="Comma list of tests, e.g. T1_static,T5_pollmach")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build case dirs + plan.json, run nothing.")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    potcar = Path(args.potcar) if args.potcar else None

    cases = build_cases(workdir, args.element, potcar, args.cmd_prefix)
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        cases = [c for c in cases if c["test"] in wanted]
    if args.dry_run:
        print(f"dry-run: built {len(cases)} cases under {workdir} "
              f"(see plan.json)")
        return 0

    meta = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "element": args.element, "cmd_prefix": args.cmd_prefix,
            "env_bin": args.env_bin, "timeout": args.timeout}
    pre = preflight(args.cmd_prefix, potcar, args.env_bin)

    results: List[Dict] = []
    if pre["missing_binaries"]:
        # No point launching VASP without the ATAT wrappers.
        write_reports(workdir, pre, results, None, meta)
        return 2

    for case in cases:
        print(f"--- {case['test']} ({case['covers']}) ---", flush=True)
        results.append(run_case(case, args.env_bin, args.timeout))

    triage = run_triage(workdir, args.env_bin)
    write_reports(workdir, pre, results, triage, meta)
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
