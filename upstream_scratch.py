#!/usr/bin/env python3
"""
upstream_scratch.py — clean-slate rebuild of the upstream binary
workflow, translated line by line from user pseudocode (2026-07-28).

Scope so far (step 1): workdir creation, SQS generation for all
phases, DLM decoration hook, and the two rich-side convergence tests
(reusing the existing converge.py verbatim). Later steps append below.

Conventions:
- Each pseudocode step is quoted in a comment directly above its
  Python translation, so the mapping stays reviewable line by line.
- Where a hard-won lesson from the previous pipeline applies (sqs2tdb
  two-pass quirks), the guard is included and marked [ADDED GUARD] —
  it is an addition to the pseudocode, not a reinterpretation of it.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

# Reuse the battle-tested modules from the existing package verbatim
# (user directive: "use the convergence.py file as is").
_UPSTREAM = Path(__file__).resolve().parent / "TDB Automated Generator" / "upstream"
sys.path.insert(0, str(_UPSTREAM))
import converge          # noqa: E402  (sweeps, selection rule, caching)
import potcar            # noqa: E402  (ENMAX parsing for the grids)

# ─────────────────────────── USER INPUTS ────────────────────────────
EL = ["Co", "Cr"]                      # [EL1, EL2]

# start level = 2, 0 — first value: single-sublattice phases;
# second value: multi-sublattice phases.
START_LEVEL_SINGLE = 2
START_LEVEL_MULTI = 0

PHASES = ["FCC_A1", "BCC_A2", "HCP_A3", "SIGMA_D8B"]

WORKING_DIR = Path("/nobackup/pjalagam")

# Known-good job scripts to clone in later steps (relax / fitfc).
ROBUSTRELAX_JOBFILE = Path(
    "/nobackup/pjalagam/CoCrNi_TC/FCC_A1_small/sqs_lev=0_a_Cr=1/run_pfe.sh")
FITFC_JOBFILE = Path(
    "/nobackup/pjalagam/CoCrNi_TC/FCC_A1_small/sqs_lev=0_a_Cr=1/"
    "vol_0/run_fitfc.sh")   # TODO: confirm the exact vol_0/... subpath

DLM_MODE = "off"                       # "off" | "on"

# Carried-over knobs the reused converge.py needs (same values as the
# existing templates; POTCAR location per the known zwu6 tree).
POTCARS = [Path(f"/home1/zwu6/vasp/POTPAW_PBE.64/{el}/POTCAR") for el in EL]
CMD_PREFIX = "mpiexec -n 32"           # VASP launcher for sweep statics
ATAT_BIN = "/home7/pjalagam/bin"       # prepended to PATH for ATAT tools
TOL_EV = 0.0001                        # 0.1 meV/atom successive-step tol
RNG_SEED = 0                           # reproducible convergence picks
# ────────────────────────── END USER INPUTS ─────────────────────────

SINGLE_SUBLATTICE = ("FCC_A1", "BCC_A2", "HCP_A3")


def run_logged(cmd: list, cwd: Path, log: Path) -> int:
    """Run one command, tee stdout+stderr to `log`, return exit code."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        fh.write(f"$ cd {cwd}\n$ {' '.join(cmd)}\n{'-' * 60}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    return proc.returncode


# ─────────────────────── step 1a: work directory ────────────────────

def make_workdir(base: Path = WORKING_DIR) -> Path:
    """
    Pseudocode:
        i = 0
        mkdir EL1_EL2_upstream. if directory already exists:
        add 1 to i, then mkdir EL1_EL2_upstream_i. if directory
        exists, continue to add 1 to i until it does not exist.
        WORKDIR becomes EL1_EL2_upstream(_i) inside WORKING_DIR.
    """
    stem = f"{EL[0]}_{EL[1]}_upstream"      # e.g. Co_Cr_upstream
    workdir = base / stem
    i = 0
    while workdir.exists():
        i += 1
        workdir = base / f"{stem}_{i}"
    workdir.mkdir(parents=True)
    return workdir


# ─────────────────────── step 1b: SQS generation ────────────────────

def sqs_target_and_level(phase: str) -> tuple:
    """
    Pseudocode: if single sublattice (BCC_A2, FCC_A1, HCP_A3), add
    _small to the phase name and use start level [0]; otherwise use
    the bare phase name and start level [1].
    """
    if phase in SINGLE_SUBLATTICE:
        return phase + "_small", START_LEVEL_SINGLE
    return phase, START_LEVEL_MULTI


def generate_sqs(workdir: Path) -> None:
    """
    Pseudocode, per phase:
        RUN: sqs2tdb -cp -sp=EL1,EL2 -lv=<level> -l=<target>
        then, run same command again.
    (sqs2tdb's two-pass protocol: pass 1 plants <target>/species.in
    and prompts; pass 2 copies the SQS directories.)
    """
    for phase in PHASES:
        target, level = sqs_target_and_level(phase)
        cmd = ["sqs2tdb", "-cp", f"-sp={EL[0]},{EL[1]}",
               f"-lv={level}", f"-l={target}"]

        # [ADDED GUARD] sqs2tdb's own passes drop a species.in in the
        # CWD; if one is present when the next phase's pass 1 runs,
        # the two-pass handshake stalls (both passes prompt, nothing
        # is copied — the 2026-07-22 missing-HCP incident). Quarantine
        # it before every generation.
        stray = workdir / "species.in"
        if stray.is_file():
            stray.rename(workdir / "species.in.stray")

        rc1 = run_logged(cmd, workdir, workdir / f"sqs2tdb_{target}.pass1.log")
        rc2 = run_logged(cmd, workdir, workdir / f"sqs2tdb_{target}.pass2.log")

        # [ADDED GUARD] fail loudly if pass 2 is still prompting or
        # produced nothing — the silent work-root fallback that once
        # aliased HCP onto the other phases is forbidden.
        pass2 = (workdir / f"sqs2tdb_{target}.pass2.log").read_text()
        if "Edit the file" in pass2:
            raise RuntimeError(
                f"{target}: pass 2 still prompting — lattice missing "
                f"from $atatdir/data/sqsdb or species.in handshake "
                f"broken (rc1={rc1}, rc2={rc2})")
        if not (workdir / target).is_dir():
            raise RuntimeError(f"{target}: sqs2tdb created no phase dir")


# ───────────────────────── step 1c: DLM hook ────────────────────────

def apply_dlm(workdir: Path) -> None:
    """
    Pseudocode: if DLM mode = off, stop here and continue. If on:
    single-sublattice phases -> cd into each phase directory, RUN
    randomspin. Multi-sublattice -> the workflow outlined in upstream.
    """
    if DLM_MODE != "on":
        return
    for phase in PHASES:
        target, _lev = sqs_target_and_level(phase)
        if phase in SINGLE_SUBLATTICE:
            run_logged(["randomspin"], workdir / target,
                       workdir / f"randomspin_{target}.log")
        else:
            # Multi-sublattice DLM (SIGMA): the existing implementation
            # is sqsgen.sigma_lev3_to_lev0_dlm (generate at lev=3, split
            # each endmember's equivalent sites into +/- spin
            # pseudo-species). Simplification candidate per user note —
            # wired up in a later step once we decide whether to reuse
            # it or replace it.
            raise NotImplementedError(
                f"DLM for multi-sublattice phase {phase}: reuse "
                f"sqsgen.sigma_lev3_to_lev0_dlm or simplify (later step)")


# ────────────── step 1d: pick the two convergence tests ─────────────

def fraction_of(dirname: str, element: str) -> float:
    """Fraction of `element` scraped from a decorated SQS dir name.

    Names look like  sqs_lev=2_a_Co=0.25,a_Cr=0.75  (FCC/BCC, site a)
    or               sqs_lev=1_c_Co=0.5,c_Cr=0.5    (HCP, site c).
    Each token is  <site>_<El>=<fraction>; for the single-sublattice
    phases used here every token carries the same site, so the
    fraction of an element is its token value divided by the token
    total (which is 1.0 for these names, but normalize anyway).
    """
    import re
    fr = {}
    for el, val in re.findall(r"[a-z]+_([A-Z][a-z]?)=([0-9.]+)", dirname):
        fr[el] = fr.get(el, 0.0) + float(val)
    total = sum(fr.values())
    return fr.get(element, 0.0) / total if total else 0.0


def pick_convergence_dirs(workdir: Path, seed: int = RNG_SEED) -> dict:
    """
    Pseudocode: use a NON-ENDMEMBER SQS in the EL1-rich region (>= 50%
    counts) of ANY single-sublattice phase, and a non-endmember SQS in
    the EL2-rich region (>= 50%) of any single-sublattice phase —
    exactly 2 convergence tests for a binary. Pick randomly among all
    candidates to reduce bias.
    """
    rich = {EL[0]: [], EL[1]: []}
    for phase in SINGLE_SUBLATTICE:
        target, _lev = sqs_target_and_level(phase)
        phase_dir = workdir / target
        if not phase_dir.is_dir():
            continue
        for d in sorted(phase_dir.iterdir()):
            # non-endmember only: skip lev=0 dirs and endmem markers
            if not d.is_dir() or "lev=0" in d.name \
                    or (d / "endmem").is_file() \
                    or not (d / "str.out").is_file():
                continue
            for el in EL:
                if fraction_of(d.name, el) >= 0.5:   # 50% counts as rich
                    rich[el].append(d)

    rng = random.Random(seed)
    if not rich[EL[0]] or not rich[EL[1]]:
        raise RuntimeError(
            f"convergence picks: empty rich-side candidate pool "
            f"({ {el: len(v) for el, v in rich.items()} }) — is the "
            f"start level high enough to produce non-endmember SQS?")
    pick1 = rng.choice(rich[EL[0]])
    pick2 = rng.choice(rich[EL[1]])
    # A 50/50 SQS is in BOTH pools; if the same dir got drawn twice,
    # redraw the second pick from the remaining candidates so two
    # distinct tests run (when only one candidate exists, keep it and
    # accept a single test — nothing else to sweep).
    if pick2 == pick1 and len(rich[EL[1]]) > 1:
        pick2 = rng.choice([d for d in rich[EL[1]] if d != pick1])
    return {EL[0]: pick1, EL[1]: pick2}


# ─────────────── step 1e: run the two convergence tests ─────────────

def run_convergence_tests(workdir: Path, picks: dict) -> tuple:
    """
    Pseudocode: run the convergence test (converge.py, unchanged
    criteria/algorithm) on each pick; the LARGEST ENCUT and KPPRA
    among the tests is then used for all actual robustrelax runs.
    """
    results = {}
    for el, sqs_dir in picks.items():
        print(f"[convergence] {el}-rich test on {sqs_dir.name}")
        encut, kppra, kres, eres = converge.converge_sqs(
            sqs_dir, sqs_dir / "convergence", POTCARS,
            dlm=None, algo="All", tol_ev=TOL_EV,
            env_bin=ATAT_BIN, timeout=7200 * 4,
            cmd_prefix=CMD_PREFIX)
        print(kres.table())
        print(eres.table())
        results[el] = (encut, kppra)
    final_encut = max(v[0] for v in results.values())
    final_kppra = max(v[1] for v in results.values())
    print(f"[convergence] GLOBAL settings: ENCUT={final_encut} eV, "
          f"KPPRA={final_kppra} (elementwise max over {results})")
    return final_encut, final_kppra


# ──────────────────────────── driver ────────────────────────────────

def main() -> int:
    workdir = make_workdir()
    print(f"[workdir] {workdir}")
    generate_sqs(workdir)
    apply_dlm(workdir)
    picks = pick_convergence_dirs(workdir)
    for el, d in picks.items():
        print(f"[picks] {el}-rich: {d.relative_to(workdir)}")
    encut, kppra = run_convergence_tests(workdir, picks)
    # (next pseudocode block continues from here: robustrelax runs at
    #  the global ENCUT/KPPRA via the known-good jobfile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
