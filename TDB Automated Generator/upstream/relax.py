#!/usr/bin/env python3
"""
Structural relaxation: runstruct (default), normal (robustrelax) or
inflection-detection (infdet).

All modes produce the relaxed geometry in ``str_relax.out``, which fitfc
and sqs2tdb consume downstream.

  runstruct : pollmach runstruct_vasp. Turns str.out + vasp.wrap into
              the real VASP inputs and drives a single ISIF=3 relaxation.
              Simplest, fastest for well-converged cases and now the
              default upstream relax method.
  normal    : robustrelax_vasp with its own crash/restart tolerance
              layer. Preceded by ``robustrelax_vasp -mk`` which builds
              the input files robustrelax needs from vasp.wrap + str.out.
              Use when runstruct fails intermittently.
  infdet    : robustrelax_vasp -id runs inflection detection / epicycle
              (van de Walle et al., PRB 95:144113) — used for
              mechanically marginal phases. Also preceded by
              ``robustrelax_vasp -mk``.

The converged (ENCUT, KPPRA) from converge.py are written into the
relax vasp.wrap so relaxation uses the calibrated basis/k-mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import runner
from vaspwrap import build_vasp_wrap
from phases import DLMConfig


RELAX_METHODS = ("infdet", "normal", "runstruct")

# Inflection detection needs a strain cutoff to work; the reference NAS
# job uses 5% ("robustrelax_vasp -id -c 0.05 mpiexec -n 128"). Applied
# by default unless the caller passes its own -c via relax_opts.
INFDET_STRAIN_CUTOFF = 0.05


def write_relax_wrap(calc_dir: Path,
                     encut: int,
                     kppra: int,
                     dlm: Optional[DLMConfig] = None,
                     algo: str = "All",
                     ranks: Optional[int] = None) -> Path:
    """Write a 'relax' vasp.wrap (all DOF) into calc_dir."""
    calc_dir = Path(calc_dir)
    try:
        from strfile import read_structure
        natoms = len(read_structure(calc_dir / "str.out").atoms) or None
    except OSError:
        natoms = None
    wrap = build_vasp_wrap("relax", encut=encut, kppra=kppra,
                           dlm=dlm, algo=algo, natoms=natoms, ranks=ranks)
    path = calc_dir / "vasp.wrap"
    path.write_text(wrap)
    return path


# The author-documented success marker of an infdet run: ALWAYS the
# last line of 01/infdet.log when inflection detection completed.
INFDET_NORMAL_TERMINATION = "infdet terminated normally"


def robustrelax_complete(calc_dir: Path) -> bool:
    """True only when robustrelax_vasp has finished an ENTIRE branch.

    THE 2026-07-22 CoCr POSTMORTEM: the old done_when was 'str_relax.out
    exists' — but robustrelax's STEP 1 (the full relaxation) already
    writes str_relax.out, so the poller declared victory and killed
    robustrelax right as it entered the 00/ volume relax. 01/ (the
    actual inflection detection) never ran on a single mixing SQS, the
    kill orphaned the mpiexec children ("vaspvol keeps running"), and
    on every restart the pre-existing str_relax.out re-triggered the
    kill 60 s after launch. checkrelax on the step-1 geometry is NOT a
    completion (or success) signal for robustrelax — only for plain
    runstruct full relaxations.

    True completion, from the robustrelax_vasp source:
      stable branch   (drift <= -c cutoff): `energy` rescaled from
          `energy_sup` — both present at the end, no energy_end.
      unstable branch (-id engaged): `energy_end` written at branch
          entry; done only when 01/cstr_relax.out exists (infdet
          finished) AND `energy` was rewritten from 01/energy (the
          inflection-point energy, the LAST file the branch writes
          before updating str_relax.out).
    The step-1 transient (str_relax.out + runstruct's own `energy`,
    neither energy_sup nor energy_end yet) matches NEITHER arm, so the
    poller keeps waiting and robustrelax simply runs to completion —
    it self-terminates, unlike pollmach.
    """
    d = Path(calc_dir)
    stable = (d / "energy_sup").is_file() and (d / "energy").is_file()
    unstable = ((d / "energy_end").is_file()
                and (d / "01" / "cstr_relax.out").is_file()
                and (d / "energy").is_file())
    return stable or unstable


def infdet_status(calc_dir: Path) -> "tuple[bool, bool, str]":
    """(engaged, ok, detail) for a ``robustrelax_vasp -id`` run.

    Semantics from the robustrelax_vasp source (verified 2026-07-20):
    when the full relaxation strays beyond the -c cutoff, robustrelax
    takes the inflection-detection branch — the FULLY-RELAXED energy
    goes to ``energy_end`` (the decayed structure: NOT the result),
    infdet runs in ``01/``, and on success the INFLECTION-POINT energy
    is written to ``energy`` (scaled from 01/energy) with
    ``str_relax.out`` = the inflection geometry.

    engaged: the -id branch was taken (01/ exists or energy_end was
             written). A large checkrelax value on an engaged run is
             EXPECTED, not a failure — the path deliberately spans a
             large deformation.
    ok:      01/infdet.log ends with "infdet terminated normally" AND
             the inflection-point energy landed in <calc_dir>/energy.
    """
    calc_dir = Path(calc_dir)
    engaged = (calc_dir / "01").is_dir() or \
        (calc_dir / "energy_end").is_file()
    if not engaged:
        return False, False, "not engaged (relaxation within -c cutoff)"
    log = calc_dir / "01" / "infdet.log"
    if not log.is_file():
        return True, False, "01/infdet.log missing (failed before infdet)"
    try:
        lines = [ln.strip() for ln in log.read_text().splitlines()
                 if ln.strip()]
    except OSError as exc:
        return True, False, f"01/infdet.log unreadable: {exc}"
    if not lines or INFDET_NORMAL_TERMINATION not in lines[-1]:
        last = lines[-1] if lines else "(empty)"
        return True, False, f"infdet did not terminate normally " \
                            f"(last log line: {last!r})"
    if not (calc_dir / "energy").is_file():
        return True, False, ("infdet terminated normally but the "
                             "inflection-point energy was not written "
                             "to `energy` (static in 01/ failed?)")
    return True, True, "infdet terminated normally"


def relax_structure(calc_dir: Path,
                    encut: int,
                    kppra: int,
                    method: str = "infdet",
                    dlm: Optional[DLMConfig] = None,
                    algo: str = "All",
                    env_bin: Optional[str] = None,
                    timeout: int = 172800,
                    infdet_opts: str = "",
                    relax_opts: str = "",
                    cmd_prefix: str = "") -> Path:
    """Relax the structure in calc_dir, producing str_relax.out.

    method   "infdet" (default) -> robustrelax_vasp -mk;
                 robustrelax_vasp -id -c 0.05 [cmd_prefix]
                 (inflection detection; the -c strain cutoff is REQUIRED
                 for it to engage — reference NAS job uses 0.05)
             "normal"    -> robustrelax_vasp -mk; robustrelax_vasp [cmd_prefix]
             "runstruct" -> pollmach runstruct_vasp [cmd_prefix]

    cmd_prefix   the command used to launch VASP, e.g. "mpiexec -n 128".
                 ATAT tools take it as TRAILING arguments (reference NAS
                 job: `robustrelax_vasp -id -c 0.05 mpiexec -n 128`).
                 Without it, runstruct_vasp launches the MPI vasp binary
                 bare, which dies before writing OSZICAR. Tokenized via
                 shlex so "mpiexec -n 128" becomes three argv elements.
    relax_opts   direct robustrelax_vasp options, e.g. "-c 0.05"
                 (constraint tolerance). Applied to both 'normal' and
                 'infdet'; ignored for 'runstruct'.
    infdet_opts  options forwarded to the infdet subprogram via -idop
                 (distinct from relax_opts, which robustrelax consumes
                 itself).
    Returns the path to str_relax.out (whether or not it was produced --
    caller should verify existence).
    """
    calc_dir = Path(calc_dir)
    from vaspwrap import ranks_from_prefix
    _ranks = ranks_from_prefix(cmd_prefix)
    try:
        from strfile import read_structure
        _natoms = len(read_structure(calc_dir / "str.out").atoms) or None
    except OSError:
        _natoms = None

    if method not in RELAX_METHODS:
        raise ValueError(
            f"unknown relax method {method!r}; expected one of {RELAX_METHODS}"
        )

    vasp_launch = runner.split_prefix(cmd_prefix)

    if method == "runstruct":
        # Simple polled runstruct_vasp: pollmach turns str.out + vasp.wrap into
        # real VASP inputs and drives the relaxation to convergence. No -mk
        # step needed here -- runstruct_vasp reads vasp.wrap directly.
        # Trailing tokens ride through pollmach to runstruct_vasp, which
        # uses them as the VASP launch command.
        write_relax_wrap(calc_dir, encut, kppra, dlm=dlm, algo=algo,
                         ranks=_ranks)
        runner.run_polled(
            ["pollmach", "runstruct_vasp"] + vasp_launch, cwd=calc_dir,
            log=calc_dir / "runstruct.log",
            done_when=runner.all_have_file([calc_dir], "str_relax.out"),
            stop_sentinel="stopcar",
            env_bin=env_bin, timeout=timeout,
            work_dirs=[calc_dir], natoms=_natoms, kind="relax",
            done_file="str_relax.out")
        return calc_dir / "str_relax.out"

    # ── Both robustrelax modes: write the TUNED vasp.wrap FIRST, then
    # run `robustrelax_vasp -mk`. Order matters (verified against the
    # robustrelax_vasp source, 2026-07-20): -mk does NOT create
    # vasp.wrap — it REQUIRES it, and derives every auxiliary wrap the
    # workflow uses (vaspvol.wrap ISIF=7, vaspstatic.wrap ISMEAR=-5,
    # vaspid.wrap for the infdet statics, vaspf.wrap, vaspneb.wrap) by
    # grep-transforming vasp.wrap. Writing ours first is therefore what
    # propagates the converged ENCUT/KPPRA, spin/MAGMOM and NCORE/KPAR
    # into EVERY step robustrelax takes (volume relax, infdet statics,
    # final static). With vasp.wrap absent, -mk exits having written
    # nothing and the -id stage later dies on missing vaspid.wrap.
    write_relax_wrap(calc_dir, encut, kppra, dlm=dlm, algo=algo,
                     ranks=_ranks)
    runner.run_logged(
        ["robustrelax_vasp", "-mk"], cwd=calc_dir,
        log=calc_dir / "robustrelax_mk.log",
        env_bin=env_bin, timeout=timeout)
    # Rerun hygiene: stale `error` makes robustrelax bail immediately
    # after step 1 ("Error during relaxation run"); stale `stop` kills
    # the 01/ infdet loop on sight. Both were littered across the
    # 2026-07-22 tree by the premature-kill bug.
    for stale in ("error", "stop"):
        f = calc_dir / stale
        if f.is_file():
            f.unlink()

    if method == "infdet":
        cmd = ["robustrelax_vasp", "-id"]
        # Strain cutoff is mandatory for inflection detection; add the
        # 5% default unless the caller supplies their own -c.
        if "-c" not in runner.split_prefix(relax_opts):
            cmd += ["-c", f"{INFDET_STRAIN_CUTOFF}"]
        if infdet_opts:
            cmd += ["-idop", infdet_opts]
    else:  # method == "normal"
        cmd = ["robustrelax_vasp"]

    if relax_opts:
        cmd += runner.split_prefix(relax_opts)
    cmd += vasp_launch          # VASP launch command LAST, per ATAT usage

    # robustrelax self-terminates; the predicate must describe FULL
    # branch completion, NOT str_relax.out (which step 1 already
    # writes — see robustrelax_complete for the 2026-07-22 postmortem).
    runner.run_polled(
        cmd, cwd=calc_dir,
        log=calc_dir / f"robustrelax_{method}.log",
        done_when=lambda _cwd: robustrelax_complete(calc_dir),
        stop_sentinel="stop",
        env_bin=env_bin, timeout=timeout,
        work_dirs=[calc_dir], natoms=_natoms, kind="relax",
        done_file="energy")

    return calc_dir / "str_relax.out"
