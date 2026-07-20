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
                     algo: str = "All") -> Path:
    """Write a 'relax' vasp.wrap (all DOF) into calc_dir."""
    calc_dir = Path(calc_dir)
    try:
        from strfile import read_structure
        natoms = len(read_structure(calc_dir / "str.out").atoms) or None
    except OSError:
        natoms = None
    wrap = build_vasp_wrap("relax", encut=encut, kppra=kppra,
                           dlm=dlm, algo=algo, natoms=natoms)
    path = calc_dir / "vasp.wrap"
    path.write_text(wrap)
    return path


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
    write_relax_wrap(calc_dir, encut, kppra, dlm=dlm, algo=algo)
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
        runner.run_polled(
            ["pollmach", "runstruct_vasp"] + vasp_launch, cwd=calc_dir,
            log=calc_dir / "runstruct.log",
            done_when=runner.all_have_file([calc_dir], "str_relax.out"),
            stop_sentinel="stopcar",
            env_bin=env_bin, timeout=timeout,
            work_dirs=[calc_dir], natoms=_natoms, kind="relax",
            done_file="str_relax.out")
        return calc_dir / "str_relax.out"

    # ── Both robustrelax modes need `robustrelax_vasp -mk` first to
    # generate the input files that the subsequent -id / plain
    # invocations expect. Skipping this was the cause of "VASP won't
    # start" in the user's job. Idempotent: -mk overwrites cleanly.
    runner.run_logged(
        ["robustrelax_vasp", "-mk"], cwd=calc_dir,
        log=calc_dir / "robustrelax_mk.log",
        env_bin=env_bin, timeout=timeout)

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

    # robustrelax stops on a 'stop' sentinel; poll until str_relax.out
    # appears.
    runner.run_polled(
        cmd, cwd=calc_dir,
        log=calc_dir / f"robustrelax_{method}.log",
        done_when=runner.all_have_file([calc_dir], "str_relax.out"),
        stop_sentinel="stop",
        env_bin=env_bin, timeout=timeout,
        work_dirs=[calc_dir], natoms=_natoms, kind="relax",
        done_file="str_relax.out")

    return calc_dir / "str_relax.out"
