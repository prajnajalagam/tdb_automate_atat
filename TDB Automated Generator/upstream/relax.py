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


RELAX_METHODS = ("runstruct", "normal", "infdet")


def write_relax_wrap(calc_dir: Path,
                     encut: int,
                     kppra: int,
                     dlm: Optional[DLMConfig] = None,
                     algo: str = "All") -> Path:
    """Write a 'relax' vasp.wrap (all DOF) into calc_dir."""
    calc_dir = Path(calc_dir)
    wrap = build_vasp_wrap("relax", encut=encut, kppra=kppra,
                           dlm=dlm, algo=algo)
    path = calc_dir / "vasp.wrap"
    path.write_text(wrap)
    return path


def relax_structure(calc_dir: Path,
                    encut: int,
                    kppra: int,
                    method: str = "runstruct",
                    dlm: Optional[DLMConfig] = None,
                    algo: str = "All",
                    env_bin: Optional[str] = None,
                    timeout: int = 172800,
                    infdet_opts: str = "") -> Path:
    """Relax the structure in calc_dir, producing str_relax.out.

    method   "runstruct" (default) -> pollmach runstruct_vasp
             "normal"               -> robustrelax_vasp -mk; robustrelax_vasp
             "infdet"               -> robustrelax_vasp -mk; robustrelax_vasp -id
    infdet_opts  extra options forwarded to infdet via -idop "...".
    Returns the path to str_relax.out (whether or not it was produced --
    caller should verify existence).
    """
    calc_dir = Path(calc_dir)
    write_relax_wrap(calc_dir, encut, kppra, dlm=dlm, algo=algo)

    if method not in RELAX_METHODS:
        raise ValueError(
            f"unknown relax method {method!r}; expected one of {RELAX_METHODS}"
        )

    if method == "runstruct":
        # Simple polled runstruct_vasp: pollmach turns str.out + vasp.wrap into
        # real VASP inputs and drives the relaxation to convergence. No -mk
        # step needed here -- runstruct_vasp reads vasp.wrap directly.
        runner.run_polled(
            ["pollmach", "runstruct_vasp"], cwd=calc_dir,
            log=calc_dir / "runstruct.log",
            done_when=runner.all_have_file([calc_dir], "str_relax.out"),
            stop_sentinel="stopcar",
            env_bin=env_bin, timeout=timeout)
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
        if infdet_opts:
            cmd += ["-idop", infdet_opts]
    else:  # method == "normal"
        cmd = ["robustrelax_vasp"]

    # robustrelax stops on a 'stop' sentinel; poll until str_relax.out
    # appears.
    runner.run_polled(
        cmd, cwd=calc_dir,
        log=calc_dir / f"robustrelax_{method}.log",
        done_when=runner.all_have_file([calc_dir], "str_relax.out"),
        stop_sentinel="stop",
        env_bin=env_bin, timeout=timeout)

    return calc_dir / "str_relax.out"
