#!/usr/bin/env python3
"""
Structural relaxation: normal (robustrelax) or inflection-detection (infdet).

Both modes produce the relaxed geometry in ``str_relax.out``, which fitfc and
sqs2tdb consume downstream.

  normal  : robustrelax_vasp drives a standard VASP relaxation (all DOF,
            ISIF=3). We hand it a 'relax' vasp.wrap.
  infdet  : robustrelax_vasp -id runs the inflection-detection / epicycle
            method (van de Walle et al., PRB 95:144113) to find the minimum
            energy geometry under the constraint that the softest phonon mode
            has zero frequency -- used for mechanically marginal phases. The
            infdet log lands in 01/infdet.log.

The converged (ENCUT, KPPRA) from converge.py are written into the relax
vasp.wrap so relaxation uses the calibrated basis/k-mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import runner
from vaspwrap import build_vasp_wrap
from phases import DLMConfig


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
                    method: str = "normal",
                    dlm: Optional[DLMConfig] = None,
                    algo: str = "All",
                    env_bin: Optional[str] = None,
                    timeout: int = 172800,
                    infdet_opts: str = "") -> Path:
    """Relax the structure in calc_dir, producing str_relax.out.

    method   "normal"  -> robustrelax_vasp
             "infdet"  -> robustrelax_vasp -id (epicycle inflection detection)
    infdet_opts  extra options forwarded to infdet via -idop "...".
    Returns the path to str_relax.out (whether or not it was produced -- caller
    should verify existence).
    """
    calc_dir = Path(calc_dir)
    write_relax_wrap(calc_dir, encut, kppra, dlm=dlm, algo=algo)

    if method == "infdet":
        cmd = ["robustrelax_vasp", "-id"]
        if infdet_opts:
            cmd += ["-idop", infdet_opts]
    elif method == "normal":
        cmd = ["robustrelax_vasp"]
    else:
        raise ValueError(f"unknown relax method {method!r}")

    # robustrelax stops on a 'stop' sentinel; poll until str_relax.out appears.
    runner.run_polled(
        cmd, cwd=calc_dir,
        log=calc_dir / f"robustrelax_{method}.log",
        done_when=runner.all_have_file([calc_dir], "str_relax.out"),
        stop_sentinel="stop",
        env_bin=env_bin, timeout=timeout)

    return calc_dir / "str_relax.out"
