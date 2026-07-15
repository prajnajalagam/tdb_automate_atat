#!/usr/bin/env python3
"""
Subprocess wrappers for the ATAT / VASP commands the upstream pipeline drives.

Two execution patterns:

run_logged    one-shot command (sqs2tdb, randomspin, fitfc, robustrelax_vasp).
              Streams stdout+stderr to a log file and returns the exit code.

run_polled    background job-manager pattern used by ATAT for VASP fan-out:
                  pollmach runstruct_vasp &
              pollmach keeps launching VASP into every str.out-bearing
              subdirectory until told to stop via a sentinel file
              (``stoppoll`` for fitfc strain runs, ``stop`` for robustrelax).
              run_polled launches it, waits for all target subdirs to produce
              their expected output, drops the sentinel, and joins.

Nothing here can be exercised without a real ATAT+VASP install, so the module
is intentionally thin and side-effect-logged; the convergence / selection
logic that *can* be tested lives in converge.py and potcar.py.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence


def _env_with_bin(env_bin: Optional[str]) -> dict:
    env = os.environ.copy()
    if env_bin:
        env["PATH"] = f"{env_bin}:{env.get('PATH', '')}"
    return env


def split_prefix(cmd_prefix: Optional[str]) -> List[str]:
    """Tokenize a VASP launch prefix like 'mpiexec -n 128' for argv use.

    ATAT's runstruct_vasp / robustrelax_vasp / pollmach take the command
    used to launch VASP as TRAILING arguments (that's how the reference
    NAS job works: `robustrelax_vasp -id -c 0.05 mpiexec -n 128`). We
    run subprocesses with shell=False, so the prefix must be split into
    separate argv tokens — appending the whole string as one element
    would hand ATAT a single argument containing spaces, which only
    works if ATAT happens to re-join args through a shell.
    """
    if not cmd_prefix:
        return []
    return shlex.split(cmd_prefix)


def run_logged(cmd: Sequence[str],
               cwd: Path,
               log: Path,
               env_bin: Optional[str] = None,
               timeout: Optional[int] = 600,
               check: bool = True) -> int:
    """Run a command, tee output to `log`, return exit code.

    Raises RuntimeError on non-zero exit when check=True so the orchestrator
    fails fast on a broken ATAT call rather than silently continuing with
    missing files.
    """
    cwd = Path(cwd)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        fh.write(f"$ cd {cwd}\n$ {' '.join(cmd)}\n{'-'*60}\n")
        fh.flush()
        try:
            proc = subprocess.run(
                list(cmd), cwd=str(cwd), env=_env_with_bin(env_bin),
                stdout=fh, stderr=subprocess.STDOUT, text=True,
                timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT after {timeout}s\n")
            rc = -1
        except FileNotFoundError as exc:
            fh.write(f"\nCOMMAND NOT FOUND: {exc}\n")
            rc = 127
    if check and rc != 0:
        raise RuntimeError(
            f"command failed (rc={rc}): {' '.join(cmd)} in {cwd}; see {log}")
    return rc


def run_polled(cmd: Sequence[str],
               cwd: Path,
               log: Path,
               done_when,
               stop_sentinel: str = "stoppoll",
               env_bin: Optional[str] = None,
               poll_interval: float = 30.0,
               timeout: float = 86400.0) -> int:
    """Launch a `pollmach ...` job manager in the background and wait until
    `done_when(cwd)` returns True (all expected outputs present), then drop the
    stop sentinel and join.

    done_when   callable(cwd: Path) -> bool, polled every poll_interval s.
    stop_sentinel  filename created in cwd to ask pollmach to stop cleanly
                   ('stoppoll' for fitfc strain runs, 'stop' for robustrelax).
    """
    cwd = Path(cwd)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w")
    fh.write(f"$ cd {cwd}\n$ {' '.join(cmd)} &\n{'-'*60}\n")
    fh.flush()
    proc = subprocess.Popen(
        list(cmd), cwd=str(cwd), env=_env_with_bin(env_bin),
        stdout=fh, stderr=subprocess.STDOUT, text=True)

    t0 = time.time()
    rc = 0
    try:
        while True:
            if done_when(cwd):
                break
            if proc.poll() is not None:
                # pollmach exited on its own (e.g. nothing left to do).
                rc = proc.returncode or 0
                break
            if time.time() - t0 > timeout:
                fh.write(f"\nTIMEOUT after {timeout}s; stopping poll\n")
                rc = -1
                break
            time.sleep(poll_interval)
    finally:
        # Ask pollmach to stop, then ensure the process is gone.
        try:
            (cwd / stop_sentinel).touch()
        except OSError:
            pass
        try:
            proc.wait(timeout=poll_interval * 2)
        except subprocess.TimeoutExpired:
            proc.terminate()
        fh.close()
    return rc


# ----- convenience predicates for run_polled's done_when ------------------

def all_have_file(subdirs: List[Path], filename: str):
    """done_when predicate: every subdir contains `filename`."""
    def _pred(_cwd: Path) -> bool:
        return all((d / filename).is_file() for d in subdirs)
    return _pred


def all_energy_present(subdirs: List[Path]):
    """done_when predicate: every subdir has a non-empty `energy` file."""
    def _pred(_cwd: Path) -> bool:
        for d in subdirs:
            e = d / "energy"
            if not (e.is_file() and e.stat().st_size > 0):
                return False
        return True
    return _pred
