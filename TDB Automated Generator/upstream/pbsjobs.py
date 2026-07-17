#!/usr/bin/env python3
"""
PBS job broker: fan the pipeline's VASP work out as individual qsub
jobs instead of driving VASP inside one monolithic allocation.

This is the cluster pattern the sqs2tdb paper itself prescribes
(Calphad 58 (2017) 70, Sec. 3.2: ``foreachfile wait sbatch
jobfile.in`` — one queued job per wait-marked directory); the broker
adds what the paper leaves to the user: resource sizing per cell,
completion tracking, bounded fan-out, and retries.

Integration point: runner.run_polled is the single chokepoint through
which every long VASP execution flows (relaxations, robustrelax/infdet,
pollmach force runs). Installing a Broker as runner's execution backend
(runner.set_backend) reroutes those commands into rendered PBS scripts
— the pipeline's control flow is unchanged; only WHERE the command runs
moves. Short local steps (sqs2tdb, fitfc, checkrelax) stay in the
orchestrator process, which is light enough for a NAS front end.

Job shapes
----------
single   one command in one job (a relaxation in its SQS dir).
loop     one job iterating a command over N work dirs (fallback for
         force runs when arrays are disabled or N == 1).
array    PBS job array (#PBS -J 0-(N-1)): one element per work dir —
         all N perturbation statics run wall-parallel for the same
         SBUs. The dir list is written to a manifest the elements
         index with $PBS_ARRAY_INDEX.

Resource sizing: size_for(natoms, kind) — ncpus/walltime chosen from
the cell that will actually run, and the trailing ``mpiexec -n K`` of
the command is REWRITTEN to the job's ncpus (retarget_launcher), so the
select-line/rank mismatch class of failure (the 2026-07-14 4x
oversubscription) is structurally impossible in this mode.

State: <workdir>/.qjob_<tag> records the submitted job id + attempt;
completion is judged by the SAME done_when predicates the local
backend uses (files in the tree), never by job exit codes alone. A job
that disappears from qstat without producing its outputs is
resubmitted up to max_retries, then reported failed. Because all state
lives in the tree + marker files, an orchestrator restart reconciles
instead of resubmitting blindly.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Resource sizing policy
# ---------------------------------------------------------------------------

# (max_natoms, ncpus, walltime) per job kind. Whole nodes are charged
# on NAS regardless of ranks, so the savings lever is WALLTIME (and
# model choice, which is a Broker field) — but matching ranks to the
# cell also avoids the small-cell decomposition crashes.
SIZING = {
    "relax": [(4, 8, "01:00:00"), (16, 16, "02:00:00"),
              (48, 32, "04:00:00"), (10**9, 64, "08:00:00")],
    "force": [(16, 16, "00:30:00"), (48, 32, "01:00:00"),
              (10**9, 64, "02:00:00")],
    "probe": [(10**9, 32, "04:00:00")],   # adaptive sweeps, sequential
    "generic": [(10**9, 32, "04:00:00")],
}


def size_for(natoms: Optional[int], kind: str) -> Dict:
    """(ncpus, walltime) for a cell of `natoms` doing `kind` work."""
    table = SIZING.get(kind, SIZING["generic"])
    n = natoms if natoms else 10**9        # unknown -> biggest tier
    for max_n, ncpus, wall in table:
        if n <= max_n:
            return {"ncpus": ncpus, "walltime": wall}
    max_n, ncpus, wall = table[-1]
    return {"ncpus": ncpus, "walltime": wall}


def retarget_launcher(cmd: Sequence[str], ncpus: int) -> List[str]:
    """Rewrite a trailing 'mpiexec -n K' to the job's ncpus."""
    cmd = list(cmd)
    for i, tok in enumerate(cmd):
        if tok in ("mpiexec", "mpirun") and i + 2 < len(cmd) + 1:
            if i + 2 < len(cmd) and cmd[i + 1] == "-n":
                cmd[i + 2] = str(ncpus)
    return cmd


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

@dataclass
class Broker:
    """Submits, tracks, retries and caps PBS jobs for the pipeline.

    site_env: shell lines sourced at the top of every job (modules,
    venv, PATH) — the orchestrator writes them once from its own
    environment config so every job matches the interactive setup.
    """
    work_root: Path
    group_list: str = "a1485"
    model: str = "mil_ait"
    queue: str = "normal"
    site_env: str = ""
    max_inflight: int = 16
    max_retries: int = 1
    poll_interval: float = 60.0
    use_arrays: bool = True
    dry_run: bool = False                      # render + record, no qsub
    submitted: List[str] = field(default_factory=list)

    # -- low-level ---------------------------------------------------------

    def _script_header(self, name: str, ncpus: int, walltime: str,
                       out: Path, array_n: int = 0) -> str:
        lines = [
            "#!/bin/bash",
            "#PBS -S /bin/bash",
            f"#PBS -N {name[:15]}",
            f"#PBS -q {self.queue}",
            f"#PBS -l select=1:ncpus={ncpus}:mpiprocs={ncpus}"
            f":model={self.model}",
            f"#PBS -l walltime={walltime}",
            "#PBS -j oe",
            f"#PBS -o {out}",
            f"#PBS -W group_list={self.group_list}",
        ]
        if array_n >= 2:
            lines.append(f"#PBS -J 0-{array_n - 1}")
        lines += ["", "set -uo pipefail", self.site_env, ""]
        return "\n".join(lines)

    def qsub(self, script: Path) -> str:
        if self.dry_run:
            jid = f"dry-{len(self.submitted)}"
            self.submitted.append(jid)
            return jid
        p = subprocess.run(["qsub", str(script)], capture_output=True,
                           text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError(f"qsub {script} failed: {p.stderr.strip()}")
        jid = p.stdout.strip().split()[0]
        self.submitted.append(jid)
        return jid

    def alive(self, job_id: str) -> bool:
        if self.dry_run:
            return False
        base = job_id.split("[")[0]            # arrays: strip [] suffix
        p = subprocess.run(["qstat", base], capture_output=True,
                           text=True, timeout=120)
        return p.returncode == 0 and base.split(".")[0] in p.stdout

    def n_inflight(self) -> int:
        if self.dry_run:
            return 0
        n = 0
        for jid in self.submitted:
            if self.alive(jid):
                n += 1
        return n

    def _throttle(self) -> None:
        while self.n_inflight() >= self.max_inflight:
            time.sleep(self.poll_interval)

    # -- job rendering -----------------------------------------------------

    def render_single(self, tag: str, cwd: Path, cmd: Sequence[str],
                      ncpus: int, walltime: str) -> Path:
        script = cwd / f"qjob_{tag}.pbs"
        body = (self._script_header(tag, ncpus, walltime,
                                    cwd / f"qjob_{tag}.log")
                + f"cd {shlex.quote(str(cwd))}\n"
                + shlex.join(retarget_launcher(cmd, ncpus)) + "\n"
                + f"echo $? > .qrc_{tag}\n")
        script.write_text(body)
        return script

    def render_loop(self, tag: str, cwd: Path, cmd: Sequence[str],
                    work_dirs: Sequence[Path], done_file: str,
                    ncpus: int, walltime: str) -> Path:
        """One job running `cmd` inside each work dir that still lacks
        done_file — the paper's `foreachfile wait ...` pattern with an
        explicit list instead of wait-marker discovery."""
        script = cwd / f"qjob_{tag}.pbs"
        runline = shlex.join(retarget_launcher(cmd, ncpus))
        dirs = " ".join(shlex.quote(str(d)) for d in work_dirs)
        body = (self._script_header(tag, ncpus, walltime,
                                    cwd / f"qjob_{tag}.log")
                + f"for d in {dirs}; do\n"
                + f"  if [ ! -f \"$d/{done_file}\" ]; then\n"
                + f"    ( cd \"$d\" && {runline} && rm -f wait )\n"
                + "  fi\n"
                + "done\n"
                + f"echo $? > .qrc_{tag}\n")
        script.write_text(body)
        return script

    def render_array(self, tag: str, cwd: Path, cmd: Sequence[str],
                     work_dirs: Sequence[Path],
                     ncpus: int, walltime: str) -> Path:
        """PBS job array: element i runs `cmd` in work_dirs[i]. All
        perturbation statics of one SQS run wall-parallel for the same
        SBUs as running them serially."""
        manifest = cwd / f"qjob_{tag}.dirs"
        manifest.write_text("\n".join(str(d) for d in work_dirs) + "\n")
        script = cwd / f"qjob_{tag}.pbs"
        runline = shlex.join(retarget_launcher(cmd, ncpus))
        body = (self._script_header(tag, ncpus, walltime,
                                    cwd / f"qjob_{tag}.^array_index^.log",
                                    array_n=len(work_dirs))
                + f"d=$(sed -n \"$((PBS_ARRAY_INDEX + 1))p\" "
                + f"{shlex.quote(str(manifest))})\n"
                + f"cd \"$d\"\n"
                + runline + "\n"
                + "rc=$?; rm -f wait\n"
                + f"echo $rc > .qrc_{tag}\n")
        script.write_text(body)
        return script

    # -- high-level: run a command as job(s), wait on done_when ------------

    def run_as_job(self, tag: str, cwd: Path, cmd: Sequence[str],
                   done_when: Callable[[Path], bool],
                   work_dirs: Optional[Sequence[Path]] = None,
                   natoms: Optional[int] = None,
                   kind: str = "generic",
                   done_file: str = "energy") -> int:
        """PBS-backed equivalent of runner.run_polled.

        Renders the appropriate job shape, submits (respecting the
        in-flight cap), then polls done_when. A job that leaves the
        queue without satisfying done_when is resubmitted up to
        max_retries. Returns 0 on done_when success, -1 otherwise.
        Restart-safe: an existing live job recorded in .qjob_<tag> is
        adopted instead of resubmitted.
        """
        cwd = Path(cwd)
        sz = size_for(natoms, kind)
        ncpus, walltime = sz["ncpus"], sz["walltime"]

        as_array = (self.use_arrays and work_dirs is not None
                    and len(work_dirs) >= 2 and cmd
                    and cmd[0] == "pollmach")
        if cmd and cmd[0] == "pollmach":
            # inside a dedicated job there is no shared machine to poll;
            # run the underlying command directly per work dir
            cmd = list(cmd[1:])

        if as_array:
            script = self.render_array(tag, cwd, cmd, work_dirs,
                                       ncpus, walltime)
        elif work_dirs is not None and len(work_dirs) > 1:
            script = self.render_loop(tag, cwd, cmd, work_dirs,
                                      done_file, ncpus, walltime)
        else:
            run_dir = Path(work_dirs[0]) if work_dirs else cwd
            script = self.render_single(tag, run_dir, cmd, ncpus, walltime)

        marker = cwd / f".qjob_{tag}"
        attempt = 0
        job_id = None
        if marker.is_file():                   # orchestrator restart
            try:
                rec = json.loads(marker.read_text())
                if self.alive(rec.get("job_id", "")):
                    job_id = rec["job_id"]
                    attempt = rec.get("attempt", 0)
            except (ValueError, OSError):
                pass

        while True:
            if done_when(cwd):
                return 0
            if job_id is None:
                if attempt > self.max_retries:
                    return -1
                self._throttle()
                job_id = self.qsub(script)
                marker.write_text(json.dumps(
                    {"job_id": job_id, "attempt": attempt,
                     "script": str(script)}))
            if self.dry_run:
                return 0 if done_when(cwd) else -1
            time.sleep(self.poll_interval)
            if done_when(cwd):
                return 0
            if not self.alive(job_id):
                # queue is done with it but outputs are absent -> retry
                job_id = None
                attempt += 1
