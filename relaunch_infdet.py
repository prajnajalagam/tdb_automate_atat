#!/usr/bin/env python3
"""
Diagnose stalled inflection-detection (robustrelax_vasp -id) runs and write one
PBS job per run, with the treatment chosen from that run's own trajectory.

    cd /nobackup/pjalagam/CoCrNi_fullpipeline/DLM
    python3 relaunch_infdet.py . --dry-run     # classify + write PBS, submit nothing
    python3 relaunch_infdet.py . --submit      # write and qsub

Standard library only.

HOW EACH RUN IS TREATED
-----------------------
Read from `01/infdet.log`: mincurv (softest-mode curvature -- the quantity the
algorithm drives to zero), energy, and grad_norm (outer convergence).

  CROSSED_*      curvature crossed zero and grad_norm is falling. The geometry
                 is on the right track, it simply ran out of walltime.
                 -> `robustrelax_vasp -cip` continues the prior run.

  STUCK_NEGATIVE curvature plateaus below zero and grad_norm plateaus at 10-70.
                 The epicycle cannot flatten the soft mode because the cell is
                 too symmetric to distort the way it needs to.
                 -> break symmetry with `cellcvrt -ja -jc` into str_hint.out,
                    delete str_beg.out/str_end.out (per the manual: "If you want
                    to restart from scratch, delete these files"), restart.

  POSITIVE_ONLY  curvature never went negative -- the structure looks
  BARELY_STARTED mechanically stable, or the run barely began.
                 -> restart with `-c 0.05` so inflection detection only fires
                    when the relaxation is actually large. With the default
                    -c 0 it fires for every structure, stable or not.

WHY NOT symbrklib
-----------------
symbrklib takes a SINGLE element and emits a plain bcc/fcc/hcp cell, so it
would discard the SQS decoration -- including the +/- DLM spin assignment that
makes these cells paramagnetic. Even for the pure-Cr cells here (which are
still Cr+/Cr- supercells), the symmetry-breaking tool that PRESERVES the
structure is cellcvrt's jitter. symbrklib is for plain single-element runs.
"""

import argparse, os, re, subprocess, sys

CURV_RE = re.compile(r"mincurv=\s*([-0-9.eE]+)\s+energy=\s*([-0-9.eE]+)"
                     r"\s+grad_norm=\s*([-0-9.eE]+)")

ACTIONS = {                       # class -> (action, walltime, queue)
    "CROSSED_NEAR_CONV": ("continue", "120:00:00", "long"),
    "CROSSED_SLOW":      ("continue", "120:00:00", "long"),
    "STUCK_NEGATIVE":    ("symbreak", "120:00:00", "long"),
    "POSITIVE_ONLY":     ("gate",     "120:00:00", "long"),
    "BARELY_STARTED":    ("gate",     "120:00:00", "long"),
}

PBS = """#!/bin/bash
#PBS -S /bin/bash
#PBS -N {name}
#PBS -q {queue}
#PBS -l select=1:ncpus={ncpus}:mpiprocs={ncpus}:model={model}
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {logname}
#PBS -W group_list={group}

# ============================================================================
#  Inflection-detection relaunch -- {cls}
#  {why}
#  last state: mincurv {c0:+.3f} -> {cN:+.3f}   grad_norm {gN:.2f}   {n} iterations
# ============================================================================
set -uo pipefail
cd "{rundir}" || exit 1

module purge
module load comp-intel/2023.2.1
module use -a /nasa/modulefiles/testing
module load mpi-intel/2021.16
export PATH="{atat_bin}:{vasp_bin}:$PATH"
export OMP_NUM_THREADS=1
ulimit -c 0

{prep}
robustrelax_vasp {flags} mpiexec -n {ncpus} > out.log 2>&1
RC=$?

echo ">> robustrelax_vasp exit $RC"
if [ -s energy ]; then
    echo ">> energy: $(cat energy)"
else
    echo ">> no energy file yet -- inspect 01/infdet.log:"
    echo ">>   grep curv 01/infdet.log | tail -20"
fi
exit $RC
"""

SEMA = """# --- clear semaphores left by the killed job -------------------------------
# infdet writes `busy` to tell the external program "run VASP now" and waits
# for it to be deleted. A job killed at walltime leaves it behind; if it is
# still there on restart the loop can sit waiting for a step that already
# died. `wait`/`running` are ATAT's queueing markers and are equally stale.
# Nothing here holds results -- only these zero-byte flags are removed.
for f in 01/busy busy running; do
    [ -f "$f" ] && rm -f "$f" && echo ">> cleared stale semaphore $f"
done

"""

PREP = {
    "continue": "# continue the previous ID run in place; nothing else is touched\n",
    "gate":     "# -c 0.05 gates ID on the relaxation magnitude (default 0 = always on)\n",
    "symbreak": """# break symmetry so the soft mode has a direction to relax along.
# cellcvrt jitters atoms and cell while PRESERVING the SQS/DLM decoration.
if [ ! -s str_hint.out ]; then
    cellcvrt -ja={ja} -jc={jc} < str.out > str_hint.out || exit 1
    echo ">> wrote str_hint.out (jitter ja={ja} jc={jc})"
fi
# manual: "str_beg.out and str_end.out contain the extremities of the path.
#          If you want to restart from scratch, delete these files."
for f in str_beg.out str_end.out; do
    [ -f "$f" ] && mv -f "$f" "$f.bak" && echo ">> parked $f -> $f.bak"
done
""",
}

FLAGS = {
    "continue": "-cip -id -c {c}",
    "gate":     "-id -c {c}",
    "symbreak": "-id -c {c} -ja {ja} -jc {jc}",
}


def parse_log(path):
    pts = []
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = CURV_RE.search(line)
                if m:
                    pts.append(tuple(float(x) for x in m.groups()))
    except OSError:
        return []
    return pts


def classify(pts):
    c = [p[0] for p in pts]; g = [p[2] for p in pts]
    n = len(c)
    if n == 0:
        return None
    crossed = min(c) < 0 < max(c)
    gt = g[max(0, n - 8):]
    mean_g = sum(gt) / len(gt)
    if n <= 4 and (abs(c[-1]) > 1 or g[-1] > 50):
        return "BARELY_STARTED"
    if crossed and mean_g < 6:
        return "CROSSED_NEAR_CONV"
    if crossed:
        return "CROSSED_SLOW"
    if max(c) < 0:
        return "STUCK_NEGATIVE"
    return "POSITIVE_ONLY"


def finished(d):
    """Inflection detection completed.

    robustrelax step 7: "Report the energy found in 6 in the file energy and
    the corresponding geometry in str_relax.out" -- so a populated `energy`
    IS the completion marker for the whole -id workflow. infdet's own
    `01/cenergy.out` / `01/cstr_relax.out` are accepted as well, for a run
    that finished the inner stage but died before robustrelax wrote out.
    """
    e = os.path.join(d, "energy")
    if os.path.exists(e) and os.path.getsize(e) > 0:
        return True
    for f in ("cenergy.out", "cstr_relax.out"):
        p = os.path.join(d, "01", f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return True
    return False


def newest_mtime(d):
    """Most recent modification anywhere in the run dir or its 01/ stage."""
    newest = 0.0
    for base in (d, os.path.join(d, "01")):
        try:
            for f in os.listdir(base):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(base, f)))
                except OSError:
                    pass
        except OSError:
            pass
    return newest


def running(d, stale_minutes):
    """Is a job actually alive here?

    `busy` is NOT the test -- infdet writes it before handing off to VASP, and
    a job killed at walltime leaves it behind, which is the single most common
    state among the runs we want to relaunch. A live run touches OSZICAR /
    vasp.out constantly, so recent file activity is the honest signal.
    """
    import time
    age_min = (time.time() - newest_mtime(d)) / 60.0
    return age_min < stale_minutes


def short_name(d, i):
    tag = os.path.basename(d.rstrip("/"))
    lat = "".join(p[:3] for p in d.split(os.sep) if p[:3].upper()
                  in ("FCC", "BCC", "HCP", "SIG"))[:3].lower() or "id"
    lev = re.search(r"lev=(\d+)", tag)
    return f"id{i:02d}{lat}{lev.group(1) if lev else ''}"[:14]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--submit", action="store_true", help="qsub the generated files")
    ap.add_argument("--dry-run", action="store_true", help="write PBS only (default)")
    ap.add_argument("--ncpus", type=int, default=128)
    ap.add_argument("--model", default="mil_ait")
    ap.add_argument("--group", default="a1485")
    ap.add_argument("--atat-bin", default="/home7/pjalagam/bin")
    ap.add_argument("--vasp-bin", default="/home1/zwu6/vasp/6.6.1/bin_PFE")
    ap.add_argument("-c", "--cutoff", default="0.05",
                    help="robustrelax -c: relaxation magnitude that activates ID "
                         "(manual recommends 0.05; default 0 fires it always)")
    ap.add_argument("--ja", default="0.01", help="atom jitter for symmetry breaking")
    ap.add_argument("--jc", default="0.01", help="cell jitter for symmetry breaking")
    ap.add_argument("--only", default="", help="comma-separated classes to act on")
    ap.add_argument("--include-lev4", action="store_true",
                    help="include sqs_lev=4 cells (excluded by default -- their "
                         "random spin configurations are not trusted)")
    ap.add_argument("--stale-minutes", type=float, default=60.0,
                    help="a run whose newest file is older than this is "
                         "treated as dead and eligible for relaunch "
                         "(default: 60)")
    ap.add_argument("--force", action="store_true",
                    help="relaunch even if files were touched recently")
    ap.add_argument("--jobfile", default="relaunch_id.pbs")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    found, plan, skipped = 0, [], []
    for dp, dn, fn in os.walk(args.root):
        log = os.path.join(dp, "01", "infdet.log")
        if not os.path.exists(log):
            continue
        dn[:] = []
        found += 1
        if not args.include_lev4 and re.search(r"lev=4", dp):
            skipped.append((dp, "lev=4 excluded")); continue
        if finished(dp):
            skipped.append((dp, "ID complete (energy written)")); continue
        if running(dp, args.stale_minutes) and not args.force:
            import time
            age = (time.time() - newest_mtime(dp)) / 60.0
            skipped.append((dp, f"active {age:.0f} min ago - may be live"))
            continue
        pts = parse_log(log)
        if not pts:
            skipped.append((dp, "no mincurv lines in infdet.log")); continue
        cls = classify(pts)
        if only and cls not in only:
            skipped.append((dp, f"{cls} not selected")); continue
        plan.append((dp, cls, pts))

    print(f"scanned {found} directories with 01/infdet.log")
    print(f"  to relaunch: {len(plan)}   skipped: {len(skipped)}\n")
    for d, why in skipped[:10]:
        print(f"    skip  {why:<28s} {os.path.relpath(d, args.root)}")
    if len(skipped) > 10:
        print(f"    ... and {len(skipped)-10} more")

    counts = {}
    for i, (d, cls, pts) in enumerate(sorted(plan), 1):
        act, wall, queue = ACTIONS[cls]
        counts[act] = counts.get(act, 0) + 1
        c = [p[0] for p in pts]; g = [p[2] for p in pts]
        why = {"continue": "on track, ran out of walltime -- continue with -cip",
               "symbreak": "soft mode never flattens -- break symmetry and restart",
               "gate":     "may not need ID at all -- gate on -c and restart"}[act]
        body = PBS.format(
            name=short_name(d, i), queue=queue, ncpus=args.ncpus,
            model=args.model, walltime=wall, group=args.group,
            logname=f"relaunch_id_{i:02d}.log", rundir=os.path.abspath(d),
            atat_bin=args.atat_bin, vasp_bin=args.vasp_bin, cls=cls, why=why,
            c0=c[0], cN=c[-1], gN=g[-1], n=len(c),
            prep=SEMA + PREP[act].format(ja=args.ja, jc=args.jc),
            flags=FLAGS[act].format(c=args.cutoff, ja=args.ja, jc=args.jc))
        path = os.path.join(d, args.jobfile)
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
        rel = os.path.relpath(d, args.root)
        marks = []
        if os.path.exists(os.path.join(d, "01", "busy")):
            marks.append("stale busy")
        if os.path.exists(os.path.join(d, "wait")):
            marks.append("stale wait")
        flag = ("  [" + ", ".join(marks) + "]") if marks else ""
        print(f"  [{i:02d}] {cls:<18s} {act:<9s} {queue:<7s} {wall}  {rel}{flag}")
        if args.submit:
            r = subprocess.run(["qsub", args.jobfile], cwd=d,
                               capture_output=True, text=True)
            print(f"        {'-> ' + r.stdout.strip() if r.returncode == 0 else 'QSUB FAILED: ' + r.stderr.strip()}")

    print(f"\nactions: {counts}")
    if not args.submit:
        print("\nPBS files written but NOT submitted (add --submit).")
        print("Inspect one with:  cat <rundir>/" + args.jobfile)


if __name__ == "__main__":
    main()
