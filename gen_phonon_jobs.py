#!/usr/bin/env python3
"""
Generate PBS jobs for the fitfc force calculations (ATAT phonon workflow step ii).

    cd /nobackup/pjalagam/CoCrNi_fullpipeline/DLM/BCC_A2_small
    python3 gen_phonon_jobs.py . --dry-run          # count + write, submit nothing
    python3 gen_phonon_jobs.py . --submit

The workflow step this replaces is:
    foreachfile -d 3 wait \; runstruct_vasp -lu -w vaspf.wrap \; rm wait

Each perturbation directory carries a `wait` marker written by fitfc. This
script finds them under every endmember (directories holding an `endmem`
file), and writes PBS jobs that run `runstruct_vasp -lu -w vaspf.wrap` in each
and remove the marker on success.

GROUPING (this is the important choice)
---------------------------------------
  --group endmember  (default)  one job per endmember, perturbations run in
                     sequence inside it. A 64-atom static point is minutes, so
                     48 of them fit comfortably in one long job. Fewest jobs,
                     kindest to the queue.
  --group chunk --chunk-size N  N perturbations per job. Use this to trade
                     queue slots for wall time.
  --group perturbation  one job per perturbation. Maximum parallelism, but a
                     full CoCrNi set is 400+ jobs -- check your queue limits
                     before choosing this.

Standard library only.
"""

import argparse, os, re, subprocess, sys

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
#  fitfc force calculations -- {ndirs} perturbation(s)
#  endmember: {endmem}
#  Equivalent to: foreachfile -d 3 wait \; runstruct_vasp -lu -w vaspf.wrap
# ============================================================================
set -uo pipefail

module purge
module load comp-intel/2023.2.1
module use -a /nasa/modulefiles/testing
module load mpi-intel/2021.16
export PATH="{atat_bin}:{vasp_bin}:$PATH"
export OMP_NUM_THREADS=1
ulimit -c 0

WRAP="{wrap}"
[ -s "$WRAP" ] || {{ echo "FATAL: no wrap file at $WRAP"; exit 1; }}

DIRS=(
{dirlist}
)

# ============================================================================
#  Time-aware loop. A 160-atom spin-polarised static point with AMIX=0.02 is
#  ~1-2 h on {ncpus} cores, and that estimate is soft -- so rather than trust it,
#  measure the first calculation and stop starting new ones when there is not
#  enough walltime left for another. Unfinished directories keep their `wait`
#  marker, so resubmitting this identical file continues where it stopped.
# ============================================================================
WALL_SECONDS={wall_seconds}
RESERVE=1800                    # leave this much slack before the hard limit
START=$SECONDS

ok=0; fail=0; skip=0; left=0
for d in "${{DIRS[@]}}"; do
    elapsed=$(( SECONDS - START ))
    remain=$(( WALL_SECONDS - RESERVE - elapsed ))
    if [ $ok -gt 0 ]; then
        avg=$(( elapsed / ok ))
        need=$(( avg + avg / 5 ))          # average + 20% margin
        if [ $remain -lt $need ]; then
            echo ">> stopping early: ${{remain}}s left, need ~${{need}}s for one more"
            left=$(( left + 1 )); continue
        fi
    elif [ $remain -lt 3600 ]; then
        echo ">> stopping early: under an hour left and no timing sample yet"
        left=$(( left + 1 )); continue
    fi

    cd "$d" || {{ echo "SKIP (no dir): $d"; skip=$((skip+1)); continue; }}
    if [ -s energy ]; then
        echo "SKIP (done): $d"; skip=$((skip+1)); continue
    fi
    echo "=== [$(( ok + fail + 1 ))/${{#DIRS[@]}}] $d   (${{elapsed}}s elapsed)"
    t0=$SECONDS
    runstruct_vasp -lu -w "$WRAP" mpiexec -n {ncpus} > runstruct.out 2>&1
    rc=$?
    dt=$(( SECONDS - t0 ))
    if [ $rc -eq 0 ] && [ -s energy ] && [ -s force.out ]; then
        rm -f wait
        echo "    ok   ${{dt}}s   energy=$(cat energy)"
        ok=$((ok+1))
    else
        echo "    FAIL rc=$rc after ${{dt}}s  (wait kept; see $d/runstruct.out)"
        fail=$((fail+1))
    fi
done

echo
echo ">> $ok done, $fail failed, $skip already complete, $left not attempted"
if [ $left -gt 0 ]; then
    echo ">> RESUBMIT THIS SAME FILE to continue: qsub $(basename "$0")"
    exit 3
fi
[ $fail -eq 0 ] || exit 2
"""


def find_endmembers(root):
    out = []
    for dp, dn, fn in os.walk(root):
        if "endmem" in fn:
            out.append(dp)
            dn[:] = [d for d in dn if d.startswith("vol")]
    return sorted(out)


def find_perturbations(endmem):
    """vol_*/p*/ directories still flagged with `wait`."""
    pert = []
    for vol in sorted(os.listdir(endmem)):
        vp = os.path.join(endmem, vol)
        if not (vol.startswith("vol") and os.path.isdir(vp)):
            continue
        for p in sorted(os.listdir(vp)):
            pp = os.path.join(vp, p)
            if not os.path.isdir(pp) or not p.startswith("p"):
                continue
            e = os.path.join(pp, "energy")
            if os.path.exists(e) and os.path.getsize(e) > 0:
                continue                      # already computed
            pert.append(pp)
    return pert


def short(endmem, i):
    tag = os.path.basename(endmem)
    el = re.search(r"[ac]_([A-Z][a-z]?)=1", tag)
    lat = "".join(c for c in endmem.split(os.sep) if c[:3].upper()
                  in ("FCC", "BCC", "HCP", "SIG"))[:3].lower()
    return f"ph{lat}{el.group(1) if el else 'x'}{i:02d}"[:14]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--group", choices=["endmember", "chunk", "perturbation"],
                    default="chunk")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="perturbations per job; 0 = fit them to the walltime")
    ap.add_argument("--ncpus", type=int, default=128)
    ap.add_argument("--model", default="mil_ait")
    ap.add_argument("--group-list", default="a1485")
    ap.add_argument("--queue", default="long")
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--minutes-per-pert", type=float, default=90.0,
                    help="per-perturbation estimate used to size chunks and "
                         "walltime. 90 min suits a ~160-atom spin-polarised "
                         "static point on 128 cores; measure yours and adjust")
    ap.add_argument("--atat-bin", default="/home7/pjalagam/bin")
    ap.add_argument("--vasp-bin", default="/home1/zwu6/vasp/6.6.1/bin_PFE")
    ap.add_argument("--wrap", default="",
                    help="absolute path to vaspf.wrap (default: the one in "
                         "each endmember directory)")
    args = ap.parse_args()

    ends = find_endmembers(args.root)
    if not ends:
        sys.exit(f"no endmember directories (no `endmem` file) under {args.root}")

    total, jobs = 0, []
    for em in ends:
        pert = find_perturbations(em)
        total += len(pert)
        if not pert:
            print(f"  {os.path.relpath(em, args.root)}: nothing outstanding")
            continue
        if args.group == "endmember":
            batches = [pert]
        elif args.group == "chunk":
            hh, mm, _ss = (int(x) for x in args.walltime.split(":"))
            usable = hh * 60 + mm - 30                 # minus the reserve
            n = max(1, int(usable // args.minutes_per_pert))
            n = args.chunk_size or n
            batches = [pert[i:i + n] for i in range(0, len(pert), n)]
        else:
            batches = [[p] for p in pert]
        for k, b in enumerate(batches, 1):
            jobs.append((em, k, b))
        print(f"  {os.path.relpath(em, args.root)}: {len(pert)} perturbations "
              f"-> {len(batches)} job(s)")

    if not jobs:
        print("\nnothing to do.")
        return
    print(f"\n{total} outstanding perturbations across {len(ends)} endmembers"
          f"  ->  {len(jobs)} PBS job(s)")

    for em, k, batch in jobs:
        wrap = args.wrap or os.path.abspath(os.path.join(em, "vaspf.wrap"))
        wall = args.walltime
        name = short(em, k)
        path = os.path.join(em, f"phonon_{k:02d}.pbs")
        with open(path, "w") as fh:
            hh, mm, ss = (int(x) for x in wall.split(":"))
            fh.write(PBS.format(
                wall_seconds=hh * 3600 + mm * 60 + ss,
                name=name, queue=args.queue, ncpus=args.ncpus,
                model=args.model, walltime=wall, group=args.group_list,
                logname=f"phonon_{k:02d}.log", ndirs=len(batch),
                endmem=os.path.abspath(em), atat_bin=args.atat_bin,
                vasp_bin=args.vasp_bin, wrap=wrap,
                dirlist="\n".join(f'  "{os.path.abspath(d)}"' for d in batch)))
        os.chmod(path, 0o755)
        print(f"  [{name}] {len(batch):3d} pert  {wall}  "
              f"{os.path.relpath(path, args.root)}")
        if args.submit:
            r = subprocess.run(["qsub", os.path.basename(path)], cwd=em,
                               capture_output=True, text=True)
            print("        " + (r.stdout.strip() if r.returncode == 0
                                else "QSUB FAILED: " + r.stderr.strip()))

    if not args.submit:
        print("\nWritten but NOT submitted (add --submit).")


if __name__ == "__main__":
    main()
