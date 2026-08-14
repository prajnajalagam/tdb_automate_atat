#!/bin/bash
#PBS -S /bin/bash
#PBS -N ncl_dlm
#PBS -q debug
#PBS -l select=1:ncpus=128:mpiprocs=128:model=mil_ait
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -W group_list=a1485

# ============================================================================
#  Constrained NONCOLLINEAR DLM run in an ATAT/sqs2tdb structure directory.
#  RESTART-SAFE: resubmit the same file after a timeout and it continues.
#
#  Fresh dir : runstruct_vasp -nr -> ncl_dlm_setup.py setup -> vasp_ncl
#  Restart   : detects the noncollinear INCAR, skips generation, and
#              vasp_ncl continues from WAVECAR automatically (ISTART default;
#              MAGMOM is ignored once a WAVECAR is read -- the magnetic
#              state continues from the wavefunction).
#  Watchdog  : writes STOPCAR (LABORT) STOP_BUFFER seconds before walltime
#              so VASP exits cleanly and WRITES WAVECAR/CHGCAR -- a hard
#              PBS kill would leave nothing to restart from.
#  Extraction: `energy` (+ penalty correction) only happens once the SCF
#              actually converged; otherwise the job exits leaving `wait`
#              in place so the dir still shows as unfinished.
#
#  Submit from inside each sqs directory:  qsub ~/path/to/run_pfe_ncl.sh
#  !! Keep WALL_SECONDS in sync with the #PBS walltime above.
# ============================================================================

set -uo pipefail
cd "$PBS_O_WORKDIR"

WALL_SECONDS=1800        # match #PBS -l walltime
STOP_BUFFER=300          # write STOPCAR this many s before the end

# --- modules / environment ---
module purge
module load python3
module load gcc
module load comp-intel/2023.2.1
module use -a /nasa/modulefiles/testing
module load mpi-intel/2021.16
source "$HOME/venvs/biniter/bin/activate"          # numpy for the editor script
export PATH="/home7/pjalagam/bin:/home1/zwu6/vasp/6.6.1/bin_PFE:$PATH"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ulimit -c 0

NCL="$HOME/tdb_automate_atat/ncl_dlm_setup.py"

# --- 1+2. generate + edit inputs, ONLY on a fresh directory ---------------
if [ -f INCAR ] && grep -qi "LNONCOLLINEAR" INCAR; then
    echo ">> restart detected: noncollinear INCAR present, skipping -nr/setup"
    [ -s WAVECAR ] && echo ">> WAVECAR found: continuing previous SCF" \
                   || echo ">> no WAVECAR: SCF restarts from scratch (same orientations)"
else
    runstruct_vasp -nr || exit 1
    python3 "$NCL" setup --lambda 10 || { echo "ncl setup failed"; exit 1; }
fi

# --- 3. run vasp_ncl with a clean-exit watchdog ----------------------------
rm -f STOPCAR
( sleep $(( WALL_SECONDS - STOP_BUFFER )) && \
  echo "LABORT = .TRUE." > STOPCAR && \
  echo ">> watchdog: STOPCAR written" ) &
WATCHDOG=$!

mpiexec -n "${NCPUS:-128}" vasp_ncl > vasp.out 2>&1

kill "$WATCHDOG" 2>/dev/null || true
rm -f STOPCAR

# --- 4. extract + penalty-correct ONLY if the SCF converged ----------------
if grep -q "aborting loop because EDIFF is reached" OUTCAR; then
    runstruct_vasp -ex || exit 1
    python3 "$NCL" fixenergy
    rm -f wait
    echo ">> converged: energy extracted and penalty-corrected"
else
    echo ">> NOT converged (timeout/NELM) -- WAVECAR kept, resubmit this same"
    echo ">> jobfile to continue. \`wait\` left in place."
    exit 2
fi
