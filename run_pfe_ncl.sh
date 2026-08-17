#!/bin/bash
#PBS -S /bin/bash
#PBS -N ncl_dlm
#PBS -q long
#PBS -l select=1:ncpus=128:mpiprocs=128:model=mil_ait
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -W group_list=a1485

# ============================================================================
#  Constrained NONCOLLINEAR DLM run in an ATAT/sqs2tdb structure directory.
#  RESTART-SAFE: resubmit the same file after a timeout and it continues.
#
#  Protocol (v2, after the mode-1 post-mortem):
#    * I_CONSTRAINED_M = 2  -- full vector. Mode 1 constrains only the AXIS
#      (its penalty is invariant under m -> -m and ignores magnitude), so
#      under ISIF=3 the moments either died or flipped: cell contracts toward
#      the NM volume and the "relaxed PM" structure is really a relaxed NM
#      one. Mode 2 holds the moments so the relaxation stays on the PM surface.
#    * ISIF=3 / NSW as set by vasp.wrap -- the structure MUST relax for the
#      thermodynamics; only the magnetic constraint changed.
#    * magnitudes + starting geometry inherited from the converged COLLINEAR
#      DLM run at the same composition (found automatically by swapping
#      /NCL/ -> /DLM/ in the path; override with COLLINEAR_DIR=... qsub -v).
#
#  Flow:  runstruct_vasp -nr -> ncl_dlm_setup.py setup -> vasp_ncl
#         -> runstruct_vasp -ex -> ncl_dlm_setup.py fixenergy
#
#  Submit from inside each sqs directory:  qsub ~/path/to/run_pfe_ncl.sh
#  Pilot first, check with check_ncl_runs.py, THEN commit the queue.
# ============================================================================

set -uo pipefail
cd "$PBS_O_WORKDIR"

WALL_SECONDS=172800      # match #PBS -l walltime
STOP_BUFFER=900          # write STOPCAR this many s before the end
LAMBDA_VAL="${LAMBDA_VAL:-5}"          # ramp: 5 -> 10 -> 20 across restarts

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

# --- locate the matching collinear DLM run (magnitudes + start geometry) ---
COLLINEAR_DIR="${COLLINEAR_DIR:-${PWD/\/NCL\//\/DLM\/}}"
SEED_ARGS=""
if [ -d "$COLLINEAR_DIR" ] && [ -s "$COLLINEAR_DIR/OUTCAR" ]; then
    SEED_ARGS="--from-collinear $COLLINEAR_DIR"
    [ -s "$COLLINEAR_DIR/CONTCAR" ] && \
        SEED_ARGS="$SEED_ARGS --seed-geometry $COLLINEAR_DIR"
    echo ">> inheriting moments/geometry from $COLLINEAR_DIR"
else
    echo ">> WARNING: no collinear DLM run at $COLLINEAR_DIR;"
    echo ">> falling back to the vasp.wrap MAGMOM magnitudes."
fi

# --- 1+2. generate + edit inputs, ONLY on a fresh directory ---------------
if [ -f INCAR ] && grep -qi "LNONCOLLINEAR" INCAR; then
    echo ">> restart detected: noncollinear INCAR present, skipping -nr/setup"
    [ -s WAVECAR ] && echo ">> WAVECAR found: continuing previous SCF" \
                   || echo ">> no WAVECAR: SCF restarts from scratch (same orientations)"
else
    runstruct_vasp -nr || exit 1
    python3 "$NCL" setup --constraint 2 --lambda "$LAMBDA_VAL" $SEED_ARGS \
        || { echo "ncl setup failed"; exit 1; }
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
    echo ">> NOW CHECK: E_p -> 0, moments held, volume vs the collinear run:"
    echo ">>   python3 ~/tdb_automate_atat/check_ncl_runs.py --dir \$PWD -v"
else
    echo ">> NOT converged (timeout/NELM) -- WAVECAR kept, resubmit this same"
    echo ">> jobfile to continue. \`wait\` left in place."
    exit 2
fi
