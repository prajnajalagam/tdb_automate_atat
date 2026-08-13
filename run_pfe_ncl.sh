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
#
#  Flow:  runstruct_vasp -nr            (inputs from the normal collinear
#                                        DLM vasp.wrap; keep it static:
#                                        NSW=0, no DOSTATIC, pre-relaxed geom)
#         ncl_dlm_setup.py setup        (INCAR -> noncollinear, MAGMOM 3N,
#                                        M_CONSTR, I_CONSTRAINED_M=1, LAMBDA)
#         mpiexec vasp_ncl              (explicit ncl binary; bypasses the
#                                        vasp_std default in ~/.ezvasp.rc)
#         runstruct_vasp -ex            (write ATAT `energy` file)
#         ncl_dlm_setup.py fixenergy    (energy -> energy - E_p, once)
#
#  Submit from inside each sqs directory:  qsub ~/path/to/run_pfe_ncl.sh
#  Resubmission note: `setup` refuses to re-edit an already-noncollinear
#  INCAR (protects the orientation assignment). For a clean redo, restore
#  INCAR.collinear and add --force, or delete INCAR and rerun from -nr
#  (POSCAR-hash seeding reproduces identical orientations).
# ============================================================================

set -uo pipefail
cd "$PBS_O_WORKDIR"

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

# 1. generate inputs from the normal collinear DLM vasp.wrap
runstruct_vasp -nr || exit 1

# 2. rewrite INCAR for constrained noncollinear
python3 "$NCL" setup --lambda 10 || { echo "ncl setup failed"; exit 1; }

# 3. run the NONCOLLINEAR binary explicitly
mpiexec -n "${NCPUS:-128}" vasp_ncl > vasp.out 2>&1

# 4. extract into ATAT's `energy` file, then subtract the penalty energy
runstruct_vasp -ex || exit 1
python3 "$NCL" fixenergy

rm -f wait
