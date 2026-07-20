#!/bin/bash
# ============================================================================
#  PBS fan-out orchestrator launcher (--submit pbs). Run on a FRONT END
#  (pfe), NOT inside a job: the orchestrator only does bookkeeping,
#  qsub/qstat, and light fitfc/sqs2tdb glue — every VASP execution is
#  submitted as its own right-sized PBS job:
#    * 2 convergence-probe jobs (parallel, one per element-rich side)
#    * 1 relaxation job per SQS (all SQS concurrently, capped)
#    * 1 PBS job ARRAY per SQS phonon stage (one element per
#      perturbation dir — all force runs wall-parallel)
#  Restart-safe: kill it and rerun; it reconciles against the work tree
#  and adopts still-running jobs instead of resubmitting.
#
#  Usage:  bash submit_orchestrator_template.sh
#  Watch:  tail -f <WORK_ROOT>/upstream_live.log ; qstat -u $USER
# ============================================================================

set -euo pipefail

# ───────────────────────── USER CONFIG (edit me) ────────────────────────────
ELEMENT_A="Co"
ELEMENT_B="Cr"
WORK_ROOT="/nobackupp27/pjalagam/${ELEMENT_A}${ELEMENT_B}_upstream"
POTCARS="/home1/zwu6/vasp/POTPAW_PBE.64/${ELEMENT_A}/POTCAR,/home1/zwu6/vasp/POTPAW_PBE.64/${ELEMENT_B}/POTCAR"
PHASES="FCC_A1,BCC_A2,HCP_A3,SIGMA_D8B"
SQS_LEVEL="2"

JOB_MODEL="mil_ait"              # node model for submitted jobs
JOB_QUEUE="normal"               # devel's job limit unfits it for fan-out
JOB_MAX_INFLIGHT="16"            # compute-cost throttle
JOB_RETRIES="1"

VENV_ACTIVATE="$HOME/venvs/biniter/bin/activate"
ATAT_BIN="/home7/pjalagam/bin"
VASP_BIN="/home1/zwu6/vasp/6.6.0/bin_PFE"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# ───────────────────────── END USER CONFIG ──────────────────────────────────

mkdir -p "${WORK_ROOT}"

# Every submitted job sources this — it must reproduce the interactive
# environment exactly (the 2026-07 merge that silently dropped module
# loads is why this is generated, not hand-copied).
JOB_ENV="${WORK_ROOT}/job_env.sh"
cat > "${JOB_ENV}" <<EOF
module purge
module load python3
module load gcc
module load comp-intel/2023.2.1
module use -a /nasa/modulefiles/testing
module load mpi-intel/2021.16
source ${VENV_ACTIVATE}
export PATH="${ATAT_BIN}:${VASP_BIN}:\$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
EOF

# Front-end environment for the orchestrator itself (fitfc/sqs2tdb glue).
source "${JOB_ENV}"

nohup python3 -u "${SCRIPT_DIR}/run_upstream.py" \
    --element1 "${ELEMENT_A}" --element2 "${ELEMENT_B}" \
    --work-root "${WORK_ROOT}" \
    --potcars "${POTCARS}" \
    --phases "${PHASES}" \
    --sqs-level "${SQS_LEVEL}" \
    --env-bin "${ATAT_BIN}" \
    --cmd-prefix "mpiexec -n 32" \
    --submit pbs \
    --job-env "${JOB_ENV}" \
    --job-model "${JOB_MODEL}" \
    --job-queue "${JOB_QUEUE}" \
    --job-max-inflight "${JOB_MAX_INFLIGHT}" \
    --job-retries "${JOB_RETRIES}" \
    > "${WORK_ROOT}/orchestrator.log" 2>&1 &

echo "orchestrator PID $! — logs:"
echo "  tail -f ${WORK_ROOT}/upstream_live.log"
echo "  qstat -u \$USER          # the fan-out"
echo "(--cmd-prefix rank counts are RETARGETED per job to each job's"
echo " ncpus by the broker, so the value above is only a placeholder.)"
