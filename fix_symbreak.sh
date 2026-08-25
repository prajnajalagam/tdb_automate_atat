#!/bin/bash
# ============================================================================
#  Undo the bad symmetry-breaking restart and redo it from the RELAXED
#  geometry.  Run from /nobackup/pjalagam/CoCrNi_fullpipeline/DLM
#
#  WHY: the first symbreak jobs ran `cellcvrt ... < str.out`, jittering the
#  IDEAL lattice. That discards the previous relaxation -- grad_norm went from
#  ~17 to ~4500 and the energy rose by ~1 eV/atom. This restores the parked
#  path endpoints and rebuilds str_hint.out from str_relax.out instead.
#
#  SCOPE: only the runs that visibly regressed. Runs that simply have not run
#  long enough yet are NOT touched -- they are progressing normally from the
#  same restart and there is nothing to undo. Their trajectories are just short.
#
#  Kill the affected jobs FIRST, then run this, then resubmit.
# ============================================================================
set -uo pipefail
JA=${JA:-0.01}; JC=${JC:-0.01}

# Regressed after the symbreak restart -- these are the only ones to repair.
DIRS=(
"BCC_A2_small/sqs_lev=5_a_Co=0.375,a_Cr=0.625"   # grad_norm  17 -> 4511
"BCC_A2_small/sqs_lev=5_a_Co=0.625,a_Cr=0.375"   # grad_norm  73 -> 4035
"FCC_A1_small/sqs_lev=5_a_Co=0.125,a_Cr=0.875"   # grad_norm  23 ->   79
)

# Deliberately EXCLUDED -- short trajectories, not failures:
#   BCC_A2_small/sqs_lev=2_a_Co=0.75,a_Cr=0.25
#   FCC_A1_small/sqs_lev=0_a_Cr=1
#   FCC_A1_small/sqs_lev=5_a_Cr=0.625,a_Ni=0.375
#   HCP_A3_small/sqs_lev=2_c_Co=0.25,c_Cr=0.75
# And handled separately via --no-id (24 iterations, genuinely not converging):
#   HCP_A3_small/sqs_lev=2_c_Cr=0.75,c_Ni=0.25

for d in "${DIRS[@]}"; do
    echo "=== $d"
    [ -d "$d" ] || { echo "    MISSING - skipped"; continue; }
    ( cd "$d" || exit 1
      if [ -f 01/busy ]; then
          echo "    01/busy present -- job may still be live. qdel it first."
          echo "    SKIPPED (nothing changed)"
          exit 0
      fi
      for f in str_beg.out str_end.out; do
          [ -f "$f.bak" ] && mv -f "$f.bak" "$f" && echo "    restored $f"
      done
      rm -f str_hint.out
      BASE=str.out
      [ -s str_relax.out ] && BASE=str_relax.out
      if cellcvrt -ja=$JA -jc=$JC < "$BASE" > str_hint.out 2>/dev/null \
         && [ -s str_hint.out ]; then
          echo "    str_hint.out rebuilt from $BASE"
      else
          rm -f str_hint.out
          echo "    cellcvrt FAILED -- no hint file written"
      fi
      rm -f 01/busy busy running
    )
done

cat <<'MSG'

Repaired the 3 regressed runs only. Resubmit just those:

  for d in "BCC_A2_small/sqs_lev=5_a_Co=0.375,a_Cr=0.625" \
           "BCC_A2_small/sqs_lev=5_a_Co=0.625,a_Cr=0.375" \
           "FCC_A1_small/sqs_lev=5_a_Co=0.125,a_Cr=0.875"; do
      (cd "$d" && qsub relaunch_id.pbs)
  done

Leave the four short-trajectory runs alone -- they are still going.
MSG
