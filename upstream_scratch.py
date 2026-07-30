#!/usr/bin/env python3
"""
upstream_scratch.py — clean-slate rebuild of the upstream binary
workflow, translated line by line from user pseudocode (2026-07-28).

Being built incrementally: each pseudocode block the user provides is
converted to Python here, in order, before any orchestration glue is
added around it.
"""

# ─────────────────────────── USER INPUTS ────────────────────────────
# Elements of the binary (order matters: EL[0] = element 1, EL[1] =
# element 2; composition axes below are fractions of EL[1]).
EL = ["Co", "Cr"]

# Starting compositional-fineness levels for sqs2tdb's -lv mesh:
#   START_LEVEL_SINGLE : single-sublattice phases (FCC_A1, BCC_A2,
#                        HCP_A3) — lev=2 -> endmembers + x=0.5 +
#                        x=0.25/0.75
#   START_LEVEL_MULTI  : multi-sublattice phases (SIGMA_D8B) — lev=0
#                        -> endmember corners only
START_LEVEL_SINGLE = 2
START_LEVEL_MULTI = 0

# Phases to calculate, in execution order.
PHASES = ["FCC_A1", "BCC_A2", "HCP_A3", "SIGMA_D8B"]
# ────────────────────────── END USER INPUTS ─────────────────────────


# ──────────────────────── ALGORITHM (pending) ───────────────────────
# The user's pseudocode is translated below this line, block by block.
# (awaiting first pseudocode block)


if __name__ == "__main__":
    pass
