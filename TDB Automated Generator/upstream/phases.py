#!/usr/bin/env python3
"""
Phase constants and DLM / SIGMA conventions for the upstream generator.

Kept deliberately self-contained (no import of the downstream
sqs2tdb_pipeline.py) so the upstream package can be unit-tested and run
on a node where only ATAT + VASP are installed.

Conventions match $ATATDIR/data/sqsdb/<PHASE>/rndstr.skel and the
existing downstream pipeline:

  FCC_A1 / BCC_A2  single sublattice, Wyckoff site 'a', mult 1
  HCP_A3           single sublattice, Wyckoff site 'c', mult 2
  SIGMA_D8B        three sublattices aj/g/ii with mult 10/4/16; in a
                   binary only endmembers are ever computed.

The *_small systems are the single-sublattice cells that sqs2tdb -cp
emits and that must be copied into a working directory before SQS
generation / randomspin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Phase geometry
# ---------------------------------------------------------------------------

# Wyckoff site label per single-sublattice phase.
PHASE_SITE: Dict[str, str] = {
    "FCC_A1": "a",
    "BCC_A2": "a",
    "HCP_A3": "c",
}

# Site multiplicities per phase.
PHASE_MULT: Dict[str, Dict[str, int]] = {
    "FCC_A1": {"a": 1},
    "BCC_A2": {"a": 1},
    "HCP_A3": {"c": 2},
    "SIGMA_D8B": {"aj": 10, "g": 4, "ii": 16},
}

# Single-sublattice phases (the ones that get *_small copies + randomspin).
SINGLE_SUBLATTICE_PHASES: List[str] = ["FCC_A1", "BCC_A2", "HCP_A3"]

# The "_small" directory names that sqs2tdb -cp -l=<name> works with for the
# single-sublattice systems. randomspin is run *inside* these directories.
SMALL_SYSTEM: Dict[str, str] = {
    "FCC_A1": "FCC_A1_small",
    "BCC_A2": "BCC_A2_small",
    "HCP_A3": "HCP_A3_small",
}

# Phases handled by the upstream pipeline, in canonical order.
ALL_PHASES: List[str] = ["FCC_A1", "BCC_A2", "HCP_A3", "SIGMA_D8B"]

# Phases that only ever use endmember calculations in a binary.
ENDMEMBER_ONLY_PHASES: List[str] = ["SIGMA_D8B"]


# ---------------------------------------------------------------------------
# DLM (disordered local moment) conventions
# ---------------------------------------------------------------------------
#
# randomspin rewrites str.out so that disordered-spin sites carry a +2 / -2
# tag appended to the element symbol, e.g. "Co" -> "Co+2" (spin up) or
# "Co-2" (spin down). The actual magnetic moment is NOT set via INCAR MAGMOM:
# ATAT's SUBATOM substitution rules in vasp.wrap rewrite "Co+2" -> "Co+1.8"
# etc. (and may switch POTCAR, e.g. Cr -> Cr_pv) at runstruct_vasp time, with
# MAGATOM= / USEPOT enabling the magnetic-atom machinery. The "+2"/"-2" are
# therefore just match tokens, not literal moments.
#
# After all phonon calculations are done, fitfc cannot read the +2/-2
# decorated str_relax.out / str_unpert.out, so the suffixes must be stripped
# (see phonon.dlm_fixup, which generalises the user's sed recipe).

DLM_SPIN_UP = "+2"      # the "_A" / spin-up tag randomspin / our SIGMA
DLM_SPIN_DOWN = "-2"    # converter append to an element symbol


@dataclass
class DLMConfig:
    """Per-run DLM settings.

    subatom maps an element symbol to (potcar_label, moment) and drives the
    SUBATOM lines written into vasp.wrap, e.g.
        {"Co": ("Co", 1.8), "Cr": ("Cr_pv", 1.5), "Ni": ("Ni", 0.7)}
    yields  s/Co+2/Co+1.8/g, s/Co-2/Co-1.8/g, s/Cr+2/Cr_pv+1.5/g, ...
    """
    enabled: bool = False
    subatom: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    # If True, SIGMA endmembers are built from lev=3 SQS via the
    # lev3 -> lev0 +/-spin conversion (sigma_lev3_to_lev0_dlm in sqsgen).
    sigma_from_lev3: bool = True


# ---------------------------------------------------------------------------
# SIGMA lev=3 -> lev=0 +/-spin conversion config
# ---------------------------------------------------------------------------
#
# For a DLM SIGMA_D8B endmember we follow the convention that each sublattice
# is occupied by a single element, but the random spin arrangement within the
# equivalent sites of that element is represented by splitting them into a
# spin-up pseudo-species (..._A, tagged +2) and a spin-down pseudo-species
# (..._B, tagged -2). sqs2tdb's lev=3 SQS already produces the correct random
# *spatial* split of two pseudo-species per site; we only need to relabel the
# two pseudo-species tokens to "<EL>+2" and "<EL>-2".

@dataclass
class SigmaDLMSpec:
    """How to relabel a SIGMA lev=3 SQS into a lev=0 DLM endmember.

    element        the single element occupying every site of this endmember
    token_up       the lev=3 pseudo-species symbol to map to spin-up (..._A)
    token_down     the lev=3 pseudo-species symbol to map to spin-down (..._B)
                   If either token is None it is auto-detected from str.out
                   (the two distinct species symbols present, sorted).
    """
    element: str
    token_up: Optional[str] = None
    token_down: Optional[str] = None


def is_single_sublattice(phase: str) -> bool:
    return phase in SINGLE_SUBLATTICE_PHASES


def site_for(phase: str) -> Optional[str]:
    return PHASE_SITE.get(phase)
