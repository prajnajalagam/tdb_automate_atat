#!/usr/bin/env python3
"""
vasp.wrap (INCAR) generation, modelled on the user's working file.

runstruct_vasp turns ``vasp.wrap`` + ``str.out`` into the real VASP inputs.
The user's reference vasp.wrap (Co-Cr-Ni DLM):

    [INCAR]
    LREAL = Auto
    LWAVE = .FALSE.
    LCHARG = .FALSE.
    NCORE = 8
    KPAR = 4
    ALGO = Fast
    PREC = Normal
    ENCUT = 520
    ISYM = 0
    NUPDOWN = 0
    LORBIT = 11
    ISMEAR = 1
    SIGMA = 0.1
    NSW = 300
    NELM = 100
    EDIFFG = -0.01
    EDIFF = 1E-6
    AMIX = 0.1
    BMIX = 0.0001
    AMIX_MAG = 0.4
    BMIX_MAG = 0.0001
    IBRION = 2
    ISIF = 3
    KPPRA = 8000
    USEPOT = PAWPBE
    DOSTATIC
    MAGATOM =
    SUBATOM = s/Co+2/Co+1.8/g
    SUBATOM = s/Co-2/Co-1.8/g
    SUBATOM = s/Ni+2/Ni+0.7/g
    SUBATOM = s/Ni-2/Ni-0.7/g
    SUBATOM = s/Cr+2/Cr_pv+1.5/g
    SUBATOM = s/Cr-2/Cr_pv-1.5/g

Key facts encoded here:
  * DLM moments are NOT set via INCAR MAGMOM. They are applied by ATAT's
    SUBATOM substitution rules that rewrite the +2 / -2 spin tags in str.out
    into "<potcar_label><+/-moment>" -- which also lets a magnetic element
    pull a different POTCAR (Cr -> Cr_pv). MAGATOM= and USEPOT=PAWPBE enable
    the magnetic-atom / pseudopotential machinery.
  * Because the moment lives in str.out (via SUBATOM), the wrap is the SAME
    for every SQS of a given binary -- no per-site MAGMOM bookkeeping.

Three modes:
  static  : single point (NSW=0), DOSTATIC -- convergence + final static E.
  relax   : full relaxation (NSW, IBRION=2, ISIF=3), DOSTATIC.
  phonon  : frozen geometry (NSW=0), no DOSTATIC, ICHARG=1 + LWAVE/LCHARG on
            so pollmach -lu can reuse WAVECAR/CHGCAR across force runs.

ALGO defaults to "All" (design spec) but is overridable to match the
reference file's "Fast".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from phases import DLMConfig


# 3d elements whose alloys must be treated spin-polarized: non-magnetic
# (ISPIN=1) energies for Co/Ni/Fe ferromagnets and Cr/Mn antiferromagnets
# are wrong by tens of meV/atom — larger than every other error source in
# this pipeline combined (the convergence tolerance is 1 meV/atom).
MAGNETIC_3D = {"Cr", "Mn", "Fe", "Co", "Ni"}

# Module default for the `spin` argument of build_vasp_wrap. run_upstream
# sets this once (auto-on when any element is in MAGNETIC_3D and the run
# is not DLM) so every wrap written by converge/relax/phonon inherits it
# without threading a flag through each signature. Tests and callers can
# always pass spin=True/False explicitly to override.
DEFAULT_SPIN = False


def wants_spin(elements) -> bool:
    """True if any element needs spin-polarized DFT (see MAGNETIC_3D)."""
    return any(str(e).capitalize() in MAGNETIC_3D for e in elements)


# Base INCAR shared by all modes (mirrors the reference file minus the
# mode-specific relaxation keys, which are layered on per mode).
_BASE_INCAR: List[Tuple[str, object]] = [
    ("LREAL", "Auto"),
    ("LWAVE", ".FALSE."),
    ("LCHARG", ".FALSE."),
    ("NCORE", 8),
    ("KPAR", 4),
    ("ALGO", "All"),        # overridden by `algo` arg
    ("PREC", "Normal"),
    ("LORBIT", 11),
    ("ISMEAR", 1),
    ("SIGMA", 0.1),
    ("NELM", 100),
    ("EDIFF", "1E-6"),
]

# Mode-specific keys (order preserved). "_dostatic"/"_extra" are pseudo-keys.
_MODE_INCAR: Dict[str, List[Tuple[str, object]]] = {
    "static": [
        ("NSW", 0),
        ("IBRION", -1),
        ("_dostatic", True),
    ],
    "relax": [
        ("NSW", 300),
        ("IBRION", 2),
        ("ISIF", 3),
        ("EDIFFG", -0.01),
        ("_dostatic", True),
    ],
    "phonon": [
        ("NSW", 0),
        ("IBRION", -1),
        ("ICHARG", 1),
        ("LWAVE", ".TRUE."),    # reuse WAVECAR/CHGCAR for pollmach -lu
        ("LCHARG", ".TRUE."),
        ("_dostatic", False),
    ],
}

# DLM-only INCAR additions (magnetic mixing + symmetry), from the reference.
_DLM_INCAR: List[Tuple[str, object]] = [
    ("ISYM", 0),
    ("NUPDOWN", 0),
    ("AMIX", 0.1),
    ("BMIX", 0.0001),
    ("AMIX_MAG", 0.4),
    ("BMIX_MAG", 0.0001),
]


def subatom_lines(subatom: Dict[str, Tuple[str, float]]) -> List[str]:
    """Build SUBATOM substitution lines from an element -> (potcar_label,
    moment) map.

    For each element EL with (potcar_label POT, moment m) this yields::

        SUBATOM = s/EL+2/POT+m/g
        SUBATOM = s/EL-2/POT-m/g

    matching the reference file (e.g. Cr -> Cr_pv at moment 1.5 gives
    s/Cr+2/Cr_pv+1.5/g and s/Cr-2/Cr_pv-1.5/g).
    """
    lines: List[str] = []
    for el, (pot, mom) in subatom.items():
        m = f"{mom:g}"
        lines.append(f"SUBATOM = s/{el}+2/{pot}+{m}/g")
        lines.append(f"SUBATOM = s/{el}-2/{pot}-{m}/g")
    return lines


def build_vasp_wrap(mode: str,
                    encut: Optional[int] = None,
                    kppra: Optional[int] = None,
                    dlm: Optional[DLMConfig] = None,
                    algo: str = "All",
                    usepot: str = "PAWPBE",
                    spin: Optional[bool] = None,
                    extra: Optional[Dict[str, object]] = None) -> str:
    """Return the text of a vasp.wrap file.

    mode    'static' | 'relax' | 'phonon'
    encut   ENCUT (eV); omitted if None (VASP falls back to POTCAR max).
    kppra   KPPRA value.
    dlm     DLMConfig; when enabled, appends magnetic mixing keys, MAGATOM=,
            USEPOT, and the SUBATOM moment-substitution lines.
    algo    ALGO value (default 'All' per spec; pass 'Fast' to match the
            reference file).
    spin    ISPIN=2 collinear spin polarization. None (default) falls back
            to the module-level DEFAULT_SPIN, which run_upstream turns on
            automatically for MAGNETIC_3D elements. Initial moments ride
            on VASP's default (1 muB/atom); that reliably finds the FM
            state for Co/Ni/Fe but can land Cr/Mn in low-moment local
            minima — for those, prefer the DLM machinery (SUBATOM
            moments) or pass explicit MAGMOM via `extra`. Not needed for
            DLM runs, where ezvasp's MAGATOM/SUBATOM path handles spin.
    extra   INCAR key->value overrides merged last (highest priority).
    """
    if mode not in _MODE_INCAR:
        raise ValueError(f"unknown mode {mode!r}; "
                         f"expected one of {sorted(_MODE_INCAR)}")

    # Assemble ordered (key, value) pairs.
    incar: List[Tuple[str, object]] = []
    seen: set = set()

    def put(k, v):
        if k in seen:
            for i, (kk, _vv) in enumerate(incar):
                if kk == k:
                    incar[i] = (k, v)
                    return
        seen.add(k)
        incar.append((k, v))

    for k, v in _BASE_INCAR:
        put(k, v)
    put("ALGO", algo)
    if encut is not None:
        put("ENCUT", int(encut))

    dostatic = False
    for k, v in _MODE_INCAR[mode]:
        if k == "_dostatic":
            dostatic = bool(v)
            continue
        put(k, v)

    if kppra is not None:
        put("KPPRA", int(kppra))
    put("USEPOT", usepot)

    if spin is None:
        spin = DEFAULT_SPIN
    dlm_on = dlm is not None and dlm.enabled
    if spin and not dlm_on:
        # Collinear spin polarization for FM/paramagnetic metals. DLM runs
        # skip this: their spin handling comes from the SUBATOM moment
        # machinery (ezvasp emits the magnetic INCAR itself).
        put("ISPIN", 2)
    if dlm_on:
        for k, v in _DLM_INCAR:
            put(k, v)

    if extra:
        for k, v in extra.items():
            put(k, v)

    lines = ["[INCAR]"]
    for k, v in incar:
        lines.append(f"{k} = {v}")
    if dostatic:
        lines.append("DOSTATIC")
    if dlm_on:
        lines.append("MAGATOM =")
        lines.extend(subatom_lines(dlm.subatom or {}))
    return "\n".join(lines) + "\n"
