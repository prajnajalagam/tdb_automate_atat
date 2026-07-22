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
  phonon  : frozen geometry (NSW=0, IBRION=-1, ISIF=2), PREC=Accurate,
            ALGO=Fast, no DOSTATIC — the user's fvasp.wrap for fitfc
            force runs.

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

# Supported ALGO modes (2026-07-17 user decision): All (default, most
# robust diagonalization), Normal (blocked Davidson, = IALGO 38 in the
# production INCARs), VeryFast (RMM-DIIS only, cheapest and least
# robust — use only on well-behaved systems). One global choice applies
# to EVERY wrap the pipeline writes (static, relax, phonon).
ALGO_MODES = ("All", "Normal", "VeryFast")


def normalize_algo(algo: str) -> str:
    """Case-insensitive ALGO normalization restricted to ALGO_MODES."""
    canon = {m.lower(): m for m in ALGO_MODES}
    key = str(algo).strip().lower()
    if key not in canon:
        raise ValueError(
            f"ALGO {algo!r} not in supported modes {ALGO_MODES}")
    return canon[key]

# Module default for the `spin` argument of build_vasp_wrap. run_upstream
# sets this once (auto-on when any element is in MAGNETIC_3D and the run
# is not DLM) so every wrap written by converge/relax/phonon inherits it
# without threading a flag through each signature. Tests and callers can
# always pass spin=True/False explicitly to override.
DEFAULT_SPIN = False

# Initial magnetic moment (muB/atom) written as a uniform MAGMOM line
# for spin-polarized non-DLM runs — the user's working INCARs initialize
# every site at 3 (high-spin start converges to the FM/ferrimagnetic
# solution far more reliably than VASP's 1.0 default; the 2026-07-14
# run's OUTCARs carried the "did not specify MAGMOM" warning on every
# calculation). Overridable per-call (magmom_init) or via
# --magmom-init in run_upstream.
DEFAULT_MAGMOM_INIT = 3.0


def wants_spin(elements) -> bool:
    """True if any element needs spin-polarized DFT (see MAGNETIC_3D)."""
    return any(str(e).capitalize() in MAGNETIC_3D for e in elements)


def ranks_from_prefix(cmd_prefix: str) -> Optional[int]:
    """Parse the MPI rank count out of a launch prefix.

    "mpiexec -n 32" / "mpirun -np 128" -> 32 / 128; None when the
    prefix is empty or carries no recognizable -n/-np <int>.
    """
    toks = (cmd_prefix or "").split()
    for i, t in enumerate(toks[:-1]):
        if t in ("-n", "-np"):
            try:
                return int(toks[i + 1])
            except ValueError:
                return None
    return None


def _kpar_dividing(ranks: Optional[int], want: int) -> int:
    """Largest KPAR <= want that divides ranks (VASP requires it)."""
    if not ranks or ranks < 2:
        return 1
    k = min(want, ranks)
    while k > 1 and ranks % k:
        k -= 1
    return k


def parallel_overrides(natoms: Optional[int],
                       ranks: Optional[int] = None) -> Dict[str, int]:
    """NCORE/KPAR safe for the cell size.

    The reference wrap's NCORE=8 + KPAR=4 assumes production-size SQS.
    On a 1-atom endmember it asks VASP to split a handful of bands over
    32 cores per k-group — in the real Co-Cr run every such cell died
    with EDWAV "gradient is not orthogonal" / "Sub-Space-Matrix is not
    hermitian" followed by MPI_Abort, which then cascaded (empty
    CONTCAR -> 0-ion static POSCAR -> bogus XC_FOCK_READER abort).
    Small cells get a conservative decomposition; production cells keep
    the wrap defaults. Safe beats optimal here — a 1-atom cell is fast
    on any layout.

    ranks: total MPI ranks the job launches (ranks_from_prefix). Small
    cells are K-POINT dominated (a 1-atom cell at KPPRA 10000 has
    hundreds of irreducible k-points), so with ranks known the saved
    band-parallelism is spent on KPAR instead of left idle: KPAR only
    splits the k-list across independent groups — each group's
    band-splitting stays at or below the validated single-group layout,
    so this is strictly gentler per group AND up to 8x faster. Without
    ranks (None) the old conservative KPAR survives unchanged, since
    VASP aborts when KPAR does not divide the rank count.
    """
    if natoms is None:
        return {}
    if natoms <= 4:
        return {"NCORE": 1,
                "KPAR": _kpar_dividing(ranks, 8) if ranks else 1}
    if natoms <= 12:
        return {"NCORE": 2,
                "KPAR": _kpar_dividing(ranks, 4) if ranks else 2}
    return {}


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
    ("SIGMA", 0.08),
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
        ("NSW", 100),
        ("IBRION", 2),
        ("ISIF", 3),
        ("EDIFFG", -0.01),
        ("_dostatic", True),
    ],
    # NOTE: no "volrelax"/"static-for-robustrelax" modes here on
    # purpose — robustrelax_vasp -mk derives vaspvol.wrap (ISIF=7),
    # vaspstatic.wrap (ISMEAR=-5), vaspid.wrap etc. from the tuned
    # vasp.wrap itself, inheriting ENCUT/KPPRA/spin/NCORE/KPAR.
    "phonon": [
        # Frozen ions (NSW=0/IBRION=-1/ISIF=2), PREC=Accurate for clean
        # forces. ALGO comes from the global --algo choice (2026-07-17:
        # one ALGO mode for all wraps; default All). No ICHARG=1 — it
        # hard-errors when no CHGCAR exists, and the first force run
        # never has one.
        ("NSW", 0),
        ("IBRION", -1),
        ("ISIF", 2),
        ("PREC", "Accurate"),
        ("_dostatic", False),
    ],
}

# Magnetic-mixing keys for spin-polarized (non-DLM) runs, from the
# user's production INCAR: damped charge mixing + fast moment mixing.
_SPIN_INCAR: List[Tuple[str, object]] = [
    ("AMIX", 0.03),
    ("BMIX", 0.0001),
    ("AMIX_MAG", 0.8),
    ("BMIX_MAG", 0.0001),
]

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
                    natoms: Optional[int] = None,
                    magmom_init: Optional[float] = None,
                    ranks: Optional[int] = None,
                    extra: Optional[Dict[str, object]] = None) -> str:
    """Return the text of a vasp.wrap file.

    mode    'static' | 'relax' | 'phonon'
    encut   ENCUT (eV); omitted if None (VASP falls back to POTCAR max).
    kppra   KPPRA value.
    dlm     DLMConfig; when enabled, appends magnetic mixing keys, MAGATOM=,
            USEPOT, and the SUBATOM moment-substitution lines.
    algo    ALGO value (default 'All' per spec; pass 'Fast' to match the
            reference file).
    natoms  atom count of the cell this wrap will run; small cells get
            conservative NCORE/KPAR (see parallel_overrides) — the
            reference NCORE=8/KPAR=4 crashes VASP on 1-atom endmembers.
    ranks   total MPI ranks of the launcher (ranks_from_prefix on the
            cmd prefix); lets small k-point-dominated cells recover the
            parallelism as KPAR. Omitted -> old conservative layout.
    spin    ISPIN=2 collinear spin polarization. None (default) falls back
            to the module-level DEFAULT_SPIN, which run_upstream turns on
            automatically for MAGNETIC_3D elements. When natoms is known
            a uniform "MAGMOM = <natoms>*<magmom_init>" line is emitted
            (default 3 muB — the user's convention; a uniform value is
            order-independent, so ezvasp's POSCAR grouping cannot
            scramble it) plus the magnetic-mixing keys from the user's
            production INCAR. Without natoms the MAGMOM line is omitted
            and VASP's 1 muB default applies (warning in OUTCAR). Not
            used for DLM runs, where ezvasp's MAGATOM/SUBATOM path
            handles moments (and can swap POTCARs, e.g. Cr -> Cr_pv).
    magmom_init  initial moment per atom for the MAGMOM line; None ->
            module-level DEFAULT_MAGMOM_INIT (3.0).
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

    for k, v in parallel_overrides(natoms, ranks).items():
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
        if magmom_init is None:
            magmom_init = DEFAULT_MAGMOM_INIT
        if natoms:
            # Uniform initial moment, VASP multiplier syntax ("32*3").
            put("MAGMOM", f"{int(natoms)}*{magmom_init:g}")
        for k, v in _SPIN_INCAR:
            put(k, v)
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
