"""
Shared library for the multicomponent TDB automation pipeline.

This module is the DFT-free foundation: phase constants, sqs2tdb directory
parsing, N-element composition handling, subsystem enumeration (binaries,
ternaries, ...), and generalized parameter counting for terms.in. All
functions here are pure (no ATAT, no pycalphad, no DFT data required) so
they can be unit-tested in isolation.

Used by select_endmembers_mc.py, sqs2tdb_pipeline_mc.py, and
score_tdb_combinations_mc.py. The binary scripts in
'../TDB Automated Generator/' are intentionally NOT touched; their copies
of the helpers (element_case, find_svib_ht, etc.) are duplicated here so
the binary pipeline keeps working unchanged.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ════════════════════════════════════════════════════════════════════
#  Phase catalogue (verified against ATAT's sqsdb/<PHASE>/rndstr.skel)
# ════════════════════════════════════════════════════════════════════

# Phase name → substrings that, if present in a path, identify the phase.
PHASE_TOKENS: Dict[str, List[str]] = {
    "FCC_A1":    ["FCC_A1", "FCC"],
    "BCC_A2":    ["BCC_A2", "BCC"],
    "HCP_A3":    ["HCP_A3", "HCP"],
    "SIGMA_D8B": ["SIGMA_D8B", "SIGMA"],
}

# Single-sublattice phases: their mixing site label.
SITE_FOR_PHASE: Dict[str, str] = {
    "FCC_A1": "a",
    "BCC_A2": "a",
    "HCP_A3": "c",
}

# Sublattice site → atom count per formula unit. SIGMA verified by counting
# rndstr.skel sites: 10 aj + 4 g + 16 ii = 30 atoms.
PHASE_MULT: Dict[str, Dict[str, int]] = {
    "FCC_A1":    {"a": 1},
    "BCC_A2":    {"a": 1},
    "HCP_A3":    {"c": 2},
    "SIGMA_D8B": {"aj": 10, "g": 4, "ii": 16},
}

# Canonical write-out order for sublattices (alphabetical site label).
PHASE_SUBLATTICES: Dict[str, List[str]] = {
    ph: sorted(PHASE_MULT[ph].keys()) for ph in PHASE_MULT
}

SIGMA_SUBLATTICE_MULT: Dict[str, int] = PHASE_MULT["SIGMA_D8B"]

# Directory-name regexes. sqs2tdb -cp has emitted both "sqs_lev=" and
# "sqsdb_lev=" prefixes across ATAT versions; accept either.
SQS_DIR_RE = re.compile(r"sqs(?:db)?_lev=(\d+)")
SQS_PREFIX_RE = re.compile(r"^sqs(?:db)?_lev=\d+_")

# Token inside a directory name: site_Element=concentration
_OCC_TOKEN_RE = re.compile(r"([a-z]+)_([A-Za-z]+)=([0-9.]+)")

# Fortran-friendly float regex for energy/svib parsing.
_SCI_FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
)


# ════════════════════════════════════════════════════════════════════
#  Small helpers (lifted from the binary pipeline, unchanged in spirit)
# ════════════════════════════════════════════════════════════════════

def element_case(sym: str) -> str:
    """Canonical chemical case: first letter upper, rest lower ('Co')."""
    if not sym:
        return sym
    return sym[0].upper() + sym[1:].lower()


def normalize_elements(elements: List[str]) -> List[str]:
    """Sort, uppercase, deduplicate an element list."""
    return sorted({e.strip().upper() for e in elements if e.strip()})


def infer_phase(path: Path) -> Optional[str]:
    """Identify the phase a directory belongs to by path substring match."""
    u = str(path).upper()
    for ph, toks in PHASE_TOKENS.items():
        if any(t in u for t in toks):
            return ph
    return None


def parse_energy(path: Path) -> Optional[float]:
    """Parse a single number (handles Fortran scientific notation)."""
    try:
        txt = path.read_text(errors="ignore").strip()
        txt = txt.replace("D", "E").replace("d", "E")
        try:
            return float(txt)
        except ValueError:
            nums = _SCI_FLOAT_RE.findall(txt)
            if not nums:
                return None
            return float(nums[-1].replace("D", "E").replace("d", "E"))
    except Exception:
        return None


def find_svib_ht(d: Path) -> Optional[Path]:
    """Locate svib_ht in d or under d/vol_0 (depth <= 3)."""
    direct = d / "svib_ht"
    if direct.is_file():
        return direct
    vol_0 = d / "vol_0"
    if vol_0.is_dir():
        for item in vol_0.rglob("svib_ht"):
            if item.is_file():
                return item
    return None


def has_mandatory_files(d: Path) -> Tuple[bool, str]:
    """Files required by sqs2tdb -fit: energy + str.out."""
    missing = []
    if not (d / "energy").is_file():
        missing.append("energy")
    if not (d / "str.out").is_file():
        missing.append("str.out")
    if missing:
        return False, ", ".join(missing)
    return True, ""


def robust_copytree(src: Path, dst: Path) -> None:
    """Recursive copy that resolves every symlink to its real bytes."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_symlink():
            real = item.resolve()
            if real.is_file():
                shutil.copy2(str(real), str(dest_item))
            elif real.is_dir():
                robust_copytree(real, dest_item)
            # else: broken — skip
        elif item.is_dir():
            robust_copytree(item, dest_item)
        elif item.is_file():
            shutil.copy2(str(item), str(dest_item))


# ════════════════════════════════════════════════════════════════════
#  N-element occupation parsing
# ════════════════════════════════════════════════════════════════════

@dataclass
class Occupation:
    """
    Per-sublattice occupation parsed from an sqs2tdb directory name.

    `sites` is the ordered list of (site_label, element, fraction)
    triples exactly as written. `species_per_site` is a derived
    {site: {element: fraction}} for ergonomic queries. Both fractions
    are unnormalized — multi-sublattice phases can have multiple species
    summing to 1 per site.
    """
    level: int
    sites: Tuple[Tuple[str, str, float], ...]
    species_per_site: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.species_per_site:
            tmp: Dict[str, Dict[str, float]] = {}
            for site, elem, val in self.sites:
                tmp.setdefault(site, {})[elem] = (
                    tmp.get(site, {}).get(elem, 0.0) + val
                )
            self.species_per_site = tmp

    def all_elements(self) -> FrozenSet[str]:
        return frozenset(e for _, e, _ in self.sites)

    def elements_on(self, site: str) -> Set[str]:
        return set(self.species_per_site.get(site, {}).keys())


def parse_occupation(dirname: str) -> Optional[Occupation]:
    """Parse an sqs2tdb dir name into an Occupation, or None on mismatch."""
    m = SQS_DIR_RE.match(dirname)
    if not m:
        return None
    lev = int(m.group(1))
    comp_str = SQS_PREFIX_RE.sub("", dirname)
    raw = _OCC_TOKEN_RE.findall(comp_str)
    if not raw:
        return None
    # element symbols normalized to UPPER for downstream comparison
    sites = tuple(
        (site, elem.upper(), float(val)) for site, elem, val in raw
    )
    return Occupation(level=lev, sites=sites)


def composition_on_phase(
    occ: Occupation,
    phase: str,
    elements: List[str],
) -> Optional[Dict[str, float]]:
    """
    Compute the N-element composition (sum to 1) on the mixing
    sublattice(s) of `phase`, restricted to `elements`.

    Returns None when:
      - any mixing-site occupant is outside `elements` (foreign-element
        rejection — matches the binary script's post-fix behavior), or
      - the phase has no mixing sites in our catalogue, or
      - no mixing-site mass is present.

    Single sublattice: site fractions are summed directly.
    Multi-sublattice (SIGMA): site occupations are multiplicity-weighted.
    """
    allowed = {e.upper() for e in elements}
    mult: Dict[str, int]
    if phase in SITE_FOR_PHASE:
        site = SITE_FOR_PHASE[phase]
        mult = {site: PHASE_MULT[phase][site]}
    elif phase in PHASE_MULT:
        mult = PHASE_MULT[phase]
    else:
        return None

    # Foreign-element rejection on any mixing sublattice
    for s, e, _ in occ.sites:
        if s in mult and e not in allowed:
            return None

    comp = {e: 0.0 for e in allowed}
    for s, e, v in occ.sites:
        if s in mult and e in allowed:
            comp[e] += mult[s] * v

    total = sum(comp.values())
    if total <= 0:
        return None
    return {e: v / total for e, v in comp.items()}


def pure_endmember_element(
    occ: Occupation,
    phase: str,
    elements: List[str],
) -> Optional[str]:
    """
    Single-sublattice lev=0 pure-element endmember check.
    Returns the element symbol if this IS such an endmember (one species
    at fraction 1 on the mixing site, and that species is in `elements`),
    None otherwise.
    """
    if phase not in SITE_FOR_PHASE:
        return None
    if occ.level != 0:
        return None
    species = occ.species_per_site.get(SITE_FOR_PHASE[phase], {})
    if len(species) != 1:
        return None
    elem, frac = next(iter(species.items()))
    if elem not in {e.upper() for e in elements}:
        return None
    if abs(frac - 1.0) > 1e-6:
        return None
    return elem


def sigma_corner_key(
    occ: Occupation,
    elements: List[str],
) -> Optional[Tuple[Tuple[str, str], ...]]:
    """
    SIGMA lev=0 corner identification.

    A "corner" is a lev=0 SIGMA occupation where every sublattice is
    occupied by exactly ONE element drawn from `elements`. Returns a
    canonical sorted tuple of (site, element) pairs identifying the
    corner, or None if the occupation isn't a corner.

    For an N-component system, SIGMA has N^N_sublattices = N**3 corners.
    """
    if occ.level != 0:
        return None
    if "SIGMA_D8B" not in PHASE_MULT:
        return None
    allowed = {e.upper() for e in elements}
    key: List[Tuple[str, str]] = []
    for site in PHASE_SUBLATTICES["SIGMA_D8B"]:
        species = occ.species_per_site.get(site, {})
        if len(species) != 1:
            return None
        elem, frac = next(iter(species.items()))
        if elem not in allowed:
            return None
        if abs(frac - 1.0) > 1e-6:
            return None
        key.append((site, elem))
    return tuple(key)


# ════════════════════════════════════════════════════════════════════
#  Subsystem enumeration
# ════════════════════════════════════════════════════════════════════

def enumerate_subsystems(
    elements: List[str],
    min_order: int = 2,
    max_order: Optional[int] = None,
) -> Dict[int, List[Tuple[str, ...]]]:
    """
    Enumerate subsystems of an N-component system by order.

        enumerate_subsystems(['Co','Cr','Ni'])
          -> {2: [('CO','CR'), ('CO','NI'), ('CR','NI')],
              3: [('CO','CR','NI')]}

    Elements are normalized; tuples are internally sorted so subsystem
    identity is canonical.
    """
    els = normalize_elements(elements)
    if max_order is None:
        max_order = len(els)
    out: Dict[int, List[Tuple[str, ...]]] = {}
    for r in range(min_order, min(max_order, len(els)) + 1):
        out[r] = [tuple(sorted(c)) for c in itertools.combinations(els, r)]
    return out


def subsystem_for_occupation(
    occ: Occupation,
    phase: str,
    elements: List[str],
) -> Optional[Tuple[str, ...]]:
    """
    Which subsystem of the full N-element system does this SQS populate?

    Returns the canonical (sorted) tuple of elements ACTUALLY present on
    the mixing sublattice(s) of `phase`. Returns None if any foreign
    element appears or the phase is unknown.

    A lev=0 endmember will return a 1-tuple (one element). A binary edge
    mixing SQS returns a 2-tuple, ternary interior a 3-tuple, etc. This
    tag is what drives Route B (per-subsystem fitting) downstream.
    """
    allowed = {e.upper() for e in elements}
    if phase in SITE_FOR_PHASE:
        mult = {SITE_FOR_PHASE[phase]: PHASE_MULT[phase][SITE_FOR_PHASE[phase]]}
    elif phase in PHASE_MULT:
        mult = PHASE_MULT[phase]
    else:
        return None

    present: Set[str] = set()
    for s, e, v in occ.sites:
        if s not in mult:
            continue
        if e not in allowed:
            return None
        if v > 0:
            present.add(e)
    return tuple(sorted(present)) if present else None


# ════════════════════════════════════════════════════════════════════
#  Discovered-SQS data class + scanner
# ════════════════════════════════════════════════════════════════════

@dataclass
class SQSCandidate:
    """A discovered sqs2tdb directory after parsing & subsystem tagging."""
    path: Path
    phase: str
    occupation: Occupation
    composition: Dict[str, float]    # {El: frac} on mixing sublattice(s)
    subsystem: Tuple[str, ...]       # sorted element tuple actually present
    energy: Optional[float]
    svib_path: Optional[Path]

    @property
    def level(self) -> int:
        return self.occupation.level

    @property
    def has_svib(self) -> bool:
        return self.svib_path is not None


def scan_sqs(
    roots: List[Path],
    elements: List[str],
    phases: Optional[List[str]] = None,
    scan_depth: int = 6,
    require_files: bool = True,
    verbose: bool = False,
) -> List[SQSCandidate]:
    """
    Walk every root and collect SQSCandidates for any sqs2tdb directory
    whose phase is in `phases` (default: all in PHASE_TOKENS) and whose
    mixing-sublattice occupants are all in `elements`.

    Foreign-element structures are silently rejected; missing-file dirs
    are skipped (with a printed reason when verbose=True).
    """
    phases_set = set(phases) if phases else set(PHASE_TOKENS.keys())
    els = normalize_elements(elements)
    out: List[SQSCandidate] = []

    for root in roots:
        if not root.exists():
            if verbose:
                print(f"  WARNING: {root} does not exist")
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            try:
                depth = len(p.relative_to(root).parts)
            except ValueError:
                continue
            if depth > scan_depth:
                dirnames.clear()
                continue

            if not SQS_DIR_RE.match(p.name):
                continue
            phase = infer_phase(p)
            if not phase or phase not in phases_set:
                continue
            occ = parse_occupation(p.name)
            if occ is None:
                continue
            if require_files:
                ok, missing = has_mandatory_files(p)
                if not ok:
                    if verbose:
                        print(f"  SKIP {p.name}: missing {missing}")
                    continue
            comp = composition_on_phase(occ, phase, els)
            if comp is None:
                continue
            sub = subsystem_for_occupation(occ, phase, els)
            if sub is None:
                continue
            out.append(SQSCandidate(
                path=p, phase=phase, occupation=occ,
                composition=comp, subsystem=sub,
                energy=parse_energy(p / "energy"),
                svib_path=find_svib_ht(p),
            ))
    return out


# ════════════════════════════════════════════════════════════════════
#  Generalized terms.in parameter counting
# ════════════════════════════════════════════════════════════════════

def n_params_for_terms(
    species_per_sublattice: Dict[str, List[str]],
    terms: List[Tuple[int, int]],
) -> int:
    """
    Generalized parameter count for a CALPHAD model fit via sqs2tdb.

    Arguments
    ---------
    species_per_sublattice : {site_label: [elements]}
        Per-sublattice species lists, e.g.
          FCC_A1 Co-Cr-Ni: {"a": ["CO","CR","NI"]}
          SIGMA Co-Cr-Ni : {"aj":[...], "g":[...], "ii":[...]}
    terms : list of (order, level)
        terms.in lines. order: 1=endmember, 2=binary, 3=ternary, ...
        level: Redlich-Kister polynomial degree for that order.

    Returns
    -------
    n_params : sum over sublattices of (per-line contribution):
        order==1 : K parameters per sublattice (one per species)
        order>=2 : C(K, order) * (level + 1) per sublattice

    This replaces the binary `n_params = order + 3` hardcoding in
    sqs2tdb_pipeline.py and is the overfit-guard's input for the
    multicomponent case (n_data > n_params).
    """
    total = 0
    for site, species in species_per_sublattice.items():
        K = len(species)
        for order, level in terms:
            if K < order:
                continue
            if order == 1:
                total += K
            else:
                total += math.comb(K, order) * (level + 1)
    return total


__all__ = [
    "PHASE_TOKENS", "SITE_FOR_PHASE", "PHASE_MULT", "PHASE_SUBLATTICES",
    "SIGMA_SUBLATTICE_MULT", "SQS_DIR_RE", "SQS_PREFIX_RE",
    "element_case", "normalize_elements", "infer_phase",
    "parse_energy", "find_svib_ht", "has_mandatory_files", "robust_copytree",
    "Occupation", "parse_occupation",
    "composition_on_phase", "pure_endmember_element", "sigma_corner_key",
    "enumerate_subsystems", "subsystem_for_occupation",
    "SQSCandidate", "scan_sqs",
    "n_params_for_terms",
]
