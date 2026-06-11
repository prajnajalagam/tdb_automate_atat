#!/usr/bin/env python3
"""
STEP 2: Automated sqs2tdb pipeline for binary CALPHAD assessment.

Strategy (from the design doc, aligned with sqs2tdb source code):
  Stage 1  Energy-only fits — iterate SQS combinations × terms.in options,
           prune on max |error| in fit_energy.out column 5.
  Stage 2  Layer svib_ht onto surviving Stage-1 fits — endmember svib_ht is
           always included when svib_ht is considered; SQS svib_ht subsets
           are combinatorially explored.  Prune on fit_svib_ht.out errors.
  Stage 3  (future) Cross-phase combination and phase-diagram scoring.

Key corrections vs prior version (verified against sqs2tdb Perl source):
  - `sqs2tdb -fit` does `ls -d sqs_*` to find structures.  The directory
    names encode composition as site_Element=value tokens parsed after
    stripping "sqs_" and "lev=N_".
  - `str.out` is MANDATORY — the code dies if it cannot open it.
    `str_relax.out` must also be present (VASP workflow produces both).
  - `species.in` and `mult.in` must be in the phase working directory.
  - HCP_A3 Wyckoff site is `c` (mult=2).
  - SIGMA_D8B has 3 sublattices (aj=10, g=4, ii=16) and for binaries
    only uses endmembers (8 configs).  terms.in is fixed.
  - Energy is in eV for the full supercell; conversion to J/mol is
    handled internally by sqs2tdb.
  - svib_ht removal is done by deleting the file from the working copy.
  - Endmember svib_ht must ALWAYS be included when svib_ht is considered.
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML required.  pip install pyyaml")


# ====================================================================
# Configuration constants
# ====================================================================

# Wyckoff site labels per phase (from rndstr.skel in sqsdb)
PHASE_SITE = {
    "FCC_A1": "a",
    "BCC_A2": "a",
    "HCP_A3": "c",
}

# Multiplicities per phase (from rndstr.skel atom counts)
PHASE_MULT = {
    "FCC_A1": {"a": 1},
    "BCC_A2": {"a": 1},
    "HCP_A3": {"c": 2},
    "SIGMA_D8B": {"aj": 10, "g": 4, "ii": 16},
}

# terms.in options for non-SIGMA binaries
BINARY_TERMS_OPTIONS = [0, 1, 2]  # the X in "2,X"

# Fixed terms.in for SIGMA
SIGMA_TERMS = "1,0:1,0:1,0\n2,0:2,0:2,0\n"

# Binary SQS counts: lev=1 gives 1, lev=2 adds 2, lev=5 adds 4 more → max 7
MIN_SQS_DEFAULT = 3
MAX_SQS_DEFAULT = 7

PHASE_TOKENS = {
    "FCC_A1": ["FCC_A1", "FCC"],
    "BCC_A2": ["BCC_A2", "BCC"],
    "HCP_A3": ["HCP_A3", "HCP"],
    "SIGMA_D8B": ["SIGMA_D8B", "SIGMA"],
}

# sqs2tdb -cp has emitted both "sqs_lev=N_..." and "sqsdb_lev=N_..." across
# ATAT versions; accept either prefix so discovery doesn't silently find zero.
SQS_DIR_RE = re.compile(r"sqs(?:db)?_lev=(\d+)")
SQS_PREFIX_RE = re.compile(r"^sqs(?:db)?_lev=\d+_")


# ====================================================================
# Data classes
# ====================================================================

@dataclass
class SQSData:
    """One non-endmember SQS directory."""
    path: Path
    name: str            # directory basename  (sqs_lev=...)
    level: int
    x1: float            # composition of element 1
    x2: float
    has_energy: bool
    has_svib: bool
    svib_path: Optional[Path]


@dataclass
class FitTask:
    """Everything needed to run one sqs2tdb -fit."""
    phase: str
    task_id: int
    work_root: Path
    endmembers: List[Path]        # lev=0 directories (always included)
    sqs_list: List[SQSData]       # mixing SQS (lev>0)
    terms_order: int              # X in "2,X"  (-1 means SIGMA fixed)
    svib_include: Set[str]        # names of SQS whose svib_ht to keep
    endmember_svib: bool          # whether endmember svib_ht is kept
    el1: str
    el2: str
    energy_cutoff: float
    svib_cutoff: float
    stage: int                    # 1 = energy-only, 2 = with svib


@dataclass
class FitResult:
    """Outcome of one fit."""
    phase: str
    task_id: int
    stage: int
    terms: str
    terms_order: int
    n_sqs: int
    sqs_names: List[str]
    endmember_svib: bool
    svib_names: List[str]
    energy_errors: List[float]
    max_energy_error: float
    svib_errors: Optional[List[float]]
    max_svib_error: Optional[float]
    tdb_path: Optional[str]
    success: bool
    error_msg: Optional[str]


# ====================================================================
# Utility
# ====================================================================

BASE_ENV = os.environ.copy()


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd: Path, log: Path, timeout: int = 600) -> int:
    try:
        with open(log, "w") as f:
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=BASE_ENV, shell=isinstance(cmd, str),
                stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        with open(log, "a") as f:
            f.write(f"\nTIMEOUT after {timeout}s\n")
        return -1
    except Exception as exc:
        with open(log, "a") as f:
            f.write(f"\nEXCEPTION: {exc}\n")
        return -1


def read_float_file(path: Path) -> Optional[float]:
    try:
        txt = path.read_text().strip().replace("D", "E").replace("d", "E")
        return float(txt)
    except Exception:
        return None


def parse_fit_file(path: Path, col_idx: int = 4) -> List[float]:
    """Parse fit_energy.out / fit_svib_ht.out — return absolute errors."""
    errors = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) > col_idx:
            try:
                errors.append(abs(float(
                    parts[col_idx].replace("D", "E").replace("d", "E"))))
            except ValueError:
                pass
    return errors


def find_svib_ht(d: Path) -> Optional[Path]:
    """Find svib_ht in d or d/vol_0 (depth ≤ 3)."""
    direct = d / "svib_ht"
    if direct.is_file():
        return direct
    vol_0 = d / "vol_0"
    if vol_0.is_dir():
        for item in vol_0.rglob("svib_ht"):
            if item.is_file():
                return item
    return None


def maybe_rename_energy_off(d: Path) -> bool:
    """
    If a directory has `energy.off` but no `energy`, rename energy.off
    to energy so sqs2tdb can use it. Returns True if a rename happened.

    `energy.off` is the ATAT convention for a "disabled" energy file
    (manually parked, e.g. because of a suspect DFT run). The user can
    re-enable everything in one pass with this rename.
    """
    energy = d / "energy"
    energy_off = d / "energy.off"
    if energy.is_file():
        return False
    if not energy_off.is_file():
        return False
    try:
        energy_off.rename(energy)
        return True
    except OSError:
        return False


def find_oszicar(d: Path) -> Optional[Path]:
    """Locate an OSZICAR file in the SQS directory (or under vol_0)."""
    direct = d / "OSZICAR"
    if direct.is_file():
        return direct
    vol_0 = d / "vol_0"
    if vol_0.is_dir():
        for item in vol_0.rglob("OSZICAR"):
            if item.is_file():
                return item
    return None


# Lazy-loaded OSZICAR scorer module (sits one directory up).
_OSZICAR_SCORER = None


def _load_oszicar_scorer():
    """Import oszicar_convergence_scorer once. Returns the module or None."""
    global _OSZICAR_SCORER
    if _OSZICAR_SCORER is not None:
        return _OSZICAR_SCORER if _OSZICAR_SCORER is not False else None
    try:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import oszicar_convergence_scorer as _mod   # type: ignore
        _OSZICAR_SCORER = _mod
        return _mod
    except Exception as exc:
        print(f"  WARNING: oszicar_convergence_scorer not importable "
              f"({exc}); --oszicar-min-score will be ignored.")
        _OSZICAR_SCORER = False
        return None


def oszicar_score(d: Path) -> Optional[float]:
    """
    Score the OSZICAR convergence for an SQS directory.
    Returns the total score in [0, 100], or None if no OSZICAR or scoring
    failed (caller should treat None as 'no information available').
    """
    osz = find_oszicar(d)
    if osz is None:
        return None
    mod = _load_oszicar_scorer()
    if mod is None:
        return None
    try:
        report = mod.score_oszicar(str(osz))
        return float(report.total_score)
    except Exception:
        return None


def has_mandatory_files(d: Path) -> bool:
    """str.out and energy must exist. str_relax.out is checked but not required."""
    return (d / "energy").is_file() and (d / "str.out").is_file()


def has_all_files(d: Path) -> Tuple[bool, List[str]]:
    """Check all expected files and return (ok, list_of_missing)."""
    missing = []
    for fn in ("energy", "str.out"):
        if not (d / fn).is_file():
            # Also check if it's a symlink that might resolve
            if (d / fn).is_symlink():
                target = os.readlink(str(d / fn))
                resolved = (d / fn).resolve()
                if not resolved.is_file():
                    missing.append(f"{fn} (broken symlink -> {target})")
                # else: symlink is valid, file exists
            else:
                missing.append(fn)
    return len(missing) == 0, missing


def infer_phase(path: Path) -> Optional[str]:
    u = str(path).upper()
    for ph, toks in PHASE_TOKENS.items():
        if any(t in u for t in toks):
            return ph
    return None


# ====================================================================
# SQS discovery
# ====================================================================

def discover_sqs(data_roots: List[Path], phase: str,
                 el1: str, el2: str,
                 verbose: bool = True,
                 rename_energy_off: bool = True,
                 oszicar_min_score: float = 0.0,
                 target_gate=None,
                 endmember_per_atom: Optional[Dict[str, float]] = None,
                 target_tol_sigma: float = 3.0) -> List[SQSData]:
    """Find all lev>0 SQS for a phase/binary, one per composition.

    Energy handling
    ---------------
    - If `rename_energy_off`, an `energy.off` file (ATAT's "parked" energy)
      is renamed to `energy` so the SQS can participate in the fit.
    - SQS whose `energy` is genuinely missing (or whose `energy` file is
      empty / unparseable) are dropped from the pool *before* sqs2tdb is
      ever asked to fit them. This prevents the "5 SQS offered but only
      3 energies available" failure mode the user reported.

    Convergence filter
    ------------------
    - When `oszicar_min_score > 0`, each candidate's OSZICAR is scored
      (if present) via oszicar_convergence_scorer; SQS scoring below the
      threshold are rejected. If the scorer is unavailable or no OSZICAR
      is present, the SQS is accepted (no information ≠ failure).

    Consensus-target gate
    ---------------------
    - When `target_gate` is supplied (a TargetGate from
      tdb_corpus/sqs_target_gate.py), each candidate SQS's DFT formation
      energy is computed relative to same-phase pure endmembers
      (provided via `endmember_per_atom` = {EL: eV/atom}) and z-scored
      against the consensus RK-excess target. SQS more than
      `target_tol_sigma` standard deviations from target are rejected.
      Compositions outside the RK's covered x-range are kept (no
      target to compare against).
    """
    el1, el2 = el1.upper(), el2.upper()
    found = []
    seen_comp: set = set()
    skipped_reasons: Dict[str, int] = {}
    rename_count = 0

    def skip(reason):
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    site = PHASE_SITE.get(phase)  # None for SIGMA (no mixing SQS)
    if site is None and phase == "SIGMA_D8B":
        return []   # SIGMA has no mixing SQS in a binary

    for root in data_roots:
        # Walk looking for sqs_lev=N directories under phase dirs
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            m = SQS_DIR_RE.match(p.name)
            if not m:
                continue
            if infer_phase(p) != phase:
                skip("wrong_phase")
                continue
            lev = int(m.group(1))
            if lev == 0:
                continue  # endmembers handled separately

            # Bring energy.off back into the pool before requiring energy.
            if rename_energy_off and maybe_rename_energy_off(p):
                rename_count += 1
                if verbose:
                    print(f"      RENAMED energy.off -> energy in {p.name}")

            ok, missing = has_all_files(p)
            if not ok:
                if verbose:
                    print(f"      SKIP {p.name}: missing {', '.join(missing)}")
                skip("missing_files")
                continue

            # Reject SQS whose energy file exists but is empty / unparseable.
            # sqs2tdb would otherwise silently fit with a wrong/zero value.
            if read_float_file(p / "energy") is None:
                if verbose:
                    print(f"      SKIP {p.name}: energy file empty or "
                          f"unparseable")
                skip("unparseable_energy")
                continue

            # Convergence quality gate.
            if oszicar_min_score > 0.0:
                score = oszicar_score(p)
                if score is not None and score < oszicar_min_score:
                    if verbose:
                        print(f"      SKIP {p.name}: OSZICAR score "
                              f"{score:.1f} < {oszicar_min_score}")
                    skip("oszicar_below_threshold")
                    continue

            # Parse composition from directory name
            comp_str = SQS_PREFIX_RE.sub("", p.name)
            tokens = re.findall(r"([a-z]+)_([A-Za-z]+)=([0-9.]+)", comp_str)
            if not tokens:
                skip("unparseable_name")
                if verbose:
                    print(f"      SKIP {p.name}: cannot parse composition tokens")
                continue

            comp = {}
            for s, elem, val in tokens:
                if s == site:
                    comp[elem.upper()] = float(val)
            x1 = comp.get(el1, 0.0)
            x2 = comp.get(el2, 0.0)
            total = x1 + x2
            if total <= 0:
                skip("no_matching_elements")
                if verbose:
                    print(f"      SKIP {p.name}: no {el1}/{el2} on site '{site}' "
                          f"(parsed: {comp})")
                continue

            x1 /= total
            x2 /= total
            comp_key = (round(x1, 5), round(x2, 5))
            if comp_key in seen_comp:
                skip("duplicate_composition")
                continue
            seen_comp.add(comp_key)

            # Consensus-target gate: same-phase excess formation energy
            # vs the RK-excess target loaded from a tdb_corpus consensus
            # JSON. Cheap (one float comparison) and applied AFTER the
            # cheap filters so we don't waste IO on obviously-bad dirs.
            if target_gate is not None and endmember_per_atom:
                try:
                    from sqs_target_gate import (
                        per_atom_energy as _per_atom_e,
                    )
                except ImportError:
                    _per_atom_e = None
                if _per_atom_e is not None:
                    pa = _per_atom_e(p / "energy", p / "str.out")
                    if pa is None:
                        skip("target_gate_no_atom_count")
                        if verbose:
                            print(f"      SKIP {p.name}: target gate could "
                                  f"not compute eV/atom (str.out unreadable)")
                        continue
                    e_per_atom, n_at = pa
                    comp_for_gate = {el1: x1, el2: x2}
                    e_ref = sum(
                        comp_for_gate[el] * endmember_per_atom[el]
                        for el in (el1, el2)
                        if el in endmember_per_atom
                    )
                    e_excess = e_per_atom - e_ref
                    passes, target, sigma, z, reason = target_gate.evaluate(
                        comp_for_gate, e_excess, n_sigma=target_tol_sigma,
                    )
                    if not passes:
                        if verbose:
                            print(f"      SKIP {p.name}: target gate "
                                  f"{reason}  (x({el2})={x2:.3f}, "
                                  f"E_excess={e_excess*1e3:+.1f} meV/atom)")
                        skip("target_gate_rejected")
                        continue

            svib_path = find_svib_ht(p)
            found.append(SQSData(
                path=p, name=p.name, level=lev,
                x1=x1, x2=x2,
                has_energy=True,
                has_svib=svib_path is not None,
                svib_path=svib_path))

    if verbose and rename_count:
        print(f"      Renamed energy.off -> energy in {rename_count} "
              f"directories")
    if verbose and skipped_reasons:
        print(f"      Discovery skip summary: {dict(skipped_reasons)}")

    found.sort(key=lambda s: s.x1)
    return found


# ====================================================================
# Working-directory setup
# ====================================================================

def element_case(sym: str) -> str:
    """
    Convert element symbol to standard chemical case: first letter uppercase,
    rest lowercase.  e.g. 'CO' -> 'Co', 'cr' -> 'Cr', 'Al' -> 'Al'.

    This is critical because sqs2tdb -fit parses directory names like
    sqs_lev=0_a_Co=1 and looks up 'Co' in the species.in hash.
    If species.in says 'CO' instead of 'Co', the lookup fails and
    every structure gets skipped.
    """
    if not sym:
        return sym
    return sym[0].upper() + sym[1:].lower()


def write_species_mult(phase: str, d: Path, el1: str, el2: str):
    """Write species.in and mult.in into phase working directory."""
    mult = PHASE_MULT[phase]
    # Element symbols MUST be in standard chemical case (Co, Cr, Al)
    # to match the directory names created by sqs2tdb -cp
    e1 = element_case(el1)
    e2 = element_case(el2)

    if phase == "SIGMA_D8B":
        sites = sorted(mult.keys())
        sp = "  ".join(f"{s}={e1},{e2}" for s in sites)
        ml = "  ".join(f"{s}={mult[s]}" for s in sites)
    else:
        site = PHASE_SITE[phase]
        sp = f"{site}={e1},{e2}"
        ml = f"{site}={mult[site]}"

    (d / "species.in").write_text(sp + "\n")
    (d / "mult.in").write_text(ml + "\n")

def write_terms(phase: str, d: Path, order: int):
    """Write terms.in.  For SIGMA, order is ignored (fixed)."""
    if phase == "SIGMA_D8B":
        (d / "terms.in").write_text(SIGMA_TERMS)
    else:
        (d / "terms.in").write_text(f"1,0\n2,{order}\n")


def robust_copytree(src: Path, dst: Path):
    """
    Copy a directory tree, fully resolving all symlinks to real file content.

    Why this is needed instead of shutil.copytree:
      - symlinks=True copies broken symlinks as-is → sqs2tdb can't read them
      - symlinks=False crashes entirely on broken symlinks
      - ATAT/VASP workflows routinely create symlinks (energy → ../../energy,
        str.out → absolute paths, chains of links, etc.)

    This function resolves every symlink to its final real target, copies the
    actual bytes, and skips anything that's broken.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name

        if item.is_symlink():
            real = item.resolve()
            if real.is_file():
                shutil.copy2(str(real), str(dest_item))
            elif real.is_dir():
                robust_copytree(real, dest_item)
            else:
                # Broken symlink — skip
                pass
        elif item.is_dir():
            robust_copytree(item, dest_item)
        elif item.is_file():
            shutil.copy2(str(item), str(dest_item))
        # else: sockets, fifos, etc. — skip


def setup_fit_directory(task: FitTask) -> Path:
    """
    Build the working tree that sqs2tdb -fit expects:
      <work_root>/<phase>/<task_id>/<phase>/
          species.in, mult.in, terms.in
          sqs_lev=0_.../  (endmembers with energy, str.out, str_relax.out)
          sqs_lev=N_.../  (mixing SQS)

    All symlinks are resolved to real file copies so that sqs2tdb -fit
    (which runs in the new directory) can open every file.
    """
    fit_dir = task.work_root / task.phase / f"{task.task_id:06d}"
    phase_dir = fit_dir / task.phase
    safe_mkdir(phase_dir)

    # Copy endmembers — fully resolve all symlinks
    for em in task.endmembers:
        dest = phase_dir / em.name
        if not dest.exists():
            robust_copytree(em, dest)
        # Handle svib removal if stage 1 or endmember_svib=False
        if not task.endmember_svib:
            sv = find_svib_ht(dest)
            if sv:
                sv.unlink()

    # Copy mixing SQS — fully resolve all symlinks
    for sqs in task.sqs_list:
        dest = phase_dir / sqs.name
        if not dest.exists():
            robust_copytree(sqs.path, dest)
        # Remove svib_ht if not in the include set
        if sqs.name not in task.svib_include:
            sv = find_svib_ht(dest)
            if sv:
                sv.unlink()

    # Write control files
    write_species_mult(task.phase, phase_dir, task.el1, task.el2)
    write_terms(task.phase, phase_dir, task.terms_order)

    return phase_dir


# ====================================================================
# Single fit execution
# ====================================================================

def do_one_fit(task: FitTask) -> FitResult:
    """Execute one sqs2tdb -fit and evaluate the result."""
    phase = task.phase
    tid = task.task_id
    terms_str = (SIGMA_TERMS.strip().replace("\n", " + ")
                 if phase == "SIGMA_D8B"
                 else f"1,0 / 2,{task.terms_order}")

    fit_dir = task.work_root / phase / f"{tid:06d}"

    def _fail(msg, e_err=None, s_err=None):
        # Preserve the sqs2tdb log into a global failure log before cleanup
        sqs_log = fit_dir / "sqs2tdb.log"
        global_fail_log = task.work_root / phase / "failures.log"
        try:
            if sqs_log.is_file():
                with open(global_fail_log, "a") as gf:
                    gf.write(f"\n{'='*60}\n")
                    gf.write(f"TASK {tid} FAILED: {msg}\n")
                    gf.write(f"SQS: {[s.name for s in task.sqs_list]}\n")
                    gf.write(f"terms: {terms_str}\n")
                    gf.write(f"{'='*60}\n")
                    gf.write(sqs_log.read_text())
                    gf.write("\n")
        except Exception:
            pass  # don't let logging failures mask the real error

        # Delete the working directory
        shutil.rmtree(fit_dir, ignore_errors=True)
        return FitResult(
            phase=phase, task_id=tid, stage=task.stage, terms=terms_str,
            terms_order=task.terms_order,
            n_sqs=len(task.sqs_list),
            sqs_names=[s.name for s in task.sqs_list],
            endmember_svib=task.endmember_svib,
            svib_names=sorted(task.svib_include),
            energy_errors=e_err or [],
            max_energy_error=max(e_err) if e_err else float("inf"),
            svib_errors=s_err, max_svib_error=max(s_err) if s_err else None,
            tdb_path=None, success=False, error_msg=msg)

    try:
        phase_dir = setup_fit_directory(task)

        # Run sqs2tdb -fit
        log = fit_dir / "sqs2tdb.log"
        rc = run_cmd(["sqs2tdb", "-fit"], phase_dir, log, timeout=1200)
        if rc != 0:
            return _fail(f"sqs2tdb exited with rc={rc}")

        # ── Evaluate energy fit ──────────────────────────────────
        fe = phase_dir / "fit_energy.out"
        if not fe.is_file():
            return _fail("fit_energy.out not produced")
        energy_errors = parse_fit_file(fe, col_idx=4)
        if not energy_errors:
            return _fail("fit_energy.out has no parseable errors")
        max_ee = max(energy_errors)

        # Overfit detection: if ALL errors are exactly 0 for a single-sublattice
        # phase, the fit is exactly determined (n_data == n_params). This means
        # the model passes through every point with zero residual — no predictive
        # value. Reject these.
        if phase in ("FCC_A1", "BCC_A2", "HCP_A3"):
            if all(e < 1e-10 for e in energy_errors):
                return _fail(
                    "overfit: all energy errors are 0 (exact fit, no DOF)",
                    e_err=energy_errors)

        if max_ee > task.energy_cutoff:
            return _fail(
                f"energy error {max_ee:.4f} > cutoff {task.energy_cutoff}",
                e_err=energy_errors)

        # ── Evaluate svib fit (stage 2 only) ─────────────────────
        svib_errors = None
        max_se = None
        if task.stage == 2 and (task.endmember_svib or task.svib_include):
            fs = phase_dir / "fit_svib_ht.out"
            if fs.is_file():
                svib_errors = parse_fit_file(fs, col_idx=4)
                if svib_errors:
                    max_se = max(svib_errors)
                    if max_se > task.svib_cutoff:
                        return _fail(
                            f"svib error {max_se:.4f} > cutoff {task.svib_cutoff}",
                            e_err=energy_errors, s_err=svib_errors)

        # ── Check TDB produced ───────────────────────────────────
        tdb = phase_dir / f"{phase}.tdb"
        if not tdb.is_file():
            return _fail("TDB file not generated", e_err=energy_errors,
                         s_err=svib_errors)

        # ── SUCCESS: strip everything except the TDB and logs ────
        # Keep: PHASE.tdb, fit_*.out, sqs2tdb.log, terms.in, species.in
        keep_names = {
            f"{phase}.tdb", "fit_energy.out", "fit_svib_ht.out",
            "cv_energy.out", "cv_svib_ht.out", "allparam.out",
            "terms.in", "species.in", "mult.in",
        }
        for item in phase_dir.iterdir():
            if item.name not in keep_names:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)

        return FitResult(
            phase=phase, task_id=tid, stage=task.stage, terms=terms_str,
            terms_order=task.terms_order,
            n_sqs=len(task.sqs_list),
            sqs_names=[s.name for s in task.sqs_list],
            endmember_svib=task.endmember_svib,
            svib_names=sorted(task.svib_include),
            energy_errors=energy_errors, max_energy_error=max_ee,
            svib_errors=svib_errors, max_svib_error=max_se,
            tdb_path=str(tdb), success=True, error_msg=None)

    except Exception as exc:
        shutil.rmtree(fit_dir, ignore_errors=True)
        return _fail(f"Exception: {exc}")


# ====================================================================
# Task generation
# ====================================================================

def gen_stage1_tasks(
    phase: str, endmembers: List[Path], sqs_list: List[SQSData],
    work_root: Path, el1: str, el2: str,
    min_sqs: int, max_sqs: int,
    energy_cutoff: float, svib_cutoff: float,
    start_id: int = 0
) -> List[FitTask]:
    """
    Stage 1: energy-only fits.
    Iterate: SQS subsets (size min_sqs..max_sqs) × terms (2,0 / 2,1 / 2,2).
    All svib_ht files are removed.
    """
    tasks = []
    tid = start_id

    if phase == "SIGMA_D8B":
        # SIGMA: endmembers only, fixed terms, single task
        tasks.append(FitTask(
            phase=phase, task_id=tid, work_root=work_root,
            endmembers=endmembers, sqs_list=[],
            terms_order=0,  # ignored for SIGMA
            svib_include=set(), endmember_svib=False,
            el1=el1, el2=el2,
            energy_cutoff=energy_cutoff, svib_cutoff=svib_cutoff,
            stage=1))
        return tasks

    cap = min(max_sqs, len(sqs_list))
    if cap < min_sqs:
        return tasks

    for n in range(min_sqs, cap + 1):
        for combo in itertools.combinations(sqs_list, n):
            for tord in BINARY_TERMS_OPTIONS:
                # Parameter count for single-sublattice binary:
                #   "1,0" contributes 2 params (one per endmember)
                #   "2,X" contributes X+1 params (L0, L1, ..., LX)
                #   Total = X + 3
                # Data points = 2 endmembers + n mixing SQS = n + 2
                #
                # MUST have n_data > n_params (strictly greater), otherwise
                # the system is exactly determined → zero residual → overfit.
                # e.g. 2,2 with 3 SQS: 5 data = 5 params → exact fit.
                n_params = tord + 3   # 2 (endmembers) + X+1 (L-params)
                n_data = 2 + n        # endmembers + SQS
                if n_data <= n_params:
                    continue  # underdetermined or exactly determined (overfit)

                tasks.append(FitTask(
                    phase=phase, task_id=tid, work_root=work_root,
                    endmembers=endmembers, sqs_list=list(combo),
                    terms_order=tord,
                    svib_include=set(), endmember_svib=False,
                    el1=el1, el2=el2,
                    energy_cutoff=energy_cutoff,
                    svib_cutoff=svib_cutoff,
                    stage=1))
                tid += 1

    return tasks


def gen_stage2_tasks(
    phase: str, endmembers: List[Path],
    stage1_winners: List[FitResult],
    sqs_pool: List[SQSData],
    work_root: Path, el1: str, el2: str,
    energy_cutoff: float, svib_cutoff: float,
    start_id: int = 0
) -> List[FitTask]:
    """
    Stage 2: add svib_ht to each surviving Stage-1 fit.
    Endmember svib_ht is ALWAYS included.
    SQS svib_ht subsets are explored combinatorially:
      all included, exclude-1, exclude-2, …, down to endmembers-only.
    """
    tasks = []
    tid = start_id

    sqs_by_name = {s.name: s for s in sqs_pool}

    # Check endmembers have svib_ht
    em_have_svib = all(find_svib_ht(em) is not None for em in endmembers)
    if not em_have_svib:
        print(f"    [{phase}] Not all endmembers have svib_ht — skipping Stage 2")
        return tasks

    for win in stage1_winners:
        # Reconstruct the SQS list for this winner
        combo = [sqs_by_name[n] for n in win.sqs_names if n in sqs_by_name]
        sqs_with_svib = [s for s in combo if s.has_svib]

        # Generate svib subsets: from all SQS-svib down to none (endmembers only)
        for k in range(len(sqs_with_svib), -1, -1):
            for svib_sub in itertools.combinations(sqs_with_svib, k):
                tasks.append(FitTask(
                    phase=phase, task_id=tid, work_root=work_root,
                    endmembers=endmembers, sqs_list=combo,
                    terms_order=win.terms_order,
                    svib_include={s.name for s in svib_sub},
                    endmember_svib=True,
                    el1=el1, el2=el2,
                    energy_cutoff=energy_cutoff,
                    svib_cutoff=svib_cutoff,
                    stage=2))
                tid += 1

    return tasks


# ====================================================================
# Parallel runner
# ====================================================================

def run_tasks_parallel(tasks: List[FitTask], n_workers: int,
                       label: str,
                       max_successes: Optional[int] = None) -> List[FitResult]:
    """
    Submit `tasks` to a ProcessPoolExecutor and wait for completion.

    First-feasible mode: when `max_successes` is set, the runner stops
    submitting / cancels pending tasks once that many successful fits
    are observed. Already-running tasks are allowed to finish (we cannot
    forcibly kill running futures). This is the pragmatic CS-style
    short-circuit suggested by Maguire et al. (2025) for the binary case
    where the cost of fitting a GPC surrogate isn't justified.
    """
    if not tasks:
        print(f"    No {label} tasks to run.")
        return []

    print(f"    {label}: {len(tasks)} tasks, {n_workers} workers"
          + (f", stop after {max_successes} successes" if max_successes else ""))
    results: List[FitResult] = []
    t0 = time.time()
    done = 0
    ok = 0
    early_stopped = False

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(do_one_fit, t): t for t in tasks}
        try:
            from concurrent.futures import CancelledError
        except ImportError:
            CancelledError = Exception  # type: ignore

        for fut in as_completed(futs):
            try:
                r = fut.result()
            except CancelledError:
                # A task we cancelled below — don't count it, don't record.
                continue
            results.append(r)
            done += 1
            if r.success:
                ok += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"      {done}/{len(tasks)}  ok={ok}  "
                      f"{rate:.1f} fits/s  ETA {eta/60:.1f}m")

            if (max_successes is not None
                    and ok >= max_successes
                    and not early_stopped):
                # Cancel everything still in the queue. Tasks currently
                # executing on a worker cannot be cancelled and will run
                # to completion; their results are still recorded above.
                cancelled = 0
                for f in futs:
                    if not f.done() and f.cancel():
                        cancelled += 1
                early_stopped = True
                print(f"      Reached {max_successes} successes; "
                      f"cancelled {cancelled} pending tasks")

    elapsed = time.time() - t0
    suffix = " (stopped early)" if early_stopped else ""
    print(f"    {label} done: {ok}/{len(tasks)} passed in {elapsed:.0f}s"
          f"{suffix}\n")
    return results


# ====================================================================
# Workdir naming
# ====================================================================

def make_workdir(base: str) -> Path:
    """Create <base>_0, _1, _2, … avoiding collisions."""
    i = 0
    while True:
        p = Path(f"{base}_{i}")
        if not p.exists():
            p.mkdir(parents=True)
            return p.resolve()
        i += 1


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Automated sqs2tdb binary pipeline")
    ap.add_argument("--endmembers-yaml", required=True,
                    help="YAML from select_endmembers.py")
    ap.add_argument("--data-roots", required=True,
                    help="Comma-separated directories to scan for SQS data")
    ap.add_argument("--workdir-prefix", default=None,
                    help="Base name for work directory (default: ElA-ElB_automate)")
    ap.add_argument("--min-sqs", type=int, default=MIN_SQS_DEFAULT)
    ap.add_argument("--max-sqs", type=int, default=MAX_SQS_DEFAULT)
    ap.add_argument("--energy-cutoff", type=float, default=0.10,
                    help="Max |error| in fit_energy.out col 5 (eV, default 0.10)")
    ap.add_argument("--svib-cutoff", type=float, default=10.0,
                    help="Max |error| in fit_svib_ht.out col 5 (default 10.0)")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--skip-svib", action="store_true",
                    help="Skip Stage 2 (svib_ht fitting)")
    ap.add_argument("--phases", default=None,
                    help="Comma-separated phase list (default: all in YAML)")
    ap.add_argument("--max-survivors-per-phase", type=int, default=None,
                    help="First-feasible mode: stop a stage once N successful "
                         "fits are found per phase. Pending tasks are "
                         "cancelled. Drastically reduces compute when the "
                         "configuration grid is large. Default: unlimited.")
    ap.add_argument("--oszicar-min-score", type=float, default=0.0,
                    help="Skip SQS whose OSZICAR convergence score (0-100) "
                         "is below this threshold. Uses "
                         "../oszicar_convergence_scorer.py. SQS with no "
                         "OSZICAR found are accepted. Default 0 (disabled).")
    ap.add_argument("--target-dir", default=None,
                    help="Directory of consensus-target JSONs produced by "
                         "tdb_corpus/reverse_engineer_targets.ipynb. When "
                         "supplied, SQS whose DFT formation energy (same-"
                         "phase pure references) deviates from the RK-"
                         "excess consensus target by more than "
                         "--target-tol-sigma standard deviations are "
                         "REJECTED before fitting. Files are picked up by "
                         "name: <A>_<B>_<phase>_consensus.json. Phases for "
                         "which no JSON is found run without a gate.")
    ap.add_argument("--target-tol-sigma", type=float, default=3.0,
                    help="Tolerance in units of consensus gate sigma "
                         "(sqrt(sigma_TDB^2 + dft_noise_floor^2)) "
                         "for the --target-dir gate. Default 3.0.")
    ap.add_argument("--target-dft-noise-floor", type=float, default=0.005,
                    help="DFT/SQS-vs-random-alloy error floor in eV/atom "
                         "added in quadrature to the cross-TDB sigma. "
                         "Default 5 meV/atom (sane lower bound for "
                         "converged VASP + reasonably sized SQS).")
    ap.add_argument("--keep-energy-off", action="store_true",
                    help="Do NOT auto-rename energy.off files to energy "
                         "during discovery (the rename is on by default so "
                         "ATAT-parked energies are brought back into the "
                         "fitting pool).")
    args = ap.parse_args()

    # ── Load inputs ──────────────────────────────────────────────
    with open(args.endmembers_yaml) as f:
        ydata = yaml.safe_load(f)
    binary = ydata["binary"]
    el1, el2 = binary.split("-")

    data_roots = [Path(r.strip()).resolve() for r in args.data_roots.split(",")]

    prefix = args.workdir_prefix or f"{el1}-{el2}_automate"
    workdir = make_workdir(prefix)

    requested_phases = (
        [p.strip() for p in args.phases.split(",")]
        if args.phases else ["FCC_A1", "BCC_A2", "HCP_A3", "SIGMA_D8B"])

    print(f"\n{'='*70}")
    print(f"  sqs2tdb Pipeline — {el1}-{el2} binary")
    print(f"{'='*70}")
    print(f"  Workdir       : {workdir}")
    print(f"  Energy cutoff : {args.energy_cutoff} eV")
    print(f"  Svib cutoff   : {args.svib_cutoff}")
    print(f"  Workers       : {args.n_workers}")
    print(f"  Min/Max SQS   : {args.min_sqs} / {args.max_sqs}")
    print(f"  Skip svib     : {args.skip_svib}")
    print(f"  Max survivors : "
          f"{args.max_survivors_per_phase if args.max_survivors_per_phase else 'unlimited'}")
    print(f"  OSZICAR min   : "
          f"{args.oszicar_min_score if args.oszicar_min_score > 0 else 'disabled'}")
    print(f"  Rename .off   : {not args.keep_energy_off}")
    print(f"  Target gate   : "
          f"{args.target_dir + f' (tol {args.target_tol_sigma}σ)' if args.target_dir else 'disabled'}")
    print(f"  Phases        : {', '.join(requested_phases)}")
    print(f"{'='*70}\n")

    # ── Load consensus-target gates (one per phase) ───────────────────
    target_gates: Dict[str, object] = {}
    if args.target_dir:
        # Make the gate module importable from anywhere this script lives.
        # tdb_corpus/sqs_target_gate.py is the canonical location; we
        # tolerate it being copied/installed elsewhere via PYTHONPATH.
        gate_path_candidates = [
            Path(__file__).resolve().parent.parent / "tdb_corpus",
            Path.cwd() / "tdb_corpus",
        ]
        for cand in gate_path_candidates:
            if (cand / "sqs_target_gate.py").is_file():
                sys.path.insert(0, str(cand))
                break
        try:
            from sqs_target_gate import load_target_dir
            target_gates = load_target_dir(
                Path(args.target_dir),
                [el1, el2],
                requested_phases,
                dft_noise_floor_ev=args.target_dft_noise_floor,
            )
        except ImportError as exc:
            print(f"  WARNING: --target-dir set but sqs_target_gate not "
                  f"importable ({exc}); gate disabled.")
        if target_gates:
            print(f"  Loaded consensus gates for: "
                  f"{sorted(target_gates.keys())}")
            for ph, gate in sorted(target_gates.items()):
                xr = gate.rk_E.x_range
                print(f"    [{ph}] {gate.system[0]}-{gate.system[1]}  "
                      f"x-range [{xr[0]:.2f}, {xr[1]:.2f}]  "
                      f"from {Path(gate.source_path).name}")
        else:
            print(f"  WARNING: no consensus JSONs matched the requested "
                  f"system+phase set in {args.target_dir}.")
        print()

    all_results: Dict[str, Dict[str, List]] = {}
    global_tid = 0

    for phase in requested_phases:
        if phase not in ydata:
            print(f"  [{phase}] No endmembers in YAML — skipping\n")
            continue

        print(f"{'='*70}")
        print(f"  Phase: {phase}")
        print(f"{'='*70}")

        # ── Resolve endmembers ───────────────────────────────────
        em_data = ydata[phase]
        if phase == "SIGMA_D8B":
            if isinstance(em_data, list):
                endmembers = [Path(p) for p in em_data]
            elif isinstance(em_data, dict) and "ALL" in em_data:
                endmembers = [Path(p) for p in em_data["ALL"]]
            else:
                print(f"    Cannot parse SIGMA endmembers — skipping\n")
                continue
        else:
            endmembers = [Path(em_data[el1]), Path(em_data[el2])]

        # Validate endmembers exist and have mandatory files. Apply the
        # same energy.off rename we use for mixing SQS. Endmembers are
        # structural anchors and we don't drop them on OSZICAR score
        # alone — only warn — because losing an endmember kills the phase.
        bad_ems = []
        for em in endmembers:
            if not em.is_dir():
                print(f"    WARNING: endmember path does not exist: {em}")
                bad_ems.append(em)
                continue
            if not args.keep_energy_off:
                if maybe_rename_energy_off(em):
                    print(f"    RENAMED energy.off -> energy in "
                          f"endmember {em.name}")
            ok, missing = has_all_files(em)
            if not ok:
                print(f"    WARNING: endmember {em.name} missing: "
                      f"{', '.join(missing)}")
                bad_ems.append(em)
                continue
            if read_float_file(em / "energy") is None:
                print(f"    WARNING: endmember {em.name} energy is empty / "
                      f"unparseable")
                bad_ems.append(em)
                continue
            if args.oszicar_min_score > 0.0:
                score = oszicar_score(em)
                if score is not None and score < args.oszicar_min_score:
                    print(f"    WARNING: endmember {em.name} OSZICAR score "
                          f"{score:.1f} < {args.oszicar_min_score} "
                          f"(kept anyway — endmember is structural anchor)")
        if bad_ems:
            print(f"    Continuing with available endmembers...\n")
            endmembers = [em for em in endmembers if em not in bad_ems]
            if len(endmembers) < 2 and phase != "SIGMA_D8B":
                print(f"    Not enough valid endmembers — skipping\n")
                continue

        print(f"    Endmembers: {len(endmembers)}")
        for em in endmembers:
            sv = "svib=YES" if find_svib_ht(em) else "svib=NO"
            print(f"      {em.name}  ({sv})")

        # ── Build per-atom endmember energies for the target gate ──
        # Same-phase formation energies: lattice stability cancels with
        # the RK-excess target, so the comparison is apples-to-apples.
        gate_for_phase = target_gates.get(phase)
        endmember_per_atom: Optional[Dict[str, float]] = None
        if gate_for_phase is not None and phase != "SIGMA_D8B":
            try:
                from sqs_target_gate import per_atom_energy as _pae
                endmember_per_atom = {}
                for em, el in zip(endmembers, (el1, el2)):
                    pa = _pae(em / "energy", em / "str.out")
                    if pa is None:
                        endmember_per_atom = None
                        break
                    endmember_per_atom[el.upper()] = pa[0]
            except Exception as exc:
                print(f"    WARNING: could not compute endmember per-atom "
                      f"energies ({exc}); gate disabled for this phase.")
                endmember_per_atom = None
            if endmember_per_atom:
                refs = ", ".join(
                    f"{el}={e:.4f}" for el, e in endmember_per_atom.items()
                )
                print(f"    Target gate enabled: per-atom refs eV/atom "
                      f"{{{refs}}}")
            else:
                print(f"    Target gate skipped for this phase "
                      f"(endmember energies unreadable).")
                gate_for_phase = None

        # ── Discover mixing SQS ──────────────────────────────────
        sqs_list = discover_sqs(
            data_roots, phase, el1, el2,
            rename_energy_off=not args.keep_energy_off,
            oszicar_min_score=args.oszicar_min_score,
            target_gate=gate_for_phase,
            endmember_per_atom=endmember_per_atom,
            target_tol_sigma=args.target_tol_sigma,
        )
        print(f"    Mixing SQS : {len(sqs_list)}")
        for s in sqs_list:
            sv = "svib=YES" if s.has_svib else "svib=NO"
            print(f"      lev={s.level}  x({el1})={s.x1:.4f}  {sv}  {s.name}")

        if phase != "SIGMA_D8B" and len(sqs_list) < args.min_sqs:
            print(f"    Only {len(sqs_list)} SQS found (need {args.min_sqs}) — skipping\n")
            continue

        # ── Stage 1: energy-only ─────────────────────────────────
        print(f"\n  ── Stage 1: energy-only fits ──")
        s1_tasks = gen_stage1_tasks(
            phase, endmembers, sqs_list, workdir,
            el1, el2, args.min_sqs, args.max_sqs,
            args.energy_cutoff, args.svib_cutoff,
            start_id=global_tid)
        global_tid += len(s1_tasks)

        s1_results = run_tasks_parallel(
            s1_tasks, args.n_workers, "Stage 1",
            max_successes=args.max_survivors_per_phase,
        )
        s1_pass = [r for r in s1_results if r.success]
        print(f"    Stage 1 survivors: {len(s1_pass)}")

        phase_results = {"stage1": [asdict(r) for r in s1_results]}

        # ── Stage 2: svib_ht ─────────────────────────────────────
        # Stage 2 also runs for SIGMA_D8B even though SIGMA has no mixing
        # SQS in a binary — gen_stage2_tasks naturally produces exactly
        # one task (empty SQS-svib subset, endmember_svib=True). This
        # ensures SIGMA's free-energy zero-point is normalized the same
        # way as FCC/BCC/HCP for Stage-3 phase-fraction scoring;
        # otherwise SIGMA endmembers are referenced to E only while the
        # other phases include the vibrational contribution, biasing
        # phase stability comparisons.
        if not args.skip_svib and s1_pass:
            print(f"\n  ── Stage 2: adding svib_ht ──")
            s2_tasks = gen_stage2_tasks(
                phase, endmembers, s1_pass, sqs_list, workdir,
                el1, el2, args.energy_cutoff, args.svib_cutoff,
                start_id=global_tid)
            global_tid += len(s2_tasks)

            s2_results = run_tasks_parallel(
                s2_tasks, args.n_workers, "Stage 2",
                max_successes=args.max_survivors_per_phase,
            )
            s2_pass = [r for r in s2_results if r.success]
            print(f"    Stage 2 survivors: {len(s2_pass)}")
            phase_results["stage2"] = [asdict(r) for r in s2_results]
        else:
            if not args.skip_svib:
                print(f"    Skipping Stage 2 (no Stage 1 survivors)")

        all_results[phase] = phase_results
        print()

    # ── Save results ─────────────────────────────────────────────
    out_file = workdir / "fit_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Build manifest of surviving TDB paths per phase ──────────
    # Stage 3 (scoring) consumes this to enumerate cross-phase combos.
    # For each phase, prefer Stage 2 survivors if available, else Stage 1.
    manifest: Dict[str, List[str]] = {}
    for phase, res in all_results.items():
        winners = []
        # Prefer best available stage
        for stage_key in ("stage2", "stage1"):
            stage_results = res.get(stage_key, [])
            for r in stage_results:
                if r.get("success") and r.get("tdb_path"):
                    winners.append(r["tdb_path"])
            if winners:
                break  # use this stage
        manifest[phase] = winners

    manifest_file = workdir / "tdb_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({
            "binary": f"{el1}-{el2}",
            "workdir": str(workdir),
            "phases": manifest,
        }, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Results  : {out_file}")
    print(f"  Manifest : {manifest_file}")
    print(f"  Workdir  : {workdir}")
    print()

    for phase, res in all_results.items():
        s1_ok = sum(1 for r in res.get("stage1", []) if r.get("success"))
        s2_ok = sum(1 for r in res.get("stage2", []) if r.get("success"))
        s1_tot = len(res.get("stage1", []))
        s2_tot = len(res.get("stage2", []))
        n_tdb = len(manifest.get(phase, []))
        print(f"    {phase:12s}  Stage1: {s1_ok}/{s1_tot}  "
              f"Stage2: {s2_ok}/{s2_tot}  TDBs for scoring: {n_tdb}")

    total_combos = 1
    for tdbs in manifest.values():
        if tdbs:
            total_combos *= len(tdbs)
    print(f"\n    Cross-phase combinations for Stage 3: {total_combos}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()