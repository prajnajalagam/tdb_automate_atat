#!/usr/bin/env python3
"""
ENCUT + KPPRA convergence testing and selection.

Procedure (from the design spec)
--------------------------------
1. Extract ENMAX from every POTCAR involved; take MAX_ENMAX.
2. KPPRA sweep at a fixed ENCUT = 1.125 x MAX_ENMAX over 4000..10000
   (step 1000). Pick the converged KPPRA.
3. ENCUT sweep at the converged KPPRA over 1.00..1.25 x MAX_ENMAX
   (5 points). Pick the converged ENCUT.

"Converged" = the smallest setting whose total energy per atom -- and every
larger setting's -- lies within `tol_ev` (default 1 meV/atom) of the
highest-setting reference. Requiring the whole tail to be within tolerance
(not just the single point) rejects accidental crossings.

The pure selection logic (select_converged) is unit-tested; the VASP-driving
orchestration (run_static_point / run_*_sweep) only runs on a real node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import potcar
import runner
from strfile import read_structure
from vaspwrap import build_vasp_wrap


def _count_atoms(str_out) -> "int | None":
    """Atom count for the NCORE/KPAR guard; None if unreadable."""
    try:
        from strfile import read_structure
        return len(read_structure(str_out).atoms) or None
    except OSError:
        return None
from phases import DLMConfig

# Default convergence tolerance: 1 meV/atom.
# Successive-step tolerance (eV/atom). 2026-07-16 user decision, backed
# by common practice (JARVIS-DFT-style protocols): a setting is
# converged when the energy change from the PREVIOUS step and to the
# NEXT step are both below tol — the next point is the deliberately
# "not needed" confirmation. 0.1 meV/atom is stricter than the usual
# 1 meV/atom total-energy criterion; appropriate because CALPHAD mixing
# energies subtract totals.
DEFAULT_TOL_EV = 0.0001


# ---------------------------------------------------------------------------
# Pure logic (testable without VASP)
# ---------------------------------------------------------------------------

def count_atoms(str_out: Path) -> int:
    """Number of atom lines in an ATAT structure file."""
    return len(read_structure(str_out).atoms)


def read_energy(calc_dir: Path) -> Optional[float]:
    """Total-cell energy (eV) from an ATAT `energy` file, or None."""
    f = Path(calc_dir) / "energy"
    try:
        return float(f.read_text().strip().replace("D", "E").replace("d", "E"))
    except Exception:
        return None


def energy_per_atom(calc_dir: Path) -> Optional[float]:
    e = read_energy(calc_dir)
    if e is None:
        return None
    n = count_atoms(Path(calc_dir) / "str.out")
    if n <= 0:
        return None
    return e / n


@dataclass
class ConvergenceResult:
    parameter: str                 # "ENCUT" or "KPPRA"
    settings: List[int]
    energy_per_atom: List[Optional[float]]
    chosen: int
    converged: bool
    tol_ev: float
    reference: int                 # the setting used as the converged target

    def table(self) -> str:
        ref_e = None
        for s, e in zip(self.settings, self.energy_per_atom):
            if s == self.reference:
                ref_e = e
        lines = [f"  {self.parameter} convergence "
                 f"(tol {self.tol_ev*1e3:.1f} meV/atom, "
                 f"{'CONVERGED' if self.converged else 'NOT CONVERGED'}):"]
        for s, e in zip(self.settings, self.energy_per_atom):
            if e is None:
                lines.append(f"    {s:>7}  <missing energy>")
                continue
            d = "" if ref_e is None else f"  d={1e3*(e-ref_e):+7.2f} meV/atom"
            mark = "  <-- chosen" if s == self.chosen else ""
            lines.append(f"    {s:>7}  {e:12.5f} eV/atom{d}{mark}")
        return "\n".join(lines)


def select_converged(settings: List[int],
                     e_per_atom: List[Optional[float]],
                     tol_ev: float = DEFAULT_TOL_EV
                     ) -> Tuple[int, bool, int]:
    """Successive-difference criterion with a confirming point above.

    Chosen = the smallest setting S_i (ascending order) such that BOTH
      |E(S_i)   - E(S_{i-1})| < tol_ev   (arrived on a plateau), and
      |E(S_{i+1}) - E(S_i)|  < tol_ev    (the next point CONFIRMS it —
                                          the deliberately "not needed"
                                          extra run).
    This replaces the old compare-to-highest-reference rule, which
    declared victory whenever the top of the grid agreed with itself —
    on the 2026-07-16 ENCUT sweep the successive steps were still
    moving by 0.4-1.4 meV/atom at the ceiling, i.e. NOT converged, and
    the reference rule couldn't see it. (Old rule on the same KPPRA
    data picked 4000; this rule picks 7000, matching manual analysis.)

    Returns (chosen_setting, converged_flag, reference_setting) where
    reference is the confirming point S_{i+1} when converged, else the
    highest usable setting. Falls back to the highest setting (flag
    False) when no triple satisfies the criterion.
    """
    pairs = [(s, e) for s, e in zip(settings, e_per_atom) if e is not None]
    if not pairs:
        ref = max(settings) if settings else 0
        return ref, False, ref
    pairs.sort(key=lambda p: p[0])  # ascending by setting

    for i in range(1, len(pairs) - 1):
        d_prev = abs(pairs[i][1] - pairs[i - 1][1])
        d_next = abs(pairs[i + 1][1] - pairs[i][1])
        if d_prev < tol_ev and d_next < tol_ev:
            return pairs[i][0], True, pairs[i + 1][0]
    ref_setting = pairs[-1][0]
    return ref_setting, False, ref_setting


# ---------------------------------------------------------------------------
# VASP-driving orchestration (real node only)
# ---------------------------------------------------------------------------

def run_static_point(src_sqs: Path,
                     dst: Path,
                     encut: int,
                     kppra: int,
                     dlm: Optional[DLMConfig] = None,
                     algo: str = "All",
                     env_bin: Optional[str] = None,
                     timeout: int = 7200,
                     cmd_prefix: str = "") -> Optional[float]:
    """Set up and run one static VASP point at (encut, kppra); return eV/atom.

    Copies str.out (and any POTCAR/species files) from src_sqs into dst, writes
    a static vasp.wrap with the requested ENCUT/KPPRA, runs runstruct_vasp, and
    reads back the energy. Returns None if the run produced no energy.

    cmd_prefix: the VASP launch command (e.g. "mpiexec -n 128"), passed to
    runstruct_vasp as trailing arguments. Without it, runstruct_vasp
    launches the MPI vasp binary bare and it dies before writing OSZICAR
    ("Problem running vasp command ... unable to open OSZICAR").
    """
    import shutil
    src_sqs = Path(src_sqs)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("str.out", "POTCAR", "species.in", "mult.in"):
        s = src_sqs / fn
        if s.is_file():
            shutil.copy2(s, dst / fn)

    natoms = _count_atoms(dst / "str.out")
    wrap = build_vasp_wrap("static", encut=encut, kppra=kppra,
                           dlm=dlm, algo=algo, natoms=natoms)
    (dst / "vasp.wrap").write_text(wrap)

    runner.run_logged(["runstruct_vasp"] + runner.split_prefix(cmd_prefix),
                      cwd=dst,
                      log=dst / "runstruct.log",
                      env_bin=env_bin, timeout=timeout, check=False)
    return energy_per_atom(dst)


def run_sweep(parameter: str,
              src_sqs: Path,
              sweep_root: Path,
              settings: List[int],
              fixed_other: int,
              dlm: Optional[DLMConfig] = None,
              algo: str = "All",
              tol_ev: float = DEFAULT_TOL_EV,
              env_bin: Optional[str] = None,
              timeout: int = 7200,
              cmd_prefix: str = "",
              extend_step: int = 0,
              extend_max: int = 0) -> ConvergenceResult:
    """Run a 1-D convergence sweep over `settings` for ENCUT or KPPRA.

    parameter   "ENCUT" or "KPPRA". The other parameter is held at
                `fixed_other`.
    extend_step when > 0, keep appending settings in steps of this size
                until select_converged succeeds (successive-difference
                + confirmation) or extend_max is reached. Used for the
                unbounded ENCUT sweep.
    """
    sweep_root = Path(sweep_root)
    settings = list(settings)

    def _point(val: int) -> Optional[float]:
        dst = sweep_root / f"{parameter.lower()}_{val}"
        if parameter == "ENCUT":
            encut, kppra = val, fixed_other
        else:
            encut, kppra = fixed_other, val
        return run_static_point(
            src_sqs, dst, encut=encut, kppra=kppra,
            dlm=dlm, algo=algo, env_bin=env_bin, timeout=timeout,
            cmd_prefix=cmd_prefix)

    e_pa: List[Optional[float]] = [_point(v) for v in settings]
    chosen, converged, reference = select_converged(settings, e_pa, tol_ev)

    # Adaptive extension (2026-07-16 user decision: NO ceiling on the
    # sweep — keep adding points until the successive-difference
    # criterion is met). extend_step > 0 enables it; extend_max is a
    # runaway guard, not a convergence ceiling: hitting it prints a
    # loud warning and falls back to the highest computed setting.
    while not converged and extend_step > 0 \
            and settings[-1] + extend_step <= extend_max:
        nxt = settings[-1] + extend_step
        settings.append(nxt)
        e_pa.append(_point(nxt))
        chosen, converged, reference = select_converged(settings, e_pa,
                                                        tol_ev)
    if not converged and extend_step > 0:
        print(f"    WARNING: {parameter} not converged to "
              f"{tol_ev*1e3:.2f} meV/atom even at {settings[-1]} "
              f"(guard {extend_max}); proceeding with the highest "
              f"setting — inspect the sweep table.")

    return ConvergenceResult(
        parameter=parameter, settings=list(settings), energy_per_atom=e_pa,
        chosen=chosen, converged=converged, tol_ev=tol_ev, reference=reference)


def converge_sqs(src_sqs: Path,
                 sweep_root: Path,
                 potcar_paths: List[Path],
                 dlm: Optional[DLMConfig] = None,
                 algo: str = "All",
                 tol_ev: float = DEFAULT_TOL_EV,
                 env_bin: Optional[str] = None,
                 timeout: int = 7200,
                 cmd_prefix: str = ""
                 ) -> Tuple[int, int, ConvergenceResult, ConvergenceResult]:
    """Full per-SQS convergence: KPPRA sweep first, then ENCUT sweep.

    Returns (chosen_encut, chosen_kppra, kppra_result, encut_result).
    """
    max_e = potcar.max_enmax(potcar_paths)
    probe_encut = potcar.kppra_probe_encut(max_e)
    kgrid = potcar.kppra_grid()
    egrid = potcar.encut_grid(max_e)

    kppra_res = run_sweep(
        "KPPRA", src_sqs, Path(sweep_root) / "kppra_sweep",
        settings=kgrid, fixed_other=probe_encut,
        dlm=dlm, algo=algo, tol_ev=tol_ev,
        env_bin=env_bin, timeout=timeout, cmd_prefix=cmd_prefix,
        extend_step=potcar.KPPRA_STEP,
        extend_max=potcar.KPPRA_EXTEND_MAX)
    chosen_kppra = kppra_res.chosen

    # ENCUT sweep has NO convergence ceiling (2026-07-16): the initial
    # 1.00-1.25 x ENMAX grid is only a starting mesh; the sweep keeps
    # climbing in grid-sized steps until successive steps agree to
    # tol_ev with a confirming point above. extend_max is a runaway
    # guard (default 3 x ENMAX), far above any physical need.
    egrid_step = max(10, egrid[1] - egrid[0]) if len(egrid) > 1 else 20
    encut_res = run_sweep(
        "ENCUT", src_sqs, Path(sweep_root) / "encut_sweep",
        settings=egrid, fixed_other=chosen_kppra,
        dlm=dlm, algo=algo, tol_ev=tol_ev,
        env_bin=env_bin, timeout=timeout, cmd_prefix=cmd_prefix,
        extend_step=egrid_step,
        extend_max=int(potcar.ENCUT_GUARD_FACTOR * max_e))
    chosen_encut = encut_res.chosen

    return chosen_encut, chosen_kppra, kppra_res, encut_res
