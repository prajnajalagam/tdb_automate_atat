#!/usr/bin/env python3
"""
OSZICAR Convergence Quality Scorer
===================================
Evaluates the convergence quality of a VASP OSZICAR file using statistical
analysis of both inner (electronic SCF) and outer (ionic) convergence behavior.

Produces a composite score in [0, 100] along with diagnostic breakdowns.

Author: Prajna / Claude collaboration
Target: Integration with sqs2tdb_pipeline or standalone QC checks on Pleiades
"""

import re
import sys
import json
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# ===========================================================================
# Data structures
# ===========================================================================
@dataclass
class InnerLoopData:
    """Data from a single ionic step's electronic (inner) SCF loop."""
    ionic_step: int
    n_steps: int
    dE: list  # dE from each DAV/RMM line
    E: list   # total energy at each electronic step
    rms: list  # RMS residual (if available)
    converged: bool = False
    final_dE: float = 0.0


@dataclass
class OuterLoopData:
    """Data from the ionic (outer) loop — the 'n F= ...' lines."""
    step_indices: list   # 1, 2, 3, ...
    energies: list       # F values
    d_E: list            # d E values from the F= lines
    mag: list            # magnetic moments (if present)


@dataclass
class ConvergenceReport:
    """Full convergence quality report."""
    filename: str
    # --- Completion flags ---
    run_completed: bool = True
    n_ionic_steps: int = 0
    n_electronic_steps_total: int = 0
    # --- Inner loop metrics ---
    inner_scores: list = field(default_factory=list)
    inner_composite: float = 0.0
    # --- Outer loop metrics ---
    outer_score: float = 0.0
    # --- Composite ---
    total_score: float = 0.0
    grade: str = ""
    diagnostics: dict = field(default_factory=dict)


# ===========================================================================
# VASP defaults (from VASP wiki, confirmed against VASP 5.x/6.x)
# ===========================================================================
VASP_DEFAULTS = {
    # Electronic convergence
    'EDIFF':   1e-4,    # eV, global break condition for electronic SC loop
    'NELM':    60,      # max electronic SC steps (VASP wiki: default 60)
    'NELMIN':  2,       # min electronic SC steps
    'NELMDL': -5,       # non-selfconsistent steps at start (negative = auto)

    # Ionic relaxation
    'NSW':     0,       # max ionic steps (0 = static)
    'EDIFFG':  None,    # None means use EDIFF*10 (VASP default behavior)
    'IBRION': -1,       # -1 for NSW=0|1, 0 otherwise. -1=static, 0=MD,
                        # 1=quasi-Newton(RMM-DIIS), 2=CG, 3=damped MD
    'ISIF':    2,       # 0-7: what to relax. 2=positions only, 3=pos+cell
    'POTIM':   0.5,     # time step (MD) or step width scaling (relaxation)

    # Smearing
    'ISMEAR':  1,       # -5=tet, -1=Fermi, 0=Gaussian, 1=MP(1st order)
    'SIGMA':   0.2,     # smearing width in eV

    # Precision / algorithm
    'PREC':    'Normal',
    'ALGO':    'Normal',  # Normal=Davidson, Fast=RMM, VeryFast=RMM-DIIS, All
    'ISPIN':   1,         # 1=non-spin-polarized, 2=spin-polarized
}


def parse_incar(filepath: str) -> dict:
    """
    Parse a VASP INCAR file into a dictionary of tag -> value.

    Handles:
    - Comments (# and !)
    - Multiple tags per line separated by ';'
    - Fortran-style values (.TRUE., .FALSE., 1E-4, etc.)

    Returns a dict with uppercase keys and parsed values (float/int/str/bool).
    """
    params = {}
    if not Path(filepath).exists():
        return params

    with open(filepath, 'r') as fh:
        for line in fh:
            # Strip comments (both # and ! style)
            line = re.split(r'[#!]', line)[0].strip()
            if not line:
                continue

            # Split on semicolons for multiple tags per line
            for segment in line.split(';'):
                segment = segment.strip()
                if '=' not in segment:
                    continue
                tag, _, val_str = segment.partition('=')
                tag = tag.strip().upper()
                val_str = val_str.strip()

                if not val_str:
                    continue

                # Parse value
                params[tag] = _parse_incar_value(val_str)

    return params


def _parse_incar_value(val_str: str):
    """Parse a single INCAR value string into a Python type."""
    v = val_str.strip()

    # Boolean
    if v.upper() in ('.TRUE.', 'TRUE', 'T'):
        return True
    if v.upper() in ('.FALSE.', 'FALSE', 'F'):
        return False

    # Try integer
    try:
        return int(v)
    except ValueError:
        pass

    # Try float (handles Fortran 1.0E-4 etc.)
    try:
        return float(v)
    except ValueError:
        pass

    # Array of values (e.g., MAGMOM = 32*5.0)
    # Don't try to fully parse these — return as string
    return v


def resolve_vasp_params(incar_params: dict) -> dict:
    """
    Resolve actual VASP parameters by merging INCAR values with defaults.

    Applies VASP's conditional default logic:
    - EDIFFG defaults to EDIFF * 10 if not set
    - IBRION defaults depend on NSW
    - NELM default is 60
    - etc.

    Returns a complete parameter dict with all relevant tags resolved.
    """
    p = {}

    # Start with defaults
    for key, default in VASP_DEFAULTS.items():
        p[key] = incar_params.get(key, default)

    # --- Conditional defaults ---

    # EDIFF
    ediff = p['EDIFF']
    if ediff is None or ediff <= 0:
        ediff = 1e-4
    p['EDIFF'] = ediff

    # EDIFFG: if not set, default is EDIFF * 10
    if p['EDIFFG'] is None:
        p['EDIFFG'] = ediff * 10
        p['_ediffg_source'] = 'default (EDIFF * 10)'
    else:
        p['_ediffg_source'] = 'INCAR'

    # IBRION: default depends on NSW
    if 'IBRION' not in incar_params:
        if p['NSW'] in (0, 1):
            p['IBRION'] = -1
        else:
            p['IBRION'] = 0
        p['_ibrion_source'] = 'default (based on NSW)'
    else:
        p['_ibrion_source'] = 'INCAR'

    # NELM
    if p['NELM'] is None or p['NELM'] <= 0:
        p['NELM'] = 60

    # Determine convergence mode
    if p['EDIFFG'] < 0:
        p['_ionic_conv_mode'] = 'force'
        p['_ionic_conv_threshold'] = abs(p['EDIFFG'])
        p['_ionic_conv_unit'] = 'eV/Å'
    elif p['EDIFFG'] > 0:
        p['_ionic_conv_mode'] = 'energy'
        p['_ionic_conv_threshold'] = p['EDIFFG']
        p['_ionic_conv_unit'] = 'eV'
    else:
        # EDIFFG = 0 means run NSW steps with no convergence check
        p['_ionic_conv_mode'] = 'none'
        p['_ionic_conv_threshold'] = None
        p['_ionic_conv_unit'] = None

    return p


# ===========================================================================
# Parser
# ===========================================================================
def parse_oszicar(filepath: str) -> tuple[list[InnerLoopData], OuterLoopData]:
    """
    Parse an OSZICAR file into structured inner-loop and outer-loop data.

    Returns
    -------
    inner_loops : list[InnerLoopData]
        One entry per ionic step, containing all electronic SCF steps.
    outer_loop : OuterLoopData
        Aggregated ionic-step data from the 'n F= ...' lines.
    """
    # Regex for DAV/RMM electronic step lines
    # Format: DAV:   N    E              dE             d_eps     ncg     rms      rms(c)
    # Note: ncg can be glued to d_eps (no space), so we parse carefully
    inner_re = re.compile(
        r'^\s*(DAV|RMM):\s*(\d+)\s+'        # algorithm and step number
        r'([-\d.E+]+)\s+'                    # total energy E
        r'([-\d.E+]+)\s+'                    # dE
        r'([-\d.E+]+)'                       # d_eps (may be glued to ncg)
        r'.*?\s+([\d.E+-]+)\s*'              # rms (second-to-last or last float)
        , re.IGNORECASE
    )

    # Simpler, more robust inner-step regex
    inner_re2 = re.compile(
        r'^\s*(DAV|RMM):\s*(\d+)\s+'
        r'([-\d.]+E[+-]?\d+)\s+'             # E
        r'([-\d.]+E[+-]?\d+)\s+'             # dE
        r'([-\d.]+E[+-]?\d+)'                # d_eps (possibly glued to ncg)
    )

    # Regex for ionic step summary lines: "  n F= ... d E = ... mag= ..."
    outer_re = re.compile(
        r'^\s*(\d+)\s+F=\s*([-\d.E+]+)\s+'
        r'E0=\s*([-\d.E+]+)\s+'
        r'd\s*E\s*=\s*([-\d.E+]+)'
        r'(?:\s+mag=\s*([-\d.]+))?'
    )

    inner_loops: list[InnerLoopData] = []
    outer_steps = []
    outer_energies = []
    outer_dE = []
    outer_mag = []

    current_inner_E = []
    current_inner_dE = []
    current_inner_rms = []
    current_ionic_step = 1

    with open(filepath, 'r') as fh:
        for line in fh:
            # Skip header lines
            if line.strip().startswith('N ') or line.strip().startswith('N\t'):
                # If we have accumulated inner data, save it before reset
                if current_inner_dE:
                    inner_loops.append(InnerLoopData(
                        ionic_step=current_ionic_step,
                        n_steps=len(current_inner_dE),
                        dE=current_inner_dE.copy(),
                        E=current_inner_E.copy(),
                        rms=current_inner_rms.copy(),
                    ))
                    current_ionic_step += 1
                current_inner_E = []
                current_inner_dE = []
                current_inner_rms = []
                continue

            # Try matching an electronic step
            m = inner_re2.match(line)
            if m:
                E_val = float(m.group(3))
                dE_val = float(m.group(4))
                current_inner_E.append(E_val)
                current_inner_dE.append(dE_val)

                # Try to extract rms from the tail of the line
                # rms values are the floating-point numbers after ncg
                tail = line[m.end():]
                floats_in_tail = re.findall(r'([\d.]+E[+-]?\d+)', tail)
                if floats_in_tail:
                    current_inner_rms.append(float(floats_in_tail[0]))
                continue

            # Try matching an ionic step summary
            m = outer_re.match(line)
            if m:
                step_idx = int(m.group(1))
                F_val = float(m.group(2))
                dE_val = float(m.group(4))
                mag_val = float(m.group(5)) if m.group(5) else None

                outer_steps.append(step_idx)
                outer_energies.append(F_val)
                outer_dE.append(dE_val)
                outer_mag.append(mag_val)

                # Finalize current inner loop
                if current_inner_dE:
                    inner_loops.append(InnerLoopData(
                        ionic_step=step_idx,
                        n_steps=len(current_inner_dE),
                        dE=current_inner_dE.copy(),
                        E=current_inner_E.copy(),
                        rms=current_inner_rms.copy(),
                    ))
                current_inner_E = []
                current_inner_dE = []
                current_inner_rms = []
                continue
    # Handle case where file ends without a final F= line (abrupt termination)
    if current_inner_dE and (not outer_steps or
                             len(inner_loops) == 0 or
                             inner_loops[-1].ionic_step != current_ionic_step):
        inner_loops.append(InnerLoopData(
            ionic_step=current_ionic_step,
            n_steps=len(current_inner_dE),
            dE=current_inner_dE.copy(),
            E=current_inner_E.copy(),
            rms=current_inner_rms.copy(),
        ))

    outer_loop = OuterLoopData(
        step_indices=outer_steps,
        energies=outer_energies,
        d_E=outer_dE,
        mag=outer_mag,
    )

    return inner_loops, outer_loop


# ===========================================================================
# Statistical metrics for a single dE series
# ===========================================================================
def compute_series_metrics(dE_series: list[float]) -> dict:
    """
    Compute statistical convergence metrics for a series of dE values
    (either inner electronic steps or outer ionic steps).

    Metrics
    -------
    1. monotonicity_score: Fraction of steps where |dE| is non-increasing.
       Ideal = 1.0 (perfectly monotonic decay).
    2. decay_rate: Slope of log10(|dE|) vs step index via linear regression.
       More negative = faster convergence.
    3. final_ratio: |dE_final| / |dE_initial|. Smaller is better.
       Measures total compression of the residual.
    4. tail_stability: Std dev of the last 30% of |dE| values relative to
       their mean. Low = stable tail. High = still noisy at end.
    5. n_steps: Number of steps (fewer is better, all else equal).
    """
    arr = np.array(dE_series, dtype=float)
    n = len(arr)
    metrics = {}

    if n < 2:
        return {
            'monotonicity_score': 1.0,
            'decay_rate': 0.0,
            'final_ratio': 1.0,
            'tail_stability': 0.0,
            'n_steps': n,
        }

    abs_dE = np.abs(arr)
    # Avoid log of zero
    abs_dE_safe = np.where(abs_dE > 0, abs_dE, 1e-30)

    # 1. Monotonicity: fraction of steps where |dE[i]| <= |dE[i-1]|
    decreasing = np.sum(abs_dE[1:] <= abs_dE[:-1])
    metrics['monotonicity_score'] = float(decreasing / (n - 1))

    # 2. Decay rate: slope of log10(|dE|) vs index
    log_abs = np.log10(abs_dE_safe)
    indices = np.arange(n, dtype=float)
    if n >= 3:
        # Use polyfit for robustness
        slope, intercept = np.polyfit(indices, log_abs, 1)
        metrics['decay_rate'] = float(slope)
    else:
        metrics['decay_rate'] = float(log_abs[-1] - log_abs[0]) if n == 2 else 0.0

    # 4. Final ratio: |dE_last| / |dE_first|
    if abs_dE[0] > 0:
        metrics['final_ratio'] = float(abs_dE[-1] / abs_dE[0])
    else:
        metrics['final_ratio'] = 0.0


    # 5. Tail stability: coefficient of variation of last 30% of |dE|
    tail_start = max(1, int(0.7 * n))
    tail = abs_dE[tail_start:]
    if len(tail) > 1 and np.mean(tail) > 0:
        metrics['tail_stability'] = float(np.std(tail) / np.mean(tail))
    else:
        metrics['tail_stability'] = 0.0

    # 6. Step count
    metrics['n_steps'] = n

    return metrics


# ===========================================================================
# Scoring functions
# ===========================================================================
def score_inner_loop(inner: InnerLoopData, ediff: float = 1e-4) -> dict:
    """
    Score a single ionic step's electronic convergence.

    Returns a dict with individual metric scores and a composite [0, 100].
    """
    dE = inner.dE
    metrics = compute_series_metrics(dE)

    # --- Did it converge to EDIFF? ---
    final_abs_dE = abs(dE[-1]) if dE else float('inf')
    converged = final_abs_dE <= ediff
    inner.converged = converged
    inner.final_dE = final_abs_dE

    # --- Sub-scores (each 0-100) ---
    scores = {}

    # Convergence achievement (30% weight)
    if converged:
        # Bonus for converging well below EDIFF
        ratio = final_abs_dE / ediff
        scores['convergence'] = 100.0 * (1.0 - 0.3 * ratio)  # 70-100
    else:
        # Penalize based on how far above EDIFF
        overshoot = np.log10(final_abs_dE / ediff)
        scores['convergence'] = max(0, 50.0 - 20.0 * overshoot)

    # Monotonicity (25% weight) — fraction of steps where |dE| is non-increasing
    scores['monotonicity'] = 100.0 * metrics['monotonicity_score']

    # Spike penalty (15% weight) — penalizes large jumps UP in |dE| magnitude
    # This catches erratic non-monotonic behavior: if |dE| has settled to 1e-3
    # then spikes back to 1e-1, that indicates poor SCF stability.
    #
    # We skip the first ~30% of steps (initialization transient), as large
    # early spikes are expected VASP behavior: the initial wavefunction is
    # far from self-consistency, and the charge mixing hasn't stabilized yet.
    # Only spikes in the "settled" regime matter for convergence quality.
    abs_dE = np.abs(np.array(dE, dtype=float))
    if len(abs_dE) > 4:
        skip = max(2, int(0.3 * len(abs_dE)))  # skip initialization transient
        settled = abs_dE[skip:]
        increases = []
        for i in range(1, len(settled)):
            if settled[i] > settled[i-1] and settled[i-1] > 0:
                increases.append(settled[i] / settled[i-1])
        if increases:
            max_spike = max(increases)
            # max_spike ~1-3 is normal SCF noise, 3-10 is concerning, >10 is bad
            if max_spike <= 3:
                scores['spike_penalty'] = 100.0
            elif max_spike <= 10:
                scores['spike_penalty'] = max(0, 100.0 - 14.3 * (max_spike - 3))
            elif max_spike <= 50:
                scores['spike_penalty'] = max(0, 10.0 - 0.25 * (max_spike - 10))
            else:
                scores['spike_penalty'] = 0.0
        else:
            scores['spike_penalty'] = 100.0  # no spikes in settled regime
    else:
        scores['spike_penalty'] = 100.0  # too short to evaluate

    # Efficiency (15% weight) — fewer steps is better
    # Heuristic: <10 steps is excellent, 10-30 is fine, 30-60 is slow, >60 is bad
    n = metrics['n_steps']
    if n <= 10:
        scores['efficiency'] = 100.0
    elif n <= 30:
        scores['efficiency'] = max(0, 100.0 - 2.5 * (n - 10))
    elif n <= 60:
        scores['efficiency'] = max(0, 50.0 - 1.67 * (n - 30))
    else:
        scores['efficiency'] = max(0, 10.0 - 0.2 * (n - 60))

    # Decay rate (15% weight) — steeper log-decay is better
    # A slope of -0.5 or steeper per step is excellent
    dr = metrics['decay_rate']
    if dr <= -0.5:
        scores['decay'] = 100.0
    elif dr <= -0.1:
        scores['decay'] = 100.0 * (-dr - 0.1) / 0.4 * 0.6 + 40.0
    elif dr <= 0:
        scores['decay'] = 40.0 * (-dr) / 0.1
    else:
        scores['decay'] = 0.0  # dE is growing — very bad

    # Weighted composite
    weights = {
        'convergence': 0.30,
        'monotonicity': 0.25,
        'spike_penalty': 0.15,
        'efficiency': 0.15,
        'decay': 0.15,
    }
    composite = sum(scores[k] * weights[k] for k in weights)

    return {
        'ionic_step': inner.ionic_step,
        'n_electronic_steps': inner.n_steps,
        'converged': converged,
        'final_abs_dE': final_abs_dE,
        'metrics': metrics,
        'sub_scores': scores,
        'composite': round(composite, 2),
    }


def score_outer_loop(outer: OuterLoopData, ediffg: float = 0.0) -> dict:
    """
    Score the ionic (outer) convergence from the F= lines.

    For a static calculation (1 ionic step), the outer score is based
    on whether the electronic loop converged and the d E value.

    For relaxations, we use step-to-step energy differences from the
    F column as the primary convergence signal, since the OSZICAR 'd E'
    column can represent cumulative energy changes rather than per-step.
    """
    n_steps = len(outer.step_indices)

    if n_steps == 0:
        return {
            'n_ionic_steps': 0,
            'converged': False,
            'sub_scores': {},
            'composite': 0.0,
            'diagnostics': {'warning': 'No ionic steps found — possible abrupt termination'},
        }

    if n_steps == 1:
        # Static calculation: outer convergence is trivially satisfied
        return {
            'n_ionic_steps': 1,
            'converged': True,
            'is_static': True,
            'final_d_E': outer.d_E[0],
            'sub_scores': {'static_ok': 100.0},
            'composite': 100.0,
            'diagnostics': {'note': 'Static calculation (single ionic step)'},
        }

    # Multi-step ionic convergence
    # Compute step-to-step energy differences from the F column
    F_arr = np.array(outer.energies, dtype=float)
    delta_F = np.diff(F_arr)  # length n_steps - 1

    # Also have the OSZICAR d_E column
    d_E_arr = np.array(outer.d_E, dtype=float)

    # Primary signal: |delta_F| should decrease toward 0
    metrics_dF = compute_series_metrics(delta_F.tolist())

    # Did the ionic loop converge?
    final_abs_dF = abs(delta_F[-1])
    if ediffg < 0:
        converged = None  # Force-based — can't determine from OSZICAR
    elif ediffg > 0:
        converged = final_abs_dF <= ediffg
    else:
        converged = None  # VASP default — uncertain

    scores = {}

    # Energy going downhill? (for relaxations, F should generally decrease)
    n_downhill = np.sum(delta_F < 0)
    scores['energy_descent'] = 100.0 * n_downhill / len(delta_F)

    # delta_F monotonic decay in magnitude
    scores['dF_monotonicity'] = 100.0 * metrics_dF['monotonicity_score']

    # delta_F magnitude decay rate
    dr = metrics_dF['decay_rate']
    if dr <= -0.5:
        scores['dF_decay'] = 100.0
    elif dr <= -0.1:
        scores['dF_decay'] = 100.0 * (-dr - 0.1) / 0.4 * 0.6 + 40.0
    elif dr <= 0:
        scores['dF_decay'] = 40.0 * (-dr) / 0.1
    else:
        scores['dF_decay'] = 0.0

    # Compression ratio: |delta_F_last| / |delta_F_first|
    abs_dF = np.abs(delta_F)
    if abs_dF[0] > 0:
        compression = abs_dF[-1] / abs_dF[0]
        scores['compression'] = max(0, min(100, 100.0 * (1.0 - compression)))
    else:
        scores['compression'] = 100.0

    # Efficiency
    if n_steps <= 5:
        scores['efficiency'] = 100.0
    elif n_steps <= 20:
        scores['efficiency'] = max(0, 100.0 - 3.33 * (n_steps - 5))
    elif n_steps <= 50:
        scores['efficiency'] = max(0, 50.0 - 1.67 * (n_steps - 20))
    else:
        scores['efficiency'] = max(0, 20.0 - 0.5 * (n_steps - 50))

    weights = {
        'energy_descent': 0.10,
        'dF_monotonicity': 0.20,
        'dF_decay': 0.25,
        'compression': 0.25,
        'efficiency': 0.20,
    }
    composite = sum(scores.get(k, 0) * weights[k] for k in weights)

    return {
        'n_ionic_steps': n_steps,
        'converged': converged,
        'final_delta_F': float(delta_F[-1]),
        'delta_F_series': delta_F.tolist(),
        'metrics': metrics_dF,
        'sub_scores': scores,
        'composite': round(composite, 2),
    }


def score_outer_loop(outer: OuterLoopData, ediffg: float = 0.0) -> dict:
    """
    Score the ionic (outer) convergence from the F= lines.

    For a static calculation (1 ionic step), the outer score is based
    on whether the electronic loop converged and the d E value.

    For relaxations, we use step-to-step energy differences from the
    F column as the primary convergence signal, since the OSZICAR 'd E'
    column can represent cumulative energy changes rather than per-step.
    """
    n_steps = len(outer.step_indices)

    if n_steps == 0:
        return {
            'n_ionic_steps': 0,
            'converged': False,
            'sub_scores': {},
            'composite': 0.0,
            'diagnostics': {'warning': 'No ionic steps found — possible abrupt termination'},
        }

    if n_steps == 1:
        # Static calculation: outer convergence is trivially satisfied
        return {
            'n_ionic_steps': 1,
            'converged': True,
            'is_static': True,
            'final_d_E': outer.d_E[0],
            'sub_scores': {'static_ok': 100.0},
            'composite': 100.0,
            'diagnostics': {'note': 'Static calculation (single ionic step)'},
        }

    # Multi-step ionic convergence
    # Compute step-to-step energy differences from the F column
    F_arr = np.array(outer.energies, dtype=float)
    delta_F = np.diff(F_arr)  # length n_steps - 1

    # Also have the OSZICAR d_E column
    d_E_arr = np.array(outer.d_E, dtype=float)

    # Primary signal: |delta_F| should decrease toward 0
    metrics_dF = compute_series_metrics(delta_F.tolist())

    # Did the ionic loop converge?
    final_abs_dF = abs(delta_F[-1])
    if ediffg < 0:
        converged = None  # Force-based — can't determine from OSZICAR
    elif ediffg > 0:
        converged = final_abs_dF <= ediffg
    else:
        converged = None  # VASP default — uncertain

    scores = {}

    # Energy going downhill? (for relaxations, F should generally decrease)
    n_downhill = np.sum(delta_F < 0)
    scores['energy_descent'] = 100.0 * n_downhill / len(delta_F)

    # delta_F monotonic decay in magnitude
    scores['dF_monotonicity'] = 100.0 * metrics_dF['monotonicity_score']

    # delta_F magnitude decay rate
    dr = metrics_dF['decay_rate']
    if dr <= -0.5:
        scores['dF_decay'] = 100.0
    elif dr <= -0.1:
        scores['dF_decay'] = 100.0 * (-dr - 0.1) / 0.4 * 0.6 + 40.0
    elif dr <= 0:
        scores['dF_decay'] = 40.0 * (-dr) / 0.1
    else:
        scores['dF_decay'] = 0.0

    # Compression ratio: |delta_F_last| / |delta_F_first|
    abs_dF = np.abs(delta_F)
    if abs_dF[0] > 0:
        compression = abs_dF[-1] / abs_dF[0]
        scores['compression'] = max(0, min(100, 100.0 * (1.0 - compression)))
    else:
        scores['compression'] = 100.0

    # Efficiency
    if n_steps <= 5:
        scores['efficiency'] = 100.0
    elif n_steps <= 20:
        scores['efficiency'] = max(0, 100.0 - 3.33 * (n_steps - 5))
    elif n_steps <= 50:
        scores['efficiency'] = max(0, 50.0 - 1.67 * (n_steps - 20))
    else:
        scores['efficiency'] = max(0, 20.0 - 0.5 * (n_steps - 50))

    weights = {
        'energy_descent': 0.10,
        'dF_monotonicity': 0.20,
        'dF_decay': 0.25,
        'compression': 0.25,
        'efficiency': 0.20,
    }
    composite = sum(scores.get(k, 0) * weights[k] for k in weights)

    return {
        'n_ionic_steps': n_steps,
        'converged': converged,
        'final_delta_F': float(delta_F[-1]),
        'delta_F_series': delta_F.tolist(),
        'metrics': metrics_dF,
        'sub_scores': scores,
        'composite': round(composite, 2),
        'diagnostics': {},
    }


def detect_anomalies(inner_loops: list[InnerLoopData],
                     outer: OuterLoopData,
                     vasp_params: dict = None) -> list[str]:
    """
    Detect common VASP convergence problems using INCAR-aware checks.
    Returns a list of warning/fail messages.
    """
    warnings = []
    if vasp_params is None:
        vasp_params = resolve_vasp_params({})

    nelm = vasp_params.get('NELM', 60)
    nsw = vasp_params.get('NSW', 0)
    ediff = vasp_params.get('EDIFF', 1e-4)
    ediffg = vasp_params.get('EDIFFG', ediff * 10)
    ionic_mode = vasp_params.get('_ionic_conv_mode', 'energy')

    if not inner_loops:
        warnings.append("FAIL: No electronic steps parsed — file may be empty or corrupt")
        return warnings

    # --- Electronic (inner) loop checks ---
    # Check for abrupt termination (inner loop without corresponding F= line)
    n_inner = len(inner_loops)
    n_outer = len(outer.step_indices)
    if n_inner > n_outer:
        warnings.append(
            f"FAIL: {n_inner} inner loops but only {n_outer} ionic summary lines — "
            f"last {n_inner - n_outer} ionic step(s) terminated without completing "
            f"electronic convergence (crashed or killed?)"
        )

    # Check for electronic steps hitting NELM (max electronic steps reached)
    for il in inner_loops:
        if il.n_steps >= nelm:
            warnings.append(
                f"WARN: Ionic step {il.ionic_step} hit NELM={nelm} electronic steps "
                f"(final |dE| = {il.final_dE:.2e} vs EDIFF = {ediff:.1e}) — "
                f"consider increasing NELM, adjusting ALGO/AMIX, or using NELMIN"
            )
        elif not il.converged:
            warnings.append(
                f"WARN: Ionic step {il.ionic_step} electronic loop did not converge "
                f"(final |dE| = {il.final_dE:.2e} vs EDIFF = {ediff:.1e})"
            )

    # Check for very large magnitude spikes in electronic steps
    for il in inner_loops:
        if len(il.dE) > 2:
            abs_dE = np.abs(il.dE)
            if np.max(abs_dE[1:]) > 10 * abs_dE[0] and abs_dE[0] > 0:
                warnings.append(
                    f"WARN: Ionic step {il.ionic_step} shows large electronic "
                    f"|dE| spike (max |dE| / initial |dE| = "
                    f"{np.max(abs_dE[1:])/abs_dE[0]:.1f}) — "
                    f"possible charge sloshing, consider AMIX=0.1, BMIX=0.001"
                )

    # --- Ionic (outer) loop checks ---

    # Check if NSW was reached without ionic convergence
    if nsw > 0 and n_outer >= nsw:
        # Relaxation hit max ionic steps
        if ionic_mode == 'energy' and len(outer.energies) >= 2:
            final_dF = abs(outer.energies[-1] - outer.energies[-2])
            if final_dF > abs(ediffg):
                warnings.append(
                    f"FAIL: Reached NSW={nsw} ionic steps without converging — "
                    f"final |ΔF| = {final_dF:.2e} eV > EDIFFG = {ediffg:.1e} eV. "
                    f"Increase NSW or improve starting geometry."
                )
            else:
                warnings.append(
                    f"INFO: Reached NSW={nsw} ionic steps; final |ΔF| = {final_dF:.2e} eV "
                    f"appears converged vs EDIFFG = {ediffg:.1e} eV."
                )
        elif ionic_mode == 'force':
            warnings.append(
                f"WARN: Reached NSW={nsw} ionic steps — cannot verify force convergence "
                f"from OSZICAR alone (EDIFFG = {ediffg:.4f} eV/Å). Check OUTCAR."
            )
        elif ionic_mode == 'none':
            warnings.append(
                f"INFO: EDIFFG=0, so VASP ran exactly NSW={nsw} steps with no "
                f"ionic convergence check."
            )

    # Check if energy went UP on the final ionic step (relaxation overshot)
    if len(outer.energies) >= 2:
        final_delta = outer.energies[-1] - outer.energies[-2]
        if final_delta > 0 and abs(final_delta) > ediff:
            warnings.append(
                f"WARN: Energy increased on final ionic step by {final_delta:.4e} eV — "
                f"possible overshoot. Final structure may not be the lowest-energy one."
            )

    # Check magnetic moment stability (if available and spin-polarized)
    mags = [m for m in outer.mag if m is not None]
    if len(mags) > 2:
        mag_arr = np.array(mags)
        mean_mag = np.abs(np.mean(mag_arr))
        if mean_mag > 0.1:  # only check if system is magnetic
            cv = np.std(mag_arr) / mean_mag
            if cv > 0.1:
                warnings.append(
                    f"WARN: Magnetic moment varies significantly across ionic steps "
                    f"(mean={np.mean(mag_arr):.2f} μB, CV={cv:.2f}) — "
                    f"consider more ionic steps or checking magnetic ground state"
                )

    # Check if ionic energies fail to descend (energy goes up frequently)
    if len(outer.energies) > 3:
        dF = np.diff(outer.energies)
        n_up = np.sum(dF > 0)
        if n_up / len(dF) > 0.5:
            ibrion = vasp_params.get('IBRION', 2)
            potim = vasp_params.get('POTIM', 0.5)
            warnings.append(
                f"WARN: Ionic energy increased in {n_up}/{len(dF)} steps — "
                f"geometry optimizer is not steadily descending. "
                f"With IBRION={ibrion}, POTIM={potim}, "
                f"consider reducing POTIM or switching IBRION"
            )

    return warnings


def assign_grade(score: float) -> str:
    """Map a 0-100 score to a letter grade."""
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "A-"
    elif score >= 80:
        return "B+"
    elif score >= 75:
        return "B"
    elif score >= 70:
        return "B-"
    elif score >= 65:
        return "C+"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


# ===========================================================================
# Main scoring pipeline
# ===========================================================================
def score_oszicar(filepath: str,
                  ediff: float = None,
                  ediffg: float = None,
                  nelm: int = None,
                  nsw: int = None,
                  incar_path: str = None,
                  inner_weight: float = 0.6,
                  outer_weight: float = 0.4) -> ConvergenceReport:
    """
    Full convergence scoring pipeline for an OSZICAR file.

    Parameters
    ----------
    filepath : str
        Path to the OSZICAR file.
    ediff : float or None
        Electronic convergence criterion. If None, read from INCAR or use
        VASP default (1e-4).
    ediffg : float or None
        Ionic convergence criterion. If None, read from INCAR or use
        VASP default (EDIFF * 10).
    nelm : int or None
        Max electronic steps. If None, read from INCAR or use default (60).
    nsw : int or None
        Max ionic steps. If None, read from INCAR or use default (0).
    incar_path : str or None
        Path to INCAR file. If provided, reads parameters from it.
        CLI auto-searches for INCAR in same directory as OSZICAR.
    inner_weight : float
        Weight for electronic convergence in composite score.
    outer_weight : float
        Weight for ionic convergence in composite score.

    Returns
    -------
    ConvergenceReport
    """
    report = ConvergenceReport(filename=str(filepath))

    # --- Resolve VASP parameters ---
    incar_params = {}
    if incar_path and Path(incar_path).exists():
        incar_params = parse_incar(incar_path)

    vasp_params = resolve_vasp_params(incar_params)

    # CLI overrides take precedence over INCAR
    if ediff is not None:
        vasp_params['EDIFF'] = ediff
        # Recompute EDIFFG default if EDIFFG wasn't explicitly set
        if ediffg is None and 'EDIFFG' not in incar_params:
            vasp_params['EDIFFG'] = ediff * 10
            vasp_params['_ediffg_source'] = f'default (EDIFF * 10 = {ediff * 10})'
    if ediffg is not None:
        vasp_params['EDIFFG'] = ediffg
        vasp_params['_ediffg_source'] = 'CLI override'
        # Recompute convergence mode
        if ediffg < 0:
            vasp_params['_ionic_conv_mode'] = 'force'
            vasp_params['_ionic_conv_threshold'] = abs(ediffg)
            vasp_params['_ionic_conv_unit'] = 'eV/Å'
        elif ediffg > 0:
            vasp_params['_ionic_conv_mode'] = 'energy'
            vasp_params['_ionic_conv_threshold'] = ediffg
            vasp_params['_ionic_conv_unit'] = 'eV'
    if nelm is not None:
        vasp_params['NELM'] = nelm
    if nsw is not None:
        vasp_params['NSW'] = nsw

    eff_ediff = vasp_params['EDIFF']
    eff_ediffg = vasp_params['EDIFFG']
    eff_nelm = vasp_params['NELM']
    eff_nsw = vasp_params['NSW']

    # --- Parse OSZICAR ---
    inner_loops, outer_loop = parse_oszicar(filepath)

    if not inner_loops:
        report.run_completed = False
        report.total_score = 0.0
        report.grade = "F"
        report.diagnostics = {'errors': ['No data parsed from OSZICAR'],
                              'vasp_params': vasp_params}
        return report

    # --- Score inner loops ---
    inner_results = []
    for il in inner_loops:
        result = score_inner_loop(il, ediff=eff_ediff)
        inner_results.append(result)

    report.inner_scores = inner_results
    report.n_ionic_steps = len(outer_loop.step_indices)
    report.n_electronic_steps_total = sum(il.n_steps for il in inner_loops)

    # Composite inner score: softmin-weighted average (penalizes bad steps)
    composites = np.array([r['composite'] for r in inner_results])
    if len(composites) > 1:
        tau = 20.0
        weights_exp = np.exp(-composites / tau)
        weights_exp /= weights_exp.sum()
        report.inner_composite = float(np.dot(weights_exp, composites))
    else:
        report.inner_composite = float(composites[0])

    # --- Score outer loop ---
    outer_result = score_outer_loop(outer_loop, ediffg=eff_ediffg)
    report.outer_score = outer_result['composite']

    # For static calculations, shift weight to inner loop
    is_static = outer_result.get('is_static', False)
    if is_static:
        effective_inner_w = 0.95
        effective_outer_w = 0.05
    else:
        effective_inner_w = inner_weight
        effective_outer_w = outer_weight

    # --- INCAR-aware anomaly detection ---
    anomalies = detect_anomalies(inner_loops, outer_loop, vasp_params)

    # Anomaly penalty: each warning costs 2 pts, each FAIL costs 15 pts
    anomaly_penalty = 0
    for a in anomalies:
        if a.startswith('FAIL'):
            anomaly_penalty += 15
        elif a.startswith('WARN'):
            anomaly_penalty += 2

    # Completion check
    if len(inner_loops) > len(outer_loop.step_indices):
        report.run_completed = False
        anomaly_penalty += 10

    # Composite total
    raw_score = (effective_inner_w * report.inner_composite +
                 effective_outer_w * report.outer_score)
    report.total_score = round(max(0, min(100, raw_score - anomaly_penalty)), 2)
    report.grade = assign_grade(report.total_score)

    report.diagnostics = {
        'anomalies': anomalies,
        'anomaly_penalty': anomaly_penalty,
        'outer_result': outer_result,
        'is_static': is_static,
        'vasp_params': {
            'EDIFF': eff_ediff,
            'EDIFFG': eff_ediffg,
            'EDIFFG_source': vasp_params.get('_ediffg_source', 'unknown'),
            'NELM': eff_nelm,
            'NSW': eff_nsw,
            'IBRION': vasp_params.get('IBRION'),
            'ISIF': vasp_params.get('ISIF'),
            'ALGO': vasp_params.get('ALGO'),
            'ISPIN': vasp_params.get('ISPIN'),
            'ionic_conv_mode': vasp_params.get('_ionic_conv_mode'),
            'ionic_conv_threshold': vasp_params.get('_ionic_conv_threshold'),
            'ionic_conv_unit': vasp_params.get('_ionic_conv_unit'),
        },
        'weights': {
            'inner': effective_inner_w,
            'outer': effective_outer_w,
        },
        'incar_path': incar_path,
    }

    return report


# ===========================================================================
# Pretty printing
# ===========================================================================
def print_report(report: ConvergenceReport, verbose: bool = False):
    """Print a human-readable convergence report."""
    print("=" * 72)
    print(f"  OSZICAR CONVERGENCE REPORT: {report.filename}")
    print("=" * 72)
    print()

    # Summary
    print(f"  Total Score:  {report.total_score:6.2f} / 100   [{report.grade}]")
    print(f"  Ionic Steps:  {report.n_ionic_steps}")
    print(f"  Electronic Steps (total): {report.n_electronic_steps_total}")
    is_static = report.diagnostics.get('is_static', False)
    print(f"  Calculation Type: {'Static' if is_static else 'Relaxation/MD'}")
    print(f"  Run Completed: {'Yes' if report.run_completed else 'NO'}")
    print()

    # VASP parameters
    vp = report.diagnostics.get('vasp_params', {})
    if vp:
        print(f"  VASP Parameters (resolved):")
        print(f"    EDIFF  = {vp.get('EDIFF', '?'):.1e}    "
              f"NELM = {vp.get('NELM', '?')}")
        ediffg_val = vp.get('EDIFFG', '?')
        ediffg_src = vp.get('EDIFFG_source', '')
        iconv = vp.get('ionic_conv_mode', '?')
        print(f"    EDIFFG = {ediffg_val}  ({ediffg_src})  "
              f"[mode: {iconv}]")
        print(f"    NSW    = {vp.get('NSW', '?')}    "                                                                           
              f"IBRION = {vp.get('IBRION', '?')}    "
              f"ISIF = {vp.get('ISIF', '?')}")
        incar_path = report.diagnostics.get('incar_path')
        if incar_path:
            print(f"    (Read from: {incar_path})")
        print()

    # Inner loop summary
    print("-" * 72)
    print("  INNER (ELECTRONIC) CONVERGENCE")
    print("-" * 72)
    print(f"  Composite Inner Score: {report.inner_composite:.2f}")
    print()
    print(f"  {'Step':>6s} {'N_elec':>7s} {'Conv?':>6s} {'|dE_final|':>12s} "
          f"{'Score':>7s}  Sub-scores")
    print(f"  {'----':>6s} {'------':>7s} {'-----':>6s} {'----------':>12s} "
          f"{'-----':>7s}  ----------")

    for r in report.inner_scores:
        ss = r['sub_scores']
        sub_str = (f"conv={ss['convergence']:.0f} mono={ss['monotonicity']:.0f} "
                   f"spike={ss['spike_penalty']:.0f} eff={ss['efficiency']:.0f} "
                   f"dec={ss['decay']:.0f}")
        print(f"  {r['ionic_step']:>6d} {r['n_electronic_steps']:>7d} "
              f"{'Yes' if r['converged'] else 'NO':>6s} "
              f"{r['final_abs_dE']:>12.2e} {r['composite']:>7.2f}  {sub_str}")

    if verbose:
        print()
        print("  Detailed metrics per ionic step:")
        for r in report.inner_scores:
            m = r['metrics']
            print(f"    Step {r['ionic_step']}: "
                  f"monotonicity={m['monotonicity_score']:.3f}  "
                  f"decay_rate={m['decay_rate']:.4f}  "
                  f"final_ratio={m['final_ratio']:.2e}  "
                  f"tail_stab={m['tail_stability']:.3f}")

    # Outer loop summary
    print()
    print("-" * 72)
    print("  OUTER (IONIC) CONVERGENCE")
    print("-" * 72)
    outer_r = report.diagnostics.get('outer_result', {})
    print(f"  Composite Outer Score: {report.outer_score:.2f}")
    if is_static:
        print("  (Static calc — outer convergence trivially satisfied)")
    elif report.n_ionic_steps > 1:
        if 'sub_scores' in outer_r:
            for k, v in outer_r['sub_scores'].items():
                print(f"    {k}: {v:.2f}")

    # Anomalies
    anomalies = report.diagnostics.get('anomalies', [])
    if anomalies:
        print()
        print("-" * 72)
        print("  DIAGNOSTICS & WARNINGS")
        print("-" * 72)
        for a in anomalies:
            print(f"  {a}")
        penalty = report.diagnostics.get('anomaly_penalty', 0)
        if penalty > 0:
            print(f"  [Total anomaly penalty: -{penalty} pts]")
    else:
        print()
        print("  No anomalies detected.")

    print()
    print("=" * 72)


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Score VASP OSZICAR convergence quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python oszicar_convergence_scorer.py OSZICAR
  python oszicar_convergence_scorer.py OSZICAR --ediff 1e-6 --verbose
  python oszicar_convergence_scorer.py OSZICAR --json output.json
        """,
    )
    parser.add_argument('oszicar', help='Path to OSZICAR file')
    parser.add_argument('--incar', type=str, default=None,
                        help='Path to INCAR file (auto-searches same dir if not given)')
    parser.add_argument('--ediff', type=float, default=None,
                        help='Override EDIFF (default: from INCAR or 1e-4)')
    parser.add_argument('--ediffg', type=float, default=None,
                        help='Override EDIFFG (default: from INCAR or EDIFF*10)')
    parser.add_argument('--nelm', type=int, default=None,
                        help='Override NELM (default: from INCAR or 60)')
    parser.add_argument('--nsw', type=int, default=None,
                        help='Override NSW (default: from INCAR or 0)')
    parser.add_argument('--inner-weight', type=float, default=0.6,
                        help='Weight for inner loop in composite (default: 0.6)')
    parser.add_argument('--outer-weight', type=float, default=0.4,
                        help='Weight for outer loop in composite (default: 0.4)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-step metrics')
    parser.add_argument('--json', type=str, default=None,
                        help='Write machine-readable report to JSON file')

    args = parser.parse_args()

    # Auto-search for INCAR in same directory as OSZICAR
    incar_path = args.incar
    if incar_path is None:
        oszicar_dir = Path(args.oszicar).parent
        # Try common INCAR naming patterns
        for candidate in ['INCAR', 'INCAR.relax', 'INCAR.static']:
            p = oszicar_dir / candidate
            if p.exists():
                incar_path = str(p)
                break

    report = score_oszicar(
        args.oszicar,
        ediff=args.ediff,
        ediffg=args.ediffg,
        nelm=args.nelm,
        nsw=args.nsw,
        incar_path=incar_path,
        inner_weight=args.inner_weight,
        outer_weight=args.outer_weight,
    )

    print_report(report, verbose=args.verbose)

    if args.json:
        # Serialize report to JSON
        out = {
            'filename': report.filename,
            'total_score': report.total_score,
            'grade': report.grade,
            'run_completed': report.run_completed,
            'n_ionic_steps': report.n_ionic_steps,
            'n_electronic_steps_total': report.n_electronic_steps_total,
            'inner_composite': report.inner_composite,
            'outer_score': report.outer_score,
            'inner_scores': report.inner_scores,
            'diagnostics': {
                k: v for k, v in report.diagnostics.items()
                if k != 'outer_result'  # avoid nested complexity
            },
        }
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  JSON report written to: {args.json}")

    # Return score for pipeline integration
    return report.total_score


if __name__ == '__main__':
    sys.exit(0 if main() >= 50 else 1)