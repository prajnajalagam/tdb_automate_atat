"""
SQS target gate — consume a consensus JSON produced by
reverse_engineer_targets.ipynb (Fable / Colab) and decide which
SQS DFT calculations should be kept or rejected from the sqs2tdb fit.

The high-level idea: a Redlich-Kister fit to the cross-TDB consensus
defines the *target* formation-energy landscape that any well-converged
DFT calculation of a representative SQS should reproduce. SQS whose
DFT energy deviates from the consensus target by more than N sigma
(N defaults to 3) are flagged as either under-converged DFT, an SQS
that doesn't represent the random alloy well, or a phase the consensus
doesn't really cover — in any case, not useful as input to the fit.

What the gate does NOT do:
  - it does not parse TDBs (that's in the notebook's loader);
  - it does not re-extract targets from raw TDBs (also in the notebook);
  - it does not try to fix DFT convergence, only reject.

Public surface
--------------
    TargetGate.from_consensus_json(path)        load a single (sys,phase)
    TargetGate.evaluate(comp, dft_excess, n_sigma=3.0)
                                                -> (passes, target,
                                                    sigma, z_score)
    load_target_dir(dir, elements, phases)      -> Dict[phase, TargetGate]
    count_atoms_str_out(p)                      atom counter for ATAT str.out
    sqs_dft_excess_eV_per_atom(em_paths, e_sqs_eV, n_sqs, x_B)
                                                same-phase excess formation
                                                energy, eV/atom

Algorithm matches the notebook's cell 12 / 13 exactly so a number
computed here equals what the notebook reports.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:                                         # pragma: no cover
    raise ImportError(
        "sqs_target_gate requires numpy. Install with: pip install numpy"
    )


# ────────────────────────────────────────────────────────────────────
#  Redlich-Kister fit — same algorithm as notebook cell 12
# ────────────────────────────────────────────────────────────────────

@dataclass
class _RkFit:
    L: np.ndarray
    x_range: Tuple[float, float]
    y_endpoints: Tuple[float, float]


def _fit_rk_excess(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    order: int = 2,
) -> _RkFit:
    """Weighted-LSQ Redlich-Kister fit of the EXCESS of y(x).

    The endpoint baseline is the chord through the covered (xlo, xhi).
    Phases with restricted sublattice models (e.g. SIGMA) only cover
    part of [0,1] — the fit and any reconstruction are valid only on
    that range. Do not extrapolate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    sg = None if sigma is None else np.asarray(sigma, dtype=float)[ok]
    xlo, xhi = float(x.min()), float(x.max())
    ylo, yhi = float(y[np.argmin(x)]), float(y[np.argmax(x)])
    t = (x - xlo) / (xhi - xlo) if xhi > xlo else np.zeros_like(x)
    exc = y - ((1 - t) * ylo + t * yhi)
    M = np.column_stack([x * (1 - x) * (1 - 2 * x) ** i for i in range(order)])
    if sg is not None:
        med = np.nanmedian(sg[sg > 0]) if np.any(sg > 0) else 0.0
        w = 1.0 / np.maximum(sg, med * 1e-2 + 1e-12) ** 2
    else:
        w = np.ones_like(x)
    sw = np.sqrt(w)
    L, *_ = np.linalg.lstsq(M * sw[:, None], exc * sw, rcond=None)
    return _RkFit(L=L, x_range=(xlo, xhi), y_endpoints=(ylo, yhi))


def _rk_excess_at(x: float, fit: _RkFit) -> float:
    """RK excess only (no baseline) — what same-phase DFT E_form matches."""
    return float(x * (1 - x) * sum(
        Li * (1 - 2 * x) ** i for i, Li in enumerate(fit.L)
    ))


# ────────────────────────────────────────────────────────────────────
#  Target gate
# ────────────────────────────────────────────────────────────────────

@dataclass
class TargetGate:
    """A loaded (system, phase) consensus target with a usable gate fn."""

    system: List[str]                    # ["CO", "CR"] uppercase
    phase: str                           # "FCC_A1"
    elsA: str
    elsB: str
    rk_E: _RkFit
    rk_S: Optional[_RkFit]
    sigma_grid_x: np.ndarray
    sigma_grid_E: np.ndarray
    n_contributing: np.ndarray
    dft_noise_floor_ev: float
    source_path: str
    # Absolute floor on the gating sigma (eV/atom). Honesty guard for
    # METASTABLE composition ranges: assessed TDBs constrain G only
    # where a phase is stable (or measured); elsewhere the "consensus"
    # is often several assessments inheriting the SAME SGTE lattice
    # stabilities, so the cross-TDB spread coincidentally shrinks and
    # a 3-sigma gate there can reject perfectly correct DFT. A floor of
    # ~0.010-0.020 eV/atom keeps the gate meaningful where data exist
    # without letting artificial consensus tighten it where none do.
    min_sigma_ev: float = 0.0

    # --- factories ---

    @classmethod
    def from_consensus_json(
        cls,
        path: Path,
        dft_noise_floor_ev: float = 0.005,
        min_sigma_ev: float = 0.0,
    ) -> "TargetGate":
        with Path(path).open() as f:
            cons = json.load(f)
        els = [e.upper() for e in cons["meta"]["els"]]
        x = np.asarray(cons["x"], dtype=float)
        eF = cons["E_form"]
        e_mean = np.asarray(eF["mean"], dtype=float)
        e_sigma = np.asarray(eF["sigma"], dtype=float)
        rk_E = _fit_rk_excess(x, e_mean, e_sigma, order=2)

        sF = cons.get("svib_ht") or {}
        s_mean = sF.get("mean")
        s_sigma = sF.get("sigma")
        rk_S = None
        if s_mean is not None:
            try:
                rk_S = _fit_rk_excess(
                    x,
                    np.asarray(s_mean, dtype=float),
                    np.asarray(s_sigma, dtype=float)
                    if s_sigma is not None else None,
                    order=2,
                )
            except Exception:
                rk_S = None

        n_contrib = np.asarray(
            eF.get("n_contributing", np.ones_like(e_mean)),
            dtype=float,
        )
        return cls(
            system=els,
            phase=cons["meta"]["phase_name"],
            elsA=els[0],
            elsB=els[1],
            rk_E=rk_E,
            rk_S=rk_S,
            sigma_grid_x=x,
            sigma_grid_E=e_sigma,
            n_contributing=n_contrib,
            dft_noise_floor_ev=float(dft_noise_floor_ev),
            source_path=str(Path(path).resolve()),
            min_sigma_ev=float(min_sigma_ev),
        )

    # --- query ---

    def gate_sigma_at(self, x_B: float) -> float:
        """Total gating-sigma at x_B (quadrature of cross-TDB sigma and
        the DFT/SQS noise floor — see notebook cell 13)."""
        ok = ~np.isnan(self.sigma_grid_E)
        if not np.any(ok):
            return max(self.dft_noise_floor_ev, self.min_sigma_ev)
        sig_tdb = float(np.interp(
            x_B,
            self.sigma_grid_x[ok],
            self.sigma_grid_E[ok],
        ))
        sig = math.sqrt(sig_tdb ** 2 + self.dft_noise_floor_ev ** 2)
        return max(sig, self.min_sigma_ev)

    def evaluate(
        self,
        comp: Dict[str, float],
        dft_excess_eV_per_atom: float,
        n_sigma: float = 3.0,
    ) -> Tuple[bool, float, float, float, str]:
        """Decide whether this SQS passes the consensus gate.

        comp                  {ELEMENT_UPPER: fraction_on_mixing_sublattice}
        dft_excess_eV_per_atom    same-phase excess formation energy of the
                                  SQS (so lattice stability cancels with
                                  the RK excess target)
        n_sigma                   tolerance in units of gate_sigma

        Returns (passes, target, gate_sigma, z_score, reason).
        """
        x_B = float(comp.get(self.elsB.upper(), 0.0))
        if x_B < self.rk_E.x_range[0] - 1e-9 or x_B > self.rk_E.x_range[1] + 1e-9:
            return (
                True,
                float("nan"),
                float("nan"),
                float("nan"),
                "x outside RK-covered range — gate skipped (kept)",
            )
        target = _rk_excess_at(x_B, self.rk_E)
        sigma = self.gate_sigma_at(x_B)
        z = (dft_excess_eV_per_atom - target) / sigma if sigma > 0 else 0.0
        passes = abs(z) <= n_sigma
        reason = (
            f"OK  z={z:+.2f}σ  target={target*1e3:+.1f} meV/atom"
            if passes else
            f"REJ z={z:+.2f}σ  > {n_sigma:.1f}σ  "
            f"target={target*1e3:+.1f}±{sigma*1e3:.2f} meV/atom"
        )
        return passes, float(target), float(sigma), float(z), reason


# ────────────────────────────────────────────────────────────────────
#  Target-directory loader
# ────────────────────────────────────────────────────────────────────

def _candidate_names(els: List[str], phase: str) -> List[str]:
    s = sorted(e.upper() for e in els)
    names = {
        f"{s[0]}_{s[1]}_{phase}_consensus.json",
        f"{s[1]}_{s[0]}_{phase}_consensus.json",
        # also accept phase aliases the notebook uses
        f"{s[0]}_{s[1]}_{phase.replace('_D8B', '')}_consensus.json",
        f"{s[1]}_{s[0]}_{phase.replace('_D8B', '')}_consensus.json",
        f"{s[0]}_{s[1]}_{phase.replace('_SGTE', '')}_consensus.json",
        f"{s[1]}_{s[0]}_{phase.replace('_SGTE', '')}_consensus.json",
    }
    return sorted(names)


def load_target_dir(
    target_dir: Optional[Path],
    elements: List[str],
    phases: List[str],
    dft_noise_floor_ev: float = 0.005,
    min_sigma_ev: float = 0.0,
) -> Dict[str, TargetGate]:
    """Walk target_dir and load the consensus JSONs whose filename
    matches `<elements>_<phase>_consensus.json`. Missing phases are
    silently skipped — the pipeline treats them as "no gate"."""
    out: Dict[str, TargetGate] = {}
    if target_dir is None:
        return out
    target_dir = Path(target_dir)
    if not target_dir.is_dir():
        return out
    for phase in phases:
        for name in _candidate_names(elements, phase):
            cand = target_dir / name
            if cand.is_file():
                try:
                    out[phase] = TargetGate.from_consensus_json(
                        cand, dft_noise_floor_ev=dft_noise_floor_ev,
                        min_sigma_ev=min_sigma_ev,
                    )
                    break
                except Exception as exc:
                    print(f"  WARNING: could not load {cand}: {exc}")
    return out


# ────────────────────────────────────────────────────────────────────
#  Per-atom DFT helpers (no pycalphad needed)
# ────────────────────────────────────────────────────────────────────

_ATOM_LINE_RE = re.compile(
    r"^\s*[-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?\s+"
    r"[-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?\s+"
    r"[-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?\s+"
    r"([A-Za-z][A-Za-z0-9_]*)\s*$"
)


def count_atoms_str_out(p: Path) -> Optional[int]:
    """Count atom lines in an ATAT str.out file.

    Each atom line has the shape `x y z ELEMENT` (4 fields, last is
    alphabetic). The first 6 lines (3 lattice + 3 cell vectors) are
    pure numbers and are filtered out naturally.
    """
    try:
        n = 0
        with Path(p).open(errors="ignore") as f:
            for line in f:
                if _ATOM_LINE_RE.match(line):
                    n += 1
        return n if n > 0 else None
    except Exception:
        return None


def per_atom_energy(
    energy_file: Path,
    str_out_file: Path,
) -> Optional[Tuple[float, int]]:
    """Read energy (eV total supercell) + atom count -> (eV/atom, N)."""
    try:
        text = Path(energy_file).read_text(errors="ignore").strip()
        e_total = float(text.replace("D", "E").replace("d", "E"))
    except Exception:
        return None
    n = count_atoms_str_out(str_out_file)
    if not n:
        return None
    return e_total / n, n


def sqs_dft_excess_eV_per_atom(
    sqs_dir: Path,
    endmember_per_atom: Dict[str, float],
    composition: Dict[str, float],
) -> Optional[float]:
    """Same-phase excess formation energy of an SQS, eV/atom.

    Parameters
    ----------
    sqs_dir : SQS directory containing 'energy' and 'str.out'
    endmember_per_atom : {EL_UPPER: eV/atom} from the pure-element
        endmember SQS in the SAME phase as `sqs_dir`. (Both pure refs
        must be in the same phase or the lattice-stability terms
        won't cancel and the comparison is meaningless.)
    composition : {EL_UPPER: x} mixing-sublattice fraction.

    Returns the excess in eV/atom, or None if any file is missing /
    unparseable.
    """
    pe = per_atom_energy(Path(sqs_dir) / "energy", Path(sqs_dir) / "str.out")
    if pe is None:
        return None
    e_sqs, _ = pe
    e_ref = 0.0
    for el, x in composition.items():
        if el not in endmember_per_atom:
            return None
        e_ref += x * endmember_per_atom[el]
    return e_sqs - e_ref


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(
            "Usage: python3 sqs_target_gate.py "
            "<consensus.json> [x_B] [dft_excess_eV_per_atom]"
        )
        sys.exit(1)
    g = TargetGate.from_consensus_json(sys.argv[1])
    print(f"Loaded {g.system} / {g.phase}  from {g.source_path}")
    print(f"  RK x-range: [{g.rk_E.x_range[0]:.3f}, "
          f"{g.rk_E.x_range[1]:.3f}]")
    print(f"  RK L coeffs (E_form, eV/atom): "
          f"{[f'{L:+.5f}' for L in g.rk_E.L]}")
    if len(sys.argv) >= 4:
        x_B = float(sys.argv[2])
        dft_excess = float(sys.argv[3])
        passes, target, sigma, z, reason = g.evaluate(
            {g.elsB: x_B}, dft_excess,
        )
        print(
            f"  At x({g.elsB})={x_B:.3f}, dft_excess={dft_excess*1e3:+.1f} meV/atom:"
        )
        print(f"    {reason}")
