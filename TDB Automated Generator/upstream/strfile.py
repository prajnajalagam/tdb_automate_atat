#!/usr/bin/env python3
"""
Minimal reader/writer for ATAT structure files (str.out, str_relax.out,
str_unpert.out).

Format (see ATAT manual, mcsqs / rndstr description):
    line 1-3  coordinate system: either one line of 6 numbers
              (a b c alpha beta gamma) OR three lines of 3 numbers each.
    next 3    lattice vectors u, v, w (3 numbers each).
    then      one line per atom: x y z Species[+/-N]

We only ever need to touch the *species token* (the last whitespace field of
each atom line) -- for randomspin tagging, DLM relabelling, and the fitfc
spin-suffix fixup -- so the parser keeps every header line verbatim and only
splits atom lines into (coords_prefix, species).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# A species token: element symbol optionally followed by a +/-N charge/spin.
_SPECIES_RE = re.compile(r"^([A-Za-z][A-Za-z]?)([+-]\d+)?$")

# Strip a trailing +/-N (e.g. Co+2 -> Co, Cr-2 -> Cr) from a species token.
_SPIN_SUFFIX_RE = re.compile(r"([A-Za-z][A-Za-z]?)[+-]\d+")


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _is_number_line(line: str) -> bool:
    toks = line.split()
    if not toks:
        return False
    return all(_is_float(t) for t in toks)


def _is_atom_line(line: str) -> bool:
    """An ATAT atom line is 'x y z Species[...]': >=4 tokens, the first three
    numeric, and the last token non-numeric. Species labels can be arbitrary
    strings (element symbols, X1/X2 pseudo-species, Co+2, etc.), so we key off
    the numeric-coordinate shape rather than the species spelling."""
    toks = line.split()
    if len(toks) < 4:
        return False
    if not all(_is_float(t) for t in toks[:3]):
        return False
    return not _is_float(toks[-1])


@dataclass
class Structure:
    header: List[str]                 # coordinate-system + lattice-vector lines
    atoms: List[Tuple[str, str]]      # (coords_prefix, species_token)
    trailing: List[str]               # any blank/extra lines after atoms

    def species(self) -> List[str]:
        return [s for _, s in self.atoms]

    def to_text(self) -> str:
        out = list(self.header)
        for coords, sp in self.atoms:
            out.append(f"{coords} {sp}")
        out.extend(self.trailing)
        return "\n".join(out) + "\n"


def read_structure(path: Path) -> Structure:
    raw = Path(path).read_text().splitlines()
    # Number of header lines: 3 (coord sys as 3x3) + 3 (lattice) = 6, OR
    # 1 (coord sys as 6 numbers) + 3 = 4. Detect by counting how many of the
    # first lines are pure-number lines before the first atom line (which ends
    # in a non-numeric species token).
    header: List[str] = []
    atoms: List[Tuple[str, str]] = []
    trailing: List[str] = []
    in_atoms = False
    for line in raw:
        if not line.strip():
            if in_atoms:
                trailing.append(line)
            else:
                header.append(line)
            continue
        toks = line.split()
        last = toks[-1]
        is_atom = _is_atom_line(line)
        if not in_atoms and not is_atom:
            header.append(line)
        else:
            in_atoms = True
            coords = " ".join(toks[:-1])
            atoms.append((coords, last))
    return Structure(header=header, atoms=atoms, trailing=trailing)


def strip_spin_suffix_text(text: str) -> str:
    """Remove every +/-N spin/charge suffix from species tokens in raw text.

    Equivalent to the user's
        sed -e 's/Co+2/Co/g' -e 's/Cr+2/Cr/g' ... -e 's/-2//g'
    chain, but element-agnostic: turns Co+2->Co, Cr-2->Cr, Fe+4->Fe, etc.
    """
    return _SPIN_SUFFIX_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Cell parsing + lattice-drift metric (checkrelax analogue)
# ---------------------------------------------------------------------------
# Pure-python 3x3 helpers — the upstream package is deliberately
# numpy-free so it runs in any bare NAS python3.

def _mat_det(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _mat_inv(m):
    d = _mat_det(m)
    if abs(d) < 1e-12:
        raise ValueError("singular cell matrix")
    c = [[(m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3]
           - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3]) / d
          for i in range(3)] for j in range(3)]
    return c


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _mat_t(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def parse_cell(struct: Structure):
    """Cartesian lattice vectors (rows) of an ATAT structure file.

    Header layout is either 3 coordinate-system rows + 3 lattice rows
    (all of 3 numbers), or one '(a b c alpha beta gamma)' line + 3
    lattice rows. Lattice rows are expressed in the coordinate system,
    so Cartesian rows are L @ A.
    """
    import math
    num_lines = [[float(t) for t in ln.split()]
                 for ln in struct.header if ln.strip()
                 and all(_is_float(t) for t in ln.split())]
    if not num_lines:
        raise ValueError("no numeric header lines in structure")
    if len(num_lines[0]) == 6:
        a, b, c, al, be, ga = num_lines[0]
        al, be, ga = (math.radians(x) for x in (al, be, ga))
        cx = c * math.cos(be)
        cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
        cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        A = [[a, 0.0, 0.0],
             [b * math.cos(ga), b * math.sin(ga), 0.0],
             [cx, cy, cz]]
        rows = num_lines[1:4]
    else:
        A = num_lines[0:3]
        rows = num_lines[3:6]
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError("could not locate 3 lattice-vector rows")
    return _mat_mul(rows, A)


def cell_distortion(ideal_cell, relaxed_cell) -> float:
    """Volume-normalized lattice drift between two cells — the analogue
    of ATAT's `checkrelax` metric.

    The relaxed cell is isotropically rescaled to the ideal volume
    (uniform expansion is physical, not drift), the deformation gradient
    F mapping ideal -> relaxed is formed, and the Frobenius norm of the
    Green-Lagrange strain E = (F^T F - I)/2 is returned. Rigid rotations
    cancel in F^T F. Values >~ 0.1 mean the SQS relaxed away from its
    parent lattice (e.g. an unstable BCC composition sliding toward
    FCC) — its energy then belongs to a DIFFERENT phase and poisons the
    mixing-energy fit if kept.
    """
    vi = abs(_mat_det(ideal_cell))
    vr = abs(_mat_det(relaxed_cell))
    if vi < 1e-12 or vr < 1e-12:
        raise ValueError("degenerate cell")
    s = (vi / vr) ** (1.0 / 3.0)
    rel = [[s * x for x in row] for row in relaxed_cell]
    # Column-vector convention: F = rel_c . ideal_c^-1 with C_c = C^T.
    F = _mat_mul(_mat_t(rel), _mat_inv(_mat_t(ideal_cell)))
    FtF = _mat_mul(_mat_t(F), F)
    acc = 0.0
    for i in range(3):
        for j in range(3):
            e = 0.5 * (FtF[i][j] - (1.0 if i == j else 0.0))
            acc += e * e
    return acc ** 0.5


def lattice_drift(ideal_path: Path, relaxed_path: Path) -> float:
    """cell_distortion() straight from two ATAT structure files."""
    return cell_distortion(parse_cell(read_structure(ideal_path)),
                           parse_cell(read_structure(relaxed_path)))
