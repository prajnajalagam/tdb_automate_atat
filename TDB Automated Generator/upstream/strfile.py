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
