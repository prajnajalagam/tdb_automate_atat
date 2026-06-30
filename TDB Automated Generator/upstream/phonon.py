#!/usr/bin/env python3
"""
fitfc phonon workflow + DLM spin-suffix fixup.

fitfc stages (ATAT manual 7.1.9):
  1. fitfc -er=.. -ns=.. -ms=.. -dr=..   -> writes vol_* strain dirs.
  2. pollmach runstruct_vasp (relax under strain) ; touch stoppoll.
  3. fitfc (same args)                    -> writes perturbation subdirs.
  4. pollmach runstruct_vasp (frozen, force runs).
  5. [DLM only] strip +2/-2 spin tags from every str_relax.out / str_unpert.out
     so fitfc -f can read plain element symbols.
  6. fitfc -f -fr=..                      -> fits force constants; svib_ht etc.

DLM fixup
---------
The user's element-specific sed recipe

    sed -e s/Co+2/Co/g -i str_relax.out ; sed -e s/Cr+2/Cr/g -i ...
    ... ; sed -e s/-2//g -i str_relax.out
    foreachfile -d 2 str_relax.out  ... (same, recursively)
    foreachfile -d 2 str_unpert.out ...

is generalised here to dlm_fixup(): walk the SQS tree and strip *any* +/-N
spin/charge suffix from species tokens in every str_relax.out and
str_unpert.out, at the top level and in all subdirectories. Element-agnostic,
so it works for Co-Cr, Fe-Ni, etc. without editing per pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import runner
from strfile import strip_spin_suffix_text
from vaspwrap import build_vasp_wrap
from phases import DLMConfig

# Files that must have spin suffixes stripped before fitfc -f.
DLM_FIXUP_FILES = ("str_relax.out", "str_unpert.out")


def dlm_fixup(sqs_dir: Path,
              filenames=DLM_FIXUP_FILES) -> List[Path]:
    """Strip +/-N spin/charge suffixes from species tokens in every
    str_relax.out / str_unpert.out under sqs_dir (recursively).

    Returns the list of files that were modified. Idempotent: running twice is
    a no-op on already-clean files.
    """
    sqs_dir = Path(sqs_dir)
    changed: List[Path] = []
    for fn in filenames:
        for path in sqs_dir.rglob(fn):
            if not path.is_file():
                continue
            text = path.read_text()
            fixed = strip_spin_suffix_text(text)
            if fixed != text:
                path.write_text(fixed)
                changed.append(path)
    return changed


def _write_phonon_wrap(calc_dir: Path, encut: int, kppra: int,
                       dlm: Optional[DLMConfig], algo: str) -> None:
    wrap = build_vasp_wrap("phonon", encut=encut, kppra=kppra,
                           dlm=dlm, algo=algo)
    (Path(calc_dir) / "vasp.wrap").write_text(wrap)


def run_fitfc(sqs_dir: Path,
              encut: int,
              kppra: int,
              er: float = 11.5,
              ns: int = 3,
              ms: float = 0.02,
              dr: float = 0.1,
              fr: Optional[float] = None,
              dlm: Optional[DLMConfig] = None,
              algo: str = "All",
              env_bin: Optional[str] = None,
              timeout: int = 172800) -> Path:
    """Drive the full fitfc workflow in sqs_dir; returns the fitfc.out path.

    Assumes str.out (unrelaxed) and str_relax.out (relaxed) already exist
    (produced by relax.relax_structure). The DLM fixup is applied just before
    the final fitfc -f.
    """
    sqs_dir = Path(sqs_dir)
    fitfc_args = [f"-er={er}", f"-ns={ns}", f"-ms={ms}", f"-dr={dr}"]
    if fr is None:
        fr = er / 2.0

    # Phonon force runs use a frozen-geometry wrap.
    _write_phonon_wrap(sqs_dir, encut, kppra, dlm, algo)

    # 1. generate strain dirs
    runner.run_logged(["fitfc"] + fitfc_args, cwd=sqs_dir,
                      log=sqs_dir / "fitfc_gen_strain.log",
                      env_bin=env_bin, timeout=600, check=False)

    vol_dirs = sorted(sqs_dir.glob("vol_*"))

    # 2. relax under strain
    if vol_dirs:
        runner.run_polled(
            ["pollmach", "runstruct_vasp"], cwd=sqs_dir,
            log=sqs_dir / "fitfc_strain_runs.log",
            done_when=runner.all_energy_present(vol_dirs),
            stop_sentinel="stoppoll",
            env_bin=env_bin, timeout=timeout)

    # 3. regenerate perturbations
    runner.run_logged(["fitfc"] + fitfc_args, cwd=sqs_dir,
                      log=sqs_dir / "fitfc_gen_pert.log",
                      env_bin=env_bin, timeout=600, check=False)

    # 4. force runs for the perturbations
    pert_dirs = [d for v in vol_dirs for d in sorted(v.glob("p*")) if d.is_dir()]
    if pert_dirs:
        runner.run_polled(
            ["pollmach", "-lu", "runstruct_vasp"], cwd=sqs_dir,
            log=sqs_dir / "fitfc_force_runs.log",
            done_when=runner.all_have_file(pert_dirs, "force.out"),
            stop_sentinel="stoppoll",
            env_bin=env_bin, timeout=timeout)

    # 5. DLM fixup BEFORE the fit (fitfc -f can't parse Co+2 etc.)
    if dlm is not None and dlm.enabled:
        changed = dlm_fixup(sqs_dir)
        (sqs_dir / "dlm_fixup.log").write_text(
            "Stripped spin suffixes from:\n"
            + "\n".join(str(p.relative_to(sqs_dir)) for p in changed) + "\n")

    # 6. fit force constants
    runner.run_logged(["fitfc", "-f", f"-fr={fr}"], cwd=sqs_dir,
                      log=sqs_dir / "fitfc_fit.log",
                      env_bin=env_bin, timeout=3600, check=False)

    return sqs_dir / "fitfc.out"
