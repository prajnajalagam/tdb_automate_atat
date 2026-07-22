#!/usr/bin/env python3
"""
SQS generation, *_small copy, randomspin, and the SIGMA lev=3 -> lev=0
+/-spin endmember conversion.

Pipeline entry points
---------------------
generate_phase_sqs   run ``sqs2tdb -cp -l=<PHASE>`` (or the *_small variant),
                     optionally apply randomspin for a DLM run.
sigma_lev3_to_lev0_dlm   convert a SIGMA_D8B lev=3 SQS into a lev=0 DLM
                     endmember by relabelling its two pseudo-species to
                     <EL>+2 / <EL>-2 (the piece that was "not implemented yet").

All ATAT commands are executed via runner.run_logged so failures land in a
per-directory log instead of vanishing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import re

from phases import (
    SMALL_SYSTEM, SINGLE_SUBLATTICE_PHASES, SigmaDLMSpec,
    DLM_SPIN_UP, DLM_SPIN_DOWN,
)

# Element-DECORATED calc dirs written by sqs2tdb -cp carry substituted
# species in their composition tokens ("sqs_lev=0_a_Co=1"); the raw
# DATABASE entries under $atatdir/data/sqsdb do not ("sqsdb_lev=0_a=1").
DECORATED_SQS_RE = re.compile(r"lev=\d+.*_[A-Z][a-z]?[+-]?\d*=")
from strfile import read_structure, Structure
import runner


def _sqs2tdb_target(phase: str, use_small: bool) -> str:
    """The -l= argument for sqs2tdb -cp. Single-sublattice phases use the
    *_small system when use_small is set (required before randomspin)."""
    if use_small and phase in SMALL_SYSTEM:
        return SMALL_SYSTEM[phase]
    return phase


def generate_phase_sqs(work_root: Path,
                       phase: str,
                       elements: Optional[List[str]] = None,
                       level: Optional[int] = None,
                       dlm: bool = False,
                       use_small: Optional[bool] = None,
                       species_edit=None,
                       env_bin: Optional[str] = None,
                       timeout: int = 600) -> Path:
    """Generate the SQS directory tree for one phase under work_root.

    sqs2tdb -cp is a TWO-PASS command: the first invocation only creates
    <target>/species.in and exits (with rc=0 and the message "Edit the file
    ... and rerun the same command"); only the second invocation copies the
    SQS structures from the database. We therefore always run it twice, with
    an optional ``species_edit(species_in_path)`` hook between the passes
    (e.g. to impose the SIGMA_D8B per-sublattice +/- spin convention).
    The double run is idempotent: if species.in already exists, pass 1 does
    the copy and pass 2 is a no-op.

    elements  passed as -sp=El1,El2. Required unless a species.in already
              exists at the work_root level (sqs2tdb falls back to it).
    level     composition-mesh cutoff passed as ``-lv=<n>`` (the actual
              flag name -- NOT -lev, which sqs2tdb silently ignores).
              CUMULATIVE semantics, per the sqs2tdb source
              (`if ($levs[1] <= $cmdline{"-lv"})`): -lv=N copies ALL
              database entries with level <= N, so -lv=2 yields the
              lev=0 endmembers, lev=1 midpoints AND lev=2 mesh in one
              invocation. Omitting it behaves like -lv=0 (endmembers
              only) because undef compares as 0 in Perl. Re-invoking
              with a larger N later only ADDS the new levels: sqs2tdb
              skips any sqs_ dir that already has str.out.
    use_small  default: True for single-sublattice phases, False otherwise.
    Runs ``randomspin`` in the produced *_small directory when dlm is set.
    Returns the directory sqs2tdb populated (work_root/<target>).
    Raises RuntimeError if the copy produced neither str.out files nor
    `link` files. (`link`-only dirs are legitimate: sqs2tdb writes a
    link instead of structure files when an SQS is symmetry-equivalent
    to a permuted-site twin or when an endmember reduces to a parent
    lattice via parentlat.in.)
    """
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    if use_small is None:
        use_small = phase in SINGLE_SUBLATTICE_PHASES
    target = _sqs2tdb_target(phase, use_small)

    cmd = ["sqs2tdb", "-cp", f"-l={target}"]
    if level is not None:
        cmd.append(f"-lv={level}")
    if elements:
        cmd.append(f"-sp={','.join(elements)}")
        # A WORK-ROOT-level species.in (as opposed to the per-lattice
        # <target>/species.in the two-pass protocol manages) breaks
        # the handshake: sqs2tdb keeps prompting on every pass and
        # never copies (2026-07-22: HCP_A3_small generated fine from a
        # clean cwd but failed in the work root, which carried a stray
        # 'c=Co,Cr' species.in from an earlier attempt). With -sp
        # given explicitly the file is never needed — quarantine it.
        stray = work_root / "species.in"
        if stray.is_file():
            bak = work_root / "species.in.stray"
            print(f"    WARNING: work-root {stray} interferes with the "
                  f"sqs2tdb -cp two-pass protocol; renaming to {bak.name}")
            stray.replace(bak)
    elif not (work_root / "species.in").is_file():
        raise RuntimeError(
            f"generate_phase_sqs({phase}): no elements given and no "
            f"species.in in {work_root}; sqs2tdb -cp would have nothing "
            f"to generate for.")

    # Pass 1: creates <target>/species.in and exits.
    runner.run_logged(cmd, cwd=work_root,
                      log=work_root / f"sqs2tdb_cp_{target}.log",
                      env_bin=env_bin, timeout=timeout)

    species_in = work_root / target / "species.in"
    if species_edit and species_in.is_file():
        species_edit(species_in)

    # Pass 2: actually copies the SQS structures.
    pass2_log = work_root / f"sqs2tdb_cp_{target}.2.log"
    runner.run_logged(cmd, cwd=work_root,
                      log=pass2_log,
                      env_bin=env_bin, timeout=timeout)

    # Pass 2 must NOT still be prompting: sqs2tdb prints "Edit the file
    # ... and rerun" only when <target>/species.in is absent, i.e. pass
    # 1 failed to plant it. Observed 2026-07-22 for HCP_A3_small (the
    # lattice was missing from $atatdir/data/sqsdb): both passes
    # prompted, nothing was generated, and the old work_root fallback
    # below silently returned the WORK ROOT as the phase dir — the
    # pipeline then "discovered" the OTHER phases' 10 SQS dirs and
    # reprocessed FCC+BCC under the HCP banner (re-relaxing finished
    # dirs, regenerating phonons). Fail loudly instead.
    try:
        pass2_text = pass2_log.read_text()
    except OSError:
        pass2_text = ""
    if "Edit the file" in pass2_text:
        raise RuntimeError(
            f"sqs2tdb -cp -l={target}: SECOND pass is still prompting "
            f"to edit {target}/species.in, i.e. pass 1 never created "
            f"it. This almost always means the lattice '{target}' is "
            f"missing from $atatdir/data/sqsdb (check with: ls "
            f"$(dirname $(which sqs2tdb))/../data/sqsdb). Install the "
            f"lattice there (or drop it from --phases); see {pass2_log}")

    target_dir = work_root / target
    if not target_dir.is_dir():
        raise RuntimeError(
            f"sqs2tdb -cp -l={target} exited cleanly but never created "
            f"{target_dir}. The old fallback of treating the WORK ROOT "
            f"as the phase dir is forbidden — it aliases this phase "
            f"onto every other phase's SQS dirs (the 2026-07-22 "
            f"missing-HCP_A3_small incident). Check that '{target}' "
            f"exists in $atatdir/data/sqsdb; see {pass2_log}")

    # Verify the copy actually happened -- both passes exit 0 even when
    # nothing was copied, so rc alone proves nothing. Only element-
    # DECORATED dirs count: a stray copy of the raw database (sqsdb_lev=*
    # with undecorated names) in the work root satisfied the old
    # any-str.out check in the 2026-07-16 e2e run while -cp had actually
    # produced nothing. Accept `link`-only decorated dirs too (permuted-
    # site twins / parentlat.in reductions have no str.out of their own).
    def _has_decorated_output(root: Path) -> bool:
        for d in root.rglob("*"):
            if d.is_dir() and DECORATED_SQS_RE.search(d.name) \
                    and ((d / "str.out").is_file()
                         or (d / "link").is_file()):
                return True
        return False

    if not _has_decorated_output(target_dir):
        raise RuntimeError(
            f"sqs2tdb -cp -l={target} produced no element-decorated "
            f"sqs dirs (str.out or link) under {target_dir} after two "
            f"passes; see {work_root / f'sqs2tdb_cp_{target}.2.log'}. "
            f"(Raw sqsdb_lev= database entries do not count.) Check "
            f"that '{target}' exists in $atatdir/data/sqsdb and that "
            f"species.in lists the intended elements.")

    if dlm and phase in SINGLE_SUBLATTICE_PHASES:
        apply_randomspin(target_dir, env_bin=env_bin, timeout=timeout)

    return target_dir


def apply_randomspin(small_dir: Path,
                     env_bin: Optional[str] = None,
                     timeout: int = 120) -> None:
    """Run ``randomspin`` inside a *_small directory so disordered-spin sites
    in str.out gain +2 / -2 tags. Idempotent-ish: randomspin re-randomises,
    so only call once per generation."""
    small_dir = Path(small_dir)
    str_out = small_dir / "str.out"
    if not str_out.is_file():
        # randomspin may operate on subdirectory str.out files; run anyway and
        # let the log capture any complaint.
        pass
    runner.run_logged(["randomspin"], cwd=small_dir,
                      log=small_dir / "randomspin.log",
                      env_bin=env_bin, timeout=timeout)


# ---------------------------------------------------------------------------
# SIGMA lev=3 -> lev=0 DLM endmember conversion
# ---------------------------------------------------------------------------

def _autodetect_tokens(struct: Structure) -> List[str]:
    """The distinct species symbols present in a structure, order-stable."""
    seen: List[str] = []
    for sp in struct.species():
        # strip any pre-existing +/-N so we compare bare symbols
        base = sp
        for sign in ("+", "-"):
            if sign in base:
                base = base.split(sign)[0]
        if base not in seen:
            seen.append(base)
    return seen


def sigma_lev3_to_lev0_dlm(src_dir: Path,
                           dst_dir: Path,
                           spec: SigmaDLMSpec,
                           copy_aux: bool = True) -> Path:
    """Convert a SIGMA_D8B lev=3 SQS into a lev=0 DLM endmember directory.

    The lev=3 SQS already has the correct random *spatial* split of two
    pseudo-species across the sites. We reinterpret those two pseudo-species
    as spin-up / spin-down of a single element:

        token_up   -> "<element>+2"   (the "_A" / spin-up species)
        token_down -> "<element>-2"   (the "_B" / spin-down species)

    so the result is a single-element endmember whose net moment averages to
    zero (DLM). When spec.token_up / token_down are None they are auto-detected
    as the two distinct species symbols present in str.out (sorted for
    determinism).

    Writes a new str.out under dst_dir and, if copy_aux, copies any energy /
    str_relax.out / POTCAR / vasp.wrap already present in src_dir.
    Returns dst_dir.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    struct = read_structure(src_dir / "str.out")

    tok_up = spec.token_up
    tok_down = spec.token_down
    if tok_up is None or tok_down is None:
        detected = sorted(_autodetect_tokens(struct))
        if len(detected) < 2:
            raise ValueError(
                f"SIGMA lev=3 str.out in {src_dir} has fewer than two "
                f"pseudo-species ({detected}); cannot build a DLM split.")
        tok_up = tok_up or detected[0]
        tok_down = tok_down or detected[1]

    up_label = f"{spec.element}{DLM_SPIN_UP}"      # e.g. Co+2
    down_label = f"{spec.element}{DLM_SPIN_DOWN}"  # e.g. Co-2

    new_atoms = []
    for coords, sp in struct.atoms:
        base = sp.split("+")[0].split("-")[0]
        if base == tok_up:
            new_atoms.append((coords, up_label))
        elif base == tok_down:
            new_atoms.append((coords, down_label))
        else:
            # Spectator / single-occupancy site -- leave untouched.
            new_atoms.append((coords, sp))
    struct.atoms = new_atoms

    (dst_dir / "str.out").write_text(struct.to_text())

    if copy_aux:
        for fn in ("energy", "str_relax.out", "POTCAR", "vasp.wrap",
                   "species.in", "mult.in"):
            src = src_dir / fn
            if src.is_file():
                shutil.copy2(src, dst_dir / fn)

    return dst_dir


def copy_small_systems(template_root: Path,
                       work_root: Path,
                       phases: List[str]) -> List[Path]:
    """DEPRECATED (2026-07-16): copying raw sqsdb database entries into
    the WORK ROOT is useless (sqs2tdb -cp reads its database from
    $atatdir/data/sqsdb, never from the work root) and polluted the
    e2e run's discovery with undecorated sqsdb_lev=* entries. If a
    *_small lattice is missing from an ATAT install, add it under
    $atatdir/data/sqsdb instead. Kept only for API compatibility;
    no longer called by run_upstream."""
    template_root = Path(template_root)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    for ph in phases:
        name = SMALL_SYSTEM.get(ph)
        if not name:
            continue
        src = template_root / name
        if not src.is_dir():
            continue
        dst = work_root / name
        if dst.exists():
            continue
        shutil.copytree(src, dst)
        made.append(dst)
    return made
