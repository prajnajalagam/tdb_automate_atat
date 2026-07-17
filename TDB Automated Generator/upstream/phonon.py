#!/usr/bin/env python3
"""
fitfc phonon workflow + DLM spin-suffix fixup.

Verified against the fitfc.c++ source ($atatdir/src/fitfc.c++). The facts
that shape this module:

* Generation mode (no -f) REQUIRES -er or -ernn and reads BOTH the ideal
  structure (-si, default str.out) and the relaxed one (-sr, default
  str_relax.out); it ERRORQUITs if either is missing. The sqs2tdb
  vibrational recipe passes ``-si=str_relax.out`` so ideal == relaxed.
* Generation writes vol_0 / vol_<strain%> dirs containing str.out (the
  relaxed structure stretched to that strain). Without -nrr it drops a
  ``wait`` marker and STOPS THERE: perturbations are only emitted for vol
  dirs that already contain str_relax.out ("Next, you need to relax the
  structures ... and rerun fitfc with the same command-line options").
  With -nrr it writes str_relax.out into the vol dir directly, so the
  SAME invocation proceeds to write the p*<dr>_<er>_<n> perturbation dirs
  (str.out perturbed, str_unpert.out, str_ideal.out, wait). fitfc's own
  help says -nrr is correct when the input is already relaxed and the
  calculation is harmonic — exactly our svib_ht use case.
* Fit mode (-f) REQUIRES -fr or -frnn. Per vol dir it reads
  str_relax.out, optional ``energy``, and per perturbation dir
  str_unpert.out + str_relax.out + force.out (NOT the perturbed str.out —
  the relaxed output of the frozen force run is what enters the fit).
  It writes fc.out, vdos.out, ``svib_ht`` and ``fvib`` INSIDE each vol
  dir, then fitfc.out / fvib / svib at the top level. It also reads
  ``../Trange.in`` (from the PHASE directory above the SQS dir) for the
  temperature grid when present.
* ``sqs2tdb -fit`` reads ``<sqs_dir>/svib_ht`` ONLY (top level), so the
  vol_0/svib_ht that fitfc produces must be copied up — the tutorial's
  ``cp vol_0/svib_ht .`` step, automated here.

Default recipe — the PUBLISHED sqs2tdb workflow (van de Walle et al.,
Calphad 58 (2017) 70, Sec. 3.3), harmonic, endmembers by default:

  1. fitfc -si=str_relax.out -ernn=4 -ns=1 -dr=0.04 -nrr
                                    -> vol_0 + p* dirs, one invocation
  2. pollmach runstruct_vasp -lu -w vaspf.wrap <launcher>
                                    -> force.out in every p* dir
     (vaspf.wrap is REWRITTEN by us over the -mk-generated one so its
      MAGMOM/NCORE/KPAR match the perturbation SUPERCELL, not the SQS)
  3. [DLM only] strip +2/-2 spin tags from str_relax.out / str_unpert.out
  4. fitfc -f -frnn=2 -si=str_relax.out            -> vol_0/svib_ht etc.
  5. robustrelax_vasp -vib                          -> svib_ht where
     sqs2tdb -fit reads it (cp vol_0/svib_ht . as fallback)
  -ernn/-frnn "may need to be adjusted depending on the alloy system"
  (paper); override via --fitfc-ernn/--fitfc-frnn/--fitfc-dr.

Quasiharmonic variant (ns > 1, nrr off): step 1 only writes vol_* dirs
with ``wait``; we then relax them (ions-only, fixed strained cell — a
per-vol ISIF=2 wrap that is removed afterwards so the p* force runs fall
through to the frozen top-level wrap), rerun fitfc with the SAME options
to emit the perturbations, and continue as above.

DLM fixup
---------
The user's element-specific sed recipe

    sed -e s/Co+2/Co/g -i str_relax.out ; sed -e s/Cr+2/Cr/g -i ...
    ... ; sed -e s/-2//g -i str_relax.out
    foreachfile -d 2 str_relax.out  ... (same, recursively)
    foreachfile -d 2 str_unpert.out ...

is generalised in dlm_fixup(): walk the SQS tree and strip *any* +/-N
spin/charge suffix from species tokens in every str_relax.out and
str_unpert.out — exactly the two files fitfc -f parses. Element-agnostic,
so it works for Co-Cr, Fe-Ni, etc. without editing per pair.
"""

from __future__ import annotations

import shutil
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


def _count_atoms(str_out: Path) -> Optional[int]:
    try:
        from strfile import read_structure
        return len(read_structure(str_out).atoms) or None
    except OSError:
        return None


def _write_force_wrap(sqs_dir: Path, pert_dirs: List[Path],
                      encut: int, kppra: int,
                      dlm: Optional[DLMConfig], algo: str) -> None:
    """Write vaspf.wrap sized for the PERTURBATION SUPERCELL.

    The force runs execute in vol_*/p*/ on the enclosing supercell
    fitfc built (e.g. 8 atoms for a 1-atom FCC endmember at -ernn=2),
    NOT on the SQS cell. MAGMOM must have exactly NIONS entries —
    VASP 6.6 hard-errors otherwise ("You have set 1 value(s) for
    MAGMOM; ... NIONS=8", seen in the 2026-07-16 e2e run when the
    wrap was sized from the 1-atom SQS cell). So this wrap is written
    AFTER fitfc generation, counting atoms from the first pending
    perturbation dir; it is also REwritten before escalated force
    runs, whose regenerated supercells are larger again. All p* dirs
    of a batch share one supercell, so one wrap per batch is exact.

    Separate wrap FILE (vaspf.wrap) so the frozen force-run settings
    never clobber (or get shadowed by) the relax-stage vasp.wrap in
    the same directory; the force runs select it with
    `runstruct_vasp -w vaspf.wrap` (the fitfc convention).
    """
    natoms = None
    for d in pert_dirs:
        for name in ("str.out", "str_unpert.out"):
            natoms = _count_atoms(d / name)
            if natoms:
                break
        if natoms:
            break
    wrap = build_vasp_wrap("phonon", encut=encut, kppra=kppra,
                           dlm=dlm, algo=algo, natoms=natoms)
    (Path(sqs_dir) / "vaspf.wrap").write_text(wrap)


def build_fitfc_gen_args(ernn: Optional[float], er: Optional[float],
                         ns: int, ms: float, dr: Optional[float],
                         nrr: bool) -> List[str]:
    """fitfc generation arguments (everything except the program name).

    -si=str_relax.out makes the relaxed structure the "ideal" one too, as
    in the sqs2tdb vibrational recipe — SQS relax hard enough that
    matching str.out against str_relax.out (fitfc's reorder_atoms) is
    fragile, and the perturbation spacegroup is found from the relaxed
    cell anyway. -er (absolute, Å) wins over -ernn (× nearest-neighbour
    distance) when both are given. -ms is only meaningful for ns > 1 but
    is harmless at ns=1 (strain grid collapses to 0), and the SAME args
    must be reused if fitfc has to be re-invoked after the vol_* relax.
    """
    if er is None and ernn is None:
        raise ValueError("fitfc generation needs -er or -ernn")
    args = ["-si=str_relax.out"]
    args.append(f"-er={er}" if er is not None else f"-ernn={ernn}")
    args.append(f"-ns={ns}")
    args.append(f"-ms={ms}")
    if dr is not None:
        args.append(f"-dr={dr}")
    if nrr:
        args.append("-nrr")
    return args


def build_fitfc_fit_args(frnn: Optional[float],
                         fr: Optional[float],
                         rl: Optional[float] = None,
                         fn: bool = False) -> List[str]:
    """fitfc fit-mode arguments: -f requires -fr or -frnn (fitfc
    ERRORQUITs without one). -fr (absolute, Å) wins over -frnn.

    Unstable-mode escape hatches, straight from the fitfc source (the
    fit aborts on ``Unstable modes found.`` unless one of these is on):
      rl  -> ``-rl=<len>``: robust-length treatment of soft modes (beta).
      fn  -> ``-fn``: force continuation even if unstable — the vibrational
             quantities then silently omit the imaginary branches, so any
             svib_ht from a -fn fit is a lower bound, not a clean value.
    """
    if fr is None and frnn is None:
        raise ValueError("fitfc -f needs -fr or -frnn")
    args = ["-f", "-si=str_relax.out"]
    args.append(f"-fr={fr}" if fr is not None else f"-frnn={frnn}")
    if rl is not None:
        args.append(f"-rl={rl}")
    if fn:
        args.append("-fn")
    return args


# What fitfc -f prints on instability (both go to stderr, which
# runner.run_logged folds into the log file):
#   per perturbation:  "Warning: <pert> is an unstable mode."  (dE < 0)
#   after the fit:     "Unstable modes found." then ERRORQUIT("Aborting.")
#                      unless -fn is set or -rl > 0.
UNSTABLE_MARKERS = ("Unstable modes found", "is an unstable mode")


def detect_unstable_modes(log_path: Path) -> List[str]:
    """Lines in a fitfc log that flag unstable/imaginary modes.
    Missing log -> [] (the fit never ran)."""
    log_path = Path(log_path)
    if not log_path.is_file():
        return []
    hits: List[str] = []
    for line in log_path.read_text(errors="replace").splitlines():
        if any(m in line for m in UNSTABLE_MARKERS):
            hits.append(line.strip())
    return hits


# Outputs of a previous fitfc -f that must not survive into (or be
# promoted after) a refit — an aborted unstable fit would otherwise
# leave stale svib_ht lying around to be promoted as if fresh.
_FIT_OUTPUTS_TOP = ("svib_ht", "fitfc.out", "fvib", "svib")


def _clear_stale_fit_outputs(sqs_dir: Path) -> None:
    for fn in _FIT_OUTPUTS_TOP:
        f = sqs_dir / fn
        if f.is_file():
            f.unlink()
    for vol in _vol_dirs(sqs_dir):
        for fn in ("svib_ht", "fvib"):
            f = vol / fn
            if f.is_file():
                f.unlink()


def _vol_dirs(sqs_dir: Path) -> List[Path]:
    return sorted(d for d in sqs_dir.glob("vol_*") if d.is_dir())


def _pert_dirs(sqs_dir: Path) -> List[Path]:
    return [d for v in _vol_dirs(sqs_dir)
            for d in sorted(v.glob("p*")) if d.is_dir()]


def all_force_runs_done(pert_dirs: List[Path]):
    """done_when predicate for the perturbation force runs: fitfc -f needs
    BOTH force.out and str_relax.out in every p* dir (it parses the
    relaxed output of the frozen run, not the input str.out)."""
    def _pred(_cwd: Path) -> bool:
        return all((d / "force.out").is_file()
                   and (d / "str_relax.out").is_file() for d in pert_dirs)
    return _pred


def promote_svib_ht(sqs_dir: Path) -> Optional[Path]:
    """Copy vol_0/svib_ht up to <sqs_dir>/svib_ht.

    fitfc -f writes svib_ht inside each vol_* dir, but ``sqs2tdb -fit``
    only ever opens ``<sqs_dir>/svib_ht`` — this is the tutorial's
    ``cp vol_0/svib_ht .`` step. Returns the top-level path, or None if
    no vol_0/svib_ht exists (fit failed or produced nothing).
    """
    sqs_dir = Path(sqs_dir)
    src = sqs_dir / "vol_0" / "svib_ht"
    if not src.is_file():
        return None
    dst = sqs_dir / "svib_ht"
    shutil.copy2(src, dst)
    return dst


def run_fitfc(sqs_dir: Path,
              encut: int,
              kppra: int,
              ernn: Optional[float] = 4.0,
              er: Optional[float] = None,
              frnn: Optional[float] = 2.0,
              fr: Optional[float] = None,
              ns: int = 1,
              ms: float = 0.02,
              dr: Optional[float] = 0.04,
              nrr: Optional[bool] = None,
              rl: Optional[float] = None,
              on_unstable: str = "mark",
              escalate_ernn: Optional[float] = None,
              dlm: Optional[DLMConfig] = None,
              algo: str = "All",
              env_bin: Optional[str] = None,
              timeout: int = 172800,
              cmd_prefix: str = "") -> Path:
    """Drive the full fitfc workflow in sqs_dir; returns the fitfc.out path.

    Requires str.out and str_relax.out in sqs_dir (produced by
    relax.relax_structure) — fitfc ERRORQUITs without the relaxed file.

    Defaults follow the sqs2tdb vibrational recipe: harmonic, single
    volume (-ernn=2 -ns=1 -nrr, fit with -frnn=1.5), which is all the
    downstream svib_ht fit consumes. Set ns > 1 for a quasiharmonic
    strain series: the vol_* dirs are then relaxed ions-only at fixed
    (strained) cell before fitfc is re-run with the same options to emit
    the perturbations, per fitfc's own two-invocation contract.

    nrr defaults to (ns == 1): at a single volume vol_0 IS the already-
    relaxed input, so re-relaxing it would waste a VASP run.

    Unstable modes: fitfc -f ABORTS on "Unstable modes found." (before
    writing svib_ht) unless -fn / -rl>0 is given. Safeguards here:
      * stale fit outputs (svib_ht, fitfc.out, fvib, svib — top level
        and per-vol) are deleted before every fit, so an aborted refit
        can never lead to a stale svib_ht being promoted as fresh;
      * any instability evidence is recorded in <sqs_dir>/unstable_modes.log;
      * on_unstable="mark" (default): leave the SQS without svib_ht —
        downstream (sqs2tdb -fit / the pipeline svib gates) then treats
        it as energy-only, which is honest: entropy from a fit that
        drops imaginary branches would bias the CALPHAD fit;
      * on_unstable="force": retry once with fitfc's own -fn (force
        continuation). The resulting svib_ht omits the unstable
        branches — the marker log records that provenance;
      * on_unstable="escalate": the dynamically-unstable-SQS workflow.
        Spurious imaginary modes are most often an artifact of a
        too-small displacement supercell (short-range force constants
        extrapolated to Γ), so the perturbations are REGENERATED at a
        larger radius (escalate_ernn, default 1.5x the original; the
        vol dirs already hold str_relax.out, so one fitfc invocation
        emits the new p* dirs), force runs are launched for the new
        dirs only (existing force.out equations stay in the fit), and
        the fit is retried. If the instability persists it is treated
        as likely GENUINE dynamical instability: the SQS is left
        energy-only and unstable_modes.log names the remaining manual
        options (tighter re-relaxation, -rl, fitfc -fu/-gu
        mode-following);
      * rl=<len> passes fitfc's -rl robust-length soft-mode treatment
        (beta) on the first attempt, which also prevents the abort.

    cmd_prefix: VASP launch command (e.g. "mpiexec -n 128") appended as
    trailing tokens to every pollmach runstruct_vasp invocation — same
    fix as converge/relax; a bare MPI vasp binary dies before writing
    output ("unable to open OSZICAR").
    """
    sqs_dir = Path(sqs_dir)
    if on_unstable not in ("mark", "force", "escalate"):
        raise ValueError(f"on_unstable must be 'mark', 'force' or "
                         f"'escalate', got {on_unstable!r}")
    if not (sqs_dir / "str_relax.out").is_file():
        raise RuntimeError(
            f"run_fitfc({sqs_dir}): no str_relax.out — fitfc generation "
            f"reads the relaxed structure and dies without it; run the "
            f"relaxation stage first.")

    if nrr is None:
        nrr = (ns == 1)
    gen_args = build_fitfc_gen_args(ernn=ernn, er=er, ns=ns, ms=ms,
                                    dr=dr, nrr=nrr)
    vasp_launch = runner.split_prefix(cmd_prefix)

    # NOTE: vaspf.wrap is written AFTER generation (see
    # _write_force_wrap) — its MAGMOM must be sized for the
    # perturbation supercell, whose atom count only exists once fitfc
    # has built the p* dirs.

    # 1. generate. With -nrr this single invocation writes vol_* AND the
    #    perturbation dirs (vol str_relax.out exists immediately).
    runner.run_logged(["fitfc"] + gen_args, cwd=sqs_dir,
                      log=sqs_dir / "fitfc_gen.log",
                      env_bin=env_bin, timeout=600, check=False)

    vol_dirs = _vol_dirs(sqs_dir)

    if not nrr:
        # 2a. relax each strained volume: ions only, cell FIXED at the
        #     strain fitfc imposed (ISIF=2 override of the relax wrap).
        #     The per-vol wrap shadows the frozen top-level one and is
        #     removed afterwards so the p* dirs underneath fall through
        #     to the frozen wrap for their force runs.
        pending = [d for d in vol_dirs if not (d / "str_relax.out").is_file()]
        if pending:
            try:
                from strfile import read_structure
                nat = len(read_structure(sqs_dir / "str.out").atoms) or None
            except OSError:
                nat = None
            relax_wrap = build_vasp_wrap("relax", encut=encut, kppra=kppra,
                                         dlm=dlm, algo=algo, natoms=nat,
                                         extra={"ISIF": 2})
            for d in pending:
                (d / "vasp.wrap").write_text(relax_wrap)
            runner.run_polled(
                ["pollmach", "runstruct_vasp"] + vasp_launch, cwd=sqs_dir,
                log=sqs_dir / "fitfc_strain_runs.log",
                done_when=runner.all_energy_present(pending),
                stop_sentinel="stoppoll",
                env_bin=env_bin, timeout=timeout)
            for d in pending:
                wrap = d / "vasp.wrap"
                if wrap.is_file():
                    wrap.unlink()

        # 2b. re-run fitfc with the SAME options — now that the vol dirs
        #     hold str_relax.out it emits the perturbation dirs (this is
        #     fitfc's own printed instruction).
        runner.run_logged(["fitfc"] + gen_args, cwd=sqs_dir,
                          log=sqs_dir / "fitfc_gen_pert.log",
                          env_bin=env_bin, timeout=600, check=False)
    else:
        # -nrr leaves no vol_0/energy; seed it from the SQS static energy
        # (same cell, same atom count) so fitfc.out / fvib include the
        # T=0 energy instead of fitfc's "assuming 0" fallback. svib_ht is
        # unaffected either way.
        top_energy = sqs_dir / "energy"
        if top_energy.is_file():
            for d in vol_dirs:
                if not (d / "energy").is_file():
                    shutil.copy2(top_energy, d / "energy")

    # 3. force runs for the perturbations (frozen vaspf.wrap via -w,
    #    launcher LAST). Wrap sized for the perturbation supercell.
    pert_dirs = _pert_dirs(sqs_dir)
    if pert_dirs:
        _write_force_wrap(sqs_dir, pert_dirs, encut, kppra, dlm, algo)
        runner.run_polled(
            ["pollmach", "runstruct_vasp", "-lu", "-w", "vaspf.wrap"]
            + vasp_launch, cwd=sqs_dir,
            log=sqs_dir / "fitfc_force_runs.log",
            done_when=all_force_runs_done(pert_dirs),
            stop_sentinel="stoppoll",
            env_bin=env_bin, timeout=timeout)

    # 4. DLM fixup BEFORE the fit — fitfc -f parses str_relax.out and
    #    str_unpert.out (top level, vol_* and p* alike) and can't match
    #    Co+2-style labels once the wrap's SUBATOM rules are out of play.
    if dlm is not None and dlm.enabled:
        changed = dlm_fixup(sqs_dir)
        (sqs_dir / "dlm_fixup.log").write_text(
            "Stripped spin suffixes from:\n"
            + "\n".join(str(p.relative_to(sqs_dir)) for p in changed) + "\n")

    # 5. fit force constants. fitfc also reads ../Trange.in (phase dir)
    #    for the T grid when present; svib_ht is T-independent.
    #    Clear previous fit outputs FIRST: an aborted unstable fit must
    #    not leave an old svib_ht behind for step 6 to promote.
    _clear_stale_fit_outputs(sqs_dir)
    fit_log = sqs_dir / "fitfc_fit.log"
    runner.run_logged(["fitfc"] + build_fitfc_fit_args(frnn=frnn, fr=fr,
                                                       rl=rl),
                      cwd=sqs_dir, log=fit_log,
                      env_bin=env_bin, timeout=3600, check=False)

    # 5b. unstable-mode safeguard. fitfc aborted before svib_ht unless
    #     -rl was on; escalate / force / mark per policy.
    unstable = detect_unstable_modes(fit_log)
    forced = False
    escalated = False
    escalation_fixed = False
    esc_desc = ""

    def _svib_missing() -> bool:
        return not (sqs_dir / "vol_0" / "svib_ht").is_file()

    if unstable and _svib_missing() and on_unstable == "escalate":
        # Dynamically-unstable-SQS workflow, step 1: rule out the
        # finite-supercell artifact by regenerating the perturbations
        # at a larger displacement radius. The vol dirs already hold
        # str_relax.out, so this single fitfc invocation writes the
        # new p* dirs directly; their names embed the new radius, so
        # the original dirs (and their force.out equations) survive
        # and stay in the refit.
        escalated = True
        if er is not None:
            esc_er, esc_ernn = er * 1.5, None
            esc_desc = f"-er={esc_er}"
        else:
            esc_er = None
            esc_ernn = escalate_ernn if escalate_ernn is not None \
                else ernn * 1.5
            esc_desc = f"-ernn={esc_ernn}"
        esc_gen = build_fitfc_gen_args(ernn=esc_ernn, er=esc_er,
                                       ns=ns, ms=ms, dr=dr, nrr=nrr)
        runner.run_logged(["fitfc"] + esc_gen, cwd=sqs_dir,
                          log=sqs_dir / "fitfc_gen_escalated.log",
                          env_bin=env_bin, timeout=600, check=False)
        new_pert = [d for d in _pert_dirs(sqs_dir)
                    if not (d / "force.out").is_file()]
        if new_pert:
            # Escalated supercells are larger (bigger -ernn): resize
            # the wrap's MAGMOM/decomposition for the NEW batch.
            _write_force_wrap(sqs_dir, new_pert, encut, kppra, dlm, algo)
            runner.run_polled(
                ["pollmach", "runstruct_vasp", "-lu", "-w", "vaspf.wrap"]
                + vasp_launch, cwd=sqs_dir,
                log=sqs_dir / "fitfc_force_runs_escalated.log",
                done_when=all_force_runs_done(new_pert),
                stop_sentinel="stoppoll",
                env_bin=env_bin, timeout=timeout)
        # The regenerated str_unpert.out / str_relax.out carry spin
        # tags again on a DLM run — re-strip before refitting.
        if dlm is not None and dlm.enabled:
            dlm_fixup(sqs_dir)
        _clear_stale_fit_outputs(sqs_dir)
        esc_log = sqs_dir / "fitfc_fit_escalated.log"
        runner.run_logged(
            ["fitfc"] + build_fitfc_fit_args(frnn=frnn, fr=fr, rl=rl),
            cwd=sqs_dir, log=esc_log,
            env_bin=env_bin, timeout=3600, check=False)
        esc_unstable = detect_unstable_modes(esc_log)
        escalation_fixed = not esc_unstable and not _svib_missing()
        unstable += esc_unstable

    if unstable and _svib_missing() and on_unstable == "force":
        forced = True
        runner.run_logged(
            ["fitfc"] + build_fitfc_fit_args(frnn=frnn, fr=fr, rl=rl,
                                             fn=True),
            cwd=sqs_dir, log=sqs_dir / "fitfc_fit_forced.log",
            env_bin=env_bin, timeout=3600, check=False)
        unstable += detect_unstable_modes(sqs_dir / "fitfc_fit_forced.log")
    if unstable:
        if forced:
            disposition = ("retried with -fn: svib_ht (if present) OMITS "
                           "the unstable branches — a lower bound, use "
                           "with care")
        elif escalation_fixed:
            disposition = (f"RESOLVED by escalating the displacement "
                           f"supercell ({esc_desc}): the imaginary modes "
                           f"were a finite-range artifact; svib_ht comes "
                           f"from the escalated fit")
        elif escalated:
            disposition = (f"PERSISTS after escalating to {esc_desc} — "
                           f"likely genuine dynamical instability at this "
                           f"composition. Left WITHOUT svib_ht "
                           f"(energy-only). Manual options: re-relax more "
                           f"tightly (EDIFFG, --relax-method normal), "
                           f"fitfc -rl=<len> robust soft-mode treatment, "
                           f"or fitfc -fu / -gu=<n> mode-following.")
        elif rl is not None:
            disposition = "soft modes handled by -rl robust-length treatment"
        else:
            disposition = ("left WITHOUT svib_ht (on_unstable='mark'): "
                           "downstream treats this SQS as energy-only")
        (sqs_dir / "unstable_modes.log").write_text(
            "fitfc -f reported unstable/imaginary modes:\n  "
            + "\n  ".join(dict.fromkeys(unstable))
            + f"\nDisposition: {disposition}\n")

    # 6. paper step (iv): `robustrelax_vasp -vib` copies the phonon
    #    outputs where sqs2tdb -fit can read them (Calphad 58 (2017)
    #    70, Sec. 3.3). Our promote_svib_ht stays as a fallback for
    #    ATAT builds where -vib is absent/quiet.
    runner.run_logged(["robustrelax_vasp", "-vib"], cwd=sqs_dir,
                      log=sqs_dir / "robustrelax_vib.log",
                      env_bin=env_bin, timeout=600, check=False)
    if not (sqs_dir / "svib_ht").is_file() \
            and promote_svib_ht(sqs_dir) is None:
        with open(fit_log, "a") as fh:
            fh.write("\nWARNING: no svib_ht (neither robustrelax -vib "
                     "nor vol_0/svib_ht) — "
                     + ("unstable modes (see unstable_modes.log)"
                        if unstable
                        else "fit failed or found no force.out")
                     + "; svib_ht NOT available.\n")

    return sqs_dir / "fitfc.out"
