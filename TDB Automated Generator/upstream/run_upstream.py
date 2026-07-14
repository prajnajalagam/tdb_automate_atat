#!/usr/bin/env python3
"""
STEP 0: Automated upstream generator for the binary sqs2tdb pipeline.

End-to-end per phase, for a binary A-B:

    generate SQS (sqs2tdb -cp [-l=*_small])         sqsgen
      |- DLM: randomspin in *_small                 sqsgen.apply_randomspin
      |- SIGMA: lev=3 -> lev=0 +/-spin endmembers   sqsgen.sigma_lev3_to_lev0_dlm
    -> ENCUT/KPPRA convergence (static, 1 meV/atom) converge
    -> structural relaxation (normal | infdet)      relax
    -> fitfc phonons (+ DLM spin-suffix fixup)      phonon
    -> hand directory tree to sqs2tdb_pipeline.py   (downstream fit)

This script *drives VASP on the node*: it calls runstruct_vasp / pollmach /
robustrelax_vasp / fitfc directly (generate-and-submit/poll model). It is meant
to run inside a PBS job on a compute node with ATAT + VASP on PATH (see
submit_upstream_template.pbs).

The convergence-selection, vasp.wrap-generation, SIGMA-spin-conversion and
DLM-fixup logic are unit-tested (tests/); the VASP-driving glue can only be
exercised on a real node.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

# Make sibling modules importable whether run as a script or as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import converge
import phonon
import potcar
import relax
import sqsgen
import vaspwrap
from phases import (
    ALL_PHASES, SINGLE_SUBLATTICE_PHASES, ENDMEMBER_ONLY_PHASES,
    SMALL_SYSTEM, DLMConfig, SigmaDLMSpec,
)
from strfile import read_structure


def parse_dlm_moments(spec: Optional[str],
                      elements: List[str]):
    """Parse '--dlm-moments' into an element -> (potcar_label, moment) map.

    Each comma-separated entry is 'EL[=POTCAR][:moment]':
      'Co'              -> ('Co', 2.0)
      'Cr=Cr_pv:1.5'    -> ('Cr_pv', 1.5)
      'Ni:0.7'          -> ('Ni', 0.7)
    Elements present in `elements` but absent from `spec` default to
    (element, 2.0) so a DLM run always has a complete SUBATOM map.
    """
    out = {}
    if spec:
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            moment = 2.0
            if ":" in entry:
                entry, mom_s = entry.split(":", 1)
                moment = float(mom_s)
            if "=" in entry:
                el, pot = entry.split("=", 1)
            else:
                el, pot = entry, entry
            out[el.strip()] = (pot.strip(), moment)
    for el in elements:
        out.setdefault(el, (el, 2.0))
    return out


class _LiveLog:
    """Mirror a stream (stdout/stderr) into a live log file.

    Why: PBS only delivers #PBS -o output after the job ENDS, and Python
    buffers stdout when it isn't a terminal — so a running upstream job
    is a black box. This tee writes every line to a file in the work
    root with a timestamp prefix and flushes both sinks per write, so

        tail -f <work_root>/upstream_live.log

    shows orchestrator progress in real time. Per-command detail (VASP
    chatter, sqs2tdb output) still goes to the per-step logs written by
    runner.run_logged / run_polled — this file is the step-level index.
    """

    def __init__(self, stream, logfile: Path):
        self._stream = stream
        self._fh = open(logfile, "a", buffering=1)
        self._at_line_start = True

    def write(self, text: str) -> int:
        n = self._stream.write(text)
        self._stream.flush()
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start and chunk.strip():
                self._fh.write(time.strftime("[%Y-%m-%d %H:%M:%S] "))
            self._fh.write(chunk)
            self._at_line_start = chunk.endswith("\n")
        self._fh.flush()
        return n

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def isatty(self) -> bool:          # keep argparse/help behaviour sane
        return False


def install_live_log(work_root: Path) -> Path:
    """Tee stdout+stderr into <work_root>/upstream_live.log."""
    logfile = Path(work_root) / "upstream_live.log"
    sys.stdout = _LiveLog(sys.stdout, logfile)
    sys.stderr = _LiveLog(sys.stderr, logfile)
    return logfile


def stamp(msg: str) -> None:
    """Step marker: timestamped both on screen and in the live log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def discover_sqs_dirs(phase_root: Path) -> List[Path]:
    """All lev>0 SQS directories produced by sqs2tdb -cp under phase_root."""
    out: List[Path] = []
    for d in sorted(phase_root.rglob("*")):
        if d.is_dir() and (d / "str.out").is_file() and "lev=" in d.name:
            out.append(d)
    if not out and (phase_root / "str.out").is_file():
        out = [phase_root]
    return out


def process_one_sqs(sqs_dir: Path,
                    potcar_paths: List[Path],
                    dlm: DLMConfig,
                    relax_method: str,
                    algo: str,
                    tol_ev: float,
                    env_bin: Optional[str],
                    skip_phonon: bool,
                    timeout: int,
                    cmd_prefix: str = "",
                    relax_opts: str = "",
                    fitfc_opts: Optional[Dict] = None,
                    preset_encut: Optional[int] = None,
                    preset_kppra: Optional[int] = None) -> Dict:
    """Convergence -> relax -> phonon for a single SQS directory.

    fitfc_opts: extra keyword args for phonon.run_fitfc, e.g.
    {"on_unstable": "force", "rl": 0.3} — the unstable-mode policy.

    preset_encut/preset_kppra: reuse settings converged on a sibling SQS
    (phase-uniform convergence scope) instead of sweeping again. Mixing
    energies subtract eV-scale totals of different structures — that
    subtraction only cancels basis/k-mesh error when every structure in
    the phase is computed at the SAME settings, so uniform settings are
    both cheaper and more correct than per-SQS sweeps.

    cmd_prefix is the VASP launch command ("mpiexec -n 128") forwarded
    to every runstruct_vasp / robustrelax_vasp invocation as trailing
    arguments — without it the MPI vasp binary is launched bare and
    dies before writing OSZICAR.
    """
    sweep_root = sqs_dir / "convergence"

    if preset_encut is not None and preset_kppra is not None:
        chosen_encut, chosen_kppra = preset_encut, preset_kppra
        kres = eres = None
        stamp(f"[{sqs_dir.name}] STAGE 1/3 convergence reused from phase "
              f"scope: ENCUT={chosen_encut} eV, KPPRA={chosen_kppra}")
    else:
        stamp(f"[{sqs_dir.name}] STAGE 1/3 convergence sweep starting "
              f"(watch {sweep_root}/*/vasp.log)")
        chosen_encut, chosen_kppra, kres, eres = converge.converge_sqs(
            sqs_dir, sweep_root, potcar_paths,
            dlm=dlm, algo=algo, tol_ev=tol_ev,
            env_bin=env_bin, timeout=timeout,
            cmd_prefix=cmd_prefix)
        print(kres.table())
        print(eres.table())
        print(f"    -> chosen ENCUT={chosen_encut} eV, KPPRA={chosen_kppra}")

    # ISIF=3 relaxations need a Pulay-safe basis: the sweep converges the
    # ENERGY, but the stress tensor converges much more slowly, and cell
    # relaxation at the energy-converged ENCUT gives systematically small
    # volumes. Floor at 1.3 x max(ENMAX) (VASP guidance); statics/phonons
    # keep the sweep-chosen value.
    relax_encut = chosen_encut
    if potcar_paths:
        relax_encut = potcar.pulay_safe_encut(
            chosen_encut, potcar.max_enmax(potcar_paths))
        if relax_encut != chosen_encut:
            stamp(f"[{sqs_dir.name}] relax ENCUT raised {chosen_encut} -> "
                  f"{relax_encut} eV (Pulay-stress floor for ISIF=3)")

    stamp(f"[{sqs_dir.name}] STAGE 2/3 relaxation starting "
          f"(method={relax_method}; watch {sqs_dir}/"
          f"{'runstruct' if relax_method == 'runstruct' else 'robustrelax_' + relax_method}.log)")
    relax.relax_structure(
        sqs_dir, encut=relax_encut, kppra=chosen_kppra,
        method=relax_method, dlm=dlm, algo=algo,
        env_bin=env_bin, timeout=timeout,
        cmd_prefix=cmd_prefix, relax_opts=relax_opts)
    stamp(f"[{sqs_dir.name}] STAGE 2/3 relaxation done "
          f"(str_relax.out present: "
          f"{(sqs_dir / 'str_relax.out').is_file()})")

    # Clear the ATAT 'wait' queue marker once the relax has produced its
    # result — sqs2tdb -cp drops a `wait` file in every to-be-computed
    # dir, and pollmach-style pollers treat its presence as "pending".
    # Mirrors the manual `rm wait` in the reference NAS workflow.
    if (sqs_dir / "str_relax.out").is_file():
        wait_marker = sqs_dir / "wait"
        if wait_marker.is_file():
            wait_marker.unlink()

    phonon_out = None
    if not skip_phonon:
        stamp(f"[{sqs_dir.name}] STAGE 3/3 fitfc phonons starting")
        phonon_out = str(phonon.run_fitfc(
            sqs_dir, encut=chosen_encut, kppra=chosen_kppra,
            dlm=dlm, algo=algo, env_bin=env_bin, timeout=timeout,
            cmd_prefix=cmd_prefix, **(fitfc_opts or {})))
        unstable_log = sqs_dir / "unstable_modes.log"
        if unstable_log.is_file():
            stamp(f"[{sqs_dir.name}] STAGE 3/3 UNSTABLE MODES reported "
                  f"by fitfc -f (see {unstable_log})")
        stamp(f"[{sqs_dir.name}] STAGE 3/3 fitfc phonons done "
              f"(svib_ht present: {(sqs_dir / 'svib_ht').is_file()})")
    else:
        stamp(f"[{sqs_dir.name}] STAGE 3/3 phonons skipped (--skip-phonon)")

    return {
        "sqs_dir": str(sqs_dir),
        "chosen_encut": chosen_encut,
        "chosen_kppra": chosen_kppra,
        "relax_encut": relax_encut,
        "convergence_reused": kres is None,
        "kppra_converged": kres.converged if kres else None,
        "encut_converged": eres.converged if eres else None,
        "relax_method": relax_method,
        "str_relax_present": (sqs_dir / "str_relax.out").is_file(),
        "phonon_out": phonon_out,
        "svib_ht_present": (sqs_dir / "svib_ht").is_file(),
        "unstable_modes": (sqs_dir / "unstable_modes.log").is_file(),
    }


def process_phase(phase: str,
                  work_root: Path,
                  potcar_paths: List[Path],
                  dlm: DLMConfig,
                  relax_method: str,
                  algo: str,
                  tol_ev: float,
                  sqs_levels: List[int],
                  sigma_elements: List[str],
                  template_root: Optional[Path],
                  env_bin: Optional[str],
                  skip_phonon: bool,
                  timeout: int,
                  cmd_prefix: str = "",
                  relax_opts: str = "",
                  fitfc_opts: Optional[Dict] = None,
                  convergence_scope: str = "phase") -> Dict:
    print(f"\n{'='*70}\n  PHASE {phase}\n{'='*70}")

    # Copy *_small template if provided (caveat 1).
    if template_root and phase in SMALL_SYSTEM:
        made = sqsgen.copy_small_systems(template_root, work_root, [phase])
        if made:
            print(f"    copied template: {made[0].name}")

    # SIGMA endmember-only handling (caveat 2).
    if phase in ENDMEMBER_ONLY_PHASES:
        # SIGMA in a binary has no composition mesh — endmember corners
        # only. Pass sqs_levels through for the (rare) DLM sigma_from_lev3
        # override; process_sigma decides what to actually do.
        return process_sigma(phase, work_root, potcar_paths, dlm, relax_method,
                             algo, tol_ev, sqs_levels, sigma_elements, env_bin,
                             skip_phonon, timeout,
                             cmd_prefix=cmd_prefix, relax_opts=relax_opts,
                             fitfc_opts=fitfc_opts,
                             convergence_scope=convergence_scope)

    # sqs2tdb -cp -lv=N has CUMULATIVE semantics (its copy loop tests
    # `level <= -lv`), so a single invocation at max(sqs_levels) copies
    # every mesh level up to and including that value — no per-level
    # loop needed. The iterative-refinement workflow still holds:
    # re-running later with a larger --sqs-level only ADDS the new
    # levels, because sqs2tdb skips any sqs_ dir that already has
    # str.out.
    gen_level = max(sqs_levels)
    print(f"    Generating SQS with -lv={gen_level} "
          f"(cumulative: copies all levels 0..{gen_level})")
    phase_root = sqsgen.generate_phase_sqs(
        work_root, phase, elements=sigma_elements,
        level=gen_level, dlm=dlm.enabled,
        env_bin=env_bin)

    sqs_dirs = discover_sqs_dirs(phase_root)
    print(f"    {len(sqs_dirs)} SQS directories")

    # Phase-uniform convergence (default): sweep on the FIRST SQS only
    # and reuse its (ENCUT, KPPRA) for every sibling — mixing energies
    # subtract eV-scale totals, and that subtraction only cancels
    # basis/k-mesh error when all structures share the same settings.
    # --convergence-scope sqs restores the old per-SQS sweeps.
    results = []
    preset: Optional[tuple] = None
    for d in sqs_dirs:
        print(f"\n  -- SQS {d.name} --")
        res = process_one_sqs(
            d, potcar_paths, dlm, relax_method, algo, tol_ev,
            env_bin, skip_phonon, timeout,
            cmd_prefix=cmd_prefix, relax_opts=relax_opts,
            fitfc_opts=fitfc_opts,
            preset_encut=preset[0] if preset else None,
            preset_kppra=preset[1] if preset else None)
        results.append(res)
        if convergence_scope == "phase" and preset is None:
            preset = (res["chosen_encut"], res["chosen_kppra"])
    return {"phase": phase, "sqs": results}


def process_sigma(phase: str,
                  work_root: Path,
                  potcar_paths: List[Path],
                  dlm: DLMConfig,
                  relax_method: str,
                  algo: str,
                  tol_ev: float,
                  sqs_levels: List[int],
                  sigma_elements: List[str],
                  env_bin: Optional[str],
                  skip_phonon: bool,
                  timeout: int,
                  cmd_prefix: str = "",
                  relax_opts: str = "",
                  fitfc_opts: Optional[Dict] = None,
                  convergence_scope: str = "phase") -> Dict:
    """SIGMA_D8B: endmembers only. Under DLM, build each endmember from a
    lev=3 SQS via the lev=3 -> lev=0 +/-spin conversion (caveat 2)."""
    # For DLM SIGMA we must generate at lev=3 (randomises each site among 2
    # species); otherwise the usual endmember (lev=0) generation is used.
    # SIGMA in a binary has no mixing composition mesh, so it is NOT
    # subject to the multi-level iteration used for FCC/BCC/HCP —
    # sqs_levels is only consulted for the DLM override path.
    gen_level = 3 if (dlm.enabled and dlm.sigma_from_lev3) else 0
    phase_root = sqsgen.generate_phase_sqs(
        work_root, phase, elements=sigma_elements,
        level=gen_level, dlm=False, use_small=False,
        env_bin=env_bin)

    endmember_dirs: List[Path] = []
    if dlm.enabled and dlm.sigma_from_lev3:
        lev3_dirs = [d for d in discover_sqs_dirs(phase_root)
                     if "lev=3" in d.name]
        print(f"    converting {len(lev3_dirs)} lev=3 SQS -> lev=0 DLM "
              f"endmembers for elements {sigma_elements}")
        dlm_root = phase_root / "dlm_endmembers"
        for el in sigma_elements:
            for i, src in enumerate(lev3_dirs):
                dst = dlm_root / f"{el}_lev0_dlm_{i}"
                sqsgen.sigma_lev3_to_lev0_dlm(
                    src, dst, SigmaDLMSpec(element=el))
                endmember_dirs.append(dst)
    else:
        endmember_dirs = [d for d in discover_sqs_dirs(phase_root)
                          if "lev=0" in d.name] or discover_sqs_dirs(phase_root)

    # Same phase-uniform convergence policy as process_phase.
    results = []
    preset: Optional[tuple] = None
    for d in endmember_dirs:
        print(f"\n  -- SIGMA endmember {d.name} --")
        res = process_one_sqs(
            d, potcar_paths, dlm, relax_method, algo, tol_ev,
            env_bin, skip_phonon, timeout,
            cmd_prefix=cmd_prefix, relax_opts=relax_opts,
            fitfc_opts=fitfc_opts,
            preset_encut=preset[0] if preset else None,
            preset_kppra=preset[1] if preset else None)
        results.append(res)
        if convergence_scope == "phase" and preset is None:
            preset = (res["chosen_encut"], res["chosen_kppra"])
    return {"phase": phase, "sqs": results, "endmember_only": True}


def main():
    ap = argparse.ArgumentParser(
        description="Upstream first-principles generator for the binary "
                    "sqs2tdb pipeline (SQS gen + convergence + relax + fitfc).")
    ap.add_argument("--element1", required=True)
    ap.add_argument("--element2", required=True)
    ap.add_argument("--work-root", required=True,
                    help="Directory to generate the SQS calculation tree in.")
    ap.add_argument("--phases", default=None,
                    help="Comma-separated phases (default: all four).")
    ap.add_argument("--potcars", required=True,
                    help="Comma-separated POTCAR paths (one per element, or a "
                         "pre-assembled multi-element POTCAR). Used for ENMAX.")
    ap.add_argument("--template-root", default=None,
                    help="Directory holding the *_small single-sublattice "
                         "template systems to copy (caveat 1).")
    ap.add_argument("--dlm", action="store_true",
                    help="Disordered-local-moment run: randomspin for the "
                         "single-sublattice phases, lev=3->lev=0 +/-spin "
                         "conversion for SIGMA, and DLM fitfc fixup.")
    ap.add_argument("--dlm-moments", default=None,
                    help="DLM SUBATOM map, comma-separated "
                         "'EL=POTCAR:moment' entries, e.g. "
                         "'Co=Co:1.8,Cr=Cr_pv:1.5'. POTCAR defaults to the "
                         "element symbol and moment to 2.0 if omitted "
                         "('Co' == 'Co=Co:2.0'). Drives the s/EL+2/POT+m/g "
                         "and s/EL-2/POT-m/g SUBATOM lines in vasp.wrap.")
    ap.add_argument("--algo", default="All",
                    help="VASP ALGO (default 'All' per spec; use 'Fast' to "
                         "match the reference vasp.wrap).")
    ap.add_argument("--relax-method",
                    choices=["runstruct", "normal", "infdet"],
                    default="runstruct",
                    help="Structural relaxation method. 'runstruct' "
                         "(default) invokes 'pollmach runstruct_vasp' — "
                         "simplest, fastest for well-converged cases. "
                         "'normal' and 'infdet' both wrap "
                         "robustrelax_vasp and are automatically "
                         "preceded by 'robustrelax_vasp -mk' to build "
                         "the input files robustrelax needs.")
    ap.add_argument("--tol-ev", type=float, default=converge.DEFAULT_TOL_EV,
                    help="Convergence tolerance, eV/atom (default 0.001 = "
                         "1 meV/atom).")
    ap.add_argument("--sqs-level", default="2",
                    help="SQS composition-mesh cutoff, passed to "
                         "sqs2tdb -cp as -lv=N. CUMULATIVE: -lv=N copies "
                         "ALL database levels <= N (so the default '2' "
                         "yields lev=0 endmembers + lev=1 midpoints + "
                         "lev=2 mesh in one shot; omitting -lv would "
                         "behave like 0 = endmembers only). A comma list "
                         "is accepted for convenience but only its MAX "
                         "matters. To refine later, re-run with a larger "
                         "value — sqs2tdb skips dirs that already have "
                         "str.out, so only the new levels are added.")
    ap.add_argument("--env-bin", default=None,
                    help="Prepend this directory to PATH for ATAT/VASP "
                         "executables.")
    ap.add_argument("--cmd-prefix", default="",
                    help="Command used to launch VASP, passed to every "
                         "runstruct_vasp / robustrelax_vasp invocation as "
                         "trailing arguments (ATAT convention), e.g. "
                         "'mpiexec -n 128'. REQUIRED on NAS mil_ait — the "
                         "MPI vasp binary dies without its launcher and "
                         "runstruct_vasp then reports 'unable to open "
                         "OSZICAR'. Default: empty (bare vasp; only valid "
                         "for serial builds).")
    ap.add_argument("--relax-opts", default="",
                    help="Direct robustrelax_vasp options for the 'normal' "
                         "and 'infdet' relax methods, e.g. '-c 0.05'. "
                         "Distinct from the -idop infdet suboptions; "
                         "ignored for --relax-method runstruct.")
    ap.add_argument("--skip-phonon", action="store_true",
                    help="Skip the fitfc phonon stage (energy-only upstream).")
    ap.add_argument("--fitfc-on-unstable",
                    choices=("mark", "force", "escalate"),
                    default="mark",
                    help="Policy when fitfc -f reports unstable modes and "
                         "aborts before writing svib_ht. 'mark' (default): "
                         "record unstable_modes.log and leave the SQS "
                         "energy-only (honest — svib from a fit that drops "
                         "imaginary branches is biased). 'force': retry once "
                         "with fitfc's -fn so a (lower-bound) svib_ht is "
                         "still produced; provenance is recorded. "
                         "'escalate': regenerate the perturbations at a "
                         "1.5x larger displacement radius and refit (rules "
                         "out the finite-supercell artifact — costs extra "
                         "VASP force runs); if the instability persists the "
                         "SQS is marked energy-only as likely genuinely "
                         "dynamically unstable.")
    ap.add_argument("--fitfc-escalate-ernn", type=float, default=None,
                    help="Displacement radius (-ernn, x nearest-neighbour "
                         "distance) for the 'escalate' retry. Default: "
                         "1.5x the original (i.e. 3.0 for the default "
                         "-ernn=2).")
    ap.add_argument("--convergence-scope", choices=("phase", "sqs"),
                    default="phase",
                    help="'phase' (default): converge ENCUT/KPPRA on the "
                         "first SQS of each phase and reuse for all its "
                         "siblings, so every structure entering a mixing-"
                         "energy fit shares identical settings (the "
                         "subtraction of eV-scale totals only cancels "
                         "basis/k-mesh error at uniform settings). 'sqs' "
                         "restores per-SQS sweeps (more compute, "
                         "inconsistent settings — for diagnostics only).")
    ap.add_argument("--no-spin", action="store_true",
                    help="Force ISPIN=1 (non-spin-polarized) even for "
                         "magnetic elements. Default: spin polarization is "
                         "AUTO-ENABLED (ISPIN=2, VASP-default initial "
                         "moments) whenever an element is in "
                         f"{sorted(vaspwrap.MAGNETIC_3D)} and the run is "
                         "not DLM — non-magnetic energies for these metals "
                         "are wrong by tens of meV/atom.")
    ap.add_argument("--spin", action="store_true",
                    help="Force ISPIN=2 even for elements outside the "
                         "magnetic-3d set.")
    ap.add_argument("--fitfc-rl", type=float, default=None,
                    help="Pass fitfc's -rl=<len> robust-length soft-mode "
                         "treatment (beta) to the fit, which also prevents "
                         "the unstable-mode abort. Off by default.")
    ap.add_argument("--timeout", type=int, default=172800,
                    help="Per-VASP/poll timeout in seconds (default 48h).")
    ap.add_argument("--out", default=None,
                    help="Write a JSON manifest of chosen params / outputs.")
    args = ap.parse_args()

    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    # From here on, everything printed also lands (timestamped, flushed)
    # in <work_root>/upstream_live.log — tail -f it while the job runs.
    live_log = install_live_log(work_root)
    stamp(f"live log: {live_log}")
    potcar_paths = [Path(p.strip()) for p in args.potcars.split(",") if p.strip()]
    template_root = Path(args.template_root) if args.template_root else None
    phases = ([p.strip() for p in args.phases.split(",")]
              if args.phases else list(ALL_PHASES))

    subatom = parse_dlm_moments(args.dlm_moments,
                                [args.element1, args.element2])
    dlm = DLMConfig(enabled=args.dlm, subatom=subatom)
    sigma_elements = [args.element1, args.element2]

    # Parse --sqs-level into a list of ints. "2" -> [2]; "2,3" -> [2, 3];
    # empty falls back to [2] so the default behaviour matches the
    # single-level flag semantics from before.
    try:
        sqs_levels = [int(x) for x in args.sqs_level.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(
            f"--sqs-level must be a comma-separated list of ints; "
            f"got {args.sqs_level!r}"
        )
    if not sqs_levels:
        sqs_levels = [2]

    # Fail fast if ENMAX can't be read -- the whole convergence stage depends
    # on it, and a silent 0 cutoff would ruin every run.
    max_e = potcar.max_enmax(potcar_paths)
    print(f"\n{'='*70}")
    print(f"  Upstream generator — {args.element1}-{args.element2}")
    print(f"{'='*70}")
    print(f"  Work root   : {work_root}")
    print(f"  Phases      : {', '.join(phases)}")
    print(f"  MAX ENMAX   : {max_e:.1f} eV")
    print(f"  ENCUT grid  : {potcar.encut_grid(max_e)} eV")
    print(f"  KPPRA grid  : {potcar.kppra_grid()} "
          f"(@ ENCUT {potcar.kppra_probe_encut(max_e)} eV)")
    print(f"  Conv tol    : {args.tol_ev*1e3:.1f} meV/atom")
    print(f"  ALGO        : {args.algo}")
    print(f"  Relax       : {args.relax_method}"
          + (f"  (opts: {args.relax_opts})" if args.relax_opts else ""))
    print(f"  SQS levels  : {sqs_levels}")
    print(f"  VASP launch : {args.cmd_prefix or '<bare vasp — serial builds only>'}")
    # Spin policy: every wrap written by converge/relax/phonon inherits
    # vaspwrap.DEFAULT_SPIN (see build_vasp_wrap docstring). DLM runs
    # handle spin through the SUBATOM machinery instead.
    if args.no_spin:
        spin_on = False
    else:
        spin_on = args.spin or vaspwrap.wants_spin(
            [args.element1, args.element2])
    vaspwrap.DEFAULT_SPIN = spin_on and not args.dlm
    print(f"  Spin        : "
          + ("DLM (SUBATOM moments)" if args.dlm else
             ("ISPIN=2, VASP-default init moments" if vaspwrap.DEFAULT_SPIN
              else "off (ISPIN=1)"))
          + ("  [--no-spin]" if args.no_spin else ""))
    print(f"  DLM         : {'on' if args.dlm else 'off'}"
          + (f"  SUBATOM={subatom}" if args.dlm else ""))
    fitfc_opts: Dict = {"on_unstable": args.fitfc_on_unstable}
    if args.fitfc_rl is not None:
        fitfc_opts["rl"] = args.fitfc_rl
    if args.fitfc_escalate_ernn is not None:
        fitfc_opts["escalate_ernn"] = args.fitfc_escalate_ernn

    print(f"  Phonons     : {'skipped' if args.skip_phonon else 'fitfc'}"
          + ("" if args.skip_phonon else
             f"  (on_unstable={args.fitfc_on_unstable}"
             + (f", rl={args.fitfc_rl}" if args.fitfc_rl is not None else "")
             + ")"))
    print(f"{'='*70}")

    if not args.cmd_prefix:
        print("  WARNING: --cmd-prefix is empty. On MPI VASP builds "
              "(NAS mil_ait) the bare binary dies before writing OSZICAR "
              "and every sweep point will fail. Pass e.g. "
              "--cmd-prefix 'mpiexec -n 128'.")

    manifest = {
        "binary": f"{args.element1}-{args.element2}",
        "work_root": str(work_root),
        "max_enmax": max_e,
        "dlm": args.dlm,
        "relax_method": args.relax_method,
        "cmd_prefix": args.cmd_prefix,
        "relax_opts": args.relax_opts,
        "fitfc_opts": fitfc_opts,
        "spin_polarized": vaspwrap.DEFAULT_SPIN,
        "convergence_scope": args.convergence_scope,
        "phases": [],
    }
    manifest["sqs_levels"] = sqs_levels
    for phase in phases:
        res = process_phase(
            phase, work_root, potcar_paths, dlm, args.relax_method,
            args.algo, args.tol_ev, sqs_levels, sigma_elements,
            template_root, args.env_bin, args.skip_phonon, args.timeout,
            cmd_prefix=args.cmd_prefix, relax_opts=args.relax_opts,
            fitfc_opts=fitfc_opts,
            convergence_scope=args.convergence_scope)
        manifest["phases"].append(res)

    out = Path(args.out) if args.out else work_root / "upstream_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  Manifest written: {out}\n")


if __name__ == "__main__":
    main()
