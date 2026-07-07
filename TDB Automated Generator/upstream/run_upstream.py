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
                    timeout: int) -> Dict:
    """Convergence -> relax -> phonon for a single SQS directory."""
    sweep_root = sqs_dir / "convergence"

    chosen_encut, chosen_kppra, kres, eres = converge.converge_sqs(
        sqs_dir, sweep_root, potcar_paths,
        dlm=dlm, algo=algo, tol_ev=tol_ev,
        env_bin=env_bin, timeout=timeout)

    print(kres.table())
    print(eres.table())
    print(f"    -> chosen ENCUT={chosen_encut} eV, KPPRA={chosen_kppra}")

    relax.relax_structure(
        sqs_dir, encut=chosen_encut, kppra=chosen_kppra,
        method=relax_method, dlm=dlm, algo=algo,
        env_bin=env_bin, timeout=timeout)

    phonon_out = None
    if not skip_phonon:
        phonon_out = str(phonon.run_fitfc(
            sqs_dir, encut=chosen_encut, kppra=chosen_kppra,
            dlm=dlm, algo=algo, env_bin=env_bin, timeout=timeout))

    return {
        "sqs_dir": str(sqs_dir),
        "chosen_encut": chosen_encut,
        "chosen_kppra": chosen_kppra,
        "kppra_converged": kres.converged,
        "encut_converged": eres.converged,
        "relax_method": relax_method,
        "str_relax_present": (sqs_dir / "str_relax.out").is_file(),
        "phonon_out": phonon_out,
    }


def process_phase(phase: str,
                  work_root: Path,
                  potcar_paths: List[Path],
                  dlm: DLMConfig,
                  relax_method: str,
                  algo: str,
                  tol_ev: float,
                  sqs_level: Optional[int],
                  sigma_elements: List[str],
                  template_root: Optional[Path],
                  env_bin: Optional[str],
                  skip_phonon: bool,
                  timeout: int) -> Dict:
    print(f"\n{'='*70}\n  PHASE {phase}\n{'='*70}")

    # Copy *_small template if provided (caveat 1).
    if template_root and phase in SMALL_SYSTEM:
        made = sqsgen.copy_small_systems(template_root, work_root, [phase])
        if made:
            print(f"    copied template: {made[0].name}")

    # SIGMA endmember-only handling (caveat 2).
    if phase in ENDMEMBER_ONLY_PHASES:
        return process_sigma(phase, work_root, potcar_paths, dlm, relax_method,
                             algo, tol_ev, sqs_level, sigma_elements, env_bin,
                             skip_phonon, timeout)

    phase_root = sqsgen.generate_phase_sqs(
        work_root, phase, elements=sigma_elements,
        level=sqs_level, dlm=dlm.enabled,
        env_bin=env_bin)

    sqs_dirs = discover_sqs_dirs(phase_root)
    print(f"    {len(sqs_dirs)} SQS directories")

    results = []
    for d in sqs_dirs:
        print(f"\n  -- SQS {d.name} --")
        results.append(process_one_sqs(
            d, potcar_paths, dlm, relax_method, algo, tol_ev,
            env_bin, skip_phonon, timeout))
    return {"phase": phase, "sqs": results}


def process_sigma(phase: str,
                  work_root: Path,
                  potcar_paths: List[Path],
                  dlm: DLMConfig,
                  relax_method: str,
                  algo: str,
                  tol_ev: float,
                  sqs_level: Optional[int],
                  sigma_elements: List[str],
                  env_bin: Optional[str],
                  skip_phonon: bool,
                  timeout: int) -> Dict:
    """SIGMA_D8B: endmembers only. Under DLM, build each endmember from a
    lev=3 SQS via the lev=3 -> lev=0 +/-spin conversion (caveat 2)."""
    # For DLM SIGMA we must generate at lev=3 (randomises each site among 2
    # species); otherwise the usual endmember (lev=0) generation is used.
    gen_level = 3 if (dlm.enabled and dlm.sigma_from_lev3) else (sqs_level or 0)
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

    results = []
    for d in endmember_dirs:
        print(f"\n  -- SIGMA endmember {d.name} --")
        results.append(process_one_sqs(
            d, potcar_paths, dlm, relax_method, algo, tol_ev,
            env_bin, skip_phonon, timeout))
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
    ap.add_argument("--relax-method", choices=["normal", "infdet"],
                    default="normal",
                    help="Structural relaxation method.")
    ap.add_argument("--tol-ev", type=float, default=converge.DEFAULT_TOL_EV,
                    help="Convergence tolerance, eV/atom (default 0.001 = "
                         "1 meV/atom).")
    ap.add_argument("--sqs-level", type=int, default=None,
                    help="Restrict generation to this sqs level (-lev=N), if "
                         "the local sqs2tdb honours it.")
    ap.add_argument("--env-bin", default=None,
                    help="Prepend this directory to PATH for ATAT/VASP "
                         "executables.")
    ap.add_argument("--skip-phonon", action="store_true",
                    help="Skip the fitfc phonon stage (energy-only upstream).")
    ap.add_argument("--timeout", type=int, default=172800,
                    help="Per-VASP/poll timeout in seconds (default 48h).")
    ap.add_argument("--out", default=None,
                    help="Write a JSON manifest of chosen params / outputs.")
    args = ap.parse_args()

    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    potcar_paths = [Path(p.strip()) for p in args.potcars.split(",") if p.strip()]
    template_root = Path(args.template_root) if args.template_root else None
    phases = ([p.strip() for p in args.phases.split(",")]
              if args.phases else list(ALL_PHASES))

    subatom = parse_dlm_moments(args.dlm_moments,
                                [args.element1, args.element2])
    dlm = DLMConfig(enabled=args.dlm, subatom=subatom)
    sigma_elements = [args.element1, args.element2]

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
    print(f"  Relax       : {args.relax_method}")
    print(f"  DLM         : {'on' if args.dlm else 'off'}"
          + (f"  SUBATOM={subatom}" if args.dlm else ""))
    print(f"  Phonons     : {'skipped' if args.skip_phonon else 'fitfc'}")
    print(f"{'='*70}")

    manifest = {
        "binary": f"{args.element1}-{args.element2}",
        "work_root": str(work_root),
        "max_enmax": max_e,
        "dlm": args.dlm,
        "relax_method": args.relax_method,
        "phases": [],
    }
    for phase in phases:
        res = process_phase(
            phase, work_root, potcar_paths, dlm, args.relax_method,
            args.algo, args.tol_ev, args.sqs_level, sigma_elements,
            template_root, args.env_bin, args.skip_phonon, args.timeout)
        manifest["phases"].append(res)

    out = Path(args.out) if args.out else work_root / "upstream_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  Manifest written: {out}\n")


if __name__ == "__main__":
    main()
