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
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

# Make sibling modules importable whether run as a script or as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import converge
import pbsjobs
import phonon
import potcar
import relax
import runner
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


# Single source of truth for "is this an element-decorated calc dir"
_DECORATED_RE = sqsgen.DECORATED_SQS_RE


def discover_sqs_dirs(phase_root: Path) -> List[Path]:
    """All element-DECORATED SQS calc dirs produced by sqs2tdb -cp.

    Raw database entries (sqsdb_lev=* with undecorated "a=1"-style
    names, as found in $atatdir/data/sqsdb) are explicitly excluded —
    in the 2026-07-16 e2e run a stray copy of the database inside the
    work root was picked up as calculations, ternary meshes included.
    """
    out: List[Path] = []
    for d in sorted(phase_root.rglob("*")):
        if d.is_dir() and (d / "str.out").is_file() \
                and _DECORATED_RE.search(d.name):
            out.append(d)
    if not out and (phase_root / "str.out").is_file():
        out = [phase_root]
    return out


# Composition tokens in a decorated calc-dir name: site_El=frac
_COMP_TOKEN_RE = re.compile(r"[a-z]+_([A-Z][a-z]?)[+-]?\d*=([0-9.]+)")


def site_fractions(dirname: str) -> Dict[str, float]:
    """Element -> overall fraction parsed from a decorated dir name
    (site multiplicities ignored — adequate for single-sublattice
    phases, which is all the probe protocol samples)."""
    fr: Dict[str, float] = {}
    for el, v in _COMP_TOKEN_RE.findall(dirname):
        fr[el] = fr.get(el, 0.0) + float(v)
    tot = sum(fr.values())
    return {el: v / tot for el, v in fr.items()} if tot > 0 else {}


def pick_probe_dirs(dirs: List[Path], elements: List[str],
                    rng) -> Dict[str, Path]:
    """One randomly chosen SQS per element from that element's RICH
    side (fraction > 0.5; pure endmembers qualify; lev irrelevant) —
    the 2026-07-17 convergence-probe protocol."""
    picks: Dict[str, Path] = {}
    for el in elements:
        key = el.capitalize()
        rich = [d for d in sorted(dirs)
                if site_fractions(d.name).get(key, 0.0) > 0.5]
        if rich:
            picks[el] = rng.choice(rich)
    return picks


def _probe_worker_argv() -> List[str]:
    """This process's argv with orchestration-only flags removed, for
    re-invoking run_upstream.py as a --probe-worker inside a job."""
    argv = list(sys.argv[1:])
    out: List[str] = []
    skip_next = False
    VALUED = {"--submit", "--job-env", "--job-model", "--job-queue",
              "--job-group-list", "--job-max-inflight", "--job-retries",
              "--probe-worker"}
    FLAGS = {"--no-job-arrays", "--job-dry-run"}
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in VALUED:
            skip_next = True
            continue
        if tok.split("=")[0] in VALUED:
            continue
        if tok in FLAGS:
            continue
        out.append(tok)
    return out


def system_probe_convergence(work_root: Path,
                             requested_phases: List[str],
                             elements: List[str],
                             potcar_paths: List[Path],
                             dlm: DLMConfig,
                             algo: str,
                             tol_ev: float,
                             sqs_levels: List[int],
                             env_bin: Optional[str],
                             timeout: int,
                             cmd_prefix: str,
                             seed: int,
                             broker=None,
                             worker_argv: Optional[List[str]] = None
                             ) -> Optional[Dict]:
    """The one-sweep-for-everything protocol (2026-07-17 user decision):

    1. Randomly pick ONE single-sublattice phase among those requested.
    2. From its generated SQS, randomly pick one structure on EACH
       element-rich side (>50% of that element; endmembers count; lev
       irrelevant).
    3. ENCUT/KPPRA sweep each probe (successive-difference criterion,
       unbounded ENCUT).
    4. Global settings = elementwise MAX over the probes, with the
       Pulay floor (1.3 x max ENMAX) folded in so relaxations, statics
       and phonon force runs all share literally ONE (ENCUT, KPPRA).

    Rich-side sampling matters because basis-completeness demand is
    element-dependent (e.g. Cr_pv is harder than Co); taking the max is
    conservative for the other side. Returns a manifest-able dict with
    the picks, per-probe results and the final settings, or None if no
    single-sublattice phase / probes are available (caller falls back
    to first-SQS reuse).
    """
    import random
    rng = random.Random(seed)
    candidates = sorted(p for p in requested_phases
                        if p in SINGLE_SUBLATTICE_PHASES)
    if not candidates:
        return None
    phase = rng.choice(candidates)
    gen_level = max(sqs_levels)
    stamp(f"[probe] convergence-probe phase: {phase} (seed {seed}), "
          f"generating at -lv={gen_level}")
    phase_root = sqsgen.generate_phase_sqs(
        work_root, phase, elements=elements,
        level=gen_level, dlm=dlm.enabled, env_bin=env_bin)
    dirs = discover_sqs_dirs(phase_root)
    picks = pick_probe_dirs(dirs, elements, rng)
    if not picks:
        return None

    probes = {}
    encuts: List[int] = []
    kppras: List[int] = []
    if broker is not None:
        # PBS mode: each probe's adaptive sweep runs INSIDE its own job
        # (--probe-worker), and the element-rich probes run in
        # PARALLEL — the 2026-07-16 run did them serially and burned
        # ~90 min each. The orchestrator only waits on the JSONs.
        from concurrent.futures import ThreadPoolExecutor

        def _one(el_d):
            el, d = el_d
            stamp(f"[probe] {el}-rich probe job: {d.name}")
            cmd = [sys.executable, "-u",
                   str(Path(__file__).resolve())] \
                + (worker_argv or []) + ["--probe-worker", str(d)]
            rc = broker.run_as_job(
                tag=f"probe_{el}", cwd=d, cmd=cmd,
                done_when=lambda _c, _d=d:
                    (_d / "probe_result.json").is_file(),
                work_dirs=[d], kind="probe",
                done_file="probe_result.json")
            if rc != 0 or not (d / "probe_result.json").is_file():
                raise RuntimeError(f"probe job for {d} failed")
            return el, d, json.loads((d / "probe_result.json").read_text())

        with ThreadPoolExecutor(max_workers=len(picks)) as pool:
            for el, d, r in pool.map(_one, sorted(picks.items())):
                probes[el] = {"dir": str(d), "encut": r["encut"],
                              "kppra": r["kppra"],
                              "encut_converged": r["encut_converged"],
                              "kppra_converged": r["kppra_converged"]}
                encuts.append(r["encut"])
                kppras.append(r["kppra"])
    else:
        for el, d in sorted(picks.items()):
            stamp(f"[probe] {el}-rich probe: {d.name} — ENCUT/KPPRA sweep")
            e_c, k_c, kres, eres = converge.converge_sqs(
                d, d / "convergence", potcar_paths,
                dlm=dlm, algo=algo, tol_ev=tol_ev,
                env_bin=env_bin, timeout=timeout, cmd_prefix=cmd_prefix)
            print(kres.table())
            print(eres.table())
            probes[el] = {"dir": str(d), "encut": e_c, "kppra": k_c,
                          "encut_converged": eres.converged,
                          "kppra_converged": kres.converged}
            encuts.append(e_c)
            kppras.append(k_c)

    max_e = potcar.max_enmax(potcar_paths)
    final_encut = potcar.pulay_safe_encut(max(encuts), max_e)
    final_kppra = max(kppras)
    stamp(f"[probe] GLOBAL settings: ENCUT={final_encut} eV "
          f"(max over probes {encuts}, Pulay floor "
          f"{potcar.PULAY_ENCUT_FACTOR} x ENMAX folded in), "
          f"KPPRA={final_kppra} (max over {kppras}) — used for ALL "
          f"energy, relaxation, inflection-detection and phonon runs")
    return {"phase": phase, "seed": seed, "probes": probes,
            "encut": final_encut, "kppra": final_kppra}


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
                    preset_kppra: Optional[int] = None,
                    max_checkrelax: float = 0.1) -> Dict:
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
    # Validate the RESULT, not just file existence: a crashed
    # robustrelax/infdet leaves a degenerate str_relax.out stub (seen in
    # the real Co-Cr run: identity cell + coordinate-less atom line),
    # which then poisons checkrelax and fitfc downstream.
    from strfile import validate_structure_file
    relax_ok, relax_msg = validate_structure_file(sqs_dir / "str_relax.out")
    stamp(f"[{sqs_dir.name}] STAGE 2/3 relaxation done "
          f"(str_relax.out: {relax_msg})")
    if not relax_ok:
        stamp(f"[{sqs_dir.name}] STAGE 2/3 RELAXATION FAILED — "
              f"str_relax.out unusable; phonons will be SKIPPED. "
              f"Triage the VASP logs (python3 vasp_triage.py {sqs_dir})")

    def _parseable(pth: Path) -> bool:
        try:
            float(pth.read_text().split()[0])
            return True
        except (OSError, ValueError, IndexError):
            return False

    # Inflection-detection bookkeeping (semantics from the
    # robustrelax_vasp source, verified 2026-07-20): when the -id
    # branch engages, `energy_end` holds the FULLY-RELAXED energy —
    # the structure the phase DECAYED INTO, which is exactly what
    # infdet exists to avoid reporting — so it is NEVER adopted as the
    # result. On success robustrelax itself writes the inflection-point
    # energy to `energy` (scaled from 01/energy). Success marker (per
    # the method's author): 01/infdet.log's LAST line is
    # 'infdet terminated normally', plus `energy` present.
    infdet_engaged = infdet_ok = False
    if relax_method == "infdet":
        infdet_engaged, infdet_ok, infdet_msg = relax.infdet_status(sqs_dir)
        if infdet_engaged:
            stamp(f"[{sqs_dir.name}] inflection detection engaged: "
                  f"{infdet_msg}")
            if infdet_ok:
                # Marker for the downstream drift gate: this SQS's
                # large checkrelax (if any) is the inflection-point
                # geometry, not decay — do not reject on drift.
                (sqs_dir / "infdet_ok.flag").write_text(
                    "inflection detection terminated normally; "
                    "energy = inflection point; drift gate waived\n")
                stale = sqs_dir / "relaxaway.flag"
                if stale.is_file():
                    stale.unlink()
            if not infdet_ok and relax_ok:
                relax_ok = False
                relax_msg = f"infdet incomplete: {infdet_msg}"
                stamp(f"[{sqs_dir.name}] INFDET FAILED — {infdet_msg}; "
                      f"energy_end (the decayed structure's energy) is "
                      f"NOT adopted; str_relax.out may be the fully-"
                      f"relaxed geometry, not the inflection point — "
                      f"rerun with `robustrelax_vasp -id -cip` or "
                      f"triage 01/ before trusting this SQS")

    # Clear the ATAT 'wait' queue marker once the relax has produced its
    # result — sqs2tdb -cp drops a `wait` file in every to-be-computed
    # dir, and pollmach-style pollers treat its presence as "pending".
    # Mirrors the manual `rm wait` in the reference NAS workflow.
    if relax_ok:
        wait_marker = sqs_dir / "wait"
        if wait_marker.is_file():
            wait_marker.unlink()

    # Lattice-drift check (ATAT checkrelax analogue): an SQS whose cell
    # relaxed away from its parent lattice carries the energy of a
    # DIFFERENT phase — it must be flagged so downstream fits can drop
    # it. Value recorded in checkrelax.out; drift beyond max_checkrelax
    # additionally drops a relaxaway.flag marker.
    checkrelax_val = None
    if relax_ok:
        try:
            from strfile import lattice_drift
            checkrelax_val = lattice_drift(sqs_dir / "str.out",
                                           sqs_dir / "str_relax.out")
            (sqs_dir / "checkrelax.out").write_text(
                f"{checkrelax_val:.6f}\n")
            if checkrelax_val > max_checkrelax \
                    and relax_method != "runstruct":
                # checkrelax is a valid keep/throw signal ONLY for
                # plain runstruct full relaxations (user directive
                # 2026-07-22). For robustrelax modes large drift is
                # part of the method: the -id branch deliberately
                # spans a large deformation and reports the inflection
                # point; success is judged by robustrelax's own
                # completion (energy_sup / 01 + energy) and the
                # 'infdet terminated normally' marker — never drift.
                note = ("inflection detection terminated normally"
                        if infdet_ok else
                        f"relax_method={relax_method} — drift is not "
                        f"a failure signal for robustrelax")
                stamp(f"[{sqs_dir.name}] checkrelax = "
                      f"{checkrelax_val:.4f} > {max_checkrelax} — "
                      f"INFORMATIONAL only ({note}); no relaxaway flag")
            elif checkrelax_val > max_checkrelax:
                (sqs_dir / "relaxaway.flag").write_text(
                    f"lattice drift {checkrelax_val:.4f} > "
                    f"{max_checkrelax} — structure left its parent "
                    f"lattice; exclude from this phase's fit\n")
                stamp(f"[{sqs_dir.name}] WARNING lattice drift "
                      f"{checkrelax_val:.4f} > {max_checkrelax} "
                      f"(relaxaway.flag written — see checkrelax.out)")
            else:
                stamp(f"[{sqs_dir.name}] checkrelax = "
                      f"{checkrelax_val:.4f} (<= {max_checkrelax}, OK)")
        except Exception as exc:                    # noqa: BLE001
            stamp(f"[{sqs_dir.name}] WARNING checkrelax metric failed: "
                  f"{exc}")

    phonon_out = None
    if not relax_ok:
        stamp(f"[{sqs_dir.name}] STAGE 3/3 skipped (failed relaxation)")
    elif not skip_phonon:
        stamp(f"[{sqs_dir.name}] STAGE 3/3 fitfc phonons starting")
        phonon_out = str(phonon.run_fitfc(
            sqs_dir, encut=relax_encut, kppra=chosen_kppra,
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
        "relax_ok": relax_ok,
        "relax_msg": relax_msg,
        "energy_present": _parseable(sqs_dir / "energy"),
        "infdet_engaged": infdet_engaged,
        "infdet_ok": infdet_ok,
        "checkrelax": checkrelax_val,
        "relaxed_away": (sqs_dir / "relaxaway.flag").is_file(),
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
                  convergence_scope: str = "system",
                  max_checkrelax: float = 0.1,
                  phonon_scope: str = "endmembers",
                  preset: Optional[tuple] = None,
                  parallel_sqs: bool = False) -> Dict:
    print(f"\n{'='*70}\n  PHASE {phase}\n{'='*70}")

    # Copy *_small template if provided (caveat 1).
    # NOTE (2026-07-16 e2e failure): the old behavior here copied the
    # RAW sqsdb database entry (sqsdb_lev=* dirs with undecorated
    # compositions + sqsgen.in) from --template-root into the work
    # root. That is useless — sqs2tdb -cp reads its database from
    # $atatdir/data/sqsdb, never from the work root — and harmful:
    # discovery then picked up the raw entries as SQS to compute.
    # The correct invocation is simply
    #     sqs2tdb -cp -l=<lattice> -sp=El1,El2 -lv=N
    # If a *_small lattice is missing from your ATAT install, add it
    # under $atatdir/data/sqsdb — not here.
    if template_root and phase in SMALL_SYSTEM:
        print(f"    NOTE: --template-root is deprecated and ignored "
              f"(sqs2tdb reads its own database under $atatdir); "
              f"NOT copying {template_root} into the work root.")

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
                             convergence_scope=convergence_scope,
                             max_checkrelax=max_checkrelax,
                             phonon_scope=phonon_scope,
                             preset=preset,
                             parallel_sqs=parallel_sqs)

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

    # Uniform convergence settings (mixing energies subtract eV-scale
    # totals; the subtraction only cancels basis/k-mesh error when all
    # structures share the same settings):
    #   system (default) one sweep for the WHOLE run — ENCUT depends on
    #                    the POTCARs (shared by every phase) far more
    #                    than on the structure, and KPPRA already
    #                    normalizes mesh density per atom; also closes
    #                    review item O1 (cross-phase consistency)
    #   phase            one sweep per phase
    #   sqs              legacy per-SQS sweeps (diagnostics only)
    # `preset` seeds from --preset-encut/--preset-kppra or, for
    # scope=system, from an earlier phase of this run.
    def _one(d: Path) -> Dict:
        print(f"\n  -- SQS {d.name} --")
        # Paper (Calphad 58 (2017) 70): phonons are "typically done for
        # the end members only"; sqs2tdb -fit then represents svib
        # linearly across composition. --phonon-scope all overrides.
        is_endmem = (d / "endmem").is_file() or "lev=0" in d.name
        skip_ph = skip_phonon or (phonon_scope == "endmembers"
                                  and not is_endmem)
        if skip_ph and not skip_phonon:
            print(f"    (phonons: endmember-only scope — skipped for "
                  f"this mixing SQS)")
        return process_one_sqs(
            d, potcar_paths, dlm, relax_method, algo, tol_ev,
            env_bin, skip_ph, timeout,
            cmd_prefix=cmd_prefix, relax_opts=relax_opts,
            fitfc_opts=fitfc_opts,
            preset_encut=preset[0] if preset else None,
            preset_kppra=preset[1] if preset else None,
            max_checkrelax=max_checkrelax)

    results = []
    if parallel_sqs and preset is not None and len(sqs_dirs) > 1:
        # PBS fan-out: every SQS advances through its
        # relax -> validate -> phonon pipeline CONCURRENTLY; the long
        # stages are broker jobs, so these threads spend their lives in
        # time.sleep polling for files. The broker's max_inflight caps
        # the actual queue load. Requires a preset (probe done) so no
        # thread runs a convergence sweep.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(sqs_dirs)) as pool:
            results = list(pool.map(_one, sqs_dirs))
    else:
        for d in sqs_dirs:
            res = _one(d)
            results.append(res)
            if convergence_scope in ("phase", "system") and preset is None:
                preset = (res["chosen_encut"], res["chosen_kppra"])
    return {"phase": phase, "sqs": results,
            "preset_out": list(preset) if preset else None}


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
                  convergence_scope: str = "system",
                  max_checkrelax: float = 0.1,
                  phonon_scope: str = "endmembers",
                  preset: Optional[tuple] = None,
                  parallel_sqs: bool = False) -> Dict:
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

    # Same uniform-convergence policy as process_phase. SIGMA dirs are
    # all endmembers, so the endmember-only phonon scope never skips
    # them. (Kept serial per-dir here; SIGMA endmember counts are small
    # and each one's long stages are broker jobs in pbs mode anyway.)
    results = []
    for d in endmember_dirs:
        print(f"\n  -- SIGMA endmember {d.name} --")
        res = process_one_sqs(
            d, potcar_paths, dlm, relax_method, algo, tol_ev,
            env_bin, skip_phonon, timeout,
            cmd_prefix=cmd_prefix, relax_opts=relax_opts,
            fitfc_opts=fitfc_opts,
            preset_encut=preset[0] if preset else None,
            preset_kppra=preset[1] if preset else None,
            max_checkrelax=max_checkrelax)
        results.append(res)
        if convergence_scope in ("phase", "system") and preset is None:
            preset = (res["chosen_encut"], res["chosen_kppra"])
    return {"phase": phase, "sqs": results, "endmember_only": True,
            "preset_out": list(preset) if preset else None}


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
                    # deprecated 2026-07-16 — kept so old PBS scripts parse

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
                    type=vaspwrap.normalize_algo,
                    help="VASP electronic algorithm, applied to EVERY "
                         "wrap the pipeline writes: All (default, most "
                         "robust), Normal (blocked Davidson, = the "
                         "production INCARs' IALGO=38), VeryFast "
                         "(RMM-DIIS, cheapest/least robust). "
                         "Case-insensitive.")
    ap.add_argument("--relax-method",
                    choices=["infdet", "normal", "runstruct"],
                    default="infdet",
                    help="Structural relaxation method. 'infdet' "
                         "(default) runs robustrelax_vasp -id -c 0.05 — "
                         "inflection detection with the 5%% strain "
                         "cutoff it needs to engage (override -c via "
                         "--relax-opts). 'runstruct' invokes 'pollmach "
                         "runstruct_vasp' — simplest for well-behaved "
                         "cases. 'normal' and 'infdet' both wrap "
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
    ap.add_argument("--max-checkrelax", type=float, default=0.1,
                    help="Lattice-drift threshold (checkrelax analogue: "
                         "Frobenius norm of the volume-normalized strain "
                         "between str.out and str_relax.out). Every SQS "
                         "gets its value written to checkrelax.out; above "
                         "the threshold a relaxaway.flag marks it as "
                         "having left its parent lattice (ATAT guidance: "
                         "~0.1). Downstream discovery drops flagged/"
                         "over-threshold SQS.")
    ap.add_argument("--convergence-scope",
                    choices=("system", "phase", "sqs"),
                    default="system",
                    help="'system' (default): ONE ENCUT/KPPRA sweep for "
                         "the whole run, reused by every phase — ENCUT "
                         "tracks the POTCARs (shared across phases) far "
                         "more than the structure, KPPRA normalizes mesh "
                         "density per atom, and uniform settings are what "
                         "make cross-phase lattice stabilities consistent "
                         "(closes review item O1) at ~1/4 the sweep "
                         "compute. 'phase': one sweep per phase. 'sqs': "
                         "legacy per-SQS sweeps (diagnostics only).")
    ap.add_argument("--phonon-scope", choices=("endmembers", "all"),
                    default="endmembers",
                    help="'endmembers' (default, the published sqs2tdb "
                         "workflow): fitfc phonons only where the endmem "
                         "marker / lev=0 name says so; sqs2tdb -fit then "
                         "fits svib linearly across composition. 'all' "
                         "adds phonons on every mixing SQS (nonideal "
                         "vibrational entropy; several x the phonon "
                         "compute).")
    ap.add_argument("--preset-encut", type=int, default=None,
                    help="Skip the ENCUT sweep entirely and use this "
                         "value (eV). With --preset-kppra, no convergence "
                         "sweeps run at all — use to REUSE settings "
                         "converged on a previous run of the same "
                         "elements (e.g. take max over the binaries when "
                         "moving to a ternary).")
    ap.add_argument("--submit", choices=("node", "pbs"), default="node",
                    help="'node' (default): drive VASP inside THIS "
                         "allocation (the monolithic mode). 'pbs': act "
                         "as an ORCHESTRATOR — every relaxation gets "
                         "its own right-sized qsub job, every SQS's "
                         "phonon force runs become a PBS job array "
                         "(one element per perturbation), and the two "
                         "convergence probes run as parallel jobs. Run "
                         "the orchestrator on a front end via nohup "
                         "(see submit_orchestrator_template.sh); it "
                         "only does bookkeeping + fitfc/sqs2tdb glue.")
    ap.add_argument("--job-env", default=None,
                    help="Shell file SOURCED at the top of every "
                         "submitted job (modules, venv, PATH). "
                         "Required with --submit pbs.")
    ap.add_argument("--job-model", default="mil_ait",
                    help="PBS node model for submitted jobs.")
    ap.add_argument("--job-queue", default="normal",
                    help="PBS queue for submitted jobs (devel's "
                         "per-user job limit makes it unsuitable for "
                         "fan-out).")
    ap.add_argument("--job-group-list", default="a1485")
    ap.add_argument("--job-max-inflight", type=int, default=16,
                    help="Cap on simultaneously queued/running jobs — "
                         "the compute-cost throttle.")
    ap.add_argument("--job-retries", type=int, default=1,
                    help="Resubmissions for a job that leaves the "
                         "queue without producing its outputs.")
    ap.add_argument("--no-job-arrays", action="store_true",
                    help="Use one looping job per SQS for force runs "
                         "instead of a PBS job array per perturbation.")
    ap.add_argument("--job-dry-run", action="store_true",
                    help="Render job scripts but do not qsub (CI/ "
                         "inspection).")
    ap.add_argument("--probe-worker", default=None, metavar="SQS_DIR",
                    help=argparse.SUPPRESS)   # internal: runs ONE probe
    ap.add_argument("--plateau-band", type=float, default=None,
                    help="Noise-plateau fallback band in eV/atom "
                         "(default 0.0005 = 0.5 meV/atom): when the "
                         "successive-step rule finds nothing, accept the "
                         "first 4 consecutive sweep points whose total "
                         "spread fits in this band (the sweep has hit "
                         "the calculation's noise floor). 0 disables.")
    ap.add_argument("--probe-seed", type=int, default=0,
                    help="Seed for the random probe-phase/probe-SQS "
                         "selection of the system convergence protocol "
                         "(deterministic by default so runs are "
                         "reproducible; picks are recorded in the "
                         "manifest).")
    ap.add_argument("--preset-kppra", type=int, default=None,
                    help="Skip the KPPRA sweep entirely and use this "
                         "value. See --preset-encut.")
    ap.add_argument("--no-spin", action="store_true",
                    help="Force ISPIN=1 (non-spin-polarized) even for "
                         "magnetic elements. Default: spin polarization is "
                         "AUTO-ENABLED (ISPIN=2, VASP-default initial "
                         "moments) whenever an element is in "
                         f"{sorted(vaspwrap.MAGNETIC_3D)} and the run is "
                         "not DLM — non-magnetic energies for these metals "
                         "are wrong by tens of meV/atom.")
    ap.add_argument("--magmom-init", type=float, default=None,
                    help="Uniform initial magnetic moment (muB/atom) for "
                         "the MAGMOM line of spin-polarized non-DLM runs "
                         "(default 3, the production-INCAR convention). "
                         "DLM runs set moments via SUBATOM instead.")
    ap.add_argument("--spin", action="store_true",
                    help="Force ISPIN=2 even for elements outside the "
                         "magnetic-3d set.")
    ap.add_argument("--fitfc-ernn", type=float, default=None,
                    help="fitfc displacement radius in nearest-neighbour "
                         "units (default 4, the published value; smaller "
                         "= cheaper force runs, shorter-ranged force "
                         "constants).")
    ap.add_argument("--fitfc-frnn", type=float, default=None,
                    help="fitfc force-constant range in nn units "
                         "(default 2, the published value).")
    ap.add_argument("--fitfc-dr", type=float, default=None,
                    help="fitfc displacement magnitude in Angstrom "
                         "(default 0.04, the published value).")
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
    if args.plateau_band is not None:
        converge.PLATEAU_BAND_EV = args.plateau_band
    if args.magmom_init is not None:
        vaspwrap.DEFAULT_MAGMOM_INIT = args.magmom_init

    # ── internal: probe worker (runs INSIDE a submitted probe job) ────
    # Executes the full adaptive ENCUT/KPPRA sweep for ONE probe SQS on
    # its own allocation and records the result for the orchestrator.
    if args.probe_worker:
        d = Path(args.probe_worker).resolve()
        stamp(f"[probe-worker] sweeping {d.name}")
        e_c, k_c, kres, eres = converge.converge_sqs(
            d, d / "convergence", potcar_paths,
            dlm=dlm, algo=args.algo, tol_ev=args.tol_ev,
            env_bin=args.env_bin, timeout=args.timeout,
            cmd_prefix=args.cmd_prefix)
        print(kres.table())
        print(eres.table())
        (d / "probe_result.json").write_text(json.dumps(
            {"encut": e_c, "kppra": k_c,
             "encut_converged": eres.converged,
             "encut_rule": eres.rule,
             "kppra_converged": kres.converged,
             "kppra_rule": kres.rule}))
        stamp(f"[probe-worker] done: ENCUT={e_c}, KPPRA={k_c}")
        return

    # ── PBS fan-out mode: install the job broker as the execution
    #    backend for every long VASP command (runner.run_polled) ──────
    broker = None
    if args.submit == "pbs":
        if args.convergence_scope != "system" and not (
                args.preset_encut and args.preset_kppra):
            raise SystemExit(
                "--submit pbs requires --convergence-scope system (the "
                "probe protocol, run as jobs) or explicit "
                "--preset-encut/--preset-kppra: sweeps must never run "
                "VASP in the orchestrator process (front end).")
        if not args.job_env:
            raise SystemExit(
                "--submit pbs requires --job-env <file> — a shell "
                "snippet sourced by every job (modules, venv, PATH); "
                "see submit_orchestrator_template.sh, which writes it.")
        broker = pbsjobs.Broker(
            work_root=work_root,
            group_list=args.job_group_list,
            model=args.job_model,
            queue=args.job_queue,
            site_env=f"source {Path(args.job_env).resolve()}",
            max_inflight=args.job_max_inflight,
            max_retries=args.job_retries,
            use_arrays=not args.no_job_arrays,
            dry_run=args.job_dry_run)
        runner.set_backend(broker)
        stamp(f"[pbs mode] broker active: model={args.job_model} "
              f"queue={args.job_queue} max_inflight="
              f"{args.job_max_inflight} arrays="
              f"{not args.no_job_arrays}"
              + (" DRY-RUN" if args.job_dry_run else ""))
    print(f"  Spin        : "
          + ("DLM (SUBATOM moments)" if args.dlm else
             (f"ISPIN=2, MAGMOM={vaspwrap.DEFAULT_MAGMOM_INIT:g} muB/atom "
              f"init" if vaspwrap.DEFAULT_SPIN
              else "off (ISPIN=1)"))
          + ("  [--no-spin]" if args.no_spin else ""))
    print(f"  DLM         : {'on' if args.dlm else 'off'}"
          + (f"  SUBATOM={subatom}" if args.dlm else ""))
    fitfc_opts: Dict = {"on_unstable": args.fitfc_on_unstable}
    if args.fitfc_rl is not None:
        fitfc_opts["rl"] = args.fitfc_rl
    if args.fitfc_escalate_ernn is not None:
        fitfc_opts["escalate_ernn"] = args.fitfc_escalate_ernn
    if args.fitfc_ernn is not None:
        fitfc_opts["ernn"] = args.fitfc_ernn
    if args.fitfc_frnn is not None:
        fitfc_opts["frnn"] = args.fitfc_frnn
    if args.fitfc_dr is not None:
        fitfc_opts["dr"] = args.fitfc_dr

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
        "magmom_init": vaspwrap.DEFAULT_MAGMOM_INIT
                       if vaspwrap.DEFAULT_SPIN else None,
        "convergence_scope": args.convergence_scope,
        "phonon_scope": args.phonon_scope,
        "max_checkrelax": args.max_checkrelax,
        "phases": [],
    }
    manifest["sqs_levels"] = sqs_levels
    run_preset: Optional[tuple] = None
    if args.preset_encut is not None and args.preset_kppra is not None:
        run_preset = (args.preset_encut, args.preset_kppra)
        print(f"  Preset      : ENCUT={args.preset_encut} eV, "
              f"KPPRA={args.preset_kppra} — convergence sweeps SKIPPED")
    elif args.convergence_scope == "system":
        probe = system_probe_convergence(
            work_root, phases, [args.element1, args.element2],
            potcar_paths, dlm, args.algo, args.tol_ev, sqs_levels,
            args.env_bin, args.timeout, args.cmd_prefix,
            seed=args.probe_seed,
            broker=broker, worker_argv=_probe_worker_argv())
        if probe:
            run_preset = (probe["encut"], probe["kppra"])
            manifest["system_probe"] = probe
        else:
            print("  WARNING: probe protocol found no single-sublattice "
                  "phase / rich-side SQS; falling back to first-SQS "
                  "convergence reuse.")
    for phase in phases:
        res = process_phase(
            phase, work_root, potcar_paths, dlm, args.relax_method,
            args.algo, args.tol_ev, sqs_levels, sigma_elements,
            template_root, args.env_bin, args.skip_phonon, args.timeout,
            cmd_prefix=args.cmd_prefix, relax_opts=args.relax_opts,
            fitfc_opts=fitfc_opts,
            convergence_scope=args.convergence_scope,
            max_checkrelax=args.max_checkrelax,
            phonon_scope=args.phonon_scope,
            preset=run_preset,
            parallel_sqs=(args.submit == "pbs"
                          and run_preset is not None))
        manifest["phases"].append(res)
        if args.convergence_scope == "system" and run_preset is None \
                and res.get("preset_out"):
            run_preset = tuple(res["preset_out"])
            print(f"  [system scope] reusing ENCUT={run_preset[0]} eV, "
                  f"KPPRA={run_preset[1]} for all remaining phases")

    out = Path(args.out) if args.out else work_root / "upstream_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  Manifest written: {out}\n")


if __name__ == "__main__":
    main()
