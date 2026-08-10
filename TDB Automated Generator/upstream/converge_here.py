#!/usr/bin/env python3
"""
converge_here.py — standalone ENCUT/KPPRA convergence test for ONE
SQS directory of your choice.

Runs the exact pipeline criteria (converge.py unchanged: KPPRA sweep
at 1.125x ENMAX, then ENCUT sweep at the chosen KPPRA; successive-
difference rule 0.1 meV/atom with a confirming point, plateau
fallback at the noise floor, unbounded ENCUT extension, PREC=Accurate
+ LREAL=.FALSE. statics, per-point caching) and prints the
recommended settings.

Usage (from anywhere):
    python3 converge_here.py                       # SQS dir = cwd
    python3 converge_here.py /path/to/sqs_lev=...  # explicit dir
    python3 converge_here.py --cmd-prefix "mpiexec -n 32" --tol-ev 1e-4

Requirements:
  * <dir>/str.out must exist (any decorated SQS calc dir works).
  * POTCARs: derived automatically from the species in str.out and
    --pot-root (default: the zwu6 POTPAW_PBE.64 tree); override with
    an explicit --potcars comma list.
  * Runs VASP INLINE — execute inside a PBS job (or an interactive
    compute-node session), not on a pfe front end.

Restart-safe: sweep points that already produced an `energy` under
<dir>/convergence/ are read back, not rerun. Results are printed AND
written to <dir>/convergence_result.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import converge                                          # noqa: E402
import potcar                                            # noqa: E402
import vaspwrap                                          # noqa: E402
from strfile import read_structure, strip_spin_suffix_text  # noqa: E402

DEFAULT_POT_ROOT = Path("/home1/zwu6/vasp/POTPAW_PBE.64")


def elements_in(sqs_dir: Path) -> list:
    """Distinct element symbols in str.out (spin suffixes stripped,
    order-stable) — e.g. ['Co', 'Cr'] from a Co/Cr SQS."""
    struct = read_structure(sqs_dir / "str.out")
    seen = []
    for sp in struct.species():
        base = strip_spin_suffix_text(sp)
        if base not in seen:
            seen.append(base)
    return seen


def resolve_potcars(args, sqs_dir: Path) -> list:
    if args.potcars:
        paths = [Path(p) for p in args.potcars.split(",")]
    else:
        els = elements_in(sqs_dir)
        paths = [Path(args.pot_root) / el / "POTCAR" for el in els]
        print(f"[potcars] elements {els} -> {[str(p) for p in paths]}")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"POTCAR(s) not found: {missing} — pass "
                         f"--potcars or --pot-root explicitly")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="ENCUT/KPPRA convergence test for one SQS dir "
                    "(pipeline criteria, standalone).")
    ap.add_argument("sqs_dir", nargs="?", default=".", type=Path,
                    help="SQS calc dir containing str.out (default: cwd)")
    ap.add_argument("--potcars", default=None,
                    help="comma list of POTCAR paths (default: derive "
                         "from str.out species + --pot-root)")
    ap.add_argument("--pot-root", default=str(DEFAULT_POT_ROOT),
                    help=f"per-element POTCAR tree (default "
                         f"{DEFAULT_POT_ROOT})")
    ap.add_argument("--tol-ev", type=float, default=0.0001,
                    help="successive-step tolerance in eV/atom "
                         "(default 0.0001 = 0.1 meV/atom)")
    ap.add_argument("--algo", default="All",
                    help="VASP ALGO for the sweep statics (default All)")
    ap.add_argument("--cmd-prefix", default="mpiexec -n 32",
                    help='VASP launcher, e.g. "mpiexec -n 32"')
    ap.add_argument("--env-bin", default=None,
                    help="dir prepended to PATH for ATAT tools")
    ap.add_argument("--timeout", type=int, default=28800,
                    help="per-VASP-point timeout in seconds")
    ap.add_argument("--vasp-wrap", type=Path, default=None,
                    help="use THIS vasp.wrap for every sweep point "
                         "(only its ENCUT/KPPRA lines are replaced "
                         "per point; spin/mixing/PREC etc. are your "
                         "file's, and no automatic settings are "
                         "injected). Default: the generated wrap.")
    args = ap.parse_args(argv)

    sqs_dir = args.sqs_dir.resolve()
    if not (sqs_dir / "str.out").is_file():
        raise SystemExit(f"{sqs_dir} has no str.out — not an SQS calc dir")

    potcar_paths = resolve_potcars(args, sqs_dir)
    enmax = potcar.max_enmax(potcar_paths)

    # Spin handling (2026-08-07 user directive: NO SUBATOM machinery):
    #  * spin-TAGGED str.out (Co+2/Co-2 from randomspin /
    #    sigma_dlm_setup): ezvasp derives per-atom MAGMOM from the
    #    tags; converge.run_static_point auto-detects this and writes
    #    an ISPIN=2 + magnetic-mixing wrap (no MAGMOM, no SUBATOM).
    #  * plain str.out with magnetic elements: standard FM spin —
    #    ISPIN=2 + uniform MAGMOM (same auto-on rule as the pipeline).
    struct = read_structure(sqs_dir / "str.out")
    tagged = any(("+" in sp or "-" in sp) for sp in struct.species())
    els = elements_in(sqs_dir)
    if args.vasp_wrap:
        if not args.vasp_wrap.is_file():
            raise SystemExit(f"--vasp-wrap {args.vasp_wrap}: not found")
        print(f"[wrap] using {args.vasp_wrap} for every point "
              f"(ENCUT/KPPRA replaced per point; no auto settings)")
    elif tagged:
        print(f"[spin] tagged str.out (DLM) — ezvasp derives MAGMOM; "
              f"wrap gets ISPIN=2 + magnetic mixing only")
    elif vaspwrap.wants_spin(els):
        vaspwrap.DEFAULT_SPIN = True
        print(f"[spin] magnetic elements {els} — ISPIN=2 + uniform "
              f"MAGMOM (FM) enabled for the sweep statics")
    print(f"[sweep] dir     : {sqs_dir}")
    print(f"[sweep] max ENMAX {enmax:.1f} eV; tol "
          f"{args.tol_ev * 1e3:.2f} meV/atom; ALGO {args.algo}")

    encut, kppra, kres, eres = converge.converge_sqs(
        sqs_dir, sqs_dir / "convergence", potcar_paths,
        dlm=None, algo=args.algo, tol_ev=args.tol_ev,
        env_bin=args.env_bin, timeout=args.timeout,
        cmd_prefix=args.cmd_prefix, wrap_template=args.vasp_wrap)

    print(kres.table())
    print(eres.table())

    # ISIF=3 cell relaxations need the Pulay-safe floor on top of the
    # energy-converged value; statics/phonons use the sweep value.
    relax_encut = potcar.pulay_safe_encut(encut, enmax)
    print(f"\nRECOMMENDED: ENCUT = {encut} eV   KPPRA = {kppra}")
    print(f"             (for ISIF=3 relaxations use ENCUT >= "
          f"{relax_encut} eV — Pulay floor 1.3 x ENMAX)")
    if not (kres.converged and eres.converged):
        print("WARNING: at least one sweep did NOT satisfy the "
              "convergence rule — inspect the tables above before "
              "trusting these values.")

    out = {"sqs_dir": str(sqs_dir), "encut": encut, "kppra": kppra,
           "relax_encut_pulay": relax_encut,
           "kppra_converged": kres.converged, "kppra_rule": kres.rule,
           "encut_converged": eres.converged, "encut_rule": eres.rule,
           "tol_ev": args.tol_ev, "algo": args.algo,
           "vasp_wrap": str(args.vasp_wrap) if args.vasp_wrap else "generated"}
    (sqs_dir / "convergence_result.json").write_text(
        json.dumps(out, indent=2))
    print(f"[sweep] written: {sqs_dir / 'convergence_result.json'}")
    return 0 if (kres.converged and eres.converged) else 1


if __name__ == "__main__":
    sys.exit(main())
