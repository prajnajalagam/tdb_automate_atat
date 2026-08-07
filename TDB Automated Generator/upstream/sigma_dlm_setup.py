#!/usr/bin/env python3
"""
sigma_dlm_setup.py — standalone DLM setup for SIGMA_D8B endmembers via
the lev=3 spin-SQS route (no pipeline run needed).

THE IDEA (user spec 2026-08-07): a lev=3 generation makes an SQS with
equal-likelihood occupancy of two species on EVERY sublattice. If the
two "species" of each sublattice are the spin-up and spin-down
pseudo-species of that sublattice's element —

    species.in for the endmember (aj=Co, g=Cr, ii=Cr):
        aj=Co+2,Co-2
        g=Cr+2,Cr-2
        ii=Cr+2,Cr-2

— then the fully-mixed lev=3 SQS that sqs2tdb produces IS the
DLM-ready endmember: each sublattice 50/50 up/down with an
SQS-optimized spin arrangement (not a random decoration), str.out
ready for ATAT + a DLM vasp.wrap (SUBATOM lines map El+2 -> El).

WHAT IT DOES, per endmember (all 2^3 = 8 sublattice->element
assignments for a binary):
  1. writes <phase>/species.in with the per-sublattice spin pairs;
  2. runs `sqs2tdb -cp -lv=3 -l=SIGMA_D8B` twice (the usual two-pass
     habit; with species.in pre-planted pass 1 already copies);
  3. locates the FULLY-mixed lev=3 dir (every sublattice carries both
     +/- tokens at 0.5) and moves it to
         <phase>/dlm_endmembers/sqs_lev=0_aj_Co=1,g_Cr=1,ii_Cr=1_dlm/
     with an `endmem` marker;
  4. deletes the other dirs of that generation round (partial-spin
     lev<3 mixtures — a sublattice locked to pure El+2 is fully
     polarized, not DLM: junk).

Usage:
    cd <WORKDIR>/SIGMA_D8B            # or pass the dir as an argument
    python3 /path/to/sigma_dlm_setup.py
    python3 sigma_dlm_setup.py --elements Co,Cr --moment 2

Idempotent: endmembers whose dlm_endmembers/ output already exists are
skipped. sqs2tdb must be on PATH (it is inside your usual job/pfe env).
"""

from __future__ import annotations

import argparse
import itertools
import re
import shutil
import subprocess
import sys
from pathlib import Path

# SIGMA_D8B sublattice letters as they appear in sqs2tdb dir names
# (sqs_lev=0_aj_Co=1,g_Co=1,ii_Cr=1): 2a+8i -> "aj", 4g -> "g",
# 8i+8j -> "ii" in the sqsdb naming.
DEFAULT_SITES = ("aj", "g", "ii")

_TOKEN = re.compile(r"([a-z]+)_([A-Za-z+\-0-9.]+?)=([0-9.]+)")


def run2(cmd, cwd: Path, log_base: Path) -> None:
    """The two-pass sqs2tdb habit, each pass logged."""
    for n in (1, 2):
        with open(f"{log_base}.pass{n}.log", "w") as fh:
            fh.write(f"$ cd {cwd}\n$ {' '.join(cmd)}\n{'-' * 60}\n")
            fh.flush()
            subprocess.run(cmd, cwd=str(cwd), stdout=fh,
                           stderr=subprocess.STDOUT, text=True)


def name_tokens(dirname: str):
    """[(site, species, fraction), ...] scraped from a decorated name."""
    return [(s, sp, float(v)) for s, sp, v in _TOKEN.findall(dirname)]


def is_full_spin_mix(dirname: str, sites, assignment, moment) -> bool:
    """True iff EVERY sublattice carries both El+m and El-m at 0.5 —
    the one dir per round that is the DLM endmember."""
    toks = name_tokens(dirname)
    for site, el in zip(sites, assignment):
        up = (site, f"{el}+{moment:g}", 0.5)
        dn = (site, f"{el}-{moment:g}", 0.5)
        if up not in toks or dn not in toks:
            return False
    return True


def setup_one_endmember(phase_dir: Path, sites, assignment,
                        moment: float, lattice: str) -> str:
    work_root = phase_dir.parent
    dlm_root = phase_dir / "dlm_endmembers"
    out_name = "sqs_lev=0_" + ",".join(
        f"{s}_{el}=1" for s, el in zip(sites, assignment)) + "_dlm"
    out_dir = dlm_root / out_name
    if (out_dir / "str.out").is_file():
        return f"skip (exists): {out_name}"

    # 1. per-sublattice spin-pair species.in for THIS endmember.
    # REAL sqs2tdb format (verified on NAS 2026-08-07): ONE line, the
    # sublattices TAB-separated —
    #     aj=Co,Cr,Ni<TAB>g=Co,Cr,Ni<TAB>ii=Co,Cr,Ni
    # A newline-per-sublattice file silently registers only its FIRST
    # line (the failed first attempt generated aj-only SQS).
    spfile = phase_dir / "species.in"
    backup = phase_dir / "species.in.orig"
    if spfile.is_file() and not backup.is_file():
        shutil.copy2(spfile, backup)       # preserve the original once
    spfile.write_text("\t".join(
        f"{s}={el}+{moment:g},{el}-{moment:g}"
        for s, el in zip(sites, assignment)) + "\n")

    # [GUARD] a work-root species.in stalls the two-pass handshake
    stray = work_root / "species.in"
    if stray.is_file():
        stray.rename(work_root / "species.in.stray")

    # 2. generate at lev=3 (NO -sp: sqs2tdb must use the species.in we
    #    just wrote, not overwrite it from the command line)
    before = {d.name for d in phase_dir.glob("sqs_lev=*") if d.is_dir()}
    run2(["sqs2tdb", "-cp", "-lv=3", f"-l={lattice}"], work_root,
         phase_dir / f"sqs2tdb_dlm_{'_'.join(assignment)}")
    new = [d for d in phase_dir.glob("sqs_lev=*")
           if d.is_dir() and d.name not in before]
    if not new:
        return (f"FAILED: sqs2tdb generated nothing for {out_name} — "
                f"check {phase_dir}/sqs2tdb_dlm_*.log (does the sqsdb "
                f"SIGMA entry provide lev=3?)")

    # 3. the fully-mixed lev=3 dir is the DLM endmember
    full = [d for d in new
            if is_full_spin_mix(d.name, sites, assignment, moment)]
    keep_msg = f"OK: {out_name}"
    if full:
        dlm_root.mkdir(exist_ok=True)
        shutil.move(str(full[0]), str(out_dir))
        (out_dir / "endmem").write_text("")
    else:
        keep_msg = (f"FAILED: no fully-mixed lev=3 dir among "
                    f"{[d.name for d in new]} for {out_name}")

    # 4. partial-spin mixtures from this round are junk (a pure El+m
    #    sublattice is ferromagnetically locked, not DLM) — but ONLY
    #    clean up after a SUCCESSFUL round; on failure keep everything
    #    for inspection (2026-08-07: the cleanup destroyed the very
    #    dirs needed to diagnose a species.in format mismatch).
    if full:
        for d in new:
            if d.exists() and d != out_dir:
                shutil.rmtree(d)
    return keep_msg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="DLM setup for SIGMA_D8B endmembers via the lev=3 "
                    "spin-SQS route.")
    ap.add_argument("phase_dir", nargs="?", default=".", type=Path,
                    help="the SIGMA_D8B directory (default: cwd)")
    ap.add_argument("--elements", default=None,
                    help="elements, comma-separated (default: AUTO — "
                         "scraped from the existing sqs_lev=0 "
                         "endmember dir names in the phase dir)")
    ap.add_argument("--moment", type=float, default=2.0,
                    help="spin pseudo-species magnitude (default 2 -> "
                         "El+2/El-2)")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES),
                    help="sublattice letters in species.in/dir-name "
                         "order (default aj,g,ii)")
    args = ap.parse_args(argv)

    phase_dir = args.phase_dir.resolve()
    lattice = phase_dir.name                     # e.g. SIGMA_D8B
    if not phase_dir.is_dir():
        raise SystemExit(f"{phase_dir} is not a directory")
    if args.elements:
        els = [e.strip() for e in args.elements.split(",")]
    else:
        # AUTO-DETECT from the existing endmember dir names, e.g.
        # sqs_lev=0_aj_Co=1,g_Ni=1,ii_Cr=1 -> {Co, Cr, Ni}. Spin-
        # tagged tokens (Co+2) from previous DLM rounds are excluded.
        found = set()
        el_re = re.compile(r"^[A-Z][a-z]?$")   # a real element symbol
        for d in phase_dir.glob("sqs_lev=0_*"):
            for _site, sp, _v in name_tokens(d.name):
                if el_re.match(sp):            # excludes Co+2 and the
                    found.add(sp)              # 'lev' of sqs_lev=0
        if not found:
            raise SystemExit(
                "could not auto-detect elements (no plain sqs_lev=0_* "
                "endmember dirs here) — pass --elements El1,El2[,El3]")
        els = sorted(found)
        print(f"[auto] elements from endmember dir names: {els} "
              f"-> {len(els) ** len(DEFAULT_SITES)} endmembers")
    sites = tuple(s.strip() for s in args.sites.split(","))

    # all sublattice->element assignments: 2^3 endmembers for a binary
    results = []
    for assignment in itertools.product(els, repeat=len(sites)):
        msg = setup_one_endmember(phase_dir, sites, assignment,
                                  args.moment, lattice)
        print(f"  {'-'.join(assignment):12s} {msg}")
        results.append(msg)

    # leave the phase dir the way we found it: restore the original
    # species.in (backup kept) so non-DLM bookkeeping stays coherent
    backup = phase_dir / "species.in.orig"
    if backup.is_file():
        shutil.copy2(backup, phase_dir / "species.in")
        print(f"\nrestored original species.in "
              f"(backup kept: {backup.name})")

    n_ok = sum(m.startswith(("OK", "skip")) for m in results)
    print(f"\n{n_ok}/{len(results)} endmembers ready under "
          f"{phase_dir / 'dlm_endmembers'}")
    print("VASP side: use a DLM vasp.wrap (SUBATOM lines mapping "
          "El+2/El-2 -> El with +/- MAGMOM — vaspwrap.build_vasp_wrap"
          "(dlm=...) writes exactly that).")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:      # e.g. `... | head` closing stdout
        sys.exit(0)
