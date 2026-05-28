#!/usr/bin/env python3
"""
STEP 1: Interactive endmember selection for a binary A-B.

Scans data roots for lev=0 SQS directories that contain the mandatory
files required by sqs2tdb -fit (energy, str.out, str_relax.out) and
optionally svib_ht.  Produces an endmembers.yaml consumed by the
pipeline.

Naming conventions (post `sqs2tdb -cp`):
  FCC_A1 / BCC_A2:  sqs_lev=0_a_Al=1       (Wyckoff site: a)
  HCP_A3:           sqs_lev=0_c_Co=1        (Wyckoff site: c)
  SIGMA_D8B:        sqs_lev=0_aj_Co=1,g_Co=1,ii_Cr=1  (sites: aj, g, ii)
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# ── Phase catalogue ──────────────────────────────────────────────────
PHASE_TOKENS = {
    "FCC_A1":    ["FCC_A1", "FCC"],
    "BCC_A2":    ["BCC_A2", "BCC"],
    "HCP_A3":    ["HCP_A3", "HCP"],
    "SIGMA_D8B": ["SIGMA_D8B", "SIGMA"],
    }

# SIGMA sublattice multiplicities (from rndstr.skel: aj=10, g=4, ii=16)
SIGMA_SUBLATTICE_MULT = {"aj": 10, "g": 4, "ii": 16}

# Wyckoff site used for the single-sublattice phases
SITE_FOR_PHASE = {
    "FCC_A1": "a",
    "BCC_A2": "a",
    "HCP_A3": "c"
    }


_SCI_FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")

# sqs2tdb -cp has emitted both "sqs_lev=N_..." and "sqsdb_lev=N_..." across
# ATAT versions; accept either prefix so the scan doesn't silently find zero.
SQS_DIR_RE = re.compile(r"sqs(?:db)?_lev=(\d+)")
SQS_PREFIX_RE = re.compile(r"^sqs(?:db)?_lev=\d+_")


# ── Helpers ──────────────────────────────────────────────────────────

def parse_energy(path: Path) -> Optional[float]:
    """Parse a single number (possibly Fortran scientific notation)."""
    try:
        txt = path.read_text(errors="ignore").strip()
        txt = txt.replace("D", "E").replace("d", "E")
        try:
            return float(txt)
        except ValueError:
            nums = _SCI_FLOAT_RE.findall(txt)
            return float(nums[-1].replace("D", "E").replace("d", "E")) if nums else None
    except Exception as exc:
        print(f"  WARNING: cannot parse {path}: {exc}")
        return None


def find_svib_ht(sqs_dir: Path) -> Optional[Path]:
    """Return path to svib_ht if it exists (direct or under vol_0, depth <= 3)."""
    direct = sqs_dir / "svib_ht"
    if direct.is_file():
        return direct
    vol_0 = sqs_dir / "vol_0"
    if vol_0.is_dir():
        for item in vol_0.rglob("svib_ht"):
            if item.is_file():
                return item
    return None


def has_mandatory_files(d: Path) -> Tuple[bool, str]:
    """Check that a directory has the files sqs2tdb -fit needs."""
    missing = []
    if not (d / "energy").is_file():
        missing.append("energy")
    if not (d / "str.out").is_file():
        missing.append("str.out")
    if missing:
        return False, ", ".join(missing)
    return True, ""


def infer_phase(path: Path) -> Optional[str]:
    u = str(path).upper()
    for ph, toks in PHASE_TOKENS.items():
        if any(t in u for t in toks):
            return ph
    return None


# ── SQS name parsing ────────────────────────────────────────────────

def parse_sqs_name(name: str, phase: str, elA: str, elB: str):
    """
    Return (level, xA, xB) or None.

    The directory name (after `sqs2tdb -cp`) looks like:
      sqs_lev=0_a_Al=1                        (FCC, BCC)
      sqs_lev=0_c_Co=1                        (HCP)
      sqs_lev=0_aj_Co=1,g_Co=1,ii_Cr=1       (SIGMA)

    After stripping "sqs_" and "lev=N_", the remainder is a
    comma-separated list of SITE_ELEMENT=CONC tokens.
    """
    elA, elB = elA.upper(), elB.upper()

    m = SQS_DIR_RE.match(name)
    if not m:
        return None
    lev = int(m.group(1))

    # Strip prefix to get the composition tokens
    comp_str = SQS_PREFIX_RE.sub("", name)
    # Parse all site_Element=value tokens
    tokens = re.findall(r"([a-z]+)_([A-Za-z]+)=([0-9.]+)", comp_str)
    if not tokens:
        return None

    if phase in ("FCC_A1", "BCC_A2", "HCP_A3"):
        expected_site = SITE_FOR_PHASE[phase]
        comp = {}
        for site, elem, val in tokens:
            if site == expected_site:
                comp[elem.upper()] = float(val)
        xA = comp.get(elA, 0.0)
        xB = comp.get(elB, 0.0)
        total = xA + xB
        if total <= 0:
            return None
        return lev, xA / total, xB / total

    if phase == "SIGMA_D8B":
        # Weight each site occupation by multiplicity
        xA = 0.0
        xB = 0.0
        for site, elem, val in tokens:
            if site not in SIGMA_SUBLATTICE_MULT:
                continue
            w = SIGMA_SUBLATTICE_MULT[site] * float(val)
            e = elem.upper()
            if e == elA:
                xA += w
            elif e == elB:
                xB += w
        total = xA + xB
        if total <= 0:
            return None
        return lev, xA / total, xB / total

    return None


def sigma_config_key(name: str) -> str:
    """
    For SIGMA deduplication: extract the site→element mapping ignoring path.
    Two directories with the same occupation pattern are duplicates.
    e.g. sqs_lev=0_aj_Co=1,g_Co=1,ii_Cr=1 → "aj_Co=1,g_Co=1,ii_Cr=1"
    """
    return SQS_PREFIX_RE.sub("", name)


def sigma_pick_best(cands: list) -> Tuple:
    """Pick best SIGMA candidate: prefer has-svib, then lowest energy, then shortest path."""
    def key(t):
        path, energy, svib_path = t
        e = energy if energy is not None else float("inf")
        return (0 if svib_path else 1, e, len(str(path)))
    return sorted(cands, key=key)[0]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Select endmembers for binary CALPHAD assessment")
    ap.add_argument("--element1", required=True)
    ap.add_argument("--element2", required=True)
    ap.add_argument("--data-roots", required=True,
                    help="Comma-separated root directories to scan")
    ap.add_argument("--scan-depth", type=int, default=6,
                    help="Max os.walk depth (default 6)")
    ap.add_argument("--out", default="endmembers.yaml")
    args = ap.parse_args()

    elA, elB = sorted([args.element1.upper(), args.element2.upper()])
    roots = [Path(r.strip()).resolve() for r in args.data_roots.split(",")]

    print(f"\n{'='*60}")
    print(f"  Endmember Selection for {elA}-{elB}")
    print(f"{'='*60}")
    print(f"  Scanning {len(roots)} root(s):")
    for r in roots:
        print(f"    {r}")
    print()

    # ── Scan ─────────────────────────────────────────────────────
    by_phase: Dict[str, list] = {ph: [] for ph in PHASE_TOKENS}
    sigma_bins: Dict[str, list] = {}
    skipped = 0

    for root in roots:
        if not root.exists():
            print(f"  WARNING: {root} does not exist, skipping")
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            # depth check
            try:
                depth = len(p.relative_to(root).parts)
                if depth > args.scan_depth:
                    dirnames.clear()
                    continue
            except ValueError:
                continue

            if not SQS_DIR_RE.match(p.name):
                continue

            phase = infer_phase(p)
            if not phase:
                continue

            parsed = parse_sqs_name(p.name, phase, elA, elB)
            if not parsed or parsed[0] != 0:
                continue

            ok, missing_msg = has_mandatory_files(p)
            if not ok:
                print(f"  SKIP {p.name}: missing {missing_msg}")
                skipped += 1
                continue

            lev, xA, xB = parsed
            energy = parse_energy(p / "energy")
            svib_path = find_svib_ht(p)

            if phase == "SIGMA_D8B":
                cfg = sigma_config_key(p.name)
                sigma_bins.setdefault(cfg, []).append(
                    (p, energy, svib_path, xA, xB))
            else:
                by_phase[phase].append((p, xA, xB, energy, svib_path))

    total = sum(len(v) for v in by_phase.values()) + sum(len(v) for v in sigma_bins.values())
    print(f"\n  Found {total} lev=0 endmembers ({skipped} skipped for missing files)\n")

    # ── Interactive selection ────────────────────────────────────
    selections: Dict = {}

    for ph in ("FCC_A1", "BCC_A2", "HCP_A3"):
        lst = by_phase.get(ph, [])
        if not lst:
            print(f"  [{ph}] No endmembers found — skipping\n")
            continue
        if len(lst) < 2:
            print(f"  [{ph}] Only {len(lst)} endmember found (need 2) — skipping\n")
            continue

        print(f"\n{'='*60}")
        print(f"  Phase: {ph}   ({len(lst)} candidates)")
        print(f"{'='*60}")

        lst.sort(key=lambda t: t[1])  # sort by xA
        for i, (p, xA, xB, e, sv) in enumerate(lst):
            e_str = f"{e:+.6f}" if e is not None else "N/A"
            sv_str = "YES" if sv else "NO"
            print(f"  [{i:2d}]  x({elA})={xA:.4f}  x({elB})={xB:.4f}"
                  f"  E={e_str}  svib={sv_str}")
            print(f"        {p}")

        while True:
            try:
                a_idx = int(input(f"\n  Select {elA}-rich endmember index: ").strip())
                b_idx = int(input(f"  Select {elB}-rich endmember index: ").strip())
                if not (0 <= a_idx < len(lst) and 0 <= b_idx < len(lst)):
                    print(f"  ERROR: index out of range [0, {len(lst)-1}]")
                elif a_idx == b_idx:
                    print("  ERROR: must select two different endmembers")
                else:
                    break
            except (ValueError, EOFError):
                print("  ERROR: enter valid integers")

        selections[ph] = {
            elA: str(lst[a_idx][0]),
            elB: str(lst[b_idx][0]),
        }
        print(f"\n  Selected for {ph}:")
        print(f"    {elA}: {selections[ph][elA]}")
        print(f"    {elB}: {selections[ph][elB]}")

    # ── SIGMA (auto-deduplicated) ────────────────────────────────
    if sigma_bins:
        print(f"\n{'='*60}")
        print(f"  Phase: SIGMA_D8B  ({len(sigma_bins)} unique configurations)")
        print(f"{'='*60}")

        sigma_selected = []
        for cfg in sorted(sigma_bins.keys()):
            cands = sigma_bins[cfg]
            xA, xB = cands[0][3], cands[0][4]
            best_path, best_e, best_sv = sigma_pick_best(
                [(p, e, sv) for p, e, sv, _, _ in cands])

            e_str = f"{best_e:+.6f}" if best_e is not None else "N/A"
            sv_str = "YES" if best_sv else "NO"
            n_dup = len(cands)
            print(f"  {cfg}")
            print(f"    x({elA})={xA:.4f}  E={e_str}  svib={sv_str}"
                  f"  ({n_dup} duplicate{'s' if n_dup>1 else ''})")
            print(f"    -> {best_path}")
            sigma_selected.append(str(best_path))

        if sigma_selected:
            selections["SIGMA_D8B"] = {"ALL": sigma_selected}
            print(f"\n  Selected {len(sigma_selected)} SIGMA endmembers")
    else:
        print(f"\n  [SIGMA_D8B] No endmembers found\n")

    # ── Write YAML ───────────────────────────────────────────────
    out = Path(args.out).resolve()
    with out.open("w") as f:
        f.write(f"# Endmember selection for {elA}-{elB} binary\n")
        f.write(f"# Generated by select_endmembers.py\n\n")
        f.write(f"binary: {elA}-{elB}\n\n")
        for ph in ("FCC_A1", "BCC_A2", "HCP_A3", "SIGMA_D8B"):
            if ph not in selections:
                continue
            f.write(f"{ph}:\n")
            if ph == "SIGMA_D8B":
                for p in selections[ph]["ALL"]:
                    f.write(f"  - {p}\n")
            else:
                for elem, path in sorted(selections[ph].items()):
                    f.write(f"  {elem}: {path}\n")
            f.write("\n")

    print(f"\n{'='*60}")
    print(f"  Output: {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()