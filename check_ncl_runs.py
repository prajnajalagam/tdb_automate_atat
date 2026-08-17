#!/usr/bin/env python3
"""
Survey constrained-noncollinear DLM runs beneath a directory and report which
ones actually succeeded -- both numerically (did VASP finish and converge) and
magnetically (did the local moments survive, or collapse toward NM).

Run it on the cluster, from e.g. .../CoCrNi_fullpipeline/NCL :

    python3 check_ncl_runs.py                 # survey ./**, table to stdout
    python3 check_ncl_runs.py /path/to/NCL --csv ncl_status.csv
    python3 check_ncl_runs.py --dir <one_sqs_dir> -v      # detail for one run

Standard library only (no numpy) -- runs on a bare login node.

Why the checks are what they are
--------------------------------
* VASP writes the constrained-moment output to OSZICAR, not OUTCAR:
  `E_p = ... lambda = ...` plus an `ion / MW_int / M_int` block (the per-site
  moment vector integrated inside the RWIGS sphere) at every electronic step.
  We read the FIRST block (initial moments) and the LAST (final moments) by
  seeking to the head and tail of the file -- so file size is irrelevant.
* A SMALL E_p IS NOT PROOF OF SUCCESS. With I_CONSTRAINED_M=1 only the
  direction is constrained; if a moment collapses to zero there is no
  direction left to penalise and E_p goes to zero trivially. The real health
  metric is the final |m_i| distribution, which is what STATUS is based on.
* Direction fidelity is checked separately: the angle between each final
  moment and its M_CONSTR target (evaluated only on surviving sites, since
  the angle is meaningless noise for a dead moment).

STATUS values
-------------
  OK          finished, converged, moments survived
  COLLAPSED   finished but the local moments died (NM-like)
  WEAK        partial collapse -- some sites survived, many did not
  UNCONVERGED electronic loop hit NELM, or ionic relaxation never reached
              the force criterion
  RUNNING     no clean VASP termination yet (or killed at walltime)
  NOT_NCL     INCAR is not a noncollinear/constrained run
  NO_DATA     no OSZICAR -- never started
"""

import argparse
import csv
import math
import os
import re
import sys

HEAD_BYTES = 400_000       # enough for the first constraint block
TAIL_BYTES = 400_000       # enough for the last constraint block + E_p
OUTCAR_TAIL = 200_000      # only for the termination / accuracy markers

ION_BLOCK_RE = re.compile(r"ion\s+MW_int\s+M_int")
EP_RE = re.compile(r"E_?p\s*=\s*([-+0-9.EeDd]+)")
FLINE_RE = re.compile(r"^\s*(\d+)\s+F=\s*([-+0-9.EeDd]+)")


# --------------------------------------------------------------- io helpers
def read_head(path, n=HEAD_BYTES):
    with open(path, "rb") as fh:
        return fh.read(n).decode("utf-8", "replace")


def read_tail(path, n=TAIL_BYTES):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - n))
        return fh.read().decode("utf-8", "replace")


# ------------------------------------------------------------- file parsing
def parse_ion_block(text, which="last"):
    """Return list of (MW_int, M_int) vectors from a constraint block."""
    starts = [m.end() for m in ION_BLOCK_RE.finditer(text)]
    if not starts:
        return None
    start = starts[0] if which == "first" else starts[-1]
    rows = []
    for line in text[start:].split("\n")[1:]:
        tok = line.split()
        if len(tok) == 7 and tok[0].isdigit():
            try:
                vals = [float(x) for x in tok[1:]]
            except ValueError:
                break
            rows.append((vals[0:3], vals[3:6]))
        elif rows:
            break
    return rows or None


def parse_incar(path):
    """Tag -> value, joining values that spill onto continuation lines."""
    tags, key, buf = {}, None, []
    if not os.path.exists(path):
        return tags
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.split("#")[0].split("!")[0].rstrip()
            m = re.match(r"\s*([A-Za-z_]+)\s*=\s*(.*)$", line)
            if m:
                if key:
                    tags[key] = " ".join(buf).strip()
                key, buf = m.group(1).upper(), [m.group(2)]
            elif key and line.strip():
                buf.append(line.strip())
    if key:
        tags[key] = " ".join(buf).strip()
    return tags


def parse_poscar_species(path):
    """Per-site element symbols.

    ezvasp writes a trailing label per atom line (e.g. `Cr_pv+1.5`), which is
    the most reliable source and also encodes the DLM up/down assignment.
    Falls back to the VASP5 symbol line + counts.
    """
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        lines = fh.read().split("\n")
    i = 5
    tok = lines[i].split()
    symbols = None
    if tok and not tok[0].lstrip("+-").isdigit():
        symbols = tok
        i += 1
        tok = lines[i].split()
    try:
        counts = [int(t) for t in tok]
    except ValueError:
        return None
    n = sum(counts)
    i += 1
    if lines[i].strip().lower().startswith("s"):
        i += 1
    i += 1                                   # Direct / Cartesian line
    labels = []
    for j in range(n):
        parts = lines[i + j].split()
        if len(parts) > 3:
            m = re.match(r"([A-Z][a-z]?)", parts[3])
            labels.append(m.group(1) if m else "?")
        else:
            labels.append(None)
    if all(labels):
        return labels
    if symbols:
        return [s for s, c in zip(symbols, counts) for _ in range(c)]
    return None


def parse_outcar_magnetization(path, tail_bytes=20_000_000):
    """Per-ion moments from the OUTCAR `magnetization (x[/y/z])` tables.

    Used for COLLINEAR runs (ISPIN=2), where OSZICAR carries only the scalar
    `mag=` total. Requires LORBIT>=10. Returns a list of 3-vectors: collinear
    moments come back as (0, 0, m) so downstream metrics are shared with the
    noncollinear path. Reads only the tail, so OUTCAR size is irrelevant.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    text = read_tail(path, tail_bytes)
    comps = {}
    for comp in ("x", "y", "z"):
        starts = [m.end() for m in
                  re.finditer(r"magnetization \(%s\)" % comp, text)]
        if not starts:
            continue
        rows = []
        for line in text[starts[-1]:].split("\n"):
            tok = line.split()
            if len(tok) >= 3 and tok[0].isdigit():
                try:
                    rows.append(float(tok[-1]))     # `tot` column
                except ValueError:
                    break
            elif rows:
                break
        if rows:
            comps[comp] = rows
    if "x" not in comps:
        return None
    n = len(comps["x"])
    if "y" in comps and "z" in comps:                # noncollinear tables
        return [[comps["x"][i], comps["y"][i], comps["z"][i]]
                for i in range(n)]
    return [[0.0, 0.0, comps["x"][i]] for i in range(n)]   # collinear


def beta_star(species, mags):
    """Xiong et al., CALPHAD 39 (2012) 11-20, Eqs. (9)-(10).

        S_max = R ln(beta* + 1) = R sum_i x_i ln(beta_i + 1)
        beta* = prod_i (beta_i + 1)^x_i - 1

    beta_i is the LOCAL moment of component i (mean |m| over that species'
    sites), x_i its mole fraction. Note a species whose moment collapses
    contributes ln(1) = 0 to the entropy, which is the correct limit --
    cf. the paper's remark that bcc Cr is assigned beta = 0.008 muB in
    CALPHAD databases precisely because its moment is ill-defined.
    """
    if not species:
        return None, None, None
    n = len(species)
    betas, ln_sum = {}, 0.0
    for el in sorted(set(species)):
        sel = [m for m, s in zip(mags, species) if s == el]
        b = sum(sel) / len(sel)                      # mean local moment
        betas[el] = b
        ln_sum += (len(sel) / n) * math.log(b + 1.0)
    bstar = math.exp(ln_sum) - 1.0
    s_max = 8.314462618 * math.log(bstar + 1.0)      # J/(mol*K)
    return bstar, s_max, betas


def parse_vector_tag(value, natoms):
    try:
        nums = [float(x) for x in value.split()]
    except ValueError:
        return None
    return [nums[3 * i:3 * i + 3] for i in range(natoms)] \
        if len(nums) >= 3 * natoms else None


# ------------------------------------------------------------------ metrics
def norm(v):
    return math.sqrt(sum(x * x for x in v))


def angle_deg(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    c = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def moment_stats(vectors, species):
    mags = [norm(v) for v in vectors]
    n = len(mags)
    out = {
        "mean": sum(mags) / n,
        "rms": math.sqrt(sum(m * m for m in mags) / n),
        "max": max(mags),
        "min": min(mags),
        "net_per_atom": norm([sum(v[k] for v in vectors) for k in range(3)]) / n,
    }
    if species:
        per = {}
        for el in sorted(set(species)):
            sel = [m for m, s in zip(mags, species) if s == el]
            per[el] = math.sqrt(sum(m * m for m in sel) / len(sel))
        out["per_species_rms"] = per
    return out, mags


# ------------------------------------------------------------ one directory
def survey(d, args):
    r = {"dir": d, "status": "NO_DATA", "note": ""}
    osz = os.path.join(d, "OSZICAR")
    if not os.path.exists(osz) or os.path.getsize(osz) == 0:
        return r

    incar = parse_incar(os.path.join(d, "INCAR"))
    ncl = incar.get("LNONCOLLINEAR", "").upper().startswith(".T")
    ispin = int(float(incar.get("ISPIN", 1) or 1))
    r["constrained"] = incar.get("I_CONSTRAINED_M", "-")
    r["lambda"] = incar.get("LAMBDA", "-")
    r["mode"] = "NCL" if ncl else ("COLLINEAR" if ispin == 2 else "NM")

    tail = read_tail(osz)
    m_init = None
    if ncl:
        # constrained-noncollinear: per-site vectors live in the OSZICAR
        head = read_head(osz)
        first = parse_ion_block(head, "first")
        last = parse_ion_block(tail, "last")
        if not last:
            r["status"] = "RUNNING"
            r["note"] = "no complete constraint block in OSZICAR"
            return r
        m_final = [v[1] for v in last]
        if first:
            m_init = [v[1] for v in first]
    elif ispin == 2:
        # collinear (FM or DLM): per-site moments only exist in the OUTCAR
        m_final = parse_outcar_magnetization(os.path.join(d, "OUTCAR"),
                                             args.outcar_tail)
        if not m_final:
            r["status"] = "RUNNING"
            r["note"] = ("no magnetization table in OUTCAR "
                         "(still running, or LORBIT<10)")
            return r
        init = parse_vector_tag("", 0)
        try:                                    # scalar MAGMOM -> z-vectors
            vals = []
            for tokn in incar.get("MAGMOM", "").split():
                if "*" in tokn:
                    c, v = tokn.split("*")
                    vals.extend([float(v)] * int(c))
                else:
                    vals.append(float(tokn))
            if len(vals) == len(m_final):
                m_init = [[0.0, 0.0, v] for v in vals]
        except ValueError:
            pass
    else:
        r["status"] = "NOT_NCL"
        r["note"] = "neither LNONCOLLINEAR nor ISPIN=2 (nonmagnetic run)"
        return r

    species = parse_poscar_species(os.path.join(d, "POSCAR"))
    r["natoms"] = len(m_final)
    if species and len(species) == len(m_final):
        r["composition"] = ",".join(
            f"{el}{species.count(el) / len(species):.3f}"
            for el in sorted(set(species)))
    else:
        species, r["composition"] = None, ""

    stats_f, mags_f = moment_stats(m_final, species)
    r["rms_final"] = stats_f["rms"]
    r["mean_final"] = stats_f["mean"]
    r["net_per_atom"] = stats_f["net_per_atom"]
    r["per_species_rms"] = stats_f.get("per_species_rms", {})
    if m_init:
        stats_i, _ = moment_stats(m_init, species)
        r["rms_init"] = stats_i["rms"]
    else:
        r["rms_init"] = float("nan")

    r["n_collapsed"] = sum(1 for m in mags_f if m < args.collapse_mag)

    # --- Xiong effective magnetic moment (CALPHAD 39 (2012) 11-20, Eq. 10)
    bstar, s_max, betas = beta_star(species, mags_f)
    r["beta_star"] = bstar if bstar is not None else float("nan")
    r["S_max_J_molK"] = s_max if s_max is not None else float("nan")
    r["beta_per_species"] = betas or {}

    # --- direction fidelity, surviving sites only
    #
    # IMPORTANT: with I_CONSTRAINED_M = 1 the penalty is
    #     E_p = lambda * sum_I |M_I - e_I (M_I . e_I)|^2
    # i.e. it penalises only the component PERPENDICULAR to the target, and
    # is therefore invariant under M_I -> -M_I. Mode 1 constrains the AXIS,
    # not the sense. A moment may flip 180 deg at zero penalty cost, which
    # silently destroys the vanishing pair correlations the SQS was built to
    # enforce. So report the two separately:
    #   axis_dev  = max angle folded into [0,90] -> is the constraint holding?
    #   n_flipped = sites past 90 deg          -> how many flipped sense?
    mc = parse_vector_tag(incar.get("M_CONSTR", ""), len(m_final)) if ncl \
        else None
    if mc:
        pairs = [(angle_deg(m, t), mag) for m, t, mag
                 in zip(m_final, mc, mags_f)]
        live = [(a, mag) for a, mag in pairs
                if a is not None and mag >= args.collapse_mag]
        if live:
            r["max_angle_deg"] = max(a for a, _ in live)
            r["axis_dev_deg"] = max(min(a, 180.0 - a) for a, _ in live)
            r["n_flipped"] = sum(1 for a, _ in live if a > 90.0)
            r["n_live"] = len(live)
        else:
            r["max_angle_deg"] = r["axis_dev_deg"] = float("nan")
            r["n_flipped"], r["n_live"] = 0, 0
    else:
        r["max_angle_deg"] = r["axis_dev_deg"] = float("nan")
        r["n_flipped"], r["n_live"] = 0, 0

    # --- energies / penalty
    eps = EP_RE.findall(tail) if ncl else []
    r["E_p"] = float(eps[-1].replace("D", "E")) if eps else float("nan")
    fl = FLINE_RE.findall(tail.replace("\r", ""))
    if not fl:
        fl = [(m.group(1), m.group(2)) for m in
              (FLINE_RE.match(l) for l in tail.split("\n")) if m]
    if fl:
        r["n_ionic"] = int(fl[-1][0])
        r["F_eV"] = float(fl[-1][1].replace("D", "E"))
    else:
        r["n_ionic"], r["F_eV"] = 0, float("nan")

    # electronic convergence: steps in the final ionic block vs NELM
    nelm = int(float(incar.get("NELM", 60)))
    tail_lines = tail.split("\n")
    last_f = max((i for i, l in enumerate(tail_lines) if FLINE_RE.match(l)),
                 default=None)
    prev_f = max((i for i, l in enumerate(tail_lines[:last_f])
                  if FLINE_RE.match(l)), default=-1) if last_f else None
    if last_f is not None:
        nelec = sum(1 for l in tail_lines[prev_f + 1:last_f]
                    if re.match(r"\s*(DAV|RMM|CGA|EDD):", l))
        r["n_elec_last"] = nelec
        elec_ok = 0 < nelec < nelm
    else:
        r["n_elec_last"], elec_ok = 0, False

    # --- VASP termination + ionic accuracy (small tail read of OUTCAR)
    out_p = os.path.join(d, "OUTCAR")
    finished = ionic_ok = False
    if os.path.exists(out_p) and os.path.getsize(out_p) > 0:
        otail = read_tail(out_p, OUTCAR_TAIL)
        finished = "General timing and accounting" in otail
        ionic_ok = "reached required accuracy" in otail
    nsw = int(float(incar.get("NSW", 0)))
    r["finished"] = finished
    r["ionic_ok"] = ionic_ok or nsw <= 1

    # --- ATAT bookkeeping
    r["has_energy"] = os.path.exists(os.path.join(d, "energy"))
    r["ep_corrected"] = os.path.exists(os.path.join(d, "energy.raw"))
    r["waiting"] = os.path.exists(os.path.join(d, "wait"))

    # --- has the cell drifted toward magnetic ORDER?
    # For a properly disordered cell of N sites the net moment is a random
    # walk: |sum m| / N ~ RMS|m| / sqrt(N). Substantially more than that means
    # the spins have aligned and this is no longer a paramagnetic sample.
    expect = r["rms_final"] / math.sqrt(r["natoms"]) if r["natoms"] else 0.0
    r["net_ratio"] = (r["net_per_atom"] / expect) if expect > 1e-9 else 0.0

    # --- verdict
    frac_dead = r["n_collapsed"] / r["natoms"]
    if not finished:
        r["status"] = "RUNNING"
        r["note"] = "no VASP termination marker"
    elif not elec_ok or not r["ionic_ok"]:
        r["status"] = "UNCONVERGED"
        r["note"] = (f"elec {r['n_elec_last']}/{nelm}"
                     + ("" if r["ionic_ok"] else ", forces not converged"))
    elif frac_dead >= args.collapse_frac:
        r["status"] = "COLLAPSED"
        r["note"] = f"{r['n_collapsed']}/{r['natoms']} sites < {args.collapse_mag} muB"
    elif r["net_ratio"] > args.order_ratio:
        r["status"] = "ORDERED"
        r["note"] = (f"net moment {r['net_ratio']:.1f}x the random-walk value"
                     f", {r['n_flipped']}/{r['n_live']} spins flipped sense"
                     " -- not a paramagnetic state")
    elif frac_dead > 0:
        r["status"] = "WEAK"
        r["note"] = f"{r['n_collapsed']}/{r['natoms']} sites collapsed"
    else:
        r["status"] = "OK"
    if r["n_flipped"] and r["status"] in ("OK", "WEAK"):
        r["note"] = (r["note"] + "; " if r["note"] else "") + \
            f"{r['n_flipped']}/{r['n_live']} spins flipped sense (mode-1 " \
            f"penalty is axis-only)"
    return r


# ------------------------------------------------------------------- driver
def find_dirs(root):
    hits = []
    for dp, dn, fn in os.walk(root):
        if "OSZICAR" in fn or "INCAR" in fn:
            hits.append(dp)
            dn[:] = []                       # do not recurse into a run dir
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".",
                    help="directory to search below (default: .)")
    ap.add_argument("--dir", default=None,
                    help="survey a single run directory instead")
    ap.add_argument("--collapse-mag", type=float, default=0.20,
                    help="a site with |m| below this (muB) counts as "
                         "collapsed (default: 0.20)")
    ap.add_argument("--collapse-frac", type=float, default=0.5,
                    help="fraction of dead sites at which the run is called "
                         "COLLAPSED (default: 0.5)")
    ap.add_argument("--outcar-tail", type=int, default=20_000_000,
                    help="bytes of OUTCAR tail scanned for the collinear "
                         "magnetization table (default: 20MB)")
    ap.add_argument("--order-ratio", type=float, default=2.5,
                    help="call the cell ORDERED when its net moment exceeds "
                         "this multiple of the random-walk expectation "
                         "RMS|m|/sqrt(N) (default: 2.5)")
    ap.add_argument("--csv", default=None, help="also write a CSV here")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="per-species detail for each run")
    ap.add_argument("--only", default=None,
                    help="print only rows with this status (e.g. OK)")
    args = ap.parse_args()

    dirs = [args.dir] if args.dir else find_dirs(args.root)
    if not dirs:
        sys.exit(f"no run directories found under {args.root}")

    rows = [survey(d, args) for d in dirs]
    base = os.path.abspath(args.root)

    hdr = (f"{'STATUS':<12}{'RMS|m| init->final':<22}{'dead':>6}"
           f"{'E_p(meV)':>10}{'axis':>6}{'flip':>7}{'net/rw':>8}"
           f"{'beta*':>7}{'ionic':>7}  run")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if args.only and r["status"] != args.only:
            continue
        name = os.path.relpath(r["dir"], base) if not args.dir else r["dir"]
        if r["status"] in ("NO_DATA", "NOT_NCL"):
            print(f"{r['status']:<12}{r.get('note',''):<46}  {name}")
            continue
        rms = f"{r.get('rms_init', float('nan')):.3f} -> {r.get('rms_final', float('nan')):.3f}"
        dead = f"{r.get('n_collapsed','?')}/{r.get('natoms','?')}"
        ep = r.get("E_p", float("nan")) * 1000
        axis = r.get("axis_dev_deg", float("nan"))
        flip = f"{r.get('n_flipped', 0)}/{r.get('n_live', 0)}"
        print(f"{r['status']:<12}{rms:<22}{dead:>6}{ep:>10.3f}"
              f"{axis:>6.1f}{flip:>7}{r.get('net_ratio', 0.0):>8.1f}"
              f"{r.get('beta_star', float('nan')):>7.3f}"
              f"{r.get('n_ionic',0):>7}  {name}")
        if args.verbose:
            per = r.get("beta_per_species", {})
            if per:
                print("            beta_i (mean local |m|, muB): " +
                      "  ".join(f"{k}={v:.3f}" for k, v in per.items())
                      + f"   -> beta*={r.get('beta_star', float('nan')):.4f}"
                      f", S_max={r.get('S_max_J_molK', float('nan')):.3f}"
                      " J/(mol K)")
            print(f"            F = {r.get('F_eV', float('nan')):.5f} eV"
                  f"   net|m|/atom = {r.get('net_per_atom', float('nan')):.4f}"
                  f"   energy={'y' if r.get('has_energy') else 'n'}"
                  f"   E_p-corrected={'y' if r.get('ep_corrected') else 'n'}")
            if r.get("note"):
                print(f"            note: {r['note']}")

    print()
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("summary: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f"   (total {len(rows)})")
    ok = [r for r in rows if r["status"] in ("OK", "WEAK")]
    if ok:
        print(f"usable for the moment/energy harvest: {len(ok)}")

    if args.csv:
        cols = ["dir", "status", "mode", "note", "composition", "natoms",
                "beta_star", "S_max_J_molK", "rms_init",
                "rms_final", "mean_final", "net_per_atom", "net_ratio",
                "n_collapsed", "max_angle_deg", "axis_dev_deg", "n_flipped",
                "n_live", "E_p", "F_eV", "n_ionic", "n_elec_last",
                "finished", "ionic_ok", "has_energy", "ep_corrected",
                "waiting", "constrained", "lambda"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
