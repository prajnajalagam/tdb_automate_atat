#!/usr/bin/env python3
"""
Audit every DLM calculation under a directory tree and tie the INCAR /
vasp.wrap settings to what actually happened to the magnetism, so that
"which settings work" is answered from evidence rather than opinion.

    cd /nobackup/pjalagam
    python3 dlm_audit.py . --csv dlm_audit.csv            # survey
    python3 dlm_audit.py . --csv dlm_audit.csv --run-magmom   # also invoke magmom_old

Standard library only.

LAYOUT IT UNDERSTANDS
---------------------
robustrelax_vasp / runstruct_vasp promote results into the SQS directory with
suffixes and gzip the OUTCARs:
    INCAR.relax   OSZICAR.relax   OUTCAR.relax.gz     <- the relaxation
    INCAR.static  OSZICAR.static  OUTCAR.static.gz    <- static (tetrahedron)
                                                         single point on the
                                                         relaxed geometry;
                                                         this is the energy
                                                         used for TDB fitting
and leave numbered stage directories (00/, 01/, ...) beneath. The HIGHEST
numbered stage dir holds the structural files of the last run, so that is
where the final geometry (and hence the final pair correlations) comes from.

DLM QUALIFICATION (a run is only counted as DLM if it passes)
-------------------------------------------------------------
  * MAGMOM carries MIXED SIGNS, and
  * the number of + and - entries is EQUAL (overall, and reported per species)
  * corroborated where possible by signed POSCAR labels (Co+2 / Co-2 style)
Anything else is recorded with dlm_ok=0 and a reason, and excluded from the
best-practice statistics.

MOMENTS
-------
Per-site moments come from `magmom_old` output (MAGMOM_raw / eff_MAGMOM) when
present, else from the OUTCAR `magnetization (x)` table. BOTH require
LORBIT >= 10 in the INCAR -- without it there is no per-ion table at all and
magmom_old cannot work, so the run is marked lorbit_ok=0 and its moment
columns are left blank rather than guessed.
"""

import argparse, csv, glob, gzip, math, os, random, re, subprocess, sys

# ------------------------------------------------------------------ file io
def _read(path, head=None, tail=None):
    if path.endswith(".gz"):
        if head:
            with gzip.open(path, "rb") as fh:
                return fh.read(head).decode("utf-8", "replace")
        buf = b""
        with gzip.open(path, "rb") as fh:
            while True:
                c = fh.read(1 << 20)
                if not c: break
                buf = (buf + c)[-(tail or (1 << 22)):]
        return buf.decode("utf-8", "replace")
    if head:
        with open(path, "rb") as fh:
            return fh.read(head).decode("utf-8", "replace")
    n = tail or (1 << 22); size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - n))
        return fh.read().decode("utf-8", "replace")

def first_of(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None

def stage_dirs(d):
    """Numbered robustrelax stage dirs, highest (most recent) first."""
    out = []
    for e in os.listdir(d) if os.path.isdir(d) else []:
        if re.fullmatch(r"\d+", e) and os.path.isdir(os.path.join(d, e)):
            out.append(e)
    return [os.path.join(d, e) for e in sorted(out, reverse=True)]

# --------------------------------------------------------------- parsing
def parse_tags(path):
    tags, key, buf = {}, None, []
    if not path or not os.path.exists(path): return tags
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.split("#")[0].split("!")[0].rstrip()
            m = re.match(r"\s*([A-Za-z_]+)\s*=\s*(.*)$", line)
            if m:
                if key: tags[key] = " ".join(buf).strip()
                key, buf = m.group(1).upper(), [m.group(2)]
            elif key and line.strip():
                buf.append(line.strip())
            elif re.match(r"\s*(DOSTATIC|USEPOT|KPPRA|SUBATOM|MAGATOM)\b", line):
                t = line.split(); tags[t[0].upper()] = " ".join(t[1:]) or "SET"
    if key: tags[key] = " ".join(buf).strip()
    return tags

def expand_magmom(v):
    out = []
    for tok in (v or "").split():
        if "*" in tok:
            try:
                n, x = tok.split("*"); out.extend([float(x)] * int(n))
            except ValueError: return []
        else:
            try: out.append(float(tok))
            except ValueError: return []
    return out

def poscar(path):
    """(cell, frac coords, per-site labels, per-site element)."""
    if not path or not os.path.exists(path): return None, None, None, None
    L = open(path, errors="replace").read().split("\n")
    try:
        sc = float(L[1].split()[0])
        cell = [[float(x) * sc for x in L[2 + k].split()[:3]] for k in range(3)]
        i = 5; tok = L[i].split(); syms = None
        if tok and not tok[0].lstrip("+-").isdigit():
            syms = tok; i += 1; tok = L[i].split()
        counts = [int(t) for t in tok]; n = sum(counts); i += 1
        if L[i].strip().lower().startswith("s"): i += 1
        cart = L[i].strip().lower().startswith(("c", "k")); i += 1
        pos, labels = [], []
        for j in range(n):
            p = L[i + j].split()
            pos.append([float(x) for x in p[:3]])
            labels.append(p[3] if len(p) > 3 else None)
    except (ValueError, IndexError):
        return None, None, None, None
    if cart:
        det = sum(cell[0][a] * (cell[1][(a+1)%3]*cell[2][(a+2)%3]
                                - cell[1][(a+2)%3]*cell[2][(a+1)%3])
                  for a in range(3))
        if abs(det) < 1e-12: return None, None, None, None
        inv = [[0.0]*3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                mm = [[cell[r][c] for c in range(3) if c != a]
                      for r in range(3) if r != b]
                inv[a][b] = ((-1)**(a+b))*(mm[0][0]*mm[1][1]-mm[0][1]*mm[1][0])/det
        pos = [[sum(p[k]*inv[k][c] for k in range(3)) for c in range(3)] for p in pos]
    els = []
    for k, lb in enumerate(labels):
        if lb:
            m = re.match(r"([A-Z][a-z]?)", lb); els.append(m.group(1) if m else "?")
        else: els.append(None)
    if not all(els) and syms:
        els = [s for s, c in zip(syms, counts) for _ in range(c)]
    return cell, pos, labels, els

def outcar_moments(path, tail=20_000_000):
    if not path: return None
    txt = _read(path, tail=tail)
    st = [m.end() for m in re.finditer(r"magnetization \(x\)", txt)]
    if not st: return None
    rows = []
    for line in txt[st[-1]:].split("\n"):
        t = line.split()
        if len(t) >= 3 and t[0].isdigit():
            try: rows.append(float(t[-1]))
            except ValueError: break
        elif rows: break
    return rows or None

def magmom_old_data(d, run=False):
    """MAGMOM_raw / eff_MAGMOM from ATAT's magmom_old (run it if asked)."""
    raw = first_of(d, ["MAGMOM_raw", "magmom_raw"])
    eff = first_of(d, ["eff_MAGMOM", "eff_magmom"])
    ran = ""
    if run and not (raw and eff):
        try:
            r = subprocess.run(["magmom_old"], cwd=d, capture_output=True,
                               text=True, timeout=180)
            ran = "ok" if r.returncode == 0 else f"rc={r.returncode}"
            if r.returncode != 0 and r.stderr:
                ran += ": " + r.stderr.strip().split("\n")[0][:60]
        except FileNotFoundError:
            ran = "magmom_old not on PATH"
        except Exception as e:
            ran = f"error {type(e).__name__}"
        raw = first_of(d, ["MAGMOM_raw", "magmom_raw"])
        eff = first_of(d, ["eff_MAGMOM", "eff_magmom"])
    def nums(p):
        if not p: return None
        v = re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?",
                       open(p, errors="replace").read())
        return [float(x.replace("D", "E").replace("d", "e")) for x in v] or None
    return nums(raw), nums(eff), ran

# ------------------------------------------------------------- correlations
def nn_list(cell, frac, tol=1.15):
    n = len(frac); dists = {}
    for a in range(n):
        for b in range(a + 1, n):
            best = None
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        df = [frac[b][k]-frac[a][k]+(sx, sy, sz)[k] for k in range(3)]
                        c = [sum(df[k]*cell[k][q] for k in range(3)) for q in range(3)]
                        d = math.sqrt(sum(x*x for x in c))
                        if best is None or d < best: best = d
            dists[(a, b)] = best
    if not dists: return [], 0.0
    dmin = min(dists.values())
    return [k for k, v in dists.items() if v <= dmin*tol], dmin

def ising_corr(pairs, sign):
    v = [sign[a]*sign[b] for a, b in pairs if sign[a] and sign[b]]
    return (sum(v)/len(v), len(v)) if v else (float("nan"), 0)

def null_stats(pairs, live, ndraw, seed=0):
    """Ideal-DLM null: balanced random +/- on the live sites."""
    if not pairs or len(live) < 2: return float("nan"), float("nan")
    lset = set(live); pr = [(a, b) for a, b in pairs if a in lset and b in lset]
    if not pr: return float("nan"), float("nan")
    rng = random.Random(seed); idx = {s: k for k, s in enumerate(live)}
    nl = len(live); vals = []
    for _ in range(ndraw):
        sg = [1]*(nl//2) + [-1]*(nl - nl//2); rng.shuffle(sg)
        vals.append(sum(sg[idx[a]]*sg[idx[b]] for a, b in pr)/len(pr))
    mu = sum(vals)/len(vals)
    sd = math.sqrt(sum((v-mu)**2 for v in vals)/len(vals))
    return mu, sd

# ------------------------------------------------------------------- audit
LATTICE_KEYS = [("FCC", "FCC_A1"), ("HCP", "HCP_A3"),
                ("BCC", "BCC_A2"), ("SIGMA", "SIGMA_D8B")]


def classify_lattice(path):
    """FCC / HCP / BCC / SIGMA from the path, case-insensitively.

    Every run in this project lives under FCC_A1, HCP_A3, BCC_A2 or
    SIGMA_D8B, so the rootword in the path is the authority. Checked from
    the deepest path element outward, so a nested dir wins over an ancestor.
    """
    parts = [q for q in os.path.normpath(path).split(os.sep) if q]
    for q in reversed(parts):
        u = q.upper()
        for key, full in LATTICE_KEYS:
            if key in u:
                return key, full
    return "", ""


SETTING_TAGS = ["ENCUT", "ENMAX", "PREC", "ALGO", "ISPIN", "NUPDOWN", "LORBIT",
                "ISMEAR", "SIGMA", "NELM", "NSW", "IBRION", "ISIF", "EDIFF",
                "EDIFFG", "AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG", "LREAL",
                "LMAXMIX", "NCORE", "KPAR", "ISYM", "LNONCOLLINEAR",
                "I_CONSTRAINED_M", "LAMBDA", "KPPRA", "USEPOT", "DOSTATIC"]

def audit_dir(d, args):
    r = {"dir": d, "dlm_ok": 0, "reason": "", "lorbit_ok": 0}
    r["lattice"], r["lattice_full"] = classify_lattice(os.path.abspath(d))
    inc = first_of(d, ["INCAR.relax", "INCAR", "INCAR.static"])
    wrap = first_of(d, ["vasp.wrap"])
    if not inc:
        for s_ in stage_dirs(d):
            inc = first_of(s_, ["INCAR", "INCAR.relax", "INCAR.static"])
            if inc: break
    if not wrap:
        for s_ in stage_dirs(d):
            wrap = first_of(s_, ["vasp.wrap"])
            if wrap: break
    if not inc and not wrap:
        return None
    tags = parse_tags(wrap); tags.update(parse_tags(inc))   # INCAR wins
    r["incar_src"] = os.path.basename(inc) if inc else ""
    r["has_wrap"] = 1 if wrap else 0
    for t in SETTING_TAGS:
        r[t.lower()] = tags.get(t, "")

    # ---------- DLM qualification, from MAGMOM signs
    mm = expand_magmom(tags.get("MAGMOM", ""))
    r["natoms_magmom"] = len(mm)
    if not mm:
        r["reason"] = "no MAGMOM (MAGATOM/DLM applied by ATAT, or missing)"
    else:
        pos_n = sum(1 for x in mm if x > 1e-8)
        neg_n = sum(1 for x in mm if x < -1e-8)
        r["n_up"], r["n_dn"] = pos_n, neg_n
        r["magmom_absmax"] = max(abs(x) for x in mm)
        r["magmom_absmean"] = sum(abs(x) for x in mm)/len(mm)
        if pos_n == 0 or neg_n == 0:
            r["reason"] = "MAGMOM single-signed -> FM/NM, not DLM"
        elif pos_n != neg_n:
            r["reason"] = f"MAGMOM unbalanced ({pos_n}+ / {neg_n}-)"
        else:
            r["dlm_ok"] = 1

    lorbit = tags.get("LORBIT", "")
    try: r["lorbit_ok"] = 1 if int(float(lorbit)) >= 10 else 0
    except ValueError: r["lorbit_ok"] = 0
    if not r["lorbit_ok"] and not r["reason"]:
        r["reason"] = "LORBIT<10: no per-ion moments, magmom_old cannot run"

    # ---------- structures: labels from the ORIGINAL, geometry from last stage
    lab_p = first_of(d, ["POSCAR.relax", "POSCAR"])
    cell0, frac0, labels, els = poscar(lab_p)
    stages = stage_dirs(d)
    geom_p = None
    for s in stages:
        geom_p = first_of(s, ["CONTCAR", "POSCAR"])
        if geom_p: break
    if not geom_p:
        geom_p = first_of(d, ["CONTCAR.static", "CONTCAR.relax", "CONTCAR",
                              "POSCAR.static"])
    r["geom_src"] = os.path.relpath(geom_p, d) if geom_p else ""
    cell, frac, _, _ = poscar(geom_p)
    if cell is None: cell, frac = cell0, frac0
    if els:
        good = [e for e in els if e]
        r["elements"] = ",".join(sorted(set(good)))
        r["natoms"] = len(els)
        r["n_species"] = len(set(good))
        r["composition"] = ",".join(
            f"{e}{good.count(e)/len(good):.4g}" for e in sorted(set(good)))
        for e in ("Co", "Cr", "Ni"):
            r["n_" + e] = good.count(e)
            r["x_" + e] = round(good.count(e) / len(good), 6) if good else ""
        r["system"] = "-".join(e for e in ("Co", "Cr", "Ni")
                               if good.count(e) > 0)
    # signed POSCAR labels are the corroborating DLM evidence
    if labels and all(labels):
        sg = [1 if re.search(r"\+\s*[\d.]+$", l) else
              (-1 if re.search(r"-\s*[\d.]+$", l) else 0) for l in labels]
        r["poscar_signed"] = 1 if any(sg) else 0

    # ---------- final moments
    mag_raw, mag_eff, ran = magmom_old_data(stages[0] if stages else d,
                                            args.run_magmom and r["lorbit_ok"])
    r["magmom_old"] = ran
    if mag_eff: r["eff_magmom_file"] = ";".join(f"{x:g}" for x in mag_eff[:8])
    final = None
    if mag_raw and r.get("natoms") and len(mag_raw) == r["natoms"]:
        final = mag_raw; r["moment_src"] = "MAGMOM_raw"
    if final is None:
        op = first_of(d, ["OUTCAR.static.gz", "OUTCAR.static", "OUTCAR.gz",
                          "OUTCAR", "OUTCAR.relax.gz", "OUTCAR.relax"])
        if not op and stages: op = first_of(stages[0], ["OUTCAR.gz", "OUTCAR"])
        final = outcar_moments(op, args.outcar_tail)
        r["moment_src"] = os.path.basename(op) if (op and final) else ""
    if final:
        mags = [abs(x) for x in final]
        r["rms_final"] = math.sqrt(sum(m*m for m in mags)/len(mags))
        r["mean_final"] = sum(mags)/len(mags)
        r["n_collapsed"] = sum(1 for m in mags if m < args.collapse_mag)
        r["net_per_atom"] = abs(sum(final))/len(final)
        if els and len(els) == len(mags):
            per = {}
            for e, m in zip(els, mags): per.setdefault(e, []).append(m)
            per = {k: sum(v)/len(v) for k, v in per.items()}
            r["beta_per_species"] = ";".join(f"{k}={v:.3f}" for k, v in sorted(per.items()))
            ln = sum((len([1 for e in els if e == k])/len(els))*math.log(v+1)
                     for k, v in per.items())
            r["beta_star"] = math.exp(ln) - 1
    if mm:
        r["rms_init"] = math.sqrt(sum(x*x for x in mm)/len(mm))

    # ---------- pair correlations: starting arrangement vs converged
    if cell and frac and not args.no_corr:
        pairs, dmin = nn_list(cell, frac, args.shell_tol)
        r["nn_pairs_all"] = len(pairs); r["d_nn"] = round(dmin, 3)
        if mm and len(mm) == len(frac):
            s0 = [1 if x > 1e-8 else (-1 if x < -1e-8 else 0) for x in mm]
            c0, n0 = ising_corr(pairs, s0)
            r["corr_init"], r["corr_init_pairs"] = c0, n0
        if final and len(final) == len(frac):
            live = [i for i, x in enumerate(final) if abs(x) >= args.collapse_mag]
            sf = [0]*len(final)
            for i in live: sf[i] = 1 if final[i] > 0 else -1
            cf, nf = ising_corr(pairs, sf)
            r["corr_final"], r["corr_final_pairs"] = cf, nf
            r["n_live"] = len(live)
            mu, sd = null_stats(pairs, live, args.ndraw)
            r["null_mean"], r["null_std"] = mu, sd
            if sd and sd == sd and sd > 1e-9 and cf == cf:
                r["corr_z"] = (cf - mu)/sd
    # ---------- energy
    ep = first_of(d, ["energy", "energy_end"])
    if ep:
        try: r["energy"] = float(open(ep).read().split()[0])
        except (ValueError, IndexError): pass
    return r

VASP_FILES = ("INCAR", "OSZICAR", "OUTCAR")


def looks_like_run(d, files, subdirs):
    """A genuine calculation directory.

    A stray vasp.wrap is NOT enough -- template wrap files sit at the top of
    a project tree, and treating the root as a run stops the walk dead. Need
    real VASP files here, or in a numbered stage dir belonging to this one
    (robustrelax puts them there before promoting them to the parent).
    """
    if any(f.startswith(VASP_FILES) for f in files):
        return True
    for sd in subdirs:
        if not re.fullmatch(r"\d+", sd):
            continue
        try:
            if any(f.startswith(VASP_FILES) for f in os.listdir(os.path.join(d, sd))):
                return True
        except OSError:
            pass
    return False

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--csv", default="dlm_audit.csv")
    ap.add_argument("--run-magmom", action="store_true",
                    help="invoke magmom_old where MAGMOM_raw/eff_MAGMOM are absent")
    ap.add_argument("--elements", default="Co,Cr,Ni",
                    help="keep runs whose species are a subset of this list "
                         "('' = keep everything)")
    ap.add_argument("--collapse-mag", type=float, default=0.20)
    ap.add_argument("--shell-tol", type=float, default=1.15)
    ap.add_argument("--ndraw", type=int, default=400)
    ap.add_argument("--outcar-tail", type=int, default=20_000_000)
    ap.add_argument("--no-corr", action="store_true",
                    help="skip pair correlations (much faster)")
    ap.add_argument("--include-sweeps", action="store_true")
    ap.add_argument("--max-depth", type=int, default=0,
                    help="stop descending below this depth (0 = no limit)")
    ap.add_argument("--max-atoms", type=int, default=80,
                    help="skip correlation maths on cells larger than this")
    args = ap.parse_args()

    # vol_<n> are ATAT volume-scan directories (E-V curves / phonons), not
    # production DLM runs -- excluded from the audit entirely, per instruction.
    VOL_RE = re.compile(r"^vol_\d+$", re.IGNORECASE)
    SKIP = ("sweep", "convergence", ".git", "__pycache__", "/venv", "site-packages")
    runs = []
    def _werr(e):
        print(f"  (skipping {getattr(e, 'filename', '?')}: {e.strerror})",
              file=sys.stderr)
    nseen = 0
    nvol = 0
    for dp, dn, fn in os.walk(args.root, onerror=_werr, followlinks=False):
        nseen += 1
        if nseen % 2000 == 0:
            print(f"  ...scanned {nseen} dirs, {len(runs)} runs so far",
                  file=sys.stderr)
        low = dp.lower().replace(os.sep, "/")
        if not args.include_sweeps and any(s in low for s in SKIP):
            dn[:] = []; continue
        if args.max_depth and dp[len(args.root):].count(os.sep) > args.max_depth:
            dn[:] = []; continue
        if any(VOL_RE.match(q) for q in dp.split(os.sep)):
            nvol += 1
            dn[:] = []                 # volume scans: skip the whole subtree
            continue
        dn[:] = [x for x in dn if not VOL_RE.match(x)]
        if looks_like_run(dp, fn, dn):
            runs.append(dp)
            # NOTHING is pruned here. Numbered stage dirs are still walked and
            # still recorded; they are simply LABELLED role="stage" below and
            # left out of the summary statistics, so the audit hides nothing
            # while the per-SQS counts stay honest.
    runs = sorted(set(runs))
    runset = set(os.path.abspath(x) for x in runs)

    def role_of(d):
        ad = os.path.abspath(d)
        if re.fullmatch(r"\d+", os.path.basename(ad)) and \
           os.path.dirname(ad) in runset:
            return "stage"
        return "sqs"

    print(f"found {len(runs)} candidate run directories under {args.root}"
          f"   ({sum(1 for d in runs if role_of(d)=='sqs')} SQS, "
          f"{sum(1 for d in runs if role_of(d)=='stage')} stage subdirs"
          + (f"; {nvol} vol_* dirs skipped)" if nvol else ")"))

    keep = [e.strip() for e in args.elements.split(",") if e.strip()]
    rows = []
    for i, d in enumerate(runs, 1):
        if i % 25 == 0: print(f"  ...{i}/{len(runs)}", file=sys.stderr)
        try: r = audit_dir(d, args)
        except Exception as e:
            r = {"dir": d, "dlm_ok": 0, "reason": f"audit error {type(e).__name__}: {e}"}
        if not r: continue
        r["role"] = role_of(d)
        r["parent_run"] = (os.path.dirname(os.path.abspath(d))
                           if r["role"] == "stage" else "")
        if keep and r.get("elements"):
            if not set(r["elements"].split(",")).issubset(set(keep)):
                continue
        rows.append(r)

    cols = ["dir", "role", "parent_run", "lattice", "lattice_full", "system", "composition",
            "x_Co", "x_Cr", "x_Ni", "n_Co", "n_Cr", "n_Ni", "n_species",
            "dlm_ok", "reason", "lorbit_ok", "incar_src", "has_wrap",
            "geom_src", "moment_src", "magmom_old", "elements",
            "natoms", "n_up", "n_dn", "poscar_signed", "magmom_absmax",
            "magmom_absmean", "rms_init", "rms_final", "mean_final",
            "n_collapsed", "n_live", "net_per_atom", "beta_star",
            "beta_per_species", "eff_magmom_file", "corr_init",
            "corr_init_pairs", "corr_final", "corr_final_pairs", "null_mean",
            "null_std", "corr_z", "nn_pairs_all", "d_nn", "energy"] + \
           [t.lower() for t in SETTING_TAGS]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)

    primary = [r for r in rows if r.get("role") != "stage"]
    dlm = [r for r in primary if r["dlm_ok"]]
    print(f"\n{len(rows)} rows written "
          f"({len(primary)} SQS + {len(rows)-len(primary)} stage) "
          f"({args.elements or 'all elements'})")
    print("  summary below counts SQS rows only; stage rows are in the CSV")
    print(f"  qualify as DLM (balanced mixed-sign MAGMOM): {len(dlm)}")
    print(f"  LORBIT>=10 (per-ion moments available):      "
          f"{sum(1 for r in primary if r['lorbit_ok'])}")
    why = {}
    for r in primary:
        if not r["dlm_ok"]:
            why[r["reason"] or "?"] = why.get(r["reason"] or "?", 0) + 1
    if why:
        print("  disqualified:")
        for k, v in sorted(why.items(), key=lambda x: -x[1]):
            print(f"     {v:4d}  {k}")
    if dlm:
        print("\n  DLM runs by lattice:")
        for key, _full in LATTICE_KEYS + [("", "unclassified")]:
            sub = [r for r in dlm if r.get("lattice", "") == key]
            if not sub: continue
            with_m = [r for r in sub if r.get("rms_final") is not None]
            kept = [r for r in with_m
                    if r.get("rms_init") and r["rms_final"] > 0.5*r["rms_init"]]
            print(f"     {(key or 'unclassified'):12s} n={len(sub):4d}   "
                  f"moments read {len(with_m):4d}   kept>half {len(kept):4d}")
        print("\n  DLM runs by subsystem:")
        sysc = {}
        for r in dlm:
            sysc.setdefault(r.get("system", "?"), []).append(r)
        for k, v in sorted(sysc.items(), key=lambda x: -len(x[1])):
            with_m = [r for r in v if r.get("rms_final") is not None]
            kept = [r for r in with_m
                    if r.get("rms_init") and r["rms_final"] > 0.5*r["rms_init"]]
            print(f"     {k:12s} n={len(v):4d}   moments read {len(with_m):4d}"
                  f"   kept>half {len(kept):4d}")

    surv = [r for r in dlm if r.get("rms_final") is not None
            and r.get("rms_init")]
    if surv:
        good = [r for r in surv if r["rms_final"] > 0.5*r["rms_init"]]
        print(f"\n  of {len(surv)} DLM runs with moments: "
              f"{len(good)} kept >half their starting moment")
    print(f"\nCSV -> {args.csv}")

if __name__ == "__main__":
    main()
