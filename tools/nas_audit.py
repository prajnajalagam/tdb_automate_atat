#!/usr/bin/env python3
"""
nas_audit.py — inventory and triage of the TDB pipeline's NAS trees.

Pipeline-aware extension of the generic single-pass auditor (2026-07-22
upload): same single bottom-up walk and reports, plus knowledge of the
upstream generator's directory shapes so its statuses and cleanup
proposals are trustworthy for THESE trees:

  * convergence sweep points (…/convergence/encut_*|kppra_*): complete
    iff `energy` exists — cached probe data, cheap and PRECIOUS
    (restart fast-forward); never proposed for deletion.
  * robustrelax SQS dirs: judged by the FULL-branch state
    (energy_sup+energy | energy_end + 01/cstr_relax.out + energy) and
    the 'infdet terminated normally' marker — NOT by OUTCAR footers,
    which only describe the last VASP substep.
  * fitfc perturbation dirs (vol_*/p*): complete iff force.out exists.
  * PROTECTED files are never proposed for deletion, whatever the dir
    status: OUTCAR.relax(.gz) is the ML-potential training corpus
    (colleague request 2026-07-16), and str.out/str_relax.out/energy*/
    svib_ht/hessian/force.out/checkrelax.out/infdet.log/manifests are
    the pipeline's actual results. Space is reclaimed from regenerable
    VASP bulk (WAVECAR/CHGCAR/…) and from whole bad runs instead.
  * top-level BUCKET report: per child of the scan root, size/inodes +
    a recommendation (stale timestamped e2e_*/smoke_* test roots,
    *failed* postmortem trees, upstream production trees).

READ-ONLY: writes CSVs + a fully commented-out nas_cleanup_review.sh.
Also proposes `gzip -9` (not rm) for uncompressed OUTCAR.relax — halves
the corpus without losing a single training frame (all readers here
stream .gz transparently).

Usage:
    python3 nas_audit.py /nobackupp27/pjalagam
    python3 nas_audit.py /nobackupp27/pjalagam --min-mb 50 --quick
"""
import os
import csv
import gzip
import time
import hashlib
import argparse
from collections import defaultdict

REGENERABLE = ["WAVECAR", "CHGCAR", "CHG", "PROCAR", "vaspout.h5",
               "WAVEDER", "TMPCAR", "LOCPOT", "ELFCAR", "AECCAR0",
               "AECCAR1", "AECCAR2"]
MARK_DONE = ["General timing and accounting",
             "reached required accuracy"]

# Never propose deleting these, regardless of run status. OUTCAR.relax
# is the ML training corpus; the rest are pipeline results/records.
PROTECT = ("OUTCAR.relax", "OUTCAR.relax.gz", "str.out", "str_relax.out",
           "energy", "energy_end", "energy_sup", "svib_ht", "svib_lt",
           "hessian.out", "force.out", "checkrelax.out", "infdet.log",
           "unstable_modes.log", "upstream_manifest.json",
           "upstream_live.log", "fit_energy.out", "svib_adaptive.json",
           "refinement_plan.json", "eci.out")


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.1f}{u}"
        n /= 1024


def open_maybe_gz(p):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", errors="ignore")
    return open(p, "r", errors="ignore")


def tail(path, nbytes=8000):
    try:
        if str(path).endswith(".gz"):
            with open_maybe_gz(path) as f:
                return f.read()[-nbytes:]
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - nbytes))
            return f.read().decode(errors="ignore")
    except Exception:
        return ""


def pick(d, n):
    for cand in (os.path.join(d, n), os.path.join(d, n + ".gz")):
        if os.path.isfile(cand):
            return cand
    return None


def classify(d, names):
    """Status of one directory, pipeline layouts first."""
    base = os.path.basename(d)
    parent = os.path.basename(os.path.dirname(d))

    # -- pipeline shapes ---------------------------------------------------
    if parent == "convergence" or base.startswith(("encut_", "kppra_")):
        return (("SWEEP_POINT_DONE", "cached probe point (keep)")
                if "energy" in names else
                ("SWEEP_POINT_UNFINISHED", "no energy — rerun regenerates"))
    if base.startswith("p") and (parent.startswith("vol_")
                                 or parent == "smsqs"):
        return (("FORCE_RUN_DONE", "phonon perturbation static")
                if "force.out" in names or "force.out" in names else
                ("FORCE_RUN_UNFINISHED", "no force.out"))
    if "str.out" in names and ("energy_sup" in names
                               or "energy_end" in names):
        stable = "energy_sup" in names and "energy" in names
        unstable = ("energy_end" in names and "energy" in names
                    and os.path.isfile(os.path.join(d, "01",
                                                    "cstr_relax.out")))
        if stable or unstable:
            return "ROBUSTRELAX_COMPLETE", ("stable branch" if stable
                                            else "infdet branch")
        return ("ROBUSTRELAX_PARTIAL",
                "branch not finished — rerun completes it (post-fix "
                "code) or wipe relax products to restart clean")

    # Pipeline results present -> NEVER a bulk-delete candidate, even
    # when the last VASP substep's OUTCAR is truncated (the 2026-07
    # kill-era trees are full of orphan-written logs inside dirs whose
    # results are fine). Validity is the manifest's / infdet markers'
    # job, not this auditor's.
    if "energy" in names and ("str_relax.out" in names
                              or "svib_ht" in names):
        return ("RESULTS_PRESENT",
                "pipeline results on disk — judge via manifest/"
                "infdet markers, clean per recovery notes, don't rm -rf")

    # -- generic VASP fallback ---------------------------------------------
    if "OUTCAR" not in names:
        if "INCAR" in names:
            return "NEVER_RAN", "INCAR present, no OUTCAR"
        return "NOT_A_RUN", ""
    txt = tail(pick(d, "OUTCAR"))
    done = any(m in txt for m in MARK_DONE)
    accurate = "reached required accuracy" in txt
    if done and accurate:
        return "COMPLETE", "converged"
    if done:
        return "FINISHED_UNCONVERGED", "timing block but no accuracy line"
    cc = pick(d, "CONTCAR")
    if cc and os.path.getsize(cc) > 0:
        return "INCOMPLETE", "killed/crashed, CONTCAR exists (restartable)"
    return "FAILED", "no timing block, no usable CONTCAR"


def bucket_advice(name):
    """Recommendation for a top-level child of the scan root."""
    low = name.lower()
    if "failed" in low:
        return ("DELETE after optional Lou archive — postmortem tree, "
                "diagnosis is closed and recorded in the repo")
    if low.startswith(("e2e_", "smoke_")):
        return ("DELETE if not the most recent of its kind — timestamped "
                "test root, fully reproducible in <2h")
    if low.endswith("_upstream"):
        return ("KEEP — production tree; clean inside it (regenerable "
                "VASP bulk, superseded bad SQS) rather than wholesale")
    if low == "ml_tables":
        return "KEEP — colleague deliverable"
    return "review"


def fingerprint(d):
    h = hashlib.sha1()
    got = False
    for name in ("POSCAR", "INCAR"):
        p = pick(d, name)
        if p:
            try:
                with open_maybe_gz(p) as f:
                    h.update(f.read().encode(errors="ignore"))
                got = True
            except Exception:
                pass
    return h.hexdigest() if got else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--min-mb", type=float, default=100.0)
    ap.add_argument("--quick", action="store_true",
                    help="skip POSCAR/INCAR fingerprinting (no dupe report)")
    ap.add_argument("--progress", type=int, default=2000)
    args = ap.parse_args()
    root = os.path.realpath(args.root)

    t0 = time.time()
    print(f"Scanning {root} (read-only, single pass)", flush=True)

    subtree_bytes = defaultdict(int)
    subtree_inodes = defaultdict(int)
    bucket_totals = {}          # root's direct children (captured at pop)
    run_dirs = {}
    bigfiles = []
    gzip_candidates = []
    ndirs = 0

    for dirpath, dirnames, filenames in os.walk(root, topdown=False,
                                                onerror=lambda e: None):
        ndirs += 1
        if args.progress and ndirs % args.progress == 0:
            print(f"  ...{ndirs} dirs, {time.time()-t0:.0f}s", flush=True)

        own_b = own_i = 0
        stripped = set()
        for fn in filenames:
            stripped.add(fn[:-3] if fn.endswith(".gz") else fn)
            try:
                own_b += os.path.getsize(os.path.join(dirpath, fn))
                own_i += 1
            except OSError:
                pass
            if fn == "OUTCAR.relax":              # uncompressed corpus
                gzip_candidates.append(os.path.join(dirpath, fn))

        tot_b, tot_i = own_b, own_i
        for sd in dirnames:
            p = os.path.join(dirpath, sd)
            b = subtree_bytes.pop(p, 0)
            i = subtree_inodes.pop(p, 0)
            if dirpath == root:      # keep totals for the bucket report
                bucket_totals[p] = (b, i)
            tot_b += b
            tot_i += i
        subtree_bytes[dirpath] = tot_b
        subtree_inodes[dirpath] = tot_i

        if not ({"OUTCAR", "INCAR", "energy", "force.out",
                 "energy_end", "energy_sup"} & stripped):
            continue
        status, note = classify(dirpath, stripped)
        if status == "NOT_A_RUN":
            continue

        regen = 0
        for name in REGENERABLE:
            p = pick(dirpath, name)
            if p and os.path.basename(p) not in PROTECT:
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                regen += sz
                if sz >= args.min_mb * 1024 * 1024:
                    bigfiles.append({"path": p, "bytes": sz,
                                     "human": human(sz), "status": status})
        run_dirs[dirpath] = (status, note, regen, tot_b, tot_i)

    runs, by_fp = [], defaultdict(list)
    for d, (status, note, regen, nb, ni) in run_dirs.items():
        fp = None if args.quick else fingerprint(d)
        if fp:
            by_fp[fp].append(d)
        runs.append({"dir": d, "status": status, "note": note,
                     "bytes": nb, "human": human(nb), "inodes": ni,
                     "regenerable_bytes": regen,
                     "regenerable_human": human(regen),
                     "fingerprint": fp})

    runs.sort(key=lambda r: -r["bytes"])
    bigfiles.sort(key=lambda b: -b["bytes"])

    with open("nas_audit_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()) if runs
                           else ["dir"])
        w.writeheader()
        w.writerows(runs)

    dupes = {k: v for k, v in by_fp.items() if len(v) > 1}
    with open("nas_audit_dupes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "n_copies", "directories"])
        for k, v in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
            w.writerow([k[:12], len(v), " | ".join(v)])

    with open("nas_audit_bigfiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "human",
                                          "status"])
        w.writeheader()
        w.writerows(bigfiles)

    # -- top-level bucket report -------------------------------------------
    print(f"\n{'top-level bucket':40s} {'size':>10s} {'inodes':>9s}  advice")
    print("-" * 100)
    try:
        kids = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        kids = []
    for e in kids:
        if e.is_dir(follow_symlinks=False):
            p = os.path.join(root, e.name)
            b, i = bucket_totals.get(p, (0, 0))
            print(f"{e.name:40s} {human(b):>10s} "
                  f"{i:9d}  {bucket_advice(e.name)}")

    tot = defaultdict(lambda: [0, 0, 0])
    for r in runs:
        s = tot[r["status"]]
        s[0] += 1
        s[1] += r["bytes"]
        s[2] += r["inodes"]

    print(f"\n{'status':26s} {'runs':>6s} {'size':>10s} {'inodes':>10s}")
    print("-" * 56)
    for k in sorted(tot, key=lambda k: -tot[k][1]):
        c, b, i = tot[k]
        print(f"{k:26s} {c:6d} {human(b):>10s} {i:10d}")
    print("-" * 56)
    print(f"{'TOTAL':26s} {len(runs):6d} "
          f"{human(sum(v[1] for v in tot.values())):>10s} "
          f"{sum(v[2] for v in tot.values()):10d}")
    print(f"\nRegenerable (WAVECAR/CHGCAR/etc): "
          f"{human(sum(r['regenerable_bytes'] for r in runs))}")
    print(f"Uncompressed OUTCAR.relax files : {len(gzip_candidates)} "
          f"(gzip proposals in the review script)")
    print(f"Duplicate fingerprint groups: {len(dupes)}"
          + (" (skipped: --quick)" if args.quick else ""))
    print(f"Directories walked: {ndirs} in {time.time()-t0:.0f}s")

    with open("nas_cleanup_review.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# REVIEW EVERY LINE. Nothing runs until you uncomment.\n")
        f.write("# nobackup is NOT backed up: shiftc --create-tar to Lou\n")
        f.write("# first for anything you might ever want again.\n\n")
        f.write("# ---- 1. Regenerable VASP bulk (safe in ANY status; "
                "results are protected files) ----\n")
        for b in bigfiles:
            f.write(f"# rm '{b['path']}'   # {b['human']} [{b['status']}]\n")
        f.write("\n# ---- 2. Compress the ML corpus (KEEPS every frame; "
                "readers stream .gz) ----\n")
        for p in gzip_candidates:
            f.write(f"# gzip -9 '{p}'\n")
        f.write("\n# ---- 3. FAILED / NEVER_RAN generic runs ----\n")
        f.write("# NOTE: dirs under an *_upstream tree that the pipeline "
                "will rerun should be CLEANED (see repo recovery "
                "instructions), not rm -rf'd wholesale.\n")
        for r in runs:
            if r["status"] in ("FAILED", "NEVER_RAN"):
                f.write(f"# rm -rf '{r['dir']}'   # {r['human']}\n")
        f.write("\n# ---- 4. Duplicate groups (identical POSCAR+INCAR) ----\n")
        f.write("# Usually a TEST tree copy of a production run: keep the\n")
        f.write("# production (*_upstream) copy, delete the e2e/smoke one.\n")
        for k, v in dupes.items():
            f.write(f"#   group {k[:12]} ({len(v)} copies):\n")
            for p in v:
                f.write(f"#     {p}\n")
            f.write("#   (keep the production copy, rm -rf the others)\n\n")

    print("\nWrote: nas_audit_runs.csv, nas_audit_dupes.csv,")
    print("       nas_audit_bigfiles.csv, nas_cleanup_review.sh")
    print("Nothing was deleted.")


if __name__ == "__main__":
    main()
