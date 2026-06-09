#!/usr/bin/env python3
"""
Walk corpus/, validate every TDB by loading it through pycalphad, attach
any sidecar citation metadata, and write a single _manifest.yaml that
records the corpus composition. That manifest is what downstream
ensemble-scoring code reads — the .TDB bytes never need to be re-parsed
or redistributed by anything else.

Usage
-----
    python3 corpus_ingest.py
        # scans ./corpus/, writes ./corpus/_manifest.yaml

    python3 corpus_ingest.py --corpus /some/other/path
    python3 corpus_ingest.py --strict   # exit non-zero on any TDB that fails to load
    python3 corpus_ingest.py --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pycalphad import Database
    _HAS_PYCALPHAD = True
except ImportError:
    _HAS_PYCALPHAD = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# Files we treat as TDBs.
TDB_SUFFIXES = (".tdb", ".TDB")
# Files we DON'T treat as TDBs even if their suffix matches.
SIDECAR_SUFFIX = ".citation.yaml"


def _find_tdb_files(corpus_dir: Path) -> List[Path]:
    """Return every .tdb / .TDB under corpus_dir, sorted, sidecars excluded."""
    out: List[Path] = []
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix in TDB_SUFFIXES:
            out.append(p)
    return out


def _load_sidecar(tdb_path: Path) -> Dict[str, Any]:
    """Read <basename>.citation.yaml if present, else return {}."""
    sidecar = tdb_path.with_name(tdb_path.stem + SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return {}
    if not _HAS_YAML:
        return {"_sidecar_present_but_pyyaml_missing": str(sidecar)}
    try:
        with sidecar.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {"_sidecar_invalid": "expected a YAML mapping at top level"}
        return data
    except Exception as exc:
        return {"_sidecar_parse_error": str(exc)}


def _describe_phase(db, ph: str) -> str:
    """Return 'mult(species)/mult(species)/...' for one phase, or '<err>'."""
    try:
        ph_obj = db.phases[ph]
        parts = []
        for mult, cons in zip(ph_obj.sublattices, ph_obj.constituents):
            sp = ",".join(sorted(str(s) for s in cons))
            parts.append(f"{mult}({sp})")
        return " / ".join(parts)
    except Exception as exc:
        return f"<error: {exc}>"


def _ingest_one(tdb_path: Path) -> Dict[str, Any]:
    """Parse one TDB and return the manifest entry."""
    entry: Dict[str, Any] = {
        "filename": tdb_path.name,
        "relative_path": None,           # set by caller
        "size_bytes": tdb_path.stat().st_size,
        "citation": _load_sidecar(tdb_path),
    }

    if not _HAS_PYCALPHAD:
        entry["status"] = "not-parsed (pycalphad unavailable)"
        return entry

    try:
        db = Database(str(tdb_path))
    except Exception as exc:
        entry["status"] = "load-failed"
        entry["error"] = str(exc)
        return entry

    entry["status"] = "ok"
    entry["elements"] = sorted(str(e) for e in db.elements)
    entry["phases"] = {ph: _describe_phase(db, ph) for ph in sorted(db.phases.keys())}
    return entry


def _emit_yaml(manifest: Dict[str, Any], out_path: Path) -> None:
    """Write manifest as YAML if PyYAML present, otherwise JSON as a fallback."""
    if _HAS_YAML:
        with out_path.open("w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    else:
        json_path = out_path.with_suffix(".json")
        with json_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  (PyYAML missing — wrote JSON manifest at {json_path} instead)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Index a local TDB corpus")
    ap.add_argument("--corpus", default=str(Path(__file__).parent / "corpus"),
                    help="Path to the corpus directory (default: ./corpus)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any TDB fails to load")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-file progress output")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus).resolve()
    if not corpus_dir.is_dir():
        print(f"ERROR: {corpus_dir} is not a directory", file=sys.stderr)
        return 2

    tdbs = _find_tdb_files(corpus_dir)
    if not args.quiet:
        print(f"  Scanning : {corpus_dir}")
        print(f"  Found    : {len(tdbs)} TDB file(s)")
        if not _HAS_PYCALPHAD:
            print(f"  WARNING  : pycalphad not importable; manifest will only "
                  f"record filename + size + sidecar metadata.")
        if not _HAS_YAML:
            print(f"  WARNING  : PyYAML not importable; manifest will be JSON.")

    entries: List[Dict[str, Any]] = []
    n_ok = n_fail = 0
    for tdb_path in tdbs:
        e = _ingest_one(tdb_path)
        e["relative_path"] = str(tdb_path.relative_to(corpus_dir))
        entries.append(e)
        status = e.get("status", "?")
        if status == "ok":
            n_ok += 1
            if not args.quiet:
                n_ph = len(e.get("phases", {}))
                els = ",".join(e.get("elements", []))
                print(f"    OK    {tdb_path.name:40s}  {n_ph} phase(s)  [{els}]")
        else:
            n_fail += 1
            if not args.quiet:
                print(f"    FAIL  {tdb_path.name:40s}  {status}  "
                      f"{e.get('error','')}")

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "n_files": len(entries),
        "n_loaded_ok": n_ok,
        "n_failed": n_fail,
        "pycalphad_available": _HAS_PYCALPHAD,
        "entries": entries,
    }

    out_path = corpus_dir / "_manifest.yaml"
    _emit_yaml(manifest, out_path)

    if not args.quiet:
        print(f"  Wrote    : {out_path}")
        print(f"  Summary  : {n_ok} ok, {n_fail} failed")

    if args.strict and n_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
