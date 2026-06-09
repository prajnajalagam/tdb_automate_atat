#!/usr/bin/env python3
"""
Fetch open-licensed TDB files into corpus/.

Reads open_sources.yaml, prints what's available, and (with
--download) pulls each `sources:` entry into corpus/<name>.tdb plus
a sidecar <name>.citation.yaml. Entries under
`references_no_auto_download:` are printed for manual fetch only —
the script never downloads them, regardless of flags.

Defaults to DRY-RUN so you see exactly what would happen before any
HTTP traffic. Skips files that already exist (idempotent).

Usage
-----
    python3 tdbdb_fetch_open.py            # dry-run: list only
    python3 tdbdb_fetch_open.py --download # actually fetch
    python3 tdbdb_fetch_open.py --download --force  # re-fetch even if present
    python3 tdbdb_fetch_open.py --sources /path/to/alt.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML required. `pip install pyyaml`")


# A safelist of licenses we'll auto-download. Anything else is
# manual-fetch-only, regardless of what the yaml says.
AUTO_DOWNLOAD_LICENSES = {
    "MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0",
    "CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "public-domain",
}

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "open_sources.yaml"
DEFAULT_CORPUS = HERE / "corpus"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        sys.exit(f"ERROR: {path} top-level must be a mapping")
    return data


def _write_sidecar(corpus_dir: Path, name: str, entry: Dict[str, Any]) -> None:
    """Write <name>.citation.yaml next to the downloaded TDB."""
    sidecar = corpus_dir / f"{name}.citation.yaml"
    payload = {
        "title": entry.get("citation", ""),
        "authors": [],            # not parsed from short citation
        "doi": entry.get("doi"),
        "source_url": entry.get("url"),
        "license": entry.get("license"),
        "system": entry.get("system", []),
        "notes": entry.get("notes", ""),
        "fetched_by": "tdbdb_fetch_open.py",
    }
    with sidecar.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _download(url: str, dest: Path) -> Optional[str]:
    """Download URL to dest. Returns None on success, error string on failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "tdb_corpus/0.1 (corpus-builder; +github)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        # Sanity: a TDB file should contain at least one ELEMENT card.
        head = data[:4096].decode("latin-1", errors="replace").upper()
        if "ELEMENT" not in head:
            return ("downloaded but content doesn't look like a TDB "
                    "(no ELEMENT card in first 4KB)")
        with dest.open("wb") as f:
            f.write(data)
        return None
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error: {e.reason}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch open-licensed TDBs")
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES),
                    help="open_sources.yaml path")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help="corpus output directory (TDBs land here)")
    ap.add_argument("--download", action="store_true",
                    help="Actually download; default is dry-run (list only)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the file already exists")
    args = ap.parse_args()

    sources_path = Path(args.sources).resolve()
    corpus_dir = Path(args.corpus).resolve()
    if not sources_path.is_file():
        sys.exit(f"ERROR: sources file not found: {sources_path}")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    data = _load_yaml(sources_path)
    autodl: List[Dict[str, Any]] = data.get("sources", []) or []
    manual: List[Dict[str, Any]] = data.get("references_no_auto_download", []) or []

    print(f"\n  Sources file : {sources_path}")
    print(f"  Corpus dir   : {corpus_dir}")
    print(f"  Mode         : {'DOWNLOAD' if args.download else 'dry-run (list only)'}")
    print(f"  Auto-download: {len(autodl)} entry/entries (license-gated)")
    print(f"  Manual-fetch : {len(manual)} entry/entries (citation only)\n")

    # ── Auto-downloadable ──────────────────────────────────────────
    print("== Auto-downloadable (license-checked) ==")
    n_ok = n_skip = n_fail = n_blocked = 0
    for entry in autodl:
        name = entry.get("name") or "<no-name>"
        url = entry.get("url")
        lic = entry.get("license", "")
        sys_ = ",".join(entry.get("system", []))
        print(f"\n  [{name}]  system={sys_}  license={lic}")
        print(f"    url : {url}")
        if lic not in AUTO_DOWNLOAD_LICENSES:
            print(f"    SKIP (license {lic!r} not in auto-download safelist)")
            n_blocked += 1
            continue
        if not url:
            print(f"    SKIP (no url)")
            n_blocked += 1
            continue
        dest = corpus_dir / f"{name}.tdb"
        if dest.exists() and not args.force:
            print(f"    SKIP (already present: {dest})")
            n_skip += 1
            continue
        if not args.download:
            print(f"    DRY-RUN (would download → {dest})")
            continue
        err = _download(url, dest)
        if err is None:
            print(f"    OK    → {dest} ({dest.stat().st_size} bytes)")
            _write_sidecar(corpus_dir, name, entry)
            n_ok += 1
        else:
            print(f"    FAIL  {err}")
            # Clean up partial file
            if dest.exists():
                dest.unlink()
            n_fail += 1

    # ── Manual-fetch references ───────────────────────────────────
    print("\n== Manual-fetch references (cite + fetch via library) ==")
    for entry in manual:
        sys_ = ",".join(entry.get("system", []))
        print(f"\n  system={sys_}  license={entry.get('license', '?')}")
        print(f"    citation : {entry.get('citation','?')}")
        if entry.get("doi"):
            print(f"    doi      : {entry['doi']}")
        if entry.get("notes"):
            print(f"    notes    : {entry['notes']}")

    print("\n  ── Summary ──")
    print(f"  downloaded   : {n_ok}")
    print(f"  already_there: {n_skip}")
    print(f"  blocked      : {n_blocked} (license or url missing)")
    print(f"  failed       : {n_fail}")
    if not args.download:
        print(f"  (dry-run — nothing was written; re-run with --download)")
    print()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
