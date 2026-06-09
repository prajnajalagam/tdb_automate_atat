#!/usr/bin/env python3
"""
TDBDB API client — STUB.

Brown's TDBDB at https://avdwgroup.engin.brown.edu exposes an HTTP API
documented in van de Walle et al., Calphad 61 (2018) 173-178 and on
https://avdwgroup.engin.brown.edu/help.html. The site is blocked from
this repository's sandbox (Brown WAF / IP allowlist), so this module
ships as a stub: the surface is defined, the implementation waits for
the user to drop the API spec into ../api_spec/help.html so we can
fill in the URL templates, parameter names, and response parser.

What this stub *does* do today
------------------------------
- Provides the public function the rest of the corpus tooling will
  call (`query_tdbdb(elements, phases=None)` -> List[CitationRecord]).
- Validates inputs and returns an explicit NotImplementedError pointing
  at the missing pieces, so accidental imports fail loudly with
  guidance.
- Documents the expected CitationRecord schema in one place.

Once api_spec/help.html lands, fill in:
    _BASE_URL, _ENDPOINT_PATH, _PARAM_NAMES, _RESPONSE_FORMAT
    _parse_response()
and remove the NotImplementedError.

Usage (after wiring)
--------------------
    python3 tdbdb_query.py --elements Co,Cr,Ni
    python3 tdbdb_query.py --elements Fe,Cr --phases SIGMA,FCC_A1 \\
                           --out coCrNi_citations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# ─── Configuration to fill in once we have help.html ───────────────
_BASE_URL: str = "https://avdwgroup.engin.brown.edu"
_ENDPOINT_PATH: Optional[str] = None     # e.g. "/query" or "/tdbdb.cgi" — TBD
_PARAM_NAMES = {
    # API's name -> our internal name
    # e.g. "elem": "elements", "phase": "phases" — TBD
}
_RESPONSE_FORMAT: Optional[str] = None   # "json" | "xml" | "html" — TBD


# ─── Public record type ────────────────────────────────────────────

@dataclass
class CitationRecord:
    """One TDBDB hit. The corpus-builder consumes this shape."""
    system: List[str]                     # element list, uppercase
    phases: List[str] = field(default_factory=list)
    citation: str = ""                    # human-readable, e.g. "Andersson 1987"
    doi: Optional[str] = None
    source_url: Optional[str] = None      # publisher / supplementary page
    tdb_url: Optional[str] = None         # direct .tdb if the API gives one
    license: str = "unknown"              # "CC-BY-4.0", "publisher-paywalled", ...
    extra: dict = field(default_factory=dict)


# ─── Public API ────────────────────────────────────────────────────

def query_tdbdb(
    elements: List[str],
    phases: Optional[List[str]] = None,
    base_url: str = _BASE_URL,
    timeout: float = 30.0,
) -> List[CitationRecord]:
    """
    Query TDBDB for assessments covering an element set (and optionally
    a phase set). Returns a list of CitationRecord; downstream
    `tdbdb_fetch_open` decides which are auto-downloadable.

    Currently NOT IMPLEMENTED — see module docstring.
    """
    elements = [e.strip().upper() for e in elements if e.strip()]
    if not elements:
        raise ValueError("query_tdbdb: at least one element required")
    if phases is not None:
        phases = [p.strip() for p in phases if p.strip()]

    if _ENDPOINT_PATH is None or _RESPONSE_FORMAT is None:
        raise NotImplementedError(
            "TDBDB API client is stubbed.\n"
            "  To wire it up: drop the API spec into\n"
            "  ../api_spec/help.html (or paste into help.txt),\n"
            "  then fill in _ENDPOINT_PATH, _PARAM_NAMES, _RESPONSE_FORMAT\n"
            "  and _parse_response() in this file."
        )

    # When implemented:
    #   url = _build_url(base_url, elements, phases)
    #   raw = _http_get(url, timeout=timeout)
    #   return _parse_response(raw)
    raise NotImplementedError  # pragma: no cover  (defensive)


# ─── Internal scaffolding (fill in when spec lands) ────────────────

def _build_url(base_url: str, elements: List[str],
               phases: Optional[List[str]]) -> str:
    """Translate (elements, phases) into the API's URL+query convention."""
    raise NotImplementedError("Define _PARAM_NAMES and the URL template "
                              "from api_spec/help.html, then implement this.")


def _http_get(url: str, timeout: float) -> bytes:
    """Plain GET — kept separate so it's easy to swap for `requests`."""
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "tdb_corpus/0.1 (+github)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_response(raw: bytes) -> List[CitationRecord]:
    """Turn the API's response (JSON/XML/HTML) into CitationRecords."""
    raise NotImplementedError("Implement once _RESPONSE_FORMAT is known.")


# ─── CLI ───────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Query TDBDB (van de Walle, Brown)")
    ap.add_argument("--elements", required=True,
                    help="Comma-separated element list, e.g. Co,Cr,Ni")
    ap.add_argument("--phases", default=None,
                    help="Optional comma-separated phase filter")
    ap.add_argument("--out", default=None,
                    help="Write results as JSON to this path (else stdout)")
    args = ap.parse_args()

    els = [e.strip() for e in args.elements.split(",") if e.strip()]
    phs = ([p.strip() for p in args.phases.split(",") if p.strip()]
           if args.phases else None)

    try:
        records = query_tdbdb(els, phs)
    except NotImplementedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = [asdict(r) for r in records]
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(payload)} record(s) -> {args.out}")
    else:
        json.dump(payload, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
