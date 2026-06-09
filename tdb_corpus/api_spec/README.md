# api_spec/ — TDBDB API specification drop-zone

The TDBDB website at https://avdwgroup.engin.brown.edu is reachable
from Brown campus / VPN / most institutional networks, but is blocked
from this repository's CI / dev sandboxes (WAF / IP allowlist).
To wire up `../tdbdb_query.py` we need the API spec from a place
we can read.

## What to drop here

**Required to write the client:**

1. `help.html` — the contents of
   https://avdwgroup.engin.brown.edu/help.html (just download via your
   browser's *Save Page As…* → "HTML Only"). The 2018 Calphad paper
   states this page contains the API conventions: base-URL pattern,
   parameter substitution, query format.

2. `sample_response.txt` — the raw response body of one query, for
   example:
   ```
   curl -s 'https://avdwgroup.engin.brown.edu/<endpoint>?<params>' \
        > sample_response.txt
   ```
   Co-Cr or Fe-Cr-Ni would be good choices because we already know
   they're well-covered (>10 assessments each).

**Optional but useful:**

3. `paper.pdf` — the 2018 Calphad paper (van de Walle, Sun, Hong,
   Kadkhodaei. *The Thermodynamic Database Database*, doi:
   `10.1016/j.calphad.2018.03.001`). The API section is in the
   "Application Programming Interface" subsection.

4. `auth_notes.md` — any non-obvious behavior you discover while
   poking the API: rate limits, required User-Agent, login state
   needed for some queries, etc.

## After you've populated this directory

Ping me and I'll fill in `../tdbdb_query.py` based on what's here.
The expected output of the client is a list of `dict`s shaped like:

```python
[
  {
    "system":     ["CO", "CR"],          # element list (uppercase)
    "phases":     ["FCC_A1", "BCC_A2", "SIGMA_D8B"],
    "citation":   "Andersson & Sundman (1987)",
    "doi":        "10.1016/0364-5916(87)90021-6",
    "source_url": "https://doi.org/10.1016/0364-5916(87)90021-6",
    "tdb_url":    None,        # or the direct .TDB link if the API gives one
    "license":    "paywalled", # or "CC-BY-4.0", "by-permission", etc.
  },
  ...
]
```

— so the client can hand straight off to `corpus_ingest.py` /
`tdbdb_fetch_open.py` without any further parsing.
