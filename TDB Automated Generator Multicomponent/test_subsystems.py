#!/usr/bin/env python3
"""
Sandbox-runnable unit tests for subsystems.py (no ATAT / no DFT needed).

Builds synthetic sqs2tdb-style directory trees in a tempdir and exercises:
  - directory-name parsing (parse_occupation)
  - composition computation on single- and multi-sublattice phases
  - foreign-element rejection
  - subsystem-tag classification (binary edge vs ternary interior vs ...)
  - SIGMA corner identification
  - terms.in parameter counting
  - scan_sqs end-to-end

Run with:  python test_subsystems.py
Exit code 0 = all tests passed.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import subsystems as sub


# ════════════════════════════════════════════════════════════════════
#  Tiny test harness (stdlib only, no pytest dependency)
# ════════════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {label}")
    else:
        _FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def section(name: str) -> None:
    print(f"\n── {name} ──")


# ════════════════════════════════════════════════════════════════════
#  Synthetic data builder
# ════════════════════════════════════════════════════════════════════

def make_sqs_dir(
    root: Path,
    phase_dir: str,
    name: str,
    energy: float = -123.456,
    with_svib: bool = False,
) -> Path:
    """Create a minimal sqs2tdb-style directory under root/phase_dir/name."""
    d = root / phase_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "energy").write_text(f"{energy}\n")
    (d / "str.out").write_text("# placeholder\n")
    if with_svib:
        (d / "svib_ht").write_text("0.0\n")
    return d


# ════════════════════════════════════════════════════════════════════
#  Tests
# ════════════════════════════════════════════════════════════════════

def test_parse_occupation() -> None:
    section("parse_occupation")

    o = sub.parse_occupation("sqs_lev=0_a_Co=1")
    check("FCC endmember basic", o is not None and o.level == 0)
    check("FCC endmember site/elem",
          o.sites == (("a", "CO", 1.0),))

    o = sub.parse_occupation("sqs_lev=2_a_Co=0.5,a_Cr=0.5")
    check("FCC binary mixing parsed",
          o is not None and o.level == 2 and len(o.sites) == 2)
    check("FCC mixing species_per_site",
          o.species_per_site == {"a": {"CO": 0.5, "CR": 0.5}})

    o = sub.parse_occupation("sqs_lev=0_aj_Co=1,g_Cr=1,ii_Ni=1")
    check("SIGMA mixed corner parsed",
          o is not None
          and o.species_per_site == {
              "aj": {"CO": 1.0}, "g": {"CR": 1.0}, "ii": {"NI": 1.0}})

    # sqsdb_ prefix should also work
    o = sub.parse_occupation("sqsdb_lev=0_a_Cr=1")
    check("sqsdb_ prefix accepted",
          o is not None and o.species_per_site == {"a": {"CR": 1.0}})

    # garbage in → None
    check("non-SQS name rejected",
          sub.parse_occupation("not_an_sqs_dir") is None)


def test_composition_on_phase() -> None:
    section("composition_on_phase")

    o = sub.parse_occupation("sqs_lev=2_a_Co=0.5,a_Cr=0.5")
    c = sub.composition_on_phase(o, "FCC_A1", ["Co", "Cr", "Ni"])
    check("FCC binary 50/50 normalized",
          c is not None and abs(c["CO"] - 0.5) < 1e-9
          and abs(c["CR"] - 0.5) < 1e-9 and c["NI"] == 0.0)

    o = sub.parse_occupation("sqs_lev=0_a_Co=1")
    c = sub.composition_on_phase(o, "FCC_A1", ["Co", "Cr", "Ni"])
    check("FCC pure-Co endmember",
          c == {"CO": 1.0, "CR": 0.0, "NI": 0.0})

    # SIGMA: multiplicity-weighted
    # aj=10 Co, g=4 Cr, ii=16 Ni  → x(Co)=10/30, x(Cr)=4/30, x(Ni)=16/30
    o = sub.parse_occupation("sqs_lev=0_aj_Co=1,g_Cr=1,ii_Ni=1")
    c = sub.composition_on_phase(o, "SIGMA_D8B", ["Co", "Cr", "Ni"])
    check("SIGMA multiplicity weighting",
          c is not None
          and abs(c["CO"] - 10 / 30) < 1e-9
          and abs(c["CR"] - 4 / 30) < 1e-9
          and abs(c["NI"] - 16 / 30) < 1e-9)

    # Foreign-element rejection (the bug we fixed in the binary script)
    o = sub.parse_occupation("sqs_lev=0_aj_Co=1,g_Cr=1,ii_Fe=1")
    c = sub.composition_on_phase(o, "SIGMA_D8B", ["Co", "Cr", "Ni"])
    check("SIGMA foreign Fe rejected", c is None)

    o = sub.parse_occupation("sqs_lev=2_a_Co=0.5,a_Fe=0.5")
    c = sub.composition_on_phase(o, "FCC_A1", ["Co", "Cr"])
    check("FCC foreign Fe rejected", c is None)


def test_subsystem_tag() -> None:
    section("subsystem_for_occupation")

    o = sub.parse_occupation("sqs_lev=0_a_Co=1")
    tag = sub.subsystem_for_occupation(o, "FCC_A1", ["Co", "Cr", "Ni"])
    check("FCC pure Co → ('CO',)", tag == ("CO",))

    o = sub.parse_occupation("sqs_lev=2_a_Co=0.5,a_Cr=0.5")
    tag = sub.subsystem_for_occupation(o, "FCC_A1", ["Co", "Cr", "Ni"])
    check("FCC binary edge Co-Cr → ('CO','CR')", tag == ("CO", "CR"))

    o = sub.parse_occupation("sqs_lev=4_a_Co=0.33,a_Cr=0.33,a_Ni=0.34")
    tag = sub.subsystem_for_occupation(o, "FCC_A1", ["Co", "Cr", "Ni"])
    check("FCC ternary interior → ('CO','CR','NI')",
          tag == ("CO", "CR", "NI"))

    # SIGMA binary corner: all three sublattices occupied by Co or Cr only
    o = sub.parse_occupation("sqs_lev=0_aj_Co=1,g_Co=1,ii_Cr=1")
    tag = sub.subsystem_for_occupation(o, "SIGMA_D8B", ["Co", "Cr", "Ni"])
    check("SIGMA binary corner Co/Co/Cr → ('CO','CR')",
          tag == ("CO", "CR"))


def test_sigma_corner_key() -> None:
    section("sigma_corner_key")
    o = sub.parse_occupation("sqs_lev=0_aj_Co=1,g_Cr=1,ii_Ni=1")
    k = sub.sigma_corner_key(o, ["Co", "Cr", "Ni"])
    check("SIGMA ternary corner key",
          k == (("aj", "CO"), ("g", "CR"), ("ii", "NI")))

    # Reject when foreign element present
    o = sub.parse_occupation("sqs_lev=0_aj_Co=1,g_Cr=1,ii_Fe=1")
    check("SIGMA corner with Fe → None",
          sub.sigma_corner_key(o, ["Co", "Cr", "Ni"]) is None)

    # Reject when a sublattice has >1 species (i.e. not a corner)
    o = sub.parse_occupation("sqs_lev=0_aj_Co=0.5,aj_Cr=0.5,g_Cr=1,ii_Ni=1")
    check("SIGMA non-corner (mixed aj) → None",
          sub.sigma_corner_key(o, ["Co", "Cr", "Ni"]) is None)


def test_pure_endmember_element() -> None:
    section("pure_endmember_element")
    o = sub.parse_occupation("sqs_lev=0_a_Co=1")
    check("FCC pure Co → 'CO'",
          sub.pure_endmember_element(o, "FCC_A1", ["Co", "Cr"]) == "CO")
    o = sub.parse_occupation("sqs_lev=2_a_Co=0.5,a_Cr=0.5")
    check("FCC binary mixing → None (not pure)",
          sub.pure_endmember_element(o, "FCC_A1", ["Co", "Cr"]) is None)
    o = sub.parse_occupation("sqs_lev=0_aj_Co=1,g_Cr=1,ii_Ni=1")
    check("SIGMA → None (single-sublattice only)",
          sub.pure_endmember_element(o, "SIGMA_D8B", ["Co", "Cr", "Ni"]) is None)


def test_enumerate_subsystems() -> None:
    section("enumerate_subsystems")
    s = sub.enumerate_subsystems(["Co", "Cr", "Ni"])
    check("Co-Cr-Ni binaries",
          s.get(2) == [("CO", "CR"), ("CO", "NI"), ("CR", "NI")])
    check("Co-Cr-Ni ternary", s.get(3) == [("CO", "CR", "NI")])

    s = sub.enumerate_subsystems(["Fe", "Cr", "Ni", "Co"], 2, 3)
    check("Quaternary, binary count C(4,2)=6", len(s.get(2, [])) == 6)
    check("Quaternary, ternary count C(4,3)=4", len(s.get(3, [])) == 4)


def test_n_params_for_terms() -> None:
    section("n_params_for_terms")

    # Binary FCC: K=2.  1,0 → 2 (endmembers).  2,L → C(2,2)*(L+1) = L+1
    binary_fcc = {"a": ["CO", "CR"]}
    check("Binary FCC 1,0 + 2,0 → 3 params",
          sub.n_params_for_terms(binary_fcc, [(1, 0), (2, 0)]) == 3)
    check("Binary FCC 1,0 + 2,2 → 5 params (matches old order+3=5)",
          sub.n_params_for_terms(binary_fcc, [(1, 0), (2, 2)]) == 5)

    # Ternary FCC: K=3. 1,0 → 3.  2,0 → C(3,2)*1 = 3 (three binary L0 pairs).
    # 3,0 → C(3,3)*1 = 1 (ternary).
    ternary_fcc = {"a": ["CO", "CR", "NI"]}
    check("Ternary FCC 1,0 + 2,0 → 6 params",
          sub.n_params_for_terms(ternary_fcc, [(1, 0), (2, 0)]) == 6)
    check("Ternary FCC 1,0 + 2,1 + 3,0 → 3+6+1 = 10 params",
          sub.n_params_for_terms(ternary_fcc, [(1, 0), (2, 1), (3, 0)]) == 10)

    # SIGMA Co-Cr (binary, 3 sublattices each with K=2). 1,0 across the three
    # sublattices = 2+2+2 = 6. 2,0 = C(2,2)*1 * 3 = 3.
    sigma_bin = {"aj": ["CO", "CR"], "g": ["CO", "CR"], "ii": ["CO", "CR"]}
    check("SIGMA binary 1,0 + 2,0 → 9 params",
          sub.n_params_for_terms(sigma_bin, [(1, 0), (2, 0)]) == 9)


def test_scan_sqs() -> None:
    section("scan_sqs end-to-end")

    tmp = Path(tempfile.mkdtemp(prefix="mc_test_"))
    try:
        root_a = tmp / "rootA"
        root_b = tmp / "rootB"

        # FCC endmembers, all three pure elements
        make_sqs_dir(root_a, "FCC_A1", "sqs_lev=0_a_Co=1", -10.0, True)
        make_sqs_dir(root_a, "FCC_A1", "sqs_lev=0_a_Cr=1",  -9.0, True)
        make_sqs_dir(root_a, "FCC_A1", "sqs_lev=0_a_Ni=1",  -8.0, True)
        # FCC binary edges
        make_sqs_dir(root_a, "FCC_A1", "sqs_lev=2_a_Co=0.5,a_Cr=0.5", -9.5, True)
        make_sqs_dir(root_a, "FCC_A1", "sqs_lev=2_a_Co=0.5,a_Ni=0.5", -9.0, True)
        # FCC ternary interior
        make_sqs_dir(root_a, "FCC_A1",
                     "sqs_lev=4_a_Co=0.333,a_Cr=0.333,a_Ni=0.334",
                     -9.2, False)

        # FCC same endmember Co from a SECOND root (different DFT run)
        make_sqs_dir(root_b, "FCC_A1", "sqs_lev=0_a_Co=1", -10.5, False)

        # SIGMA pure Co/Co/Co corner
        make_sqs_dir(root_a, "SIGMA_D8B",
                     "sqs_lev=0_aj_Co=1,g_Co=1,ii_Co=1", -11.0, True)
        # SIGMA binary corner Co/Co/Cr
        make_sqs_dir(root_a, "SIGMA_D8B",
                     "sqs_lev=0_aj_Co=1,g_Co=1,ii_Cr=1", -10.9, True)
        # SIGMA ternary corner Co/Cr/Ni
        make_sqs_dir(root_a, "SIGMA_D8B",
                     "sqs_lev=0_aj_Co=1,g_Cr=1,ii_Ni=1", -10.8, True)
        # SIGMA with foreign Fe — should be rejected
        make_sqs_dir(root_a, "SIGMA_D8B",
                     "sqs_lev=0_aj_Co=1,g_Cr=1,ii_Fe=1", -10.5, True)
        # Missing files — should be skipped
        bad = root_a / "FCC_A1" / "sqs_lev=0_a_Fe=1"
        bad.mkdir(parents=True)
        # no energy / no str.out

        cands = sub.scan_sqs(
            roots=[root_a, root_b],
            elements=["Co", "Cr", "Ni"],
            phases=["FCC_A1", "SIGMA_D8B"],
            verbose=False,
        )

        # 3 endmembers + 2 binary edges + 1 ternary + 1 from rootB = 7 FCC
        # 1 + 1 + 1 = 3 SIGMA (the Fe one rejected)
        fcc = [c for c in cands if c.phase == "FCC_A1"]
        sig = [c for c in cands if c.phase == "SIGMA_D8B"]
        check("FCC candidates collected", len(fcc) == 7,
              f"got {len(fcc)}: {[c.path.name for c in fcc]}")
        check("SIGMA candidates collected (Fe rejected)", len(sig) == 3,
              f"got {len(sig)}: {[c.path.name for c in sig]}")

        co_em = [c for c in fcc if c.subsystem == ("CO",)]
        check("Two pure-Co FCC candidates (one per root)", len(co_em) == 2)

        bin_edges = [c for c in fcc if len(c.subsystem) == 2]
        ter_int   = [c for c in fcc if len(c.subsystem) == 3]
        check("Two FCC binary-edge SQS detected", len(bin_edges) == 2)
        check("One FCC ternary interior SQS detected", len(ter_int) == 1)

        # Energies parsed (round-trip)
        all_es = [c.energy for c in cands]
        check("All energies parsed (no None)", all(e is not None for e in all_es))

        # svib detection: with_svib=True was set on
        #   FCC: Co, Cr, Ni, CoCr, CoNi  (5) — not the ternary, not the rootB Co
        #   SIGMA: CoCoCo, CoCoCr, CoCrNi  (3) — Fe one is rejected pre-svib
        # → 8 candidates with svib
        n_svib = sum(1 for c in cands if c.has_svib)
        check("svib detection picked up the right count",
              n_svib == 8, f"expected 8 with svib, got {n_svib}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════

def main() -> int:
    print("Running subsystems.py unit tests\n")
    test_parse_occupation()
    test_composition_on_phase()
    test_subsystem_tag()
    test_sigma_corner_key()
    test_pure_endmember_element()
    test_enumerate_subsystems()
    test_n_params_for_terms()
    test_scan_sqs()
    print(f"\n{'=' * 50}")
    print(f"  {_PASS} passed, {_FAIL} failed")
    print(f"{'=' * 50}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
