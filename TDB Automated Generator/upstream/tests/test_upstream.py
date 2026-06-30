#!/usr/bin/env python3
"""
Unit tests for the pure (VASP-free) logic of the upstream generator:
POTCAR ENMAX parsing, ENCUT/KPPRA grids, vasp.wrap generation, 1 meV/atom
convergence selection, str.out parsing, SIGMA lev=3->lev=0 spin conversion,
and the DLM spin-suffix fixup.

Run:  cd "TDB Automated Generator/upstream" && python3 -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest

# Make the package modules importable.
PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

import potcar
import vaspwrap
import converge
import sqsgen
import phonon
import run_upstream
from strfile import read_structure, strip_spin_suffix_text
from phases import DLMConfig, SigmaDLMSpec, DLM_SPIN_UP, DLM_SPIN_DOWN


# ---- POTCAR / grids --------------------------------------------------------

POTCAR_CO = "  PAW_PBE Co 06Sep2000\n   ENMAX  =  267.882; ENMIN  =  200.911 eV\n"
POTCAR_CR = "  PAW_PBE Cr 06Sep2000\n   ENMAX  =  227.388; ENMIN  =  170.541 eV\n"


def test_parse_enmax(tmp_path):
    p = tmp_path / "POTCAR"
    p.write_text(POTCAR_CO)
    vals = potcar.parse_enmax(p)
    assert vals and abs(vals[0] - 267.882) < 1e-6


def test_max_enmax_multi(tmp_path):
    a = tmp_path / "CoP"; a.write_text(POTCAR_CO)
    b = tmp_path / "CrP"; b.write_text(POTCAR_CR)
    assert abs(potcar.max_enmax([a, b]) - 267.882) < 1e-6


def test_max_enmax_raises_when_empty(tmp_path):
    p = tmp_path / "POTCAR"; p.write_text("no cutoff here")
    with pytest.raises(ValueError):
        potcar.max_enmax([p])


def test_encut_grid():
    g = potcar.encut_grid(200.0)            # 200..250, 5 pts (step 12.5)
    assert g == [200, 212, 225, 238, 250]   # 212.5 -> 212 (round-half-even)
    assert len(g) == 5
    assert g[0] == 200 and g[-1] == 250


def test_kppra_grid_and_probe():
    assert potcar.kppra_grid() == [4000, 5000, 6000, 7000, 8000, 9000, 10000]
    assert potcar.kppra_probe_encut(200.0) == 225   # 1.125 * 200


# ---- vasp.wrap -------------------------------------------------------------

def test_wrap_static_has_dostatic_and_algo_all():
    w = vaspwrap.build_vasp_wrap("static", encut=300, kppra=6000)
    assert "DOSTATIC" in w
    assert "ALGO = All" in w
    assert "ENCUT = 300" in w
    assert "KPPRA = 6000" in w
    assert "NSW = 0" in w
    assert "USEPOT = PAWPBE" in w


def test_wrap_relax_has_dostatic_full_dof():
    # The reference workflow keeps DOSTATIC on relax (relax + final static E).
    w = vaspwrap.build_vasp_wrap("relax", encut=300, kppra=6000)
    assert "DOSTATIC" in w
    assert "ISIF = 3" in w
    assert "IBRION = 2" in w
    assert "NSW = 300" in w


def test_wrap_phonon_icharg_frozen():
    w = vaspwrap.build_vasp_wrap("phonon", encut=300, kppra=6000)
    assert "DOSTATIC" not in w
    assert "ICHARG = 1" in w
    assert "NSW = 0" in w
    assert "LWAVE = .TRUE." in w


def test_wrap_algo_override():
    w = vaspwrap.build_vasp_wrap("static", encut=520, kppra=8000, algo="Fast")
    assert "ALGO = Fast" in w


def test_parse_dlm_moments_defaults_and_overrides():
    m = run_upstream.parse_dlm_moments("Co=Co:1.8,Cr=Cr_pv:1.5", ["Co", "Cr"])
    assert m["Co"] == ("Co", 1.8)
    assert m["Cr"] == ("Cr_pv", 1.5)
    # missing element falls back to (element, 2.0)
    m2 = run_upstream.parse_dlm_moments("Co=Co:1.8", ["Co", "Cr"])
    assert m2["Cr"] == ("Cr", 2.0)
    # bare element and ':moment' forms
    m3 = run_upstream.parse_dlm_moments("Ni:0.7,Fe", ["Ni", "Fe"])
    assert m3["Ni"] == ("Ni", 0.7) and m3["Fe"] == ("Fe", 2.0)


def test_subatom_lines():
    sub = {"Co": ("Co", 1.8), "Cr": ("Cr_pv", 1.5)}
    lines = vaspwrap.subatom_lines(sub)
    assert "SUBATOM = s/Co+2/Co+1.8/g" in lines
    assert "SUBATOM = s/Co-2/Co-1.8/g" in lines
    assert "SUBATOM = s/Cr+2/Cr_pv+1.5/g" in lines
    assert "SUBATOM = s/Cr-2/Cr_pv-1.5/g" in lines


def test_wrap_dlm_subatom_and_magatom():
    dlm = DLMConfig(enabled=True,
                    subatom={"Co": ("Co", 1.8), "Cr": ("Cr_pv", 1.5)})
    w = vaspwrap.build_vasp_wrap("static", encut=520, kppra=8000,
                                 dlm=dlm, algo="Fast")
    assert "MAGATOM =" in w
    assert "NUPDOWN = 0" in w
    assert "AMIX_MAG = 0.4" in w
    assert "SUBATOM = s/Cr-2/Cr_pv-1.5/g" in w
    assert "MAGMOM" not in w     # moments come from SUBATOM, not MAGMOM


# ---- convergence selection (1 meV/atom) ------------------------------------

def test_select_converged_picks_smallest_stable():
    settings = [4000, 5000, 6000, 7000]
    # energies drift then settle within 1 meV of the 7000 reference from 6000 on
    e = [-5.010, -5.0035, -5.0006, -5.0000]
    chosen, conv, ref = converge.select_converged(settings, e, tol_ev=0.001)
    assert ref == 7000
    assert chosen == 6000
    assert conv is True


def test_select_converged_not_converged_falls_back():
    settings = [4000, 5000, 6000]
    e = [-5.10, -5.05, -5.00]               # 50 meV gaps, never within 1 meV
    chosen, conv, ref = converge.select_converged(settings, e, tol_ev=0.001)
    assert conv is False
    assert chosen == 6000                   # reference / highest


def test_select_converged_ignores_missing():
    settings = [4000, 5000, 6000]
    e = [None, -5.0005, -5.0000]
    chosen, conv, ref = converge.select_converged(settings, e, tol_ev=0.001)
    assert chosen == 5000 and conv is True


# ---- str.out parsing + DLM fixup -------------------------------------------

FCC_DLM_STR = """\
3.5 3.5 3.5 90 90 90
1 0 0
0 1 0
0 0 1
0 0 0 Co+2
0.5 0.5 0 Co-2
0.5 0 0.5 Co+2
0 0.5 0.5 Co-2
"""


def test_read_structure_counts_atoms():
    import tempfile, os
    p = Path(tempfile.mkdtemp()) / "str.out"
    p.write_text(FCC_DLM_STR)
    s = read_structure(p)
    assert len(s.atoms) == 4
    assert s.species() == ["Co+2", "Co-2", "Co+2", "Co-2"]


def test_strip_spin_suffix_text():
    assert strip_spin_suffix_text("0 0 0 Co+2\n") == "0 0 0 Co\n"
    assert strip_spin_suffix_text("0 0 0 Cr-2\n") == "0 0 0 Cr\n"
    assert strip_spin_suffix_text("0 0 0 Fe+4\n") == "0 0 0 Fe\n"


def test_dlm_fixup_walks_tree(tmp_path):
    top = tmp_path / "sqs"
    (top / "vol_0" / "p1").mkdir(parents=True)
    (top / "str_relax.out").write_text(FCC_DLM_STR)
    (top / "vol_0" / "str_unpert.out").write_text(FCC_DLM_STR)
    (top / "vol_0" / "p1" / "str_relax.out").write_text(FCC_DLM_STR)
    changed = phonon.dlm_fixup(top)
    assert len(changed) == 3
    txt = (top / "str_relax.out").read_text()
    assert "+2" not in txt and "-2" not in txt
    assert "Co" in txt
    # idempotent second pass
    assert phonon.dlm_fixup(top) == []


# ---- SIGMA lev=3 -> lev=0 +/- spin conversion ------------------------------

SIGMA_LEV3_STR = """\
4.5 4.5 2.8 90 90 90
1 0 0
0 1 0
0 0 1
0 0 0 X1
0.1 0.1 0 X2
0.2 0.2 0 X1
0.3 0.3 0 X2
"""


def test_sigma_lev3_to_lev0_autodetect(tmp_path):
    src = tmp_path / "sqs_lev=3"; src.mkdir()
    (src / "str.out").write_text(SIGMA_LEV3_STR)
    (src / "energy").write_text("-42.0\n")
    dst = tmp_path / "Co_lev0_dlm"
    sqsgen.sigma_lev3_to_lev0_dlm(src, dst, SigmaDLMSpec(element="Co"))
    s = read_structure(dst / "str.out")
    # X1 -> Co+2, X2 -> Co-2
    assert s.species() == ["Co+2", "Co-2", "Co+2", "Co-2"]
    assert (dst / "energy").is_file()       # aux copied


def test_sigma_lev3_explicit_tokens(tmp_path):
    src = tmp_path / "sqs_lev=3"; src.mkdir()
    (src / "str.out").write_text(SIGMA_LEV3_STR)
    dst = tmp_path / "Cr_lev0_dlm"
    sqsgen.sigma_lev3_to_lev0_dlm(
        src, dst, SigmaDLMSpec(element="Cr", token_up="X2", token_down="X1"))
    s = read_structure(dst / "str.out")
    # tokens swapped: X2 -> up (+2), X1 -> down (-2)
    assert s.species() == ["Cr-2", "Cr+2", "Cr-2", "Cr+2"]
