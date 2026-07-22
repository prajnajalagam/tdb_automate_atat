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
import relax
import runner
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
    # ISIF=3 + IBRION=2 = full relaxation; NSW=100 per the production INCAR.
    w = vaspwrap.build_vasp_wrap("relax", encut=300, kppra=6000)
    assert "DOSTATIC" in w
    assert "ISIF = 3" in w
    assert "IBRION = 2" in w
    assert "NSW = 100" in w
    assert "ISMEAR = 1" in w and "SIGMA = 0.08" in w


def test_wrap_phonon_matches_fvasp_conventions():
    """Frozen force-run wrap = the user's vaspf.wrap: NSW=0/IBRION=-1/
    ISIF=2, PREC=Accurate, ALGO=Fast, and NO ICHARG=1 (hard-errors when
    no CHGCAR exists — the first force run never has one)."""
    w = vaspwrap.build_vasp_wrap("phonon", encut=300, kppra=6000)
    assert "DOSTATIC" not in w
    assert "ICHARG" not in w
    assert "NSW = 0" in w and "IBRION = -1" in w and "ISIF = 2" in w
    assert "PREC = Accurate" in w and "ALGO = All" in w


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


# ---- convergence selection: successive-difference + confirmation ----------
# (2026-07-16 user criterion: step from PREVIOUS < tol AND step to NEXT
#  < tol; the next point is the deliberately "unneeded" confirmation.)

def test_select_converged_real_kppra_data_picks_7000():
    """The user's actual 2026-07-16 KPPRA sweep: manual analysis says
    7000 (|7000-6000| = 0.082 meV < 0.1, |8000-7000| = 0.000 confirms).
    4000/5000 must NOT win despite their small mutual step, because the
    5000->6000 step (0.176 meV) breaks the plateau."""
    settings = [4000, 5000, 6000, 7000, 8000, 9000, 10000]
    e = [-6.2614504, -6.2615232, -6.2616992, -6.2616170,
         -6.2616170, -6.2617104, -6.2615474]
    chosen, conv, ref, rule = converge.select_converged(settings, e,
                                                         tol_ev=0.0001)
    assert conv is True
    assert chosen == 7000
    assert ref == 8000                      # the confirming point
    assert rule == "successive"             # primary rule, not plateau


def test_select_converged_real_encut_data_not_converged():
    """The user's actual ENCUT sweep: successive steps still 0.4-1.4
    meV/atom at the top of the grid — must report NOT converged (the
    old compare-to-reference rule wrongly accepted this)."""
    settings = [268, 285, 301, 318, 335]
    e = [-6.2500555, -6.2579381, -6.2614504, -6.2610447, -6.2596040]
    chosen, conv, ref, rule = converge.select_converged(
        settings, e, tol_ev=0.0001, plateau_band_ev=0)
    assert conv is False
    assert chosen == 335                    # fall back to highest


def test_select_converged_needs_confirming_point_above():
    # plateau reached at the LAST point only -> no confirmation -> not
    # converged (drives the adaptive extension to add one more run)
    settings = [300, 320, 340]
    e = [-5.010, -5.0002, -5.00015]
    chosen, conv, ref, rule = converge.select_converged(
        settings, e, tol_ev=0.0001, plateau_band_ev=0)
    assert conv is False and chosen == 340


def test_select_converged_ignores_missing():
    settings = [4000, 5000, 6000, 7000]
    e = [None, -5.00005, -5.00002, -5.00001]
    chosen, conv, ref, rule = converge.select_converged(settings, e,
                                                         tol_ev=0.0001)
    assert chosen == 6000 and conv is True and ref == 7000


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


# ---- sqs2tdb -cp two-pass behaviour -----------------------------------------

def _fake_sqs2tdb(work_root, target, calls):
    """Simulate sqs2tdb -cp: pass 1 only writes <target>/species.in;
    pass 2 (species.in exists) copies the SQS structures."""
    def fake_run_logged(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        calls.append(list(cmd))
        tdir = Path(cwd) / target
        sp = tdir / "species.in"
        if not sp.is_file():
            tdir.mkdir(parents=True, exist_ok=True)
            sp.write_text("Co,Cr\n")
        else:
            d = tdir / "sqs_lev=1_a_Co=0.5,a_Cr=0.5"
            d.mkdir(parents=True, exist_ok=True)
            (d / "str.out").write_text("1 0 0\n0 1 0\n0 0 1\n")
        return 0
    return fake_run_logged


def test_generate_phase_sqs_runs_cp_twice(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sqsgen.runner, "run_logged",
                        _fake_sqs2tdb(tmp_path, "BCC_A2_small", calls))
    out = sqsgen.generate_phase_sqs(tmp_path, "BCC_A2", elements=["Co", "Cr"],
                                    dlm=False)
    cp_calls = [c for c in calls if c[:2] == ["sqs2tdb", "-cp"]]
    assert len(cp_calls) == 2, "sqs2tdb -cp must be run twice"
    assert all("-sp=Co,Cr" in c for c in cp_calls)
    assert any(out.rglob("str.out"))


def test_generate_phase_sqs_uses_lv_flag(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sqsgen.runner, "run_logged",
                        _fake_sqs2tdb(tmp_path, "SIGMA_D8B", calls))
    sqsgen.generate_phase_sqs(tmp_path, "SIGMA_D8B", elements=["Co", "Cr"],
                              level=3, use_small=False)
    assert all("-lv=3" in c for c in calls if c[:2] == ["sqs2tdb", "-cp"])
    assert not any("-lev=3" in c for c in calls)


def test_generate_phase_sqs_species_edit_between_passes(tmp_path, monkeypatch):
    calls = []
    seen = {}
    monkeypatch.setattr(sqsgen.runner, "run_logged",
                        _fake_sqs2tdb(tmp_path, "SIGMA_D8B", calls))

    def edit(species_in):
        # Hook fires after pass 1 (species.in exists, no SQS dirs yet).
        seen["cp_calls_at_edit"] = len(
            [c for c in calls if c[:2] == ["sqs2tdb", "-cp"]])
        species_in.write_text("Co+2,Co-2\n")

    sqsgen.generate_phase_sqs(tmp_path, "SIGMA_D8B", elements=["Co", "Cr"],
                              level=3, use_small=False, species_edit=edit)
    assert seen["cp_calls_at_edit"] == 1
    assert (tmp_path / "SIGMA_D8B" / "species.in").read_text() == "Co+2,Co-2\n"


def test_generate_phase_sqs_requires_elements_or_species_in(tmp_path):
    with pytest.raises(RuntimeError, match="species.in"):
        sqsgen.generate_phase_sqs(tmp_path, "BCC_A2")


def test_generate_phase_sqs_fails_if_nothing_copied(tmp_path, monkeypatch):
    def noop(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        (Path(cwd) / "BCC_A2_small").mkdir(exist_ok=True)
        return 0
    monkeypatch.setattr(sqsgen.runner, "run_logged", noop)
    with pytest.raises(RuntimeError, match="no element-decorated"):
        sqsgen.generate_phase_sqs(tmp_path, "BCC_A2", elements=["Co", "Cr"])


# ---- relax.py: -mk prep and runstruct default ------------------------------

class _RecCalls:
    def __init__(self):
        self.logged = []
        self.polled = []

    def logged_fn(self):
        def f(cmd, cwd, log, env_bin=None, timeout=None, check=True):
            self.logged.append(list(cmd))
            return 0
        return f

    def polled_fn(self, touch_str_relax=True):
        def f(cmd, cwd, log, done_when=None, stop_sentinel=None,
              env_bin=None, timeout=None, check=True, **kw):
            self.polled.append(list(cmd))
            if touch_str_relax:
                (Path(cwd) / "str_relax.out").write_text("stub\n")
            return 0
        return f


def _stub_encut_kppra(monkeypatch, calc_dir):
    """relax_structure calls write_relax_wrap which needs a valid POTCAR
    only via vaspwrap.build_vasp_wrap. We stub vaspwrap so the test
    doesn't need a real POTCAR."""
    monkeypatch.setattr(
        relax, "build_vasp_wrap",
        lambda kind, *a, **k: f"# stub {kind}\n",
    )


def test_infdet_is_now_the_default(tmp_path, monkeypatch):
    """--relax-method default is 'infdet' (user decision 2026-07-15):
    robustrelax_vasp -id with the -c 0.05 strain cutoff it REQUIRES."""
    rec = _RecCalls()
    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged", rec.logged_fn())
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=400, kppra=8000)  # no method= arg
    assert rec.logged == [["robustrelax_vasp", "-mk"]], rec.logged
    assert rec.polled == [["robustrelax_vasp", "-id", "-c", "0.05"]], \
        rec.polled


_STR_OUT_2ATOM = ("1 0 0\n0 1 0\n0 0 1\n"
                  "1 0 0\n0 1 0\n0 0 1\n"
                  "0 0 0 Co\n0.5 0.5 0.5 Cr\n")


def test_robustrelax_tuned_wrap_exists_before_mk(tmp_path, monkeypatch):
    """Order verified against the robustrelax_vasp SOURCE (2026-07-20):
    -mk does NOT create vasp.wrap — it requires it, and derives every
    auxiliary wrap (vaspvol/vaspstatic/vaspid/vaspf) by grep-
    transforming it. The TUNED vasp.wrap must therefore be on disk
    BEFORE -mk runs, or every derived wrap loses the converged
    ENCUT/KPPRA/spin settings (and -id later dies on missing
    vaspid.wrap)."""
    (tmp_path / "str.out").write_text(_STR_OUT_2ATOM)
    seen = {}

    def fake_logged(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        if "-mk" in cmd:
            wrap = Path(cwd) / "vasp.wrap"
            seen["wrap_present"] = wrap.is_file()
            seen["wrap_text"] = wrap.read_text() if wrap.is_file() else ""
        return 0

    rec = _RecCalls()
    monkeypatch.setattr(relax.runner, "run_logged", fake_logged)
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=454, kppra=7000, method="infdet",
                          cmd_prefix="mpiexec -n 128")
    assert seen.get("wrap_present"), "vasp.wrap must exist when -mk runs"
    assert "ENCUT = 454" in seen["wrap_text"]
    assert "KPPRA = 7000" in seen["wrap_text"]


def test_infdet_status_reads_termination_marker(tmp_path):
    """Success criterion per the method's author (2026-07-20): the LAST
    line of 01/infdet.log is 'infdet terminated normally', plus the
    inflection-point energy propagated to <dir>/energy. checkrelax
    magnitude is NOT part of the criterion."""
    # not engaged: no 01/, no energy_end
    assert relax.infdet_status(tmp_path) == (
        False, False, "not engaged (relaxation within -c cutoff)")

    # engaged via energy_end but infdet never ran -> failed
    (tmp_path / "energy_end").write_text("-130.2\n")
    eng, ok, msg = relax.infdet_status(tmp_path)
    assert eng and not ok and "infdet.log missing" in msg

    # infdet ran but aborted mid-flight
    d01 = tmp_path / "01"
    d01.mkdir()
    (d01 / "infdet.log").write_text("step 1\nstep 2\nvasp\n")
    eng, ok, msg = relax.infdet_status(tmp_path)
    assert eng and not ok and "did not terminate normally" in msg

    # normal termination but energy never propagated
    (d01 / "infdet.log").write_text(
        "step 1\nstep 2\ninfdet terminated normally\n")
    eng, ok, msg = relax.infdet_status(tmp_path)
    assert eng and not ok and "energy" in msg

    # the real success shape
    (tmp_path / "energy").write_text("-129.8\n")
    assert relax.infdet_status(tmp_path) == (
        True, True, "infdet terminated normally")


def test_robustrelax_complete_predicate(tmp_path):
    """2026-07-22 postmortem: str_relax.out appears after robustrelax
    STEP 1, so it must NOT count as completion — the old predicate made
    the poller kill robustrelax entering the 00/ volume relax, so 01/
    inflection detection never ran anywhere."""
    d = tmp_path
    # step-1 transient: full relax done, branch not chosen yet
    (d / "str_relax.out").write_text("x\n")
    (d / "energy").write_text("-130.2\n")       # runstruct's own energy
    assert relax.robustrelax_complete(d) is False
    # unstable branch entered: energy moved to energy_end
    (d / "energy").unlink()
    (d / "energy_end").write_text("-130.2\n")
    assert relax.robustrelax_complete(d) is False
    # infdet finished but inflection energy not yet propagated
    (d / "01").mkdir()
    (d / "01" / "cstr_relax.out").write_text("x\n")
    assert relax.robustrelax_complete(d) is False
    # full unstable completion
    (d / "energy").write_text("-129.8\n")
    assert relax.robustrelax_complete(d) is True
    # stable branch shape
    for f in ("energy_end", "01/cstr_relax.out"):
        (d / f).unlink()
    (d / "01").rmdir()
    assert relax.robustrelax_complete(d) is False
    (d / "energy_sup").write_text("-130.0\n")
    assert relax.robustrelax_complete(d) is True


def test_relax_infdet_done_when_ignores_str_relax(tmp_path, monkeypatch):
    """The done_when passed to run_polled for robustrelax modes must be
    robustrelax_complete, not str_relax.out existence."""
    captured = {}

    def fake_polled(cmd, cwd, log, done_when=None, **kw):
        captured["done_when"] = done_when
        (Path(cwd) / "str_relax.out").write_text("stub\n")
        return 0

    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged",
                        lambda *a, **k: 0)
    monkeypatch.setattr(relax.runner, "run_polled", fake_polled)
    relax.relax_structure(tmp_path, encut=400, kppra=8000, method="infdet")

    done_when = captured["done_when"]
    assert not done_when(tmp_path)      # str_relax.out alone: NOT done
    (tmp_path / "energy_sup").write_text("-1\n")
    (tmp_path / "energy").write_text("-1\n")
    assert done_when(tmp_path)          # stable-branch completion: done


def test_relax_clears_stale_error_and_stop(tmp_path, monkeypatch):
    """Stale error/stop litter from a killed run must be removed before
    relaunching robustrelax (error makes it bail after step 1; stop
    kills the 01/ infdet loop on sight)."""
    (tmp_path / "error").write_text("old\n")
    (tmp_path / "stop").write_text("")
    _stub_encut_kppra(monkeypatch, tmp_path)
    seen = {}

    def fake_polled(cmd, cwd, log, done_when=None, **kw):
        seen["error"] = (Path(cwd) / "error").exists()
        seen["stop"] = (Path(cwd) / "stop").exists()
        (Path(cwd) / "str_relax.out").write_text("stub\n")
        return 0

    monkeypatch.setattr(relax.runner, "run_logged", lambda *a, **k: 0)
    monkeypatch.setattr(relax.runner, "run_polled", fake_polled)
    relax.relax_structure(tmp_path, encut=400, kppra=8000, method="infdet")
    assert seen == {"error": False, "stop": False}


def test_run_polled_no_sentinel_litter_for_self_terminating(tmp_path):
    """A command that exits on its own (robustrelax) must not leave the
    stop sentinel behind — stale sentinels poisoned every rerun in the
    2026-07-22 tree. Also: a PRE-existing stale sentinel is removed
    before launch."""
    (tmp_path / "stop").write_text("stale\n")
    rc = runner.run_polled(
        ["bash", "-c", "test ! -e stop && echo clean"],
        cwd=tmp_path, log=tmp_path / "p.log",
        done_when=lambda _d: False,     # never satisfied -> exit path
        stop_sentinel="stop", poll_interval=0.05, timeout=10)
    assert rc == 0
    assert not (tmp_path / "stop").exists()
    assert "clean" in (tmp_path / "p.log").read_text()  # stale removed


def test_run_polled_kills_whole_process_group(tmp_path):
    """Killing only the shell parent orphaned mpiexec children (the
    UCX 'failed to create UD QP' node exhaustion of 2026-07-22). The
    poller must take down the entire process group."""
    import time as _t
    # parent spawns a child that would outlive a parent-only terminate
    marker = tmp_path / "child_alive"
    cmd = ["bash", "-c",
           f"(sleep 3 && touch {marker}) & sleep 30"]
    t0 = _t.time()
    runner.run_polled(cmd, cwd=tmp_path, log=tmp_path / "k.log",
                      done_when=lambda _d: True,   # done immediately
                      stop_sentinel="stop", poll_interval=0.05,
                      timeout=10)
    assert _t.time() - t0 < 20
    _t.sleep(3.5)          # give the (killed) child time to NOT appear
    assert not marker.exists(), \
        "child survived the kill — process group not terminated"


def test_robustrelax_normal_runs_mk_first(tmp_path, monkeypatch):
    """method='normal' must be preceded by robustrelax_vasp -mk."""
    rec = _RecCalls()
    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged", rec.logged_fn())
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=400, kppra=8000, method="normal")
    assert rec.logged == [["robustrelax_vasp", "-mk"]], rec.logged
    assert rec.polled == [["robustrelax_vasp"]], rec.polled


def test_robustrelax_infdet_runs_mk_first(tmp_path, monkeypatch):
    """method='infdet' must also be preceded by robustrelax_vasp -mk."""
    rec = _RecCalls()
    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged", rec.logged_fn())
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=400, kppra=8000, method="infdet",
                          infdet_opts="-t 1e-3")
    assert rec.logged == [["robustrelax_vasp", "-mk"]], rec.logged
    assert rec.polled == [["robustrelax_vasp", "-id", "-c", "0.05",
                           "-idop", "-t 1e-3"]], rec.polled


def test_relax_rejects_unknown_method(tmp_path, monkeypatch):
    _stub_encut_kppra(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="unknown relax method"):
        relax.relax_structure(tmp_path, encut=400, kppra=8000, method="bogus")


# ---- run_upstream.py: multi-level SQS iteration ---------------------------

def test_sqs_level_default_is_2(monkeypatch):
    """CLI default for --sqs-level must be '2' and the help must state
    the CUMULATIVE -lv semantics (level <= N per the sqs2tdb source)."""
    import re
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, str(Path(run_upstream.__file__)), "--help"],
        capture_output=True, text=True,
    )
    flat = re.sub(r"\s+", " ", r.stdout)   # undo argparse line-wrapping
    assert "the default '2'" in flat
    assert "CUMULATIVE" in flat
    assert "levels <= N" in flat


def test_process_phase_single_cumulative_lv_call(tmp_path, monkeypatch):
    """sqs2tdb -lv=N is CUMULATIVE (copies all levels <= N per its
    `$levs[1] <= $cmdline{"-lv"}` test), so process_phase must make ONE
    generate_phase_sqs call at max(sqs_levels) — not one per level."""
    calls = []

    def fake_gen(work_root, phase, elements=None, level=None, dlm=False,
                 use_small=None, species_edit=None, env_bin=None, timeout=600):
        calls.append({"phase": phase, "level": level, "dlm": dlm})
        d = Path(work_root) / f"{phase}_small"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fake_discover(_root):
        return []

    monkeypatch.setattr(run_upstream.sqsgen, "generate_phase_sqs", fake_gen)
    monkeypatch.setattr(run_upstream, "discover_sqs_dirs", fake_discover)

    run_upstream.process_phase(
        phase="FCC_A1",
        work_root=tmp_path,
        potcar_paths=[],
        dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="runstruct",
        algo="All",
        tol_ev=1e-3,
        sqs_levels=[2, 3, 4],
        sigma_elements=["Co", "Cr"],
        template_root=None,
        env_bin=None,
        skip_phonon=True,
        timeout=60,
    )
    fcc_calls = [c for c in calls if c["phase"] == "FCC_A1"]
    assert len(fcc_calls) == 1, fcc_calls
    assert fcc_calls[0]["level"] == 4, fcc_calls


def test_process_sigma_ignores_sqs_levels_binary(tmp_path, monkeypatch):
    """SIGMA in a binary must NOT iterate over sqs_levels — it's
    endmember-only (unless DLM sigma_from_lev3 overrides)."""
    calls = []

    def fake_gen(work_root, phase, elements=None, level=None, dlm=False,
                 use_small=None, species_edit=None, env_bin=None, timeout=600):
        calls.append({"phase": phase, "level": level})
        d = Path(work_root) / "SIGMA_D8B"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fake_discover(_root):
        return []

    monkeypatch.setattr(run_upstream.sqsgen, "generate_phase_sqs", fake_gen)
    monkeypatch.setattr(run_upstream, "discover_sqs_dirs", fake_discover)

    run_upstream.process_sigma(
        phase="SIGMA_D8B",
        work_root=tmp_path,
        potcar_paths=[],
        dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="runstruct",
        algo="All",
        tol_ev=1e-3,
        sqs_levels=[2, 3, 4],   # deliberately noisy; must be ignored
        sigma_elements=["Co", "Cr"],
        env_bin=None,
        skip_phonon=True,
        timeout=60,
    )
    # exactly one call, at level 0 (endmembers), regardless of sqs_levels
    assert len(calls) == 1, calls
    assert calls[0]["level"] == 0, calls


# ---- cmd_prefix (VASP MPI launcher) threading ------------------------------

def test_split_prefix_tokenizes():
    import runner as _runner
    assert _runner.split_prefix("mpiexec -n 128") == ["mpiexec", "-n", "128"]
    assert _runner.split_prefix("") == []
    assert _runner.split_prefix(None) == []


def test_runstruct_gets_vasp_launcher_tokens(tmp_path, monkeypatch):
    """runstruct method must append the launcher as SEPARATE argv tokens
    (not one space-containing string) after 'pollmach runstruct_vasp'."""
    rec = _RecCalls()
    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged", rec.logged_fn())
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=400, kppra=8000,
                          method="runstruct",
                          cmd_prefix="mpiexec -n 128")
    assert rec.polled == [
        ["pollmach", "runstruct_vasp", "mpiexec", "-n", "128"]], rec.polled


def test_robustrelax_gets_launcher_and_relax_opts(tmp_path, monkeypatch):
    """normal method: -mk first, then robustrelax with direct opts and the
    launcher LAST (matches the reference NAS job
    `robustrelax_vasp -id -c 0.05 mpiexec -n 128`)."""
    rec = _RecCalls()
    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged", rec.logged_fn())
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=400, kppra=8000, method="infdet",
                          relax_opts="-c 0.05",
                          cmd_prefix="mpiexec -n 128")
    assert rec.logged == [["robustrelax_vasp", "-mk"]], rec.logged
    assert rec.polled == [
        ["robustrelax_vasp", "-id", "-c", "0.05",
         "mpiexec", "-n", "128"]], rec.polled


def test_empty_prefix_leaves_commands_unchanged(tmp_path, monkeypatch):
    rec = _RecCalls()
    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged", rec.logged_fn())
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=400, kppra=8000,
                          method="runstruct")
    assert rec.polled == [["pollmach", "runstruct_vasp"]], rec.polled


def test_static_point_gets_launcher(tmp_path, monkeypatch):
    """The convergence sweep — where the user's OSZICAR failure occurred —
    must pass the launcher to runstruct_vasp."""
    calls = []

    def fake_logged(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        calls.append(list(cmd))
        return 0

    monkeypatch.setattr(converge.runner, "run_logged", fake_logged)
    monkeypatch.setattr(converge, "build_vasp_wrap",
                        lambda kind, *a, **k: "# stub\n")
    monkeypatch.setattr(converge, "energy_per_atom", lambda d: -5.0)

    src = tmp_path / "sqs"; src.mkdir()
    (src / "str.out").write_text("stub\n")
    e = converge.run_static_point(src, tmp_path / "pt", encut=268, kppra=6000,
                                  cmd_prefix="mpiexec -n 128")
    assert e == -5.0
    assert calls == [
        ["runstruct_vasp", "mpiexec", "-n", "128"]], calls


def test_generate_phase_sqs_raises_when_pass2_still_prompts(
        tmp_path, monkeypatch):
    """The missing-HCP_A3_small incident (2026-07-22): both sqs2tdb -cp
    passes printed the 'Edit the file ... and rerun' prompt (lattice
    absent from $atatdir/data/sqsdb), nothing was generated, and the
    old work_root fallback aliased the HCP phase onto the OTHER
    phases' SQS dirs (log showed 'PHASE HCP_A3 ... 10 SQS directories'
    all living in FCC/BCC). Must raise loudly instead."""
    def fake_run(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        Path(log).write_text("Using species: Co,Cr\nEdit the file "
                             "HCP_A3_small/species.in (if needed) and "
                             "rerun the same command.\n")
        return 0

    monkeypatch.setattr(sqsgen.runner, "run_logged", fake_run)
    with pytest.raises(RuntimeError, match="missing from .atatdir"):
        sqsgen.generate_phase_sqs(tmp_path, "HCP_A3",
                                  elements=["Co", "Cr"])


def test_generate_phase_sqs_never_falls_back_to_work_root(
        tmp_path, monkeypatch):
    """Clean pass-2 log but no target dir: the work-root fallback is
    forbidden (it made discovery reprocess every other phase)."""
    def fake_run(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        Path(log).write_text("Using species: Co,Cr\nCopied SQSs\n")
        return 0    # ...but never creates HCP_A3_small/

    monkeypatch.setattr(sqsgen.runner, "run_logged", fake_run)
    with pytest.raises(RuntimeError, match="never created"):
        sqsgen.generate_phase_sqs(tmp_path, "HCP_A3",
                                  elements=["Co", "Cr"])


# ---- link-only dirs and wait-marker semantics (per sqs2tdb source) ----------

def test_generate_phase_sqs_accepts_link_only_dirs(tmp_path, monkeypatch):
    """sqs2tdb writes only a `link` file (no str.out) for SQS equivalent
    to a permuted-site twin or endmembers reducible to a parent lattice.
    The post-copy verification must accept a tree of link-only dirs."""
    def fake_run(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        tdir = Path(cwd) / "BCC_B2"
        sp = tdir / "species.in"
        if not sp.is_file():
            tdir.mkdir(parents=True, exist_ok=True)
            sp.write_text("Co,Cr\n")
        else:
            d = tdir / "sqs_lev=0_a_Co=1,b_Co=1"
            d.mkdir(parents=True, exist_ok=True)
            (d / "link").write_text("BCC_A2/sqs_lev=0_a_Co=1\n")
        return 0

    monkeypatch.setattr(sqsgen.runner, "run_logged", fake_run)
    out = sqsgen.generate_phase_sqs(tmp_path, "BCC_B2",
                                    elements=["Co", "Cr"], use_small=False)
    assert any(out.rglob("link"))     # verification passed on links alone


def test_process_one_sqs_clears_wait_marker(tmp_path, monkeypatch):
    """sqs2tdb -cp drops a `wait` queue marker in every to-be-computed
    dir; after a successful relax we must remove it (the reference NAS
    workflow's manual `rm wait`)."""
    import types

    sqs = tmp_path / "sqs_lev=2_a_Co=0.5,a_Cr=0.5"
    sqs.mkdir()
    (sqs / "str.out").write_text("stub\n")
    (sqs / "wait").write_text("")

    fake_res = types.SimpleNamespace(table=lambda: "", converged=True)
    monkeypatch.setattr(run_upstream.converge, "converge_sqs",
                        lambda *a, **k: (400, 6000, fake_res, fake_res))

    def fake_relax(calc_dir, **kwargs):
        (Path(calc_dir) / "str_relax.out").write_text(
            "1 0 0\n0 1 0\n0 0 1\n3.5 0 0\n0 3.5 0\n0 0 3.5\n"
            "0 0 0 Co\n")
        return Path(calc_dir) / "str_relax.out"

    monkeypatch.setattr(run_upstream.relax, "relax_structure", fake_relax)

    run_upstream.process_one_sqs(
        sqs, potcar_paths=[], dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="runstruct", algo="All", tol_ev=1e-3,
        env_bin=None, skip_phonon=True, timeout=60)

    assert not (sqs / "wait").exists(), "wait marker must be cleared"
    assert (sqs / "str_relax.out").is_file()


# ---- fitfc orchestration (verified against fitfc.c++ source) ---------------

def test_fitfc_gen_args_harmonic_default():
    """Default recipe = sqs2tdb vibrational workflow: relative radius,
    single volume, no re-relax, ideal structure = relaxed structure."""
    args = phonon.build_fitfc_gen_args(ernn=2.0, er=None, ns=1, ms=0.02,
                                       dr=None, nrr=True)
    assert args[0] == "-si=str_relax.out"
    assert "-ernn=2.0" in args
    assert "-ns=1" in args
    assert "-nrr" in args
    assert not any(a.startswith("-dr") for a in args)  # fitfc default 0.2
    assert not any(a.startswith("-er=") for a in args)


def test_fitfc_gen_args_er_wins_and_requires_radius():
    args = phonon.build_fitfc_gen_args(ernn=2.0, er=5.5, ns=3, ms=0.02,
                                       dr=0.1, nrr=False)
    assert "-er=5.5" in args and not any(a.startswith("-ernn") for a in args)
    assert "-dr=0.1" in args and "-nrr" not in args
    with pytest.raises(ValueError):
        phonon.build_fitfc_gen_args(ernn=None, er=None, ns=1, ms=0.02,
                                    dr=None, nrr=True)


def test_fitfc_fit_args():
    """fitfc -f ERRORQUITs without -fr/-frnn; -si must match generation."""
    args = phonon.build_fitfc_fit_args(frnn=1.5, fr=None)
    assert "-f" in args and "-frnn=1.5" in args and "-si=str_relax.out" in args
    args = phonon.build_fitfc_fit_args(frnn=1.5, fr=4.0)
    assert "-fr=4.0" in args and not any(a.startswith("-frnn") for a in args)
    with pytest.raises(ValueError):
        phonon.build_fitfc_fit_args(frnn=None, fr=None)


def test_run_fitfc_requires_str_relax(tmp_path):
    """fitfc generation dies without str_relax.out; fail fast with a
    message instead of a cryptic fitfc log."""
    with pytest.raises(RuntimeError, match="str_relax.out"):
        phonon.run_fitfc(tmp_path, encut=400, kppra=6000)


def test_run_fitfc_harmonic_single_gen_call(tmp_path, monkeypatch):
    """With -nrr (ns=1 default) fitfc emits vol_0 AND the perturbations in
    ONE invocation, so run_fitfc must call fitfc exactly twice (gen, fit)
    and pollmach exactly once (force runs), and must copy vol_0/svib_ht
    to the top level (the only path sqs2tdb -fit reads)."""
    sqs = tmp_path / "sqs_lev=0_a_Co=1"
    sqs.mkdir()
    (sqs / "str.out").write_text("stub\n")
    (sqs / "str_relax.out").write_text("stub\n")
    (sqs / "energy").write_text("-42.0\n")

    calls = []

    def fake_run_logged(cmd, cwd, log, **kw):
        calls.append(("logged", list(cmd)))
        if cmd[0] == "fitfc" and "-f" not in cmd:
            # simulate -nrr generation: vol_0 + pert dirs in one shot
            vol = Path(cwd) / "vol_0"
            pert = vol / "p+0.2_5.1_0"
            pert.mkdir(parents=True, exist_ok=True)
            (vol / "str.out").write_text("s\n")
            (vol / "str_relax.out").write_text("s\n")
            (pert / "str.out").write_text("s\n")
            (pert / "str_unpert.out").write_text("s\n")
            (pert / "wait").write_text("")
        if "-f" in cmd:
            (Path(cwd) / "vol_0" / "svib_ht").write_text("3.21\n")
            (Path(cwd) / "fitfc.out").write_text("fit\n")
        return 0

    def fake_run_polled(cmd, cwd, log, done_when, **kw):
        calls.append(("polled", list(cmd)))
        for d in Path(cwd).glob("vol_*/p*"):
            (d / "force.out").write_text("0 0 0\n")
            (d / "str_relax.out").write_text("s\n")
        assert done_when(Path(cwd))
        return 0

    monkeypatch.setattr(phonon.runner, "run_logged", fake_run_logged)
    monkeypatch.setattr(phonon.runner, "run_polled", fake_run_polled)

    out = phonon.run_fitfc(sqs, encut=400, kppra=6000,
                           cmd_prefix="mpiexec -n 128")

    fitfc_calls = [c for k, c in calls if k == "logged" and c[0] == "fitfc"]
    assert len(fitfc_calls) == 2, "harmonic path: gen once, fit once"
    assert "-nrr" in fitfc_calls[0] and "-f" in fitfc_calls[1]
    polled = [c for k, c in calls if k == "polled"]
    assert len(polled) == 1, "only the force runs need pollmach"
    assert polled[0][:5] == ["pollmach", "runstruct_vasp",
                             "-lu", "-w", "vaspf.wrap"]
    assert polled[0][-3:] == ["mpiexec", "-n", "128"], "launcher trails"
    # svib_ht promoted top-level for sqs2tdb -fit
    assert (sqs / "svib_ht").read_text() == "3.21\n"
    # -nrr: static energy seeded into vol_0 so fitfc.out has a T=0 term
    assert (sqs / "vol_0" / "energy").read_text() == "-42.0\n"
    assert out == sqs / "fitfc.out"


def test_run_fitfc_quasiharmonic_two_gen_calls(tmp_path, monkeypatch):
    """ns>1 without -nrr follows fitfc's two-invocation contract: gen
    (vol_* + wait), relax vols with a per-vol ISIF=2 wrap (removed
    afterwards so p* runs inherit the frozen top wrap), gen again with
    the SAME args, force runs, fit."""
    sqs = tmp_path / "sqs"
    sqs.mkdir()
    (sqs / "str.out").write_text("stub\n")
    (sqs / "str_relax.out").write_text("stub\n")

    fitfc_gen_calls = []

    def fake_run_logged(cmd, cwd, log, **kw):
        if cmd[0] == "fitfc" and "-f" not in cmd:
            fitfc_gen_calls.append(list(cmd))
            for name in ("vol_0", "vol_2"):
                vol = Path(cwd) / name
                vol.mkdir(exist_ok=True)
                (vol / "str.out").write_text("s\n")
                if (vol / "str_relax.out").is_file():
                    pert = vol / "p+0.2_5.1_0"
                    pert.mkdir(exist_ok=True)
                    (pert / "str.out").write_text("s\n")
                    (pert / "str_unpert.out").write_text("s\n")
                else:
                    (vol / "wait").write_text("")
        if "-f" in cmd:
            (Path(cwd) / "vol_0" / "svib_ht").write_text("3.3\n")
        return 0

    def fake_run_polled(cmd, cwd, log, done_when, **kw):
        for vol in Path(cwd).glob("vol_*"):
            if not (vol / "str_relax.out").is_file():
                # vol relax stage: relax wrap must be present NOW
                assert (vol / "vasp.wrap").is_file()
                assert "ISIF = 2" in (vol / "vasp.wrap").read_text()
                (vol / "str_relax.out").write_text("s\n")
                (vol / "energy").write_text("-1.0\n")
            for d in vol.glob("p*"):
                (d / "force.out").write_text("0\n")
                (d / "str_relax.out").write_text("s\n")
        return 0

    monkeypatch.setattr(phonon.runner, "run_logged", fake_run_logged)
    monkeypatch.setattr(phonon.runner, "run_polled", fake_run_polled)

    phonon.run_fitfc(sqs, encut=400, kppra=6000, ns=3)

    assert len(fitfc_gen_calls) == 2
    assert fitfc_gen_calls[0] == fitfc_gen_calls[1], \
        "fitfc must be re-run with the SAME command-line options"
    assert "-nrr" not in fitfc_gen_calls[0] and "-ns=3" in fitfc_gen_calls[0]
    # per-vol relax wraps removed; force runs use the separate top-level
    # vaspf.wrap (selected with -w), never the relax-stage vasp.wrap
    for vol in sqs.glob("vol_*"):
        assert not (vol / "vasp.wrap").is_file()
    assert (sqs / "vaspf.wrap").is_file()
    assert (sqs / "svib_ht").read_text() == "3.3\n"


def test_promote_svib_ht_noop_without_source(tmp_path):
    assert phonon.promote_svib_ht(tmp_path) is None
    (tmp_path / "vol_0").mkdir()
    (tmp_path / "vol_0" / "svib_ht").write_text("1.1\n")
    dst = phonon.promote_svib_ht(tmp_path)
    assert dst == tmp_path / "svib_ht" and dst.read_text() == "1.1\n"


# ---- fitfc unstable-mode safeguards ----------------------------------------

def test_fitfc_fit_args_rl_and_fn():
    args = phonon.build_fitfc_fit_args(frnn=1.5, fr=None, rl=0.3, fn=True)
    assert "-rl=0.3" in args and "-fn" in args


def test_detect_unstable_modes(tmp_path):
    log = tmp_path / "fitfc_fit.log"
    assert phonon.detect_unstable_modes(log) == []  # missing log = no fit
    log.write_text("Reading...\nvol_0\n"
                   "Warning: p+0.2_5.1_3 is an unstable mode.\n"
                   "Unstable modes found.\nAborting.\n")
    hits = phonon.detect_unstable_modes(log)
    assert len(hits) == 2 and "Unstable modes found." in hits[1]


def _unstable_sqs(tmp_path):
    """SQS dir with STALE fit outputs from a previous (good) run — the
    refit will hit unstable modes and must not resurrect these."""
    sqs = tmp_path / "sqs_lev=1_a_Co=0.5,a_Cr=0.5"
    (sqs / "vol_0").mkdir(parents=True)
    (sqs / "str.out").write_text("stub\n")
    (sqs / "str_relax.out").write_text("stub\n")
    (sqs / "svib_ht").write_text("STALE\n")
    (sqs / "fitfc.out").write_text("STALE\n")
    (sqs / "vol_0" / "svib_ht").write_text("STALE\n")
    return sqs


def test_run_fitfc_unstable_mark_no_stale_svib(tmp_path, monkeypatch):
    """Default policy: unstable fit aborts -> stale svib_ht cleared, NOT
    promoted; unstable_modes.log written; no -fn retry."""
    sqs = _unstable_sqs(tmp_path)
    fit_cmds = []

    def fake_run_logged(cmd, cwd, log, **kw):
        if cmd[0] == "fitfc" and "-f" not in cmd:
            vol = Path(cwd) / "vol_0"
            pert = vol / "p+0.2_5.1_0"
            pert.mkdir(parents=True, exist_ok=True)
            (vol / "str_relax.out").write_text("s\n")
            (pert / "str_unpert.out").write_text("s\n")
            (pert / "wait").write_text("")
        if "-f" in cmd:
            fit_cmds.append(list(cmd))
            Path(log).write_text(
                "Warning: p+0.2_5.1_0 is an unstable mode.\n"
                "Unstable modes found.\nAborting.\n")  # no svib_ht written
        return 0

    def fake_run_polled(cmd, cwd, log, done_when, **kw):
        for d in Path(cwd).glob("vol_*/p*"):
            (d / "force.out").write_text("0\n")
            (d / "str_relax.out").write_text("s\n")
        return 0

    monkeypatch.setattr(phonon.runner, "run_logged", fake_run_logged)
    monkeypatch.setattr(phonon.runner, "run_polled", fake_run_polled)

    phonon.run_fitfc(sqs, encut=400, kppra=6000)  # on_unstable="mark"

    assert len(fit_cmds) == 1 and "-fn" not in fit_cmds[0]
    assert not (sqs / "svib_ht").exists(), "stale svib_ht must be gone"
    assert not (sqs / "vol_0" / "svib_ht").exists()
    marker = sqs / "unstable_modes.log"
    assert marker.is_file() and "energy-only" in marker.read_text()
    assert "unstable" in (sqs / "fitfc_fit.log").read_text().lower()


def test_run_fitfc_unstable_force_retries_with_fn(tmp_path, monkeypatch):
    """on_unstable='force': one retry with -fn; the forced svib_ht is
    promoted and the marker records the -fn provenance."""
    sqs = _unstable_sqs(tmp_path)
    fit_cmds = []

    def fake_run_logged(cmd, cwd, log, **kw):
        if cmd[0] == "fitfc" and "-f" not in cmd:
            vol = Path(cwd) / "vol_0"
            pert = vol / "p+0.2_5.1_0"
            pert.mkdir(parents=True, exist_ok=True)
            (vol / "str_relax.out").write_text("s\n")
            (pert / "wait").write_text("")
        if "-f" in cmd:
            fit_cmds.append(list(cmd))
            if "-fn" in cmd:
                Path(log).write_text("Unstable modes found.\n")
                (Path(cwd) / "vol_0" / "svib_ht").write_text("2.2\n")
                (Path(cwd) / "fitfc.out").write_text("forced\n")
            else:
                Path(log).write_text("Unstable modes found.\nAborting.\n")
        return 0

    def fake_run_polled(cmd, cwd, log, done_when, **kw):
        for d in Path(cwd).glob("vol_*/p*"):
            (d / "force.out").write_text("0\n")
            (d / "str_relax.out").write_text("s\n")
        return 0

    monkeypatch.setattr(phonon.runner, "run_logged", fake_run_logged)
    monkeypatch.setattr(phonon.runner, "run_polled", fake_run_polled)

    phonon.run_fitfc(sqs, encut=400, kppra=6000, on_unstable="force")

    assert len(fit_cmds) == 2
    assert "-fn" not in fit_cmds[0] and "-fn" in fit_cmds[1]
    assert (sqs / "svib_ht").read_text() == "2.2\n", "forced svib promoted"
    marker = (sqs / "unstable_modes.log").read_text()
    assert "-fn" in marker and "lower bound" in marker


def test_run_fitfc_rl_passthrough_and_bad_policy(tmp_path, monkeypatch):
    sqs = tmp_path / "sqs"
    sqs.mkdir()
    (sqs / "str.out").write_text("s\n")
    (sqs / "str_relax.out").write_text("s\n")
    with pytest.raises(ValueError, match="on_unstable"):
        phonon.run_fitfc(sqs, encut=400, kppra=6000, on_unstable="ignore")

    fit_cmds = []

    def fake_run_logged(cmd, cwd, log, **kw):
        if "-f" in cmd:
            fit_cmds.append(list(cmd))
        return 0

    monkeypatch.setattr(phonon.runner, "run_logged", fake_run_logged)
    monkeypatch.setattr(phonon.runner, "run_polled",
                        lambda *a, **k: 0)
    phonon.run_fitfc(sqs, encut=400, kppra=6000, rl=0.3)
    assert any("-rl=0.3" in c for cmd in fit_cmds for c in cmd)


def _escalation_fakes(monkeypatch, resolves: bool):
    """Fake runner where the first fit is unstable; the escalated fit
    succeeds iff `resolves`. Returns (gen_cmds, fit_cmds, forced_pert)."""
    gen_cmds, fit_cmds, forced_pert = [], [], []

    def fake_run_logged(cmd, cwd, log, **kw):
        cwd = Path(cwd)
        if cmd[0] == "fitfc" and "-f" not in cmd:
            gen_cmds.append(list(cmd))
            vol = cwd / "vol_0"
            er = [a for a in cmd if a.startswith(("-er", "-ernn"))][0]
            radius = er.split("=")[1]
            pert = vol / f"p+0.2_{radius}_0"
            pert.mkdir(parents=True, exist_ok=True)
            (vol / "str_relax.out").write_text("s\n")
            (pert / "str_unpert.out").write_text("s\n")
            (pert / "wait").write_text("")
        if "-f" in cmd:
            fit_cmds.append(list(cmd))
            if len(fit_cmds) == 1 or not resolves:
                Path(log).write_text("Unstable modes found.\nAborting.\n")
            else:
                Path(log).write_text("fitted fine\n")
                (cwd / "vol_0" / "svib_ht").write_text("4.4\n")
                (cwd / "fitfc.out").write_text("ok\n")
        return 0

    def fake_run_polled(cmd, cwd, log, done_when, **kw):
        for d in Path(cwd).glob("vol_*/p*"):
            if not (d / "force.out").is_file():
                forced_pert.append(d.name)
                (d / "force.out").write_text("0\n")
                (d / "str_relax.out").write_text("s\n")
        return 0

    monkeypatch.setattr(phonon.runner, "run_logged", fake_run_logged)
    monkeypatch.setattr(phonon.runner, "run_polled", fake_run_polled)
    return gen_cmds, fit_cmds, forced_pert


def test_run_fitfc_escalate_resolves(tmp_path, monkeypatch):
    """escalate: regenerate at 1.5x -ernn, force-run ONLY the new p*
    dirs, refit; resolved -> escalated svib_ht promoted + marker says so."""
    sqs = tmp_path / "sqs"
    sqs.mkdir()
    (sqs / "str.out").write_text("s\n")
    (sqs / "str_relax.out").write_text("s\n")
    gen_cmds, fit_cmds, forced_pert = _escalation_fakes(monkeypatch, True)

    phonon.run_fitfc(sqs, encut=400, kppra=6000, on_unstable="escalate")

    assert len(gen_cmds) == 2, "one normal gen + one escalated gen"
    assert "-ernn=4.0" in gen_cmds[0] and "-ernn=6.0" in gen_cmds[1]
    assert len(fit_cmds) == 2 and all("-fn" not in c for c in fit_cmds)
    # the original pert kept its force.out; only the new dir was run
    assert forced_pert.count("p+0.2_4.0_0") == 1
    assert forced_pert.count("p+0.2_6.0_0") == 1
    assert (sqs / "svib_ht").read_text() == "4.4\n"
    marker = (sqs / "unstable_modes.log").read_text()
    assert "RESOLVED" in marker and "-ernn=6.0" in marker


def test_run_fitfc_escalate_persists_marks_energy_only(tmp_path, monkeypatch):
    """escalate, still unstable: no svib_ht, no -fn fallback; marker
    calls it likely genuine dynamical instability with manual options."""
    sqs = tmp_path / "sqs"
    sqs.mkdir()
    (sqs / "str.out").write_text("s\n")
    (sqs / "str_relax.out").write_text("s\n")
    gen_cmds, fit_cmds, _ = _escalation_fakes(monkeypatch, False)

    phonon.run_fitfc(sqs, encut=400, kppra=6000, on_unstable="escalate",
                     escalate_ernn=4.0)

    assert "-ernn=4.0" in gen_cmds[1], "explicit escalate_ernn honoured"
    assert len(fit_cmds) == 2 and all("-fn" not in c for c in fit_cmds)
    assert not (sqs / "svib_ht").exists()
    marker = (sqs / "unstable_modes.log").read_text()
    assert "PERSISTS" in marker and "genuine dynamical instability" in marker
    assert "-fu" in marker and "-rl" in marker, "manual options named"


# ---- spin polarization (advisor review F1) ---------------------------------

def test_wrap_spin_flag_sets_ispin2():
    w = vaspwrap.build_vasp_wrap("static", encut=400, kppra=6000, spin=True)
    assert "ISPIN = 2" in w
    w = vaspwrap.build_vasp_wrap("static", encut=400, kppra=6000, spin=False)
    assert "ISPIN" not in w


def test_wrap_spin_module_default(monkeypatch):
    """run_upstream sets vaspwrap.DEFAULT_SPIN once; every wrap written by
    converge/relax/phonon then inherits it without explicit plumbing."""
    monkeypatch.setattr(vaspwrap, "DEFAULT_SPIN", True)
    for mode in ("static", "relax", "phonon"):
        assert "ISPIN = 2" in vaspwrap.build_vasp_wrap(mode, encut=400,
                                                       kppra=6000)
    monkeypatch.setattr(vaspwrap, "DEFAULT_SPIN", False)
    assert "ISPIN" not in vaspwrap.build_vasp_wrap("static", encut=400,
                                                   kppra=6000)


def test_wrap_spin_skipped_for_dlm():
    """DLM spin handling comes from SUBATOM moments (ezvasp emits the
    magnetic INCAR); no bare ISPIN=2 on top."""
    dlm = DLMConfig(enabled=True, subatom={"Co": ("Co", 1.8)})
    w = vaspwrap.build_vasp_wrap("static", encut=400, kppra=6000,
                                 dlm=dlm, spin=True)
    assert "ISPIN" not in w
    assert "NUPDOWN = 0" in w and "SUBATOM = s/Co+2/Co+1.8/g" in w


def test_wants_spin_detects_magnetic_3d():
    assert vaspwrap.wants_spin(["Co", "Cr"])
    assert vaspwrap.wants_spin(["Al", "Ni"])
    assert not vaspwrap.wants_spin(["Al", "Ti"])


# ---- Pulay ENCUT floor + phase-uniform convergence (review F2/F4) ----------

def test_pulay_safe_encut():
    # 1.3 x 267.882 = 348.2 -> ceil to 350; sweep choice below the floor
    # is raised, above it is kept.
    assert potcar.pulay_safe_encut(300, 267.882) == 350
    assert potcar.pulay_safe_encut(400, 267.882) == 400
    assert potcar.pulay_safe_encut(350, 267.882) == 350


def test_process_one_sqs_preset_skips_sweep_and_bumps_relax_encut(
        tmp_path, monkeypatch):
    """Preset (phase-scope) settings must skip converge_sqs entirely, and
    the relax wrap must get the Pulay-floored ENCUT while the record
    keeps the sweep-chosen one for statics/phonons."""
    sqs = tmp_path / "sqs_lev=1_a_Co=0.5,a_Cr=0.5"
    sqs.mkdir()
    (sqs / "str.out").write_text("stub\n")
    pot = tmp_path / "POTCAR"
    pot.write_text(POTCAR_CO)   # ENMAX 267.882 -> floor 350

    def boom(*a, **k):
        raise AssertionError("convergence sweep must not run with preset")

    monkeypatch.setattr(run_upstream.converge, "converge_sqs", boom)

    seen = {}

    def fake_relax(calc_dir, encut, kppra, **kwargs):
        seen["encut"], seen["kppra"] = encut, kppra
        (Path(calc_dir) / "str_relax.out").write_text("relaxed\n")
        return Path(calc_dir) / "str_relax.out"

    monkeypatch.setattr(run_upstream.relax, "relax_structure", fake_relax)

    res = run_upstream.process_one_sqs(
        sqs, potcar_paths=[pot], dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="runstruct", algo="All", tol_ev=1e-3,
        env_bin=None, skip_phonon=True, timeout=60,
        preset_encut=300, preset_kppra=6000)

    assert res["convergence_reused"] is True
    assert res["chosen_encut"] == 300 and res["relax_encut"] == 350
    assert seen["encut"] == 350, "relax must use the Pulay-floored ENCUT"
    assert seen["kppra"] == 6000


# ---- lattice-drift metric + gate (review F3) --------------------------------

from strfile import parse_cell, cell_distortion, lattice_drift


def _write_str(path, cell_rows, coord="1 0 0\n0 1 0\n0 0 1"):
    rows = "\n".join(" ".join(f"{v}" for v in r) for r in cell_rows)
    path.write_text(f"{coord}\n{rows}\n0 0 0 Co\n0.5 0.5 0.5 Cr\n")


def test_parse_cell_both_header_layouts(tmp_path):
    p = tmp_path / "a.out"
    _write_str(p, [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])
    c = parse_cell(read_structure(p))
    assert c[0][0] == pytest.approx(3.5) and c[1][2] == pytest.approx(0.0)
    # (a b c alpha beta gamma) header with identity lattice rows
    q = tmp_path / "b.out"
    q.write_text("3.5 3.5 3.5 90 90 90\n1 0 0\n0 1 0\n0 0 1\n0 0 0 Co\n")
    c2 = parse_cell(read_structure(q))
    assert c2[0][0] == pytest.approx(3.5)
    assert c2[2][2] == pytest.approx(3.5)


def test_cell_distortion_invariances():
    ident = [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]]
    # pure volume change is NOT drift
    scaled = [[3.9, 0, 0], [0, 3.9, 0], [0, 0, 3.9]]
    assert cell_distortion(ident, scaled) == pytest.approx(0.0, abs=1e-10)
    # rigid rotation is NOT drift (90 deg about z)
    rot = [[0, 3.5, 0], [-3.5, 0, 0], [0, 0, 3.5]]
    assert cell_distortion(ident, rot) == pytest.approx(0.0, abs=1e-10)
    # shear IS drift
    shear = [[3.5, 1.4, 0], [0, 3.5, 0], [0, 0, 3.5]]
    assert cell_distortion(ident, shear) > 0.1


def test_process_one_sqs_flags_relaxed_away(tmp_path, monkeypatch):
    """A relax that shears the cell beyond max_checkrelax must leave
    checkrelax.out + relaxaway.flag and mark the manifest record."""
    import types
    sqs = tmp_path / "sqs_lev=1_a_Co=0.5,a_Cr=0.5"
    sqs.mkdir()
    _write_str(sqs / "str.out", [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])

    fake_res = types.SimpleNamespace(table=lambda: "", converged=True)
    monkeypatch.setattr(run_upstream.converge, "converge_sqs",
                        lambda *a, **k: (400, 6000, fake_res, fake_res))

    def fake_relax(calc_dir, **kwargs):
        _write_str(Path(calc_dir) / "str_relax.out",
                   [[3.5, 1.4, 0], [0, 3.5, 0], [0, 0, 3.5]])
        return Path(calc_dir) / "str_relax.out"

    monkeypatch.setattr(run_upstream.relax, "relax_structure", fake_relax)

    res = run_upstream.process_one_sqs(
        sqs, potcar_paths=[], dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="runstruct", algo="All", tol_ev=1e-3,
        env_bin=None, skip_phonon=True, timeout=60)

    assert (sqs / "checkrelax.out").is_file()
    assert res["checkrelax"] > 0.1 and res["relaxed_away"] is True
    assert (sqs / "relaxaway.flag").is_file()
    # and a faithful relax must NOT be flagged
    sqs2 = tmp_path / "sqs_lev=2_a_Co=0.25,a_Cr=0.75"
    sqs2.mkdir()
    _write_str(sqs2 / "str.out", [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])

    def faithful_relax(calc_dir, **kwargs):
        _write_str(Path(calc_dir) / "str_relax.out",
                   [[3.62, 0, 0], [0, 3.62, 0], [0, 0, 3.62]])
        return Path(calc_dir) / "str_relax.out"

    monkeypatch.setattr(run_upstream.relax, "relax_structure",
                        faithful_relax)
    res2 = run_upstream.process_one_sqs(
        sqs2, potcar_paths=[], dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="runstruct", algo="All", tol_ev=1e-3,
        env_bin=None, skip_phonon=True, timeout=60)
    assert res2["relaxed_away"] is False
    assert not (sqs2 / "relaxaway.flag").exists()


# ---- NAS smoke suite (dry-run only — VASP paths run on the node) -----------

def test_nas_smoke_dry_run_builds_all_call_paths(tmp_path):
    sys.path.insert(0, str(PKG / "nas_smoke"))
    import run_smoke
    rc = run_smoke.main(["--dry-run", "--workdir", str(tmp_path),
                         "--element", "Co",
                         "--cmd-prefix", "mpiexec -n 8"])
    assert rc == 0
    import json
    plan = {c["test"]: c for c in
            json.loads((tmp_path / "plan.json").read_text())}
    assert set(plan) == {"T1_static", "T2_runstruct", "T3_robustrelax",
                         "T4_fitfc_wrap", "T5_pollmach"}
    # each distinct call form, launcher trailing
    assert plan["T1_static"]["argv"] == ["runstruct_vasp",
                                         "mpiexec", "-n", "8"]
    assert plan["T3_robustrelax"]["pre_argv"] == ["robustrelax_vasp", "-mk"]
    assert plan["T3_robustrelax"]["argv"][:4] == \
        ["robustrelax_vasp", "-id", "-c", "0.05"]
    assert plan["T4_fitfc_wrap"]["argv"][:4] == ["runstruct_vasp", "-lu",
                                                 "-w", "vaspf.wrap"]
    assert plan["T5_pollmach"]["argv"][:2] == ["pollmach", "runstruct_vasp"]
    # inputs on disk: frozen wrap under the separate name, wait markers,
    # displaced structure for the force run
    assert (tmp_path / "T4_fitfc_wrap" / "vaspf.wrap").is_file()
    assert "NSW = 0" in (tmp_path / "T4_fitfc_wrap" / "vaspf.wrap").read_text()
    assert "0.52" in (tmp_path / "T4_fitfc_wrap" / "str.out").read_text()
    assert (tmp_path / "T5_pollmach" / "p_1" / "wait").is_file()
    assert (tmp_path / "T5_pollmach" / "vasp.wrap").is_file()
    # smoke wraps are tiny-run tuned and spin-off (plumbing test)
    w = (tmp_path / "T1_static" / "vasp.wrap").read_text()
    assert "NELM = 25" in w and "NCORE = 1" in w and "ISPIN" not in w


# ---- CoCr-run regression fixes (diagnosed from real NAS outputs) -----------

from strfile import validate_structure_file


def test_parallel_overrides_small_cells():
    assert vaspwrap.parallel_overrides(1) == {"NCORE": 1, "KPAR": 1}
    assert vaspwrap.parallel_overrides(8) == {"NCORE": 2, "KPAR": 2}
    assert vaspwrap.parallel_overrides(30) == {}
    assert vaspwrap.parallel_overrides(None) == {}
    w = vaspwrap.build_vasp_wrap("static", encut=300, kppra=1000, natoms=1)
    assert "NCORE = 1" in w and "KPAR = 1" in w
    w = vaspwrap.build_vasp_wrap("static", encut=300, kppra=1000, natoms=30)
    assert "NCORE = 8" in w and "KPAR = 4" in w


def test_parallel_overrides_rank_aware_kpar():
    """Small cells are k-point dominated (1 atom @ KPPRA 10000 = hundreds
    of IBZ k-points). With the rank count known, the idle band
    parallelism is recovered as KPAR — each k-group's band split stays
    at or below the hardware-validated single-group layout."""
    # 32 ranks, 1-atom probe cell: KPAR=8 -> 4 ranks/group, NCORE=1
    assert vaspwrap.parallel_overrides(1, ranks=32) == \
        {"NCORE": 1, "KPAR": 8}
    # KPAR must divide the rank count (VASP aborts otherwise)
    assert vaspwrap.parallel_overrides(1, ranks=2) == \
        {"NCORE": 1, "KPAR": 2}
    assert vaspwrap.parallel_overrides(1, ranks=6) == \
        {"NCORE": 1, "KPAR": 6}
    assert vaspwrap.parallel_overrides(1, ranks=1) == \
        {"NCORE": 1, "KPAR": 1}
    # mid tier caps at 4
    assert vaspwrap.parallel_overrides(8, ranks=32) == \
        {"NCORE": 2, "KPAR": 4}
    # big cells keep the reference NCORE=8/KPAR=4 regardless of ranks
    assert vaspwrap.parallel_overrides(30, ranks=32) == {}
    w = vaspwrap.build_vasp_wrap("static", encut=300, kppra=10000,
                                 natoms=1, ranks=32)
    assert "NCORE = 1" in w and "KPAR = 8" in w


def test_ranks_from_prefix():
    assert vaspwrap.ranks_from_prefix("mpiexec -n 32") == 32
    assert vaspwrap.ranks_from_prefix("mpirun -np 128") == 128
    assert vaspwrap.ranks_from_prefix("") is None
    assert vaspwrap.ranks_from_prefix("mpiexec") is None
    assert vaspwrap.ranks_from_prefix("mpiexec -n lots") is None


def test_static_point_returns_cached_energy(tmp_path, monkeypatch):
    """A sweep point with an energy already on disk (from a killed
    submission) must be read back, not rerun — resubmission after the
    2026-07-20 early-termination fix relies on this fast-forward."""
    def boom(*a, **k):
        raise AssertionError("VASP must not be launched for cached point")
    monkeypatch.setattr(converge.runner, "run_logged", boom)

    dst = tmp_path / "encut_301"
    dst.mkdir()
    (dst / "energy").write_text("-12.4\n")
    (dst / "str.out").write_text(
        "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n"
        "0 0 0 Co\n0.5 0.5 0.5 Co\n")
    e = converge.run_static_point(tmp_path / "sqs", dst,
                                  encut=301, kppra=7000)
    assert e == -6.2                      # -12.4 eV / 2 atoms, cached


def test_sweep_early_terminates_on_successive_rule(tmp_path, monkeypatch):
    """Once chosen + confirmation satisfy the successive rule, trailing
    grid points cannot change the answer (the rule takes the FIRST
    qualifying triple) — they must be SKIPPED, not computed."""
    energies = {4000: -6.0000, 5000: -6.0010, 6000: -6.00105,
                7000: -6.00104, 8000: -6.5, 9000: -6.6, 10000: -6.7}
    ran = []

    def fake_point(src, dst, encut, kppra, **kw):
        ran.append(kppra)
        return energies[kppra]

    monkeypatch.setattr(converge, "run_static_point", fake_point)
    res = converge.run_sweep("KPPRA", tmp_path, tmp_path / "sw",
                             settings=sorted(energies), fixed_other=300)
    assert ran == [4000, 5000, 6000, 7000]     # 8000+ never run
    assert res.converged and res.rule == "successive"
    assert res.chosen == 6000 and res.reference == 7000
    assert res.settings == [4000, 5000, 6000, 7000]


def test_sweep_noise_floor_still_runs_full_grid(tmp_path, monkeypatch):
    """The plateau fallback must NOT terminate the sweep early — it only
    engages after the whole base grid has failed the pointwise rule
    (otherwise it could preempt a later, stricter successive hit)."""
    vals = [-6.00000, -6.00015, -6.00000, -6.00015,
            -6.00000, -6.00015, -6.00000]        # +-0.15 meV noise
    grid = [4000 + 1000 * i for i in range(7)]
    energies = dict(zip(grid, vals))
    ran = []

    def fake_point(src, dst, encut, kppra, **kw):
        ran.append(kppra)
        return energies[kppra]

    monkeypatch.setattr(converge, "run_static_point", fake_point)
    res = converge.run_sweep("KPPRA", tmp_path, tmp_path / "sw",
                             settings=grid, fixed_other=300)
    assert ran == grid                           # every point computed
    assert res.converged and res.rule == "plateau"
    assert res.chosen == 4000


def test_validate_structure_file_catches_degenerate_stub(tmp_path):
    """Exact stub robustrelax/infdet left behind in the real Co-Cr run
    after its inner VASP crashed."""
    stub = tmp_path / "str_relax.out"
    stub.write_text("1 0 0\n0 1 0\n0 0 1\n\tCo\n")
    ok, why = validate_structure_file(stub)
    assert not ok and "no atom lines" in why
    ok, why = validate_structure_file(tmp_path / "nope.out")
    assert not ok and why == "missing"
    good = tmp_path / "good.out"
    good.write_text("1 0 0\n0 1 0\n0 0 1\n3.5 0 0\n0 3.5 0\n0 0 3.5\n"
                    "0 0 0 Co\n")
    ok, why = validate_structure_file(good)
    assert ok and "1 atoms" in why


def test_process_one_sqs_failed_relax_skips_phonons(tmp_path, monkeypatch):
    import types
    sqs = tmp_path / "sqs_lev=0_a_Co=1"
    sqs.mkdir()
    _write_str(sqs / "str.out", [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])
    (sqs / "wait").write_text("")

    fake_res = types.SimpleNamespace(table=lambda: "", converged=True)
    monkeypatch.setattr(run_upstream.converge, "converge_sqs",
                        lambda *a, **k: (400, 6000, fake_res, fake_res))

    def stub_relax(calc_dir, **kwargs):
        (Path(calc_dir) / "str_relax.out").write_text(
            "1 0 0\n0 1 0\n0 0 1\n\tCo\n")     # the degenerate stub
        (Path(calc_dir) / "energy").write_text("")
        return Path(calc_dir) / "str_relax.out"

    monkeypatch.setattr(run_upstream.relax, "relax_structure", stub_relax)
    monkeypatch.setattr(run_upstream.phonon, "run_fitfc",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("phonons must not run")))

    res = run_upstream.process_one_sqs(
        sqs, potcar_paths=[], dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="infdet", algo="All", tol_ev=1e-3,
        env_bin=None, skip_phonon=False, timeout=60)

    assert res["relax_ok"] is False and "no atom lines" in res["relax_msg"]
    assert res["phonon_out"] is None
    assert res["energy_present"] is False
    assert (sqs / "wait").exists(), "wait must NOT be cleared on failure"


def test_process_one_sqs_infdet_failure_never_adopts_energy_end(
        tmp_path, monkeypatch):
    """Per the robustrelax_vasp source: energy_end is the FULLY-RELAXED
    (decayed) structure's energy — adopting it as the result is exactly
    the error inflection detection exists to prevent. An engaged-but-
    incomplete infdet run (energy_end present, no normal termination)
    is a FAILURE: no adoption, relax_ok False, phonons skipped."""
    import types
    sqs = tmp_path / "sqs_lev=1_a_Co=0.5,a_Cr=0.5"
    sqs.mkdir()
    _write_str(sqs / "str.out", [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])

    fake_res = types.SimpleNamespace(table=lambda: "", converged=True)
    monkeypatch.setattr(run_upstream.converge, "converge_sqs",
                        lambda *a, **k: (400, 6000, fake_res, fake_res))

    def infdet_relax(calc_dir, **kwargs):
        _write_str(Path(calc_dir) / "str_relax.out",
                   [[3.6, 0, 0], [0, 3.6, 0], [0, 0, 3.6]])
        (Path(calc_dir) / "energy").write_text("")            # empty!
        (Path(calc_dir) / "energy_end").write_text("-.13017812E+03\n")
        return Path(calc_dir) / "str_relax.out"

    monkeypatch.setattr(run_upstream.relax, "relax_structure", infdet_relax)

    res = run_upstream.process_one_sqs(
        sqs, potcar_paths=[], dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="infdet", algo="All", tol_ev=1e-3,
        env_bin=None, skip_phonon=True, timeout=60)

    assert res["infdet_engaged"] is True and res["infdet_ok"] is False
    assert res["relax_ok"] is False
    assert res["energy_present"] is False        # energy_end NOT adopted
    assert "infdet incomplete" in res["relax_msg"]


def test_process_one_sqs_infdet_success_waives_drift_gate(
        tmp_path, monkeypatch):
    """User directive 2026-07-20: large checkrelax does NOT mean the
    robustrelax workflow failed. Success = energy present + 01/
    infdet.log ends with 'infdet terminated normally'. Such an SQS must
    keep relax_ok, get infdet_ok.flag (downstream gate waiver) and NO
    relaxaway.flag despite drift over the threshold."""
    import types
    sqs = tmp_path / "sqs_lev=1_a_Co=0.5,a_Cr=0.5"
    sqs.mkdir()
    _write_str(sqs / "str.out", [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])

    fake_res = types.SimpleNamespace(table=lambda: "", converged=True)
    monkeypatch.setattr(run_upstream.converge, "converge_sqs",
                        lambda *a, **k: (400, 6000, fake_res, fake_res))

    def infdet_relax(calc_dir, **kwargs):
        # inflection geometry with LARGE drift vs str.out (> 0.1)
        _write_str(Path(calc_dir) / "str_relax.out",
                   [[4.4, 0, 0], [0, 3.5, 0], [0, 0, 3.2]])
        (Path(calc_dir) / "energy").write_text("-129.8\n")
        (Path(calc_dir) / "energy_end").write_text("-130.2\n")
        d01 = Path(calc_dir) / "01"
        d01.mkdir()
        (d01 / "infdet.log").write_text(
            "vasp\nwaiting\ninfdet terminated normally\n")
        return Path(calc_dir) / "str_relax.out"

    monkeypatch.setattr(run_upstream.relax, "relax_structure", infdet_relax)

    res = run_upstream.process_one_sqs(
        sqs, potcar_paths=[], dlm=DLMConfig(enabled=False, subatom={}),
        relax_method="infdet", algo="All", tol_ev=1e-3,
        env_bin=None, skip_phonon=True, timeout=60)

    assert res["infdet_engaged"] and res["infdet_ok"]
    assert res["relax_ok"] is True and res["energy_present"] is True
    assert res["checkrelax"] is not None and res["checkrelax"] > 0.1
    assert not (sqs / "relaxaway.flag").exists()
    assert (sqs / "infdet_ok.flag").is_file()
    assert res["relaxed_away"] is False


def test_vasp_triage_reads_gz_and_suffixed_logs(tmp_path):
    import gzip as _gzip
    import vasp_triage
    d = tmp_path / "case"
    d.mkdir()
    with _gzip.open(d / "OUTCAR.static.gz", "wt") as fh:
        fh.write("|  ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ..."
                 " BYE!!! <----  |\n")
    (d / "vasp.out.static").write_text(
        " POSCAR found :  0 types and       0 ions\n")
    reports = vasp_triage.scan_tree(tmp_path)
    assert len(reports) == 1
    ids = {f.error_id for f in reports[0].findings}
    assert "sick_job" in ids and "empty_poscar" in ids
    assert reports[0].outcar_seen is True


# ---- MAGMOM generation (user correction 2026-07-15) -------------------------

def test_wrap_spin_writes_uniform_magmom_and_mixing():
    """Spin-on wraps must carry an explicit MAGMOM (the 2026-07-14 run's
    OUTCARs all warned about the missing tag) — uniform init moment via
    VASP multiplier syntax, order-independent — plus the production
    INCAR's magnetic-mixing keys."""
    w = vaspwrap.build_vasp_wrap("relax", encut=400, kppra=6000,
                                 spin=True, natoms=32)
    assert "ISPIN = 2" in w and "MAGMOM = 32*3" in w
    assert "AMIX = 0.03" in w and "AMIX_MAG = 0.8" in w
    w = vaspwrap.build_vasp_wrap("relax", encut=400, kppra=6000,
                                 spin=True, natoms=8, magmom_init=1.5)
    assert "MAGMOM = 8*1.5" in w


def test_wrap_spin_without_natoms_omits_magmom():
    w = vaspwrap.build_vasp_wrap("static", encut=400, kppra=6000, spin=True)
    assert "ISPIN = 2" in w and "MAGMOM" not in w


def test_wrap_dlm_keeps_subatom_route_no_magmom():
    dlm = DLMConfig(enabled=True, subatom={"Co": ("Co", 1.8)})
    w = vaspwrap.build_vasp_wrap("relax", encut=400, kppra=6000,
                                 dlm=dlm, spin=True, natoms=32)
    assert "MAGMOM" not in w and "SUBATOM = s/Co+2/Co+1.8/g" in w


# ---- endmember end-to-end verifier (nas_smoke/run_endmember_e2e.py) --------

def _mk_e2e_endmember(root, name, good=True, svib=True):
    d = root / "FCC_A1_small" / name
    d.mkdir(parents=True)
    _write_str(d / "str.out", [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])
    if good:
        _write_str(d / "str_relax.out",
                   [[3.55, 0, 0], [0, 3.55, 0], [0, 0, 3.55]])
        (d / "energy").write_text("-7.1\n")
        (d / "vasp.wrap").write_text("[INCAR]\nISPIN = 2\nMAGMOM = 1*3\n")
        (d / "INCAR.relax").write_text("ISPIN = 2\nMAGMOM = 1*3\n")
        (d / "robustrelax_infdet.log").write_text(
            "$ robustrelax_vasp -id -c 0.05 mpiexec -n 32\n")
        (d / "checkrelax.out").write_text("0.012\n")
        (d / "vaspf.wrap").write_text("[INCAR]\nNSW = 0\n")
        pert = d / "vol_0" / "p+0.2_5.1_0"
        pert.mkdir(parents=True)
        (pert / "force.out").write_text("0 0 0\n")
        if svib:
            (d / "svib_ht").write_text("3.4\n")
        else:
            (d / "unstable_modes.log").write_text(
                "Unstable modes found.\nDisposition: energy-only\n")
    else:
        (d / "str_relax.out").write_text("1 0 0\n0 1 0\n0 0 1\n\tCo\n")
        (d / "energy").write_text("")
        (d / "vasp.wrap").write_text("[INCAR]\nISPIN = 2\n")   # no MAGMOM
        (d / "robustrelax_infdet.log").write_text(
            "$ robustrelax_vasp -id mpiexec -n 128\n")         # no -c
    return d


def test_e2e_verifier_grades_good_tree(tmp_path):
    sys.path.insert(0, str(PKG / "nas_smoke"))
    import run_endmember_e2e as e2e
    _mk_e2e_endmember(tmp_path, "sqs_lev=0_a_Co=1", good=True, svib=True)
    _mk_e2e_endmember(tmp_path, "sqs_lev=0_a_Cr=1", good=True, svib=False)
    results, ok = e2e.verify_tree(tmp_path)
    assert ok and len(results) == 2
    cr = next(r for r in results if "Cr" in r["dir"])
    # unstable disposition counts as correct phonon-machinery behavior
    assert cr["checks"]["svib_or_disposition"]["pass"]
    assert "energy-only by policy" in \
        cr["checks"]["svib_or_disposition"]["detail"]


def test_e2e_verifier_fails_regressed_tree(tmp_path):
    sys.path.insert(0, str(PKG / "nas_smoke"))
    import run_endmember_e2e as e2e
    _mk_e2e_endmember(tmp_path, "sqs_lev=0_a_Co=1", good=True)
    bad = _mk_e2e_endmember(tmp_path, "sqs_lev=0_a_Cr=1", good=False)
    results, ok = e2e.verify_tree(tmp_path)
    assert not ok
    r = next(x for x in results if x["dir"] == str(bad))
    c = r["checks"]
    assert not c["str_relax_valid"]["pass"]      # degenerate stub caught
    assert not c["energy_present"]["pass"]       # empty energy caught
    assert not c["wrap_spin_magmom"]["pass"]     # missing MAGMOM caught
    assert not c["infdet_with_cutoff"]["pass"]   # missing -c 0.05 caught
    assert not c["checkrelax_recorded"]["pass"]
    # report writing works and flags the suite as FAIL
    e2e.write_report(tmp_path, results, ok, {"timestamp": "t"})
    txt = (tmp_path / "e2e_report.txt").read_text()
    assert "SUITE: FAIL" in txt and "XX str_relax_valid" in txt


def test_triage_tet_not_triggered_by_routine_bzints(tmp_path):
    """'BZINTS: Fermi energy: 17.9; 18 electrons' is routine ISMEAR=-5
    output, not an error (false positive on the 2026-07-16 smoke run);
    only genuine tetrahedron failures may flag kpoints_tet."""
    import vasp_triage
    d = tmp_path / "case"
    d.mkdir()
    (d / "vasp.out").write_text(
        " BZINTS: Fermi energy:   17.925019;   18.000000 electrons\n")
    reports = vasp_triage.scan_tree(tmp_path)
    assert all("kpoints_tet" not in r.categories for r in reports)
    (d / "vasp.out").write_text("Tetrahedron method fails (number of "
                                "k-points < 4)\n")
    reports = vasp_triage.scan_tree(tmp_path)
    assert any("kpoints_tet" in r.categories for r in reports)



# ---- raw-sqsdb pollution regression (2026-07-16 e2e failure) ----------------

def test_discover_sqs_dirs_ignores_raw_sqsdb_entries(tmp_path):
    """A stray copy of the ATAT database inside the work root must not
    be treated as calculations: raw entries are undecorated
    (sqsdb_lev=0_a=1); only element-decorated dirs count."""
    root = tmp_path / "FCC_A1_small"
    for name in ("sqsdb_lev=0_a=1", "sqsdb_lev=1_a=0.5,0.5",
                 "sqsdb_lev=3_a=0.33333,0.33333,0.33333"):
        d = root / name
        d.mkdir(parents=True)
        (d / "str.out").write_text("raw db entry\n")
    good = root / "sqs_lev=0_a_Co=1"
    good.mkdir()
    (good / "str.out").write_text("decorated\n")
    found = run_upstream.discover_sqs_dirs(root)
    assert found == [good]


def test_generate_phase_sqs_rejects_raw_db_only_output(tmp_path,
                                                       monkeypatch):
    """If -cp copies nothing but a raw database tree sits in the work
    root, verification must FAIL (the old any-str.out check passed)."""
    def fake(cmd, cwd, log, env_bin=None, timeout=None, check=True):
        tdir = Path(cwd) / "FCC_A1_small"
        sp = tdir / "species.in"
        if not sp.is_file():
            tdir.mkdir(parents=True, exist_ok=True)
            sp.write_text("Co,Cr\n")
            raw = tdir / "sqsdb_lev=0_a=1"     # raw DB pollution only
            raw.mkdir()
            (raw / "str.out").write_text("raw\n")
        return 0
    monkeypatch.setattr(sqsgen.runner, "run_logged", fake)
    with pytest.raises(RuntimeError, match="no element-decorated"):
        sqsgen.generate_phase_sqs(tmp_path, "FCC_A1",
                                  elements=["Co", "Cr"], level=0)


def test_e2e_find_endmembers_skips_raw_db_dirs(tmp_path):
    sys.path.insert(0, str(PKG / "nas_smoke"))
    import run_endmember_e2e as e2e
    _mk_e2e_endmember(tmp_path, "sqs_lev=0_a_Co=1", good=True)
    raw = tmp_path / "FCC_A1_small" / "sqsdb_lev=0_a=1"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "str.out").write_text("raw db\n")
    dirs = e2e.find_endmember_dirs(tmp_path)
    assert [d.name for d in dirs] == ["sqs_lev=0_a_Co=1"]



# ---- vaspf.wrap sized for the perturbation supercell (2026-07-16) ----------

def test_force_wrap_magmom_matches_pert_supercell(tmp_path, monkeypatch):
    """VASP 6.6: MAGMOM must have exactly NIONS values. The force runs
    execute in the perturbation SUPERCELL (8 atoms for a 1-atom FCC
    endmember at -ernn=2), so vaspf.wrap must be sized from the p* dirs,
    not the SQS cell — the e2e run died on "1 value(s) for MAGMOM ...
    NIONS=8" when it was sized from the 1-atom cell."""
    monkeypatch.setattr(vaspwrap, "DEFAULT_SPIN", True)
    sqs = tmp_path / "sqs_lev=0_a_Co=1"
    sqs.mkdir()
    _write_str(sqs / "str.out", [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]])
    (sqs / "str_relax.out").write_text(
        "1 0 0\n0 1 0\n0 0 1\n3.5 0 0\n0 3.5 0\n0 0 3.5\n0 0 0 Co\n")

    def fake_run_logged(cmd, cwd, log, **kw):
        if cmd[0] == "fitfc" and "-f" not in cmd:
            pert = Path(cwd) / "vol_0" / "p+0.2_5.1_0"
            pert.mkdir(parents=True, exist_ok=True)
            rows = "2 0 0\n0 2 0\n0 0 2\n"     # 8-atom supercell
            atoms = "\n".join(f"0 0 {i} Co" for i in range(8))
            (pert / "str.out").write_text(
                "1 0 0\n0 1 0\n0 0 1\n" + rows + atoms + "\n")
            (pert / "wait").write_text("")
            (Path(cwd) / "vol_0" / "str_relax.out").write_text("s\n")
        if "-f" in cmd:
            (Path(cwd) / "vol_0" / "svib_ht").write_text("1.0\n")
        return 0

    def fake_run_polled(cmd, cwd, log, done_when, **kw):
        # the wrap must ALREADY be sized for the supercell when the
        # force runs launch
        wrap = (Path(cwd) / "vaspf.wrap").read_text()
        assert "MAGMOM = 8*3" in wrap, wrap
        assert "MAGMOM = 1*3" not in wrap
        for d in Path(cwd).glob("vol_*/p*"):
            (d / "force.out").write_text("0\n")
            (d / "str_relax.out").write_text("s\n")
        return 0

    monkeypatch.setattr(phonon.runner, "run_logged", fake_run_logged)
    monkeypatch.setattr(phonon.runner, "run_polled", fake_run_polled)

    phonon.run_fitfc(sqs, encut=400, kppra=6000)
    wrap = (sqs / "vaspf.wrap").read_text()
    assert "MAGMOM = 8*3" in wrap and "ISPIN = 2" in wrap



# ---- adaptive ENCUT extension (no convergence ceiling) ----------------------

def test_run_sweep_extends_until_converged(tmp_path, monkeypatch):
    """ENCUT sweep must climb past the initial grid until the
    successive-difference + confirmation criterion is met."""
    # plateau only reached at 400: initial grid [300..335] can't satisfy
    energies = {300: -5.020, 317: -5.010, 335: -5.004,
                352: -5.001, 369: -5.00030, 386: -5.00025, 403: -5.00022}

    def fake_point(src, dst, encut, kppra, **kw):
        return energies[encut]

    monkeypatch.setattr(converge, "run_static_point", fake_point)
    res = converge.run_sweep(
        "ENCUT", tmp_path / "sqs", tmp_path / "sweep",
        settings=[300, 317, 335], fixed_other=7000,
        tol_ev=0.0001, extend_step=17, extend_max=500)
    assert res.converged is True
    assert res.chosen == 386                # 369->386 and 386->403 < tol
    assert res.reference == 403             # confirmation point
    assert res.settings[-1] == 403          # extended past the old grid


def test_run_sweep_extension_hits_guard_and_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(converge, "run_static_point",
                        lambda src, dst, encut, kppra, **kw:
                        -5.0 - encut * 1e-4)   # never converges
    res = converge.run_sweep(
        "ENCUT", tmp_path / "sqs", tmp_path / "sweep",
        settings=[300, 320], fixed_other=7000,
        tol_ev=0.0001, extend_step=20, extend_max=400)
    assert res.converged is False
    assert res.settings[-1] <= 400          # guard respected
    assert res.chosen == res.settings[-1]   # falls back to highest



# ---- ALGO modes + convergence-probe protocol (2026-07-17) -------------------

def test_normalize_algo_modes():
    assert vaspwrap.normalize_algo("all") == "All"
    assert vaspwrap.normalize_algo("NORMAL") == "Normal"
    assert vaspwrap.normalize_algo("VeryFast") == "VeryFast"
    for bad in ("Fast", "damped", ""):
        with pytest.raises(ValueError):
            vaspwrap.normalize_algo(bad)


def test_site_fractions_and_rich_side_picks(tmp_path):
    import random
    names = ["sqs_lev=0_a_Co=1", "sqs_lev=0_a_Cr=1",
             "sqs_lev=1_a_Co=0.5,a_Cr=0.5",
             "sqs_lev=2_a_Co=0.75,a_Cr=0.25",
             "sqs_lev=2_a_Co=0.25,a_Cr=0.75"]
    dirs = []
    for n in names:
        d = tmp_path / n
        d.mkdir()
        dirs.append(d)
    fr = run_upstream.site_fractions("sqs_lev=2_a_Co=0.75,a_Cr=0.25")
    assert fr["Co"] == pytest.approx(0.75) and fr["Cr"] == pytest.approx(0.25)

    picks = run_upstream.pick_probe_dirs(dirs, ["Co", "Cr"],
                                         random.Random(0))
    # rich = strictly > 0.5: endmember or 0.75 side; NEVER the midpoint
    assert set(picks) == {"Co", "Cr"}
    assert run_upstream.site_fractions(picks["Co"].name)["Co"] > 0.5
    assert run_upstream.site_fractions(picks["Cr"].name)["Cr"] > 0.5
    assert "0.5,a_Cr=0.5" not in picks["Co"].name
    # deterministic under a fixed seed
    again = run_upstream.pick_probe_dirs(dirs, ["Co", "Cr"],
                                         random.Random(0))
    assert picks == again


def test_system_probe_takes_max_and_pulay_floor(tmp_path, monkeypatch):
    """Probe protocol: sweep one rich-side SQS per element, take the
    elementwise MAX, fold in the Pulay floor -> ONE global (ENCUT,
    KPPRA) for every energy/relax/infdet/phonon run."""
    import types
    root = tmp_path

    def fake_gen(work_root, phase, elements=None, level=None, dlm=False,
                 env_bin=None, **kw):
        pr = Path(work_root) / f"{phase}_small"
        for n in ("sqs_lev=0_a_Co=1", "sqs_lev=0_a_Cr=1",
                  "sqs_lev=1_a_Co=0.5,a_Cr=0.5"):
            d = pr / n
            d.mkdir(parents=True, exist_ok=True)
            (d / "str.out").write_text("s\n")
        return pr

    res = types.SimpleNamespace(table=lambda: "", converged=True)
    sweeps = {}

    def fake_converge(src, sweep_root, pots, **kw):
        # Cr probe demands more than Co probe
        if "Cr=1" in Path(src).name:
            sweeps["cr"] = True
            return 420, 8000, res, res
        sweeps["co"] = True
        return 360, 7000, res, res

    monkeypatch.setattr(run_upstream.sqsgen, "generate_phase_sqs", fake_gen)
    monkeypatch.setattr(run_upstream.converge, "converge_sqs", fake_converge)
    monkeypatch.setattr(run_upstream.potcar, "max_enmax", lambda p: 268.0)

    probe = run_upstream.system_probe_convergence(
        root, ["FCC_A1", "SIGMA_D8B"], ["Co", "Cr"],
        potcar_paths=["x"], dlm=DLMConfig(enabled=False, subatom={}),
        algo="All", tol_ev=1e-4, sqs_levels=[2], env_bin=None,
        timeout=60, cmd_prefix="", seed=0)

    assert sweeps == {"co": True, "cr": True}, "one sweep per rich side"
    assert probe["kppra"] == 8000                    # max over probes
    # max encut 420 already exceeds the 1.3 x 268 = 350 Pulay floor
    assert probe["encut"] == 420
    assert probe["phase"] == "FCC_A1"                # only 1-sublattice pick
    assert set(probe["probes"]) == {"Co", "Cr"}


def test_system_probe_none_without_single_sublattice(tmp_path):
    out = run_upstream.system_probe_convergence(
        tmp_path, ["SIGMA_D8B"], ["Co", "Cr"], potcar_paths=[],
        dlm=DLMConfig(enabled=False, subatom={}), algo="All",
        tol_ev=1e-4, sqs_levels=[0], env_bin=None, timeout=60,
        cmd_prefix="", seed=0)
    assert out is None


def test_e2e_probe_check_reads_manifest(tmp_path):
    sys.path.insert(0, str(PKG / "nas_smoke"))
    import json as _json
    import run_endmember_e2e as e2e
    r = e2e.verify_probe(tmp_path)
    assert not r["pass"] and "missing" in r["detail"]
    (tmp_path / "upstream_manifest.json").write_text(_json.dumps({
        "system_probe": {"phase": "FCC_A1", "seed": 0,
                         "encut": 420, "kppra": 8000,
                         "probes": {"Co": {"dir": "x", "encut": 360,
                                           "kppra": 7000},
                                    "Cr": {"dir": "y", "encut": 420,
                                           "kppra": 8000}}}}))
    r = e2e.verify_probe(tmp_path)
    assert r["pass"]
    assert "GLOBAL ENCUT=420" in r["detail"] and "KPPRA=8000" in r["detail"]



# ---- noise-floor plateau fallback (2026-07-17, real FCC-Co sweep) ----------

_REAL_ENCUT_S = [268, 285, 301, 318, 335, 352, 369, 386, 403, 420, 437,
                 454, 471, 488, 505, 522, 539, 556, 573, 590, 607, 624,
                 641, 658, 675, 692, 709, 726, 743, 760, 777]
_REAL_ENCUT_E = [-6.2498443, -6.2580797, -6.2616170, -6.2612131,
                 -6.2597746, -6.2572824, -6.2552779, -6.2535186,
                 -6.2540986, -6.2528565, -6.2520979, -6.2520142,
                 -6.2516278, -6.2519271, -6.2514212, -6.2515857,
                 -6.2515278, -6.2518319, -6.2523146, -6.2518121,
                 -6.2514408, -6.2517460, -6.2513132, -6.2515541,
                 -6.2511311, -6.2512216, -6.2515604, -6.2517727,
                 -6.2515866, -6.2516322, -6.2517257]


def test_run_sweep_real_noisy_data_stops_via_plateau(tmp_path, monkeypatch):
    """The 2026-07-16 e2e run: past ~437 eV the energy only fluctuates
    in a ~0.5 meV band (noise floor of PREC=Normal/LREAL=Auto statics);
    the pointwise 0.1 meV rule only fired at 760 eV on a noise
    coincidence. Incrementally, the plateau fallback must terminate the
    sweep at 488 and choose 437."""
    table = dict(zip(_REAL_ENCUT_S, _REAL_ENCUT_E))
    monkeypatch.setattr(converge, "run_static_point",
                        lambda src, dst, encut, kppra, **kw: table[encut])
    res = converge.run_sweep(
        "ENCUT", tmp_path / "sqs", tmp_path / "sweep",
        settings=_REAL_ENCUT_S[:5], fixed_other=7000,
        tol_ev=0.0001, extend_step=17, extend_max=804)
    assert res.converged is True
    assert res.rule == "plateau"
    assert res.chosen == 437
    assert res.reference == 488             # plateau window end
    assert res.settings[-1] == 488, \
        "sweep must STOP at the plateau, not wander to 760 on noise"


def test_plateau_never_preempts_successive_rule():
    """Plateau is a FALLBACK: on the KPPRA data the pointwise rule
    fires (7000) and the plateau band must not undercut it to 4000."""
    ks = [4000, 5000, 6000, 7000, 8000, 9000, 10000]
    ke = [-6.2614504, -6.2615232, -6.2616992, -6.2616170, -6.2616170,
          -6.2617104, -6.2615474]
    chosen, conv, ref, rule = converge.select_converged(ks, ke,
                                                        tol_ev=0.0001)
    assert (chosen, rule) == (7000, "successive")


def test_sweep_statics_use_high_precision_wrap(tmp_path, monkeypatch):
    """Sweep points must run PREC=Accurate + LREAL=.FALSE. — the noise
    that defeated the pointwise rule came from PREC=Normal FFT-grid
    jumps and LREAL=Auto projector re-optimization."""
    seen = {}

    def fake_wrap(mode, **kw):
        seen.update(kw.get("extra") or {})
        return "# wrap\n"

    monkeypatch.setattr(converge, "build_vasp_wrap", fake_wrap)
    monkeypatch.setattr(converge.runner, "run_logged",
                        lambda *a, **k: 0)
    monkeypatch.setattr(converge, "energy_per_atom", lambda d: -5.0)
    src = tmp_path / "sqs"
    src.mkdir()
    (src / "str.out").write_text("1 0 0\n0 1 0\n0 0 1\n3 0 0\n0 3 0\n"
                                 "0 0 3\n0 0 0 Co\n")
    converge.run_static_point(src, tmp_path / "pt", encut=300, kppra=7000)
    assert seen.get("PREC") == "Accurate"
    assert seen.get("LREAL") == ".FALSE."


def test_cr_kppra_noise_band_stops_on_initial_grid():
    """Cr-rich probe KPPRA data from the 2026-07-16 17:31 run (pre-fix
    code extended toward 20000 and the job died mid-extension): the
    whole 4000-10000 grid is a +-0.17 meV noise band with NO successive
    triple. The plateau fallback must terminate on the INITIAL grid —
    zero extension points — and the global max over probes still takes
    KPPRA from the Co side (7000)."""
    ks = [4000, 5000, 6000, 7000, 8000, 9000, 10000]
    ke = [-8.0796959, -8.0794469, -8.0796206, -8.0797775, -8.0797775,
          -8.0795921, -8.0798591]
    chosen, conv, ref, rule = converge.select_converged(ks, ke,
                                                        tol_ev=0.0001)
    assert conv is True and rule == "plateau"
    assert chosen == 4000 and ref == 7000
    # old behavior (plateau disabled): not converged -> endless extension
    _c, conv_off, _r, _rule = converge.select_converged(
        ks, ke, tol_ev=0.0001, plateau_band_ev=0)
    assert conv_off is False


# ---- PBS fan-out broker (--submit pbs, 2026-07-17) --------------------------

import pbsjobs


def test_sizing_and_launcher_retarget():
    assert pbsjobs.size_for(1, "relax")["ncpus"] == 8
    assert pbsjobs.size_for(30, "relax")["ncpus"] == 32
    assert pbsjobs.size_for(100, "force")["ncpus"] == 64
    assert pbsjobs.size_for(None, "relax")["ncpus"] == 64  # unknown->big
    cmd = ["robustrelax_vasp", "-id", "-c", "0.05", "mpiexec", "-n", "128"]
    out = pbsjobs.retarget_launcher(cmd, 16)
    assert out[-3:] == ["mpiexec", "-n", "16"]
    assert out[:4] == cmd[:4]                 # options untouched


def test_render_single_job_script(tmp_path):
    b = pbsjobs.Broker(work_root=tmp_path, dry_run=True,
                       site_env="source /x/job_env.sh", queue="normal",
                       model="mil_ait")
    d = tmp_path / "sqs"
    d.mkdir()
    script = b.render_single("relax", d,
                             ["robustrelax_vasp", "-id", "-c", "0.05",
                              "mpiexec", "-n", "999"], 8, "01:00:00")
    t = script.read_text()
    assert "#PBS -l select=1:ncpus=8:mpiprocs=8:model=mil_ait" in t
    assert "#PBS -l walltime=01:00:00" in t
    assert "#PBS -q normal" in t
    assert "#PBS -W group_list=a1485" in t
    assert "source /x/job_env.sh" in t
    assert "mpiexec -n 8" in t                # retargeted to job ncpus
    assert "-n 999" not in t
    assert ".qrc_relax" in t


def test_render_array_one_element_per_pert_dir(tmp_path):
    b = pbsjobs.Broker(work_root=tmp_path, dry_run=True)
    sqs = tmp_path / "sqs"
    dirs = []
    for i in range(28):
        d = sqs / "vol_0" / f"p+0.2_9.9_{i}"
        d.mkdir(parents=True)
        dirs.append(d)
    script = b.render_array("forces", sqs,
                            ["runstruct_vasp", "-lu", "-w", "vaspf.wrap",
                             "mpiexec", "-n", "1"], dirs, 32, "01:00:00")
    t = script.read_text()
    assert "#PBS -J 0-27" in t                # 28 wall-parallel elements
    assert "PBS_ARRAY_INDEX" in t
    manifest = (sqs / "qjob_forces.dirs").read_text().splitlines()
    assert len(manifest) == 28 and manifest[0].endswith("p+0.2_9.9_0")
    assert "mpiexec -n 32" in t


def test_run_as_job_array_for_pollmach_force_runs(tmp_path):
    """A pollmach force-run command with >=2 work dirs becomes a job
    ARRAY with pollmach stripped (each element runs runstruct directly
    in its own dir — no shared machine to poll inside a job)."""
    b = pbsjobs.Broker(work_root=tmp_path, dry_run=True)
    sqs = tmp_path / "sqs"
    dirs = []
    for i in range(3):
        d = sqs / "vol_0" / f"p+0.2_9.9_{i}"
        d.mkdir(parents=True)
        (d / "force.out").write_text("0\n")   # already done -> rc 0
        dirs.append(d)
    rc = b.run_as_job(
        "fitfc_force_runs", sqs,
        ["pollmach", "runstruct_vasp", "-lu", "-w", "vaspf.wrap",
         "mpiexec", "-n", "1"],
        done_when=lambda _c: all((d / "force.out").is_file()
                                 for d in dirs),
        work_dirs=dirs, natoms=32, kind="force", done_file="force.out")
    assert rc == 0
    t = (sqs / "qjob_fitfc_force_runs.pbs").read_text()
    assert "#PBS -J 0-2" in t
    assert "pollmach" not in t                # stripped inside the job
    assert "runstruct_vasp -lu -w vaspf.wrap" in t


def test_run_as_job_retries_then_fails(tmp_path, monkeypatch):
    """A job that leaves the queue without outputs is resubmitted up to
    max_retries, then reported failed."""
    b = pbsjobs.Broker(work_root=tmp_path, max_retries=1,
                       poll_interval=0)
    subs = []
    monkeypatch.setattr(b, "qsub", lambda script: subs.append(str(script))
                        or f"job{len(subs)}")
    monkeypatch.setattr(b, "alive", lambda jid: False)   # dies instantly
    monkeypatch.setattr(b, "n_inflight", lambda: 0)
    monkeypatch.setattr(pbsjobs.time, "sleep", lambda s: None)
    d = tmp_path / "sqs"
    d.mkdir()
    rc = b.run_as_job("relax", d, ["robustrelax_vasp", "-id"],
                      done_when=lambda _c: False,
                      work_dirs=[d], natoms=1, kind="relax")
    assert rc == -1
    assert len(subs) == 2                     # original + 1 retry
    import json as _json
    marker = _json.loads((d / ".qjob_relax").read_text())
    assert marker["job_id"] == "job2"


def test_runner_backend_routes_run_polled(tmp_path):
    """With a broker installed, run_polled must delegate with the work
    metadata instead of launching locally."""
    calls = {}

    class FakeBroker:
        def run_as_job(self, tag, cwd, cmd, done_when, work_dirs=None,
                       natoms=None, kind="generic", done_file="energy"):
            calls.update(tag=tag, cmd=list(cmd), natoms=natoms,
                         kind=kind, work_dirs=list(work_dirs or []))
            return 0

    import runner as _runner
    _runner.set_backend(FakeBroker())
    try:
        rc = _runner.run_polled(
            ["robustrelax_vasp", "-id", "-c", "0.05"],
            cwd=tmp_path, log=tmp_path / "robustrelax_infdet.log",
            done_when=lambda c: True,
            work_dirs=[tmp_path], natoms=30, kind="relax",
            done_file="str_relax.out")
    finally:
        _runner.set_backend(None)
    assert rc == 0
    assert calls["tag"] == "robustrelax_infdet"
    assert calls["kind"] == "relax" and calls["natoms"] == 30


def test_probe_worker_argv_strips_orchestration_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_upstream.py", "--element1", "Co",
                         "--element2", "Cr", "--submit", "pbs",
                         "--job-env", "/x/env.sh", "--job-queue",
                         "normal", "--no-job-arrays",
                         "--potcars", "/p/Co,/p/Cr"])
    argv = run_upstream._probe_worker_argv()
    assert "--submit" not in argv and "pbs" not in argv
    assert "--job-env" not in argv and "/x/env.sh" not in argv
    assert "--no-job-arrays" not in argv
    assert argv[:4] == ["--element1", "Co", "--element2", "Cr"]
    assert "--potcars" in argv


# ---- refine.py: fit-error mesh refinement + adaptive svib (2026-07-22) -----

import refine


def test_parse_fit_energy_and_worst_point(tmp_path):
    f = tmp_path / "fit_energy.out"
    f.write_text("# comment\n"
                 "0.00  -6.900 -6.899\n"
                 "0.25  -7.100 -7.080\n"     # err 20 meV  <-- worst
                 "0.50  -7.300 -7.295\n"
                 "1.00  -8.000 -8.001\n")
    rows = refine.parse_fit_energy(f)
    assert len(rows) == 4
    x_star, err = refine.worst_fit_point(rows)
    assert x_star == 0.25 and err == pytest.approx(0.020)


def test_refinement_targets_bracket_worst_point():
    xs = [0.0, 0.25, 0.5, 1.0]
    # midpoints toward BOTH neighbours of x*=0.25
    assert refine.refinement_targets(xs, 0.25) == [0.125, 0.375]
    # endmember-adjacent: only one side exists
    assert refine.refinement_targets(xs, 0.0) == [0.125]


def test_select_new_dirs_marks_unchosen(tmp_path):
    """Freshly generated dirs not bracketing the worst point get
    refine_skip; discovery must then ignore them."""
    def mk(name, computed=False):
        d = tmp_path / name
        d.mkdir()
        (d / "str.out").write_text("s\n")
        if computed:
            (d / "energy").write_text("-1\n")
        return d

    mk("sqs_lev=0_a_Co=1", computed=True)          # existing: untouched
    d125 = mk("sqs_lev=3_a_Co=0.875,a_Cr=0.125")   # x_Cr = 0.125
    d375 = mk("sqs_lev=3_a_Co=0.625,a_Cr=0.375")   # x_Cr = 0.375
    d625 = mk("sqs_lev=3_a_Co=0.375,a_Cr=0.625")   # x_Cr = 0.625 (extra)

    chosen = refine.select_new_dirs(tmp_path, [0.125, 0.375], "Cr")
    assert chosen["0.1250"] == d125.name
    assert chosen["0.3750"] == d375.name
    assert (d125 / "refine_pick").is_file()
    assert (d625 / "refine_skip").is_file()
    assert not (tmp_path / "sqs_lev=0_a_Co=1" / "refine_skip").exists()

    disc = run_upstream.discover_sqs_dirs(tmp_path)
    names = {d.name for d in disc}
    assert d625.name not in names and d125.name in names


def _svib_dir(root, name, natoms, svib=None, energy=None):
    d = root / name
    d.mkdir()
    coord = "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n"
    atoms = "".join(f"0 0 {i} Co\n" for i in range(natoms))
    (d / "str.out").write_text(coord + atoms)
    if svib is not None:
        (d / "svib_ht").write_text(f"{svib}\n")
    if energy is not None:
        (d / "energy").write_text(f"{energy}\n")
    return d


def test_adaptive_svib_linear_holds(tmp_path):
    """|dev| <= tol: linearity kept, NO lev=2 phonons purchased."""
    _svib_dir(tmp_path, "sqs_lev=0_a_Co=1", 1, svib=3.0)
    _svib_dir(tmp_path, "sqs_lev=0_a_Cr=1", 1, svib=4.0)
    _svib_dir(tmp_path, "sqs_lev=1_a_Co=0.5,a_Cr=0.5", 4,
              svib=4 * 3.52)                     # per atom 3.52 vs 3.5
    bought = []
    out = refine.adaptive_svib_phase(tmp_path, "Cr", bought.append,
                                     tol=0.1, log=lambda *_: None)
    assert bought == []
    assert out["model"]["kind"] == "linear"
    assert "HOLDS" in out["decision"]
    assert (tmp_path / "svib_adaptive.json").is_file()


def test_adaptive_svib_refuted_buys_stable_side_first(tmp_path):
    """Refuted linearity: phonons bought on the LOWER-mixing-energy
    side; quadratic kept when the 4-point RMSE beats the lev=1
    prediction error at the new point."""
    _svib_dir(tmp_path, "sqs_lev=0_a_Co=1", 1, svib=3.0, energy=-7.0)
    _svib_dir(tmp_path, "sqs_lev=0_a_Cr=1", 1, svib=4.0, energy=-9.0)
    # strongly non-linear midpoint: dev = 4.3 - 3.5 = 0.8 > tol
    _svib_dir(tmp_path, "sqs_lev=1_a_Co=0.5,a_Cr=0.5", 4,
              svib=4 * 4.3, energy=4 * -8.5)
    # Cr-rich side has the lower (more negative) mixing energy:
    #   e_mix(0.75) = -8.9 - (-8.5)  = -0.4 ;  e_mix(0.25) = +0.1
    d25 = _svib_dir(tmp_path, "sqs_lev=2_a_Co=0.75,a_Cr=0.25", 4,
                    energy=4 * -7.4)
    d75 = _svib_dir(tmp_path, "sqs_lev=2_a_Co=0.25,a_Cr=0.75", 4,
                    energy=4 * -8.9)

    quad = lambda x: 3.0 + 2.4 * x - 1.9 * x * x   # noqa: E731
    # exact quadratic through the 3 base points (3.0, 4.3, 3.5):
    # a=3.0, b=... solve: at .5: a+.5b+.25c=4.3; at 1: a+b+c=4.0
    # b=2.1? compute inside test instead:
    def run_phonon(d):
        x = refine.composition_fraction(d.name, "Cr")
        # value ON the exact 3-point quadratic -> rmse ~ 0 -> keep
        import numpy as np
        A = np.array([[1, 0, 0], [1, .5, .25], [1, 1, 1]], float)
        a, b, c = np.linalg.solve(A, np.array([3.0, 4.3, 4.0]))
        s = a + b * x + c * x * x
        (Path(d) / "svib_ht").write_text(f"{4 * s}\n")

    out = refine.adaptive_svib_phase(tmp_path, "Cr", run_phonon,
                                     tol=0.1, log=lambda *_: None)
    assert out["tried"][0]["x"] == 0.75      # stable side first
    assert (d75 / "svib_ht").is_file()
    assert not (d25 / "svib_ht").exists()    # other side never bought
    assert out["model"]["kind"] == "quadratic"
    assert out["model"]["rmse"] < 0.05


def test_adaptive_svib_falls_back_to_other_side(tmp_path):
    """If the first side's 4-point quadratic fits WORSE than the lev=1
    prediction error, the other lev=2 side is computed and the lower-
    RMSE fit wins (user spec)."""
    _svib_dir(tmp_path, "sqs_lev=0_a_Co=1", 1, svib=3.0, energy=-7.0)
    _svib_dir(tmp_path, "sqs_lev=0_a_Cr=1", 1, svib=4.0, energy=-9.0)
    _svib_dir(tmp_path, "sqs_lev=1_a_Co=0.5,a_Cr=0.5", 4,
              svib=4 * 4.3, energy=4 * -8.5)
    _svib_dir(tmp_path, "sqs_lev=2_a_Co=0.75,a_Cr=0.25", 4,
              energy=4 * -7.4)
    _svib_dir(tmp_path, "sqs_lev=2_a_Co=0.25,a_Cr=0.75", 4,
              energy=4 * -8.9)

    def run_phonon(d):
        x = refine.composition_fraction(d.name, "Cr")
        # first (stable, x=0.75) side: WILD outlier -> bad quadratic;
        # second side (x=0.25): on-model value
        s = 20.0 if x == 0.75 else 3.9
        (Path(d) / "svib_ht").write_text(f"{4 * s}\n")

    out = refine.adaptive_svib_phase(tmp_path, "Cr", run_phonon,
                                     tol=0.1, log=lambda *_: None)
    assert [t["x"] for t in out["tried"]] == [0.75, 0.25]
    assert out["model"]["kind"] == "quadratic"
    # the kept fit is the second side's (lower RMSE)
    assert out["tried"][1]["quad4_rmse"] < out["tried"][0]["quad4_rmse"]
    assert out["model"]["rmse"] == pytest.approx(
        out["tried"][1]["quad4_rmse"])
