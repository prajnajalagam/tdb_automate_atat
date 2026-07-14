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
            d = tdir / "sqsdb_lev=1_a_0.5_b_0.5"
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
    with pytest.raises(RuntimeError, match="no str.out"):
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
              env_bin=None, timeout=None, check=True):
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
        lambda kind, encut, kppra, dlm=None, algo="All": f"# stub {kind}\n",
    )


def test_runstruct_is_now_the_default(tmp_path, monkeypatch):
    """--relax-method default must be 'runstruct' (was 'normal')."""
    rec = _RecCalls()
    _stub_encut_kppra(monkeypatch, tmp_path)
    monkeypatch.setattr(relax.runner, "run_logged", rec.logged_fn())
    monkeypatch.setattr(relax.runner, "run_polled", rec.polled_fn())
    relax.relax_structure(tmp_path, encut=400, kppra=8000)  # no method= arg
    # runstruct: no -mk prep, one polled pollmach runstruct_vasp call.
    assert rec.logged == [], f"runstruct should not run robustrelax_vasp -mk; got {rec.logged}"
    assert rec.polled == [["pollmach", "runstruct_vasp"]], rec.polled


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
    assert rec.polled == [["robustrelax_vasp", "-id", "-idop", "-t 1e-3"]], rec.polled


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
    relax.relax_structure(tmp_path, encut=400, kppra=8000)
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
                        lambda kind, encut, kppra, dlm=None, algo="All":
                        "# stub\n")
    monkeypatch.setattr(converge, "energy_per_atom", lambda d: -5.0)

    src = tmp_path / "sqs"; src.mkdir()
    (src / "str.out").write_text("stub\n")
    e = converge.run_static_point(src, tmp_path / "pt", encut=268, kppra=6000,
                                  cmd_prefix="mpiexec -n 128")
    assert e == -5.0
    assert calls == [
        ["runstruct_vasp", "mpiexec", "-n", "128"]], calls


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
        (Path(calc_dir) / "str_relax.out").write_text("relaxed\n")
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
    assert polled[0][:2] == ["pollmach", "runstruct_vasp"]
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
    # per-vol relax wraps removed so p* force runs see the frozen top wrap
    for vol in sqs.glob("vol_*"):
        assert not (vol / "vasp.wrap").is_file()
    assert (sqs / "vasp.wrap").is_file()
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
    assert "-ernn=2.0" in gen_cmds[0] and "-ernn=3.0" in gen_cmds[1]
    assert len(fit_cmds) == 2 and all("-fn" not in c for c in fit_cmds)
    # the original pert kept its force.out; only the new dir was run
    assert forced_pert.count("p+0.2_2.0_0") == 1
    assert forced_pert.count("p+0.2_3.0_0") == 1
    assert (sqs / "svib_ht").read_text() == "4.4\n"
    marker = (sqs / "unstable_modes.log").read_text()
    assert "RESOLVED" in marker and "-ernn=3.0" in marker


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
