"""Unit tests for count_ml_frames.py — run: python3 -m pytest tools/ -q"""
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_ml_frames as cmf


def _outcar_text(elements, nions, ispin, conv, unconv, footer=True):
    lines = []
    for el in elements:
        lines.append(f"   VRHFIN ={el}: d7 s2")
    lines.append(f"   number of ions     NIONS =      {nions}")
    lines.append(f"   ISPIN  =      {ispin}    spin polarized?")
    for _ in range(conv):
        lines.append(" aborting loop because EDIFF is reached")
    for _ in range(unconv):
        lines.append(" aborting loop EDIFF was not reached (unconverged)")
    lines.append(" reached required accuracy - stopping")
    if footer:
        lines.append(" General timing and accounting informations")
    return "\n".join(lines) + "\n"


def test_scan_and_rollup(tmp_path):
    d1 = tmp_path / "CrNi" / "sqs_1"
    d1.mkdir(parents=True)
    (d1 / "OUTCAR.relax").write_text(
        _outcar_text(["Cr", "Ni"], 16, 2, conv=12, unconv=2))
    d2 = tmp_path / "CoCr" / "sqs_2"
    d2.mkdir(parents=True)
    with gzip.open(d2 / "OUTCAR.relax.gz", "wt") as fh:   # gz path
        fh.write(_outcar_text(["Co", "Cr"], 32, 1, conv=7, unconv=0))
    d3 = tmp_path / "pure"
    d3.mkdir()
    (d3 / "OUTCAR.relax").write_text(
        _outcar_text(["Cr"], 1, 2, conv=5, unconv=1))

    recs = [cmf.scan_outcar(p) for p in
            sorted(tmp_path.rglob("OUTCAR.relax*"))]
    by = cmf.rollup(recs)

    assert by["CR-NI"]["frames"] == 12          # unconverged excluded
    assert by["CR-NI"]["rejected_steps"] == 2
    assert by["CR-NI"]["frames_spin"] == 12
    assert by["CR-NI"]["envs"] == 12 * 16
    assert by["CO-CR"]["frames"] == 7           # read through gzip
    assert by["CO-CR"]["frames_nospin"] == 7    # ISPIN=1 -> no magmoms
    assert by["CR"]["frames"] == 5              # pure bucket kept apart
    assert "CO-CR-NI" not in by


def test_cr_pv_reads_as_cr(tmp_path):
    p = tmp_path / "OUTCAR.relax"
    # VRHFIN gives the bare element even for Cr_pv POTCARs
    p.write_text("   VRHFIN =Cr: d5 s1\n"
                 "   NIONS =      2\n   ISPIN  =      2\n"
                 " aborting loop because EDIFF is reached\n")
    rec = cmf.scan_outcar(p)
    assert rec["system"] == "CR" and rec["frames"] == 1
