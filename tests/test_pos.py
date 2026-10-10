"""geo_dataread.pos: the PBO .pos reader and the openGlobkTimes(tType="POS") path."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest

from geo_dataread import pos

FIX = Path(__file__).parent / "fixtures" / "pos"


def test_read_pos_fields() -> None:
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    assert s.station == "AKUR" and len(s) == 5
    assert str(s.epochs[0]) == "2026-09-01T11:59:00"
    np.testing.assert_allclose(s.xyz[0], [2502918.04409, -819167.02669, 5789715.19547])
    np.testing.assert_allclose(s.sxyz[0], [0.00212, 0.00116, 0.00422])
    np.testing.assert_allclose(s.rxyz[0], [-0.453, 0.818, -0.460])
    np.testing.assert_allclose(s.ref_xyz, [2502918.04456, -819167.02792, 5789715.19606])
    assert s.soln[0] == "final"
    assert s.header["Format Version"] == "1.1.0"


def test_yearf_is_globk_epoch_convention() -> None:
    # 2026-09-01 11:59 = doy 244: 2026 + (243 + 11/24 + 59/1440) / 365; mb_ TOT prints 2026.66712
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    assert s.yearf[0] == pytest.approx(2026 + (243 + 11 / 24 + 59 / 1440) / 365)
    assert round(s.yearf[0], 5) == 2026.66712


def test_covariance_from_sigmas_and_correlations() -> None:
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    c = s.cov_neu()[0]
    sn, se, su = s.sneu[0]
    rne, rnu, reu = s.rneu[0]
    np.testing.assert_allclose(np.diag(c), [sn**2, se**2, su**2])
    assert c[0, 1] == pytest.approx(rne * sn * se) and c[1, 0] == c[0, 1]
    assert c[0, 2] == pytest.approx(rnu * sn * su) and c[1, 2] == pytest.approx(
        reu * se * su
    )


def test_gz_and_open_pos_times(tmp_path: Path) -> None:
    with (
        open(FIX / "AKUR.imo.final_itr08.pos", "rb") as fi,
        gzip.open(tmp_path / "AKUR.imo.final_itr08.pos.gz", "wb") as fo,
    ):
        shutil.copyfileobj(fi, fo)
    yearf, data, ddata = pos.open_pos_times("akur", tmp_path)
    s = pos.read_pos(FIX / "AKUR.imo.final_itr08.pos")
    assert data.shape == ddata.shape == (3, 5)
    np.testing.assert_array_equal(data, s.dneu.T)
    np.testing.assert_array_equal(yearf, s.yearf)


def test_missing_station_and_bad_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pos.pos_path("NONE", tmp_path)
    (tmp_path / "BAD.imo.final_itr08.pos").write_text("not a pos file\n")
    with pytest.raises(pos.PosError):
        pos.read_pos(tmp_path / "BAD.imo.final_itr08.pos")


def test_openGlobkTimes_pos_scheme() -> None:
    from geo_dataread.gps_read import openGlobkTimes

    yearf, data, ddata = openGlobkTimes("AKUR", Dir=str(FIX), tType="POS")
    assert data.shape == (3, 5) and yearf[0] == pytest.approx(2026.667122, abs=1e-6)


def test_numpy_only() -> None:
    # like sinex.py: usable from gps_api / notebooks without the legacy stack
    imports = [
        ln
        for ln in Path(pos.__file__).read_text().splitlines()
        if ln.startswith(("import ", "from "))
    ]
    assert not [
        ln for ln in imports if "pandas" in ln or "gps_parser" in ln or "gps_read" in ln
    ]
