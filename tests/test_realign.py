"""write_pos and realign: the .pos writer and the day re-alignment."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

from geo_dataread import org, pos, realign, sinex, stabilize

FIX = Path(__file__).parent / "fixtures"
SNX = FIX / "sinex" / "IMO0OPSFIN_20141500000_01D_01D_SOL.SNX.gz"
AKUR = FIX / "pos" / "AKUR.imo.final_itr08.pos"


def test_write_pos_reproduces_a_tssum_file(tmp_path: Path) -> None:
    src = pos.read_pos(AKUR)
    out = pos.write_pos(
        tmp_path / "AKUR.x.pos",
        "AKUR",
        src.epochs,
        src.xyz,
        src.cov_xyz(),
        frame="Unknown",
        ref_xyz=src.ref_xyz,
    )
    got = pos.read_pos(out)
    np.testing.assert_array_equal(got.epochs, src.epochs)
    np.testing.assert_array_equal(got.yearf, src.yearf)
    np.testing.assert_allclose(got.mjd, src.mjd, atol=1e-4)
    np.testing.assert_allclose(got.xyz, src.xyz, atol=6e-6)
    np.testing.assert_allclose(got.sxyz, src.sxyz, atol=6e-6)
    np.testing.assert_allclose(got.rxyz, src.rxyz, atol=6e-4)
    np.testing.assert_allclose(got.llh[:, :2], src.llh[:, :2], atol=1e-9)
    np.testing.assert_allclose(got.dneu, src.dneu, atol=2e-5)  # GLOBK Loc. like tssum
    np.testing.assert_allclose(got.sneu, src.sneu, atol=2e-5)
    np.testing.assert_allclose(got.ref_xyz, src.ref_xyz, atol=6e-6)
    assert got.header["Format Version"] == "1.1.1"


def test_write_pos_truncates_to_the_minute_and_sorts(tmp_path: Path) -> None:
    t = np.array(["2024-04-10T11:59:45", "2024-04-09T11:59:45"], dtype="datetime64[s]")
    x = np.array([[2587384.0, -1043034.0, 5716564.0]] * 2) + [[0.001, 0, 0], [0, 0, 0]]
    c = np.broadcast_to(np.diag([1e-6, 2e-6, 4e-6]), (2, 3, 3))
    got = pos.read_pos(pos.write_pos(tmp_path / "S.pos", "s", t, x, c, frame="IGb14"))
    assert [str(e) for e in got.epochs] == [
        "2024-04-09T11:59:00",
        "2024-04-10T11:59:00",
    ]
    assert got.station == "S" and got.header["4-character ID"] == "S"
    np.testing.assert_allclose(got.xyz[1] - got.xyz[0], [0.001, 0, 0], atol=1e-9)


def test_realign_day_glorg_frame_and_external_identity(tmp_path: Path) -> None:
    day = realign.realign_day(SNX, "itrf2008")
    pos_org = org.read_org_positions(str(SNX).replace(".SNX.gz", ".org.gz"))
    assert day.n_ref == 47 and not day.dropped and not day.error
    assert day.epoch == datetime.datetime(2014, 5, 30, 11, 59, 45)
    np.testing.assert_allclose(
        day.sites["ONSA"][0], pos_org["ONSA_1PS"].xyz, atol=1.5e-5
    )
    # an external apriori equal to the SINEX one is the same fit
    sol = sinex.read_sinex(SNX)
    setup = org.read_org_setup(str(SNX).replace(".SNX.gz", ".org.gz"))
    own = {
        c: sol.apriori[sol.xyz_index(c)]
        for c in setup.reference
        if (sol.code == c).sum() == 3
    }
    ext = stabilize.stabilize(sol, setup.reference, setup.config, apriori=own)
    np.testing.assert_allclose(
        ext.solution.estimate[sol.xyz_index("ONSA")], day.sites["ONSA"][0], atol=1e-12
    )


def test_archive_days(tmp_path: Path) -> None:
    (tmp_path / "2014").mkdir()
    for doy in (149, 150, 151):
        (tmp_path / "2014" / f"IMO0OPSFIN_2014{doy}0000_01D_01D_SOL.SNX.gz").touch()
    got = realign.archive_days(
        tmp_path, datetime.date(2014, 5, 30), datetime.date(2014, 5, 31)
    )
    assert [p.name[11:18] for p in got] == ["2014150", "2014151"]
