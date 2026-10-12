"""globk_apr: GLOBK apr/eq readers and the a priori at an epoch."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
from geofunc import local

from geo_dataread import globk_apr as ga

X0 = np.array([2587384.0, -1043034.0, 5716564.0])
APR = f"""# comment
+REFERENCE_FRAME IGb14
 REYK_GPS {X0[0]:.5f} {X0[1]:.5f} {X0[2]:.5f}  -0.01000   0.02000   0.00500 2010.0  0.001 0.001 0.001
 REYK_2PS {X0[0] + 1:.5f} {X0[1]:.5f} {X0[2]:.5f}   0.00000   0.00000   0.00000 2010.0
 EXTENDED REYK_2PS EXP 2012 01 01 00 00  100.0  0.01000  0.00000  0.00000 note
 EXTENDED REYK_2PS LOG 2012 01 01 00 00   10.0  0.00000  0.00000  0.00200
 EXTENDED REYK_2PS EXP 2012 01 01 00 00  100.0  0.02000  0.00000  0.00000 replaces the first
"""
EQ = """# renames
 rename REYK     REYK_2PS 2012 01 01 00 00 2100 01 01 00 00
 rename REYK_2PS REYK_XCL 2013 05 01 00 00 2013 05 02 24 00
 rename HOFN_GPS HOFN_1PS 2000 01 01 00 00 2100 01 01 00 00
 rename HOFN_1PS HOFN_3PS 2005 01 01 00 00 2100 01 01 00 00
#rename HOFN_GPS HOFN_9PS 2000 01 01 00 00 2100 01 01 00 00
"""


@pytest.fixture()
def files(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "a.apr").write_text(APR)
    (tmp_path / "a.eq").write_text(EQ)
    return tmp_path / "a.apr", tmp_path / "a.eq"


def test_decimal_year() -> None:
    assert ga.decimal_year_to_datetime(2005.0) == datetime.datetime(2005, 1, 1)
    assert ga.decimal_year_to_datetime(2024.5) == datetime.datetime(
        2024, 7, 2
    )  # 366-day year


def test_renames_chain_code_and_exclusion(files) -> None:  # type: ignore[no-untyped-def]
    ren = ga.read_renames(files[1])
    assert len(ren) == 4  # the commented line is skipped
    t = datetime.datetime
    assert ga.site_name("REYK", t(2011, 6, 1), ren) == "REYK_GPS"
    assert ga.site_name("reyk", t(2013, 1, 1), ren) == "REYK_2PS"
    assert ga.site_name("REYK", t(2013, 5, 2, 12), ren) == "REYK_XCL"  # hour 24 end
    assert ga.site_name("HOFN", t(2006, 1, 1), ren) == "HOFN_3PS"  # chained
    assert ga.site_name("HOFN", t(2001, 1, 1), ren) == "HOFN_1PS"


def test_velocity_and_post_seismic_terms(files) -> None:  # type: ignore[no-untyped-def]
    apr = ga.read_apr(files[0])
    when = datetime.datetime(2012, 4, 10)  # 100 days after the event
    dt = (when - datetime.datetime(2010, 1, 1)).days / 365.25
    np.testing.assert_allclose(
        apr["REYK_GPS"].at(when), X0 + np.array([-0.01, 0.02, 0.005]) * dt
    )
    s = apr["REYK_2PS"]
    assert len(s.terms) == 2  # the repeated EXP replaced the first
    dneu = np.array([0.02 * (1 - np.exp(-1.0)), 0.0, 0.002 * np.log(11.0)])
    g = local.ecef_to_geodetic(s.xyz)
    np.testing.assert_allclose(
        s.at(when), s.xyz + local.local_to_ecef(dneu, g.lat, g.lon).d, atol=1e-12
    )
    np.testing.assert_allclose(
        s.at(datetime.datetime(2011, 1, 1)), s.xyz
    )  # before the event


def test_apriori_xyz_reports_skips(files) -> None:  # type: ignore[no-untyped-def]
    apr, ren = ga.read_apr(files[0]), ga.read_renames(files[1])
    pos, skipped = ga.apriori_xyz(
        ["REYK", "HOFN"], datetime.datetime(2013, 5, 2), apr, ren
    )
    assert pos == {} and skipped == {"REYK": "excluded", "HOFN": "no apr HOFN_3PS"}
    pos, _ = ga.apriori_xyz(["REYK"], datetime.datetime(2011, 1, 1), apr, ren)
    assert set(pos) == {"REYK"}


def test_later_files_override(tmp_path: Path, files) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "b.apr").write_text(" REYK_GPS 1.0 2.0 3.0 0 0 0 2010.0\n")
    apr = ga.read_apr([files[0], tmp_path / "b.apr"])
    np.testing.assert_array_equal(apr["REYK_GPS"].xyz, [1.0, 2.0, 3.0])
