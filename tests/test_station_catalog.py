"""station_catalog: plate/coordinate lookups and the N/E/U plate velocity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geo_dataread import station_catalog as sc

CFG = Path(__file__).parent / "goldenmaster" / "fixtures" / "realdata" / "gpsconfig"
COORD = CFG / "station_coord.xyz"
PLATE = CFG / "station-plate"


def test_lookups() -> None:
    assert sc.plate_table(PLATE) == {"HOFN": "EURA", "REYK": "NOAM", "SENG": "NOAM"}
    assert sc.station_plate("SENG", PLATE) == "NOAM"
    assert sc.station_plate("NONE", PLATE) is None
    x = sc.station_xyz("REYK", COORD)
    assert x is not None
    np.testing.assert_allclose(x, [2587383.91122, -1043033.58722, 5716564.19740])
    assert sc.station_xyz("NONE", COORD) is None


def test_duplicate_site_resolves_to_last_row(tmp_path: Path) -> None:
    f = tmp_path / "coord.xyz"
    f.write_text(" 1.0 2.0 3.0 HESA\n 4.0 5.0 6.0 HESA\n")
    np.testing.assert_array_equal(sc.station_xyz("HESA", f), [4.0, 5.0, 6.0])


@pytest.mark.parametrize("sta", ["HOFN", "REYK", "SENG"])
@pytest.mark.parametrize("reference", ["ITRF2008", "ITRF2014"])
def test_matches_the_legacy_plateVelo_chain(sta: str, reference: str) -> None:
    # geofunc.geofunc.plateVelo = platePoles → rotpole → eccell (rows E, N, U)
    geo = pytest.importorskip("geofunc.geo")
    x = sc.station_xyz(sta, COORD)
    assert x is not None
    w = geo.platePoles(sc.station_plate(sta, PLATE), reference=reference)
    llh = np.array(geo.itrf08towgs84.transform(*x, radians=True))
    enu = geo.eccell(llh, geo.rotpole(w, x))
    neu = sc.plate_velocity_neu(
        sta, reference=reference, coord_file=COORD, plate_file=PLATE
    )
    np.testing.assert_allclose(neu, enu[[1, 0, 2]], atol=1e-12)


def test_explicit_plate_and_missing_station() -> None:
    eura = sc.plate_velocity_neu("SENG", "EURA", coord_file=COORD, plate_file=PLATE)
    noam = sc.plate_velocity_neu("SENG", coord_file=COORD, plate_file=PLATE)
    # EURA − NOAM at Svartsengi: ~16 mm/yr east (the plate-boundary rate)
    assert 0.012 < eura[1] - noam[1] < 0.020
    with pytest.warns(UserWarning, match="Cannot determine plate velocity"):
        z = sc.plate_velocity_neu("NONE", coord_file=COORD, plate_file=PLATE)
    np.testing.assert_array_equal(z, np.zeros(3))
