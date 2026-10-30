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


# N, E, U [m/yr] from the legacy plateVelo chain (platePoles → rotpole → eccell,
# E,N,U rows reordered), frozen 2026-10-12 before geofunc.geo retired
LEGACY = {
    ("HOFN", "ITRF2008"): (0.016562758572750195, 0.008435024581705248, 4.360223801871254e-05),
    ("HOFN", "ITRF2014"): (0.016489647873740628, 0.008779640598814921, 4.3409770677845364e-05),
    ("REYK", "ITRF2008"): (0.018530986455884538, -0.009111887907600187, 4.895682345376907e-05),
    ("REYK", "ITRF2014"): (0.019573082710926074, -0.008660304101246363, 5.170992688414283e-05),
    ("SENG", "ITRF2008"): (0.018458507369191354, -0.009245774352904567, 4.910679146803832e-05),
    ("SENG", "ITRF2014"): (0.019500126792532407, -0.008797269812110655, 5.187790328049527e-05),
}  # fmt: skip


@pytest.mark.parametrize(("sta", "reference"), list(LEGACY))
def test_matches_the_legacy_plateVelo_chain(sta: str, reference: str) -> None:
    neu = sc.plate_velocity_neu(
        sta, reference=reference, coord_file=COORD, plate_file=PLATE
    )
    np.testing.assert_allclose(neu, LEGACY[sta, reference], atol=1e-12)


def test_explicit_plate_and_missing_station() -> None:
    eura = sc.plate_velocity_neu("SENG", "EURA", coord_file=COORD, plate_file=PLATE)
    noam = sc.plate_velocity_neu("SENG", coord_file=COORD, plate_file=PLATE)
    # EURA − NOAM at Svartsengi: ~16 mm/yr east (the plate-boundary rate)
    assert 0.012 < eura[1] - noam[1] < 0.020
    with pytest.warns(UserWarning, match="Cannot determine plate velocity"):
        z = sc.plate_velocity_neu("NONE", coord_file=COORD, plate_file=PLATE)
    np.testing.assert_array_equal(z, np.zeros(3))


def test_apriori_baseline_in_the_reference_frame() -> None:
    d = sc.apriori_baseline("SENG", "REYK", COORD)
    assert d is not None
    xs, xr = sc.station_xyz("SENG", COORD), sc.station_xyz("REYK", COORD)
    assert xs is not None and xr is not None
    # length preserved, horizontal: SENG is ~40 km SW of Reykjavík
    assert np.isclose(np.linalg.norm(d), np.linalg.norm(xs - xr))
    assert d[0] < 0 and d[1] < 0 and abs(d[2]) < 0.01 * np.linalg.norm(d)
    # swapped: the opposite vector, turned by the angle between the two
    # stations' local frames (tilt L/R plus meridian convergence: ~0.5° over
    # these 37 km; negligible for the mm changes plotted on a baseline)
    back = sc.apriori_baseline("REYK", "SENG", COORD)
    assert back is not None
    length = np.linalg.norm(d)
    angle = np.arccos(-back @ d / length**2)
    assert length / 6.37e6 < angle < 2.0 * length / 6.37e6
    assert sc.apriori_baseline("SENG", "NONE", COORD) is None
