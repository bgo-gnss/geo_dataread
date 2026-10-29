"""The velocity product's files, frozen before it moved into gpsprod (step 3).

A synthetic run through the whole product — reader, gates, short tier,
k(T) from a table, refusal, pre-onset, the three writers — compared byte
for byte with ``tests/goldenmaster/velocity/``. The ``software`` key (the
package versions) is the only thing normalised. Recapture ONLY when a
change to the product is meant (``GPSPROD_CAPTURE=1 pytest
tests/test_velocity_golden.py``), never to make a refactor pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from geo_dataread import velocity as vel

GOLDEN = Path(__file__).parent / "goldenmaster" / "velocity"
DAY = 1 / 365.25
TABLE = {
    "schema": vel.K_TABLE_SCHEMA,
    "frame": "itrf2014",
    "series": "imo.24hr_itrf14",
    "view": "declared",
    "uncert": 10.0,
    "stations": ["AAAA"],
    "residuals": {
        res: {
            "days": [7, 14, 30, 61, 91, 182, 365, 730],
            "k": [[1.5 + 0.1 * i, 1.6 + 0.1 * i, 1.2 + 0.05 * i] for i in range(8)],
            "n_stations": [10] * 8,
        }
        for res in ("all", "secular")
    },
}
RATES = {
    "AAAA": (12.0, -25.0, 4.0),
    "BBBB": (-3.0, 8.0, -1.5),
    "CCCC": (40.0, -90.0, 120.0),
}


def _series(sta: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = sum(ord(c) for c in sta)
    rng = np.random.default_rng(seed)
    t = 2015.0 + (np.arange(int(2.0 * 365.25)) + 0.5) * DAY
    r = RATES[sta]
    y = np.vstack([r[c] * (t - 2015.0) for c in range(3)])
    if sta == "CCCC":  # an episode: a faster rate from 2016.3
        y += np.vstack([np.where(t > 2016.3, 300.0 * (t - 2016.3), 0.0)] * 3)
    y += rng.normal(0.0, 1.5, y.shape)
    s = np.full(y.shape, 1.2)
    s[2] *= 3.0
    keep = rng.random(t.size) > 0.05  # a few missing days
    return t[keep], y[:, keep], s[:, keep]


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    def read(sta, start, end, settings, *, terms, doc):
        if sta not in RATES:
            raise ValueError(f"no data for station {sta}")
        t, y, s = _series(sta)
        k = (t >= start) & (t < end)
        return t[k], y[:, k], s[:, k], {}

    monkeypatch.setattr(vel, "_read", read)
    monkeypatch.setattr(vel, "_params_doc", lambda settings: {"stations": {}})
    monkeypatch.setattr(vel, "_coords", lambda sta: (-22.0 - len(sta) * 0.01, 64.0))
    monkeypatch.setattr(vel, "_data_end", lambda *a, **k: None)
    monkeypatch.setattr(
        vel, "load_k_table", lambda settings: (TABLE, "k_table.json#sha256:fixed")
    )
    monkeypatch.setattr(
        vel, "_software", lambda: {"geo_dataread": "golden", "gps_analysis": "golden"}
    )


RUNS = {
    "year": ((2015.2, 2016.2), {}),
    "episode": ((2016.25, 2016.6), {"ref": "plate"}),
    "short": ((2015.5, 2015.5 + 3 * DAY), {}),
    "refuse": ((2015.2, 2015.4), {"no_record": "refuse", "ref": "background"}),
    "preonset": (
        (2016.35, 2016.5),
        {"ref": "pre-onset", "pre_window": (2015.8, 2016.25)},
    ),
}


@pytest.mark.parametrize("name", sorted(RUNS))
def test_velocity_files_are_unchanged(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch)
    (start, end), settings = RUNS[name]
    recs = vel.estimate_velocities(
        # the estimator as captured: the common mode is the read path's
        ["CCCC", "AAAA", "BBBB", "DDDD"],
        start,
        end,
        vel.VelocitySettings(**settings, common_mode=False),
    )
    paths = vel.write_velocity_set(recs, tmp_path, name)
    if os.environ.get("GPSPROD_CAPTURE") == "1":
        GOLDEN.mkdir(parents=True, exist_ok=True)
        for p in paths.values():
            (GOLDEN / p.name).write_bytes(p.read_bytes())
        pytest.skip("captured")
    for p in paths.values():
        want = GOLDEN / p.name
        assert want.is_file(), f"no golden {want.name}: capture it before the move"
        assert p.read_bytes() == want.read_bytes(), p.name
    # the records themselves, through JSON
    lines = (GOLDEN / f"{name}.jsonl").read_text().splitlines()
    assert [json.loads(x)["station"] for x in lines] == ["AAAA", "BBBB", "CCCC", "DDDD"]
