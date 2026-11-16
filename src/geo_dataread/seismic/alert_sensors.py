"""Cross-check of the alertMap field-6 code 22 against FDSN sensor models (B4).

PLAN-seismicity.md §9c B4: the page method drops ``.event`` rows whose
field 6 is **22** (on the golden event that removes ``hei`` (Lennartz 1 Hz),
``gha`` and ``ada``). The rule is **provisional**: the code is a property of
the detection, not of the station (``hei`` is 22 on 2026-11-12 and 12 on
2026-11-10), and the hraun rule is unknown. This report puts the code next to
the sensor the FDSN station service lists for the station **at the
detection time** (channel epochs: ``gha`` changed sensor in 2023, ``hei`` in
2026-04), so the two can be compared:

* ``code22_one_hz``: code 22 and a 1 Hz sensor (consistent with "no Lennartz
  1 Hz");
* ``code22_other_sensor``: code 22 but the sensor is not 1 Hz by the patterns;
* ``code22_unknown_sensor``: code 22, no sensor description (many FDSN
  channels have an empty one);
* ``one_hz_without_code22``: a 1 Hz sensor the code does **not** mark.

Sensors are classified by their description with :data:`ONE_HZ_PATTERNS`
(configurable); an empty description is ``unknown``, never "not 1 Hz".
:func:`read_fdsn_channels` reads ``fdsnws/station/1`` (``level=channel``,
``format=text``) from ``seiscomp-p01`` (internal network only; set
``$GPS_FDSN_STATION_URL`` for another).
"""

from __future__ import annotations

import io
import os
import re
import urllib.parse
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Final

import pandas as pd

from geo_dataread.seismic.quakes_api import Session, UrllibSession
from geo_dataread.seismic.schema import STRING_DTYPE, TIME_DTYPE
from geo_dataread.seismic.sil_alert import AlertEvent

__all__ = [
    "DEFAULT_STATION_URL",
    "ENV_STATION_URL",
    "FDSN_CHANNEL_COLUMNS",
    "ONE_HZ_PATTERNS",
    "SENSOR_REPORT_COLUMNS",
    "SensorServiceError",
    "parse_fdsn_channels",
    "read_fdsn_channels",
    "sensor_code_report",
    "sensors_at",
]

#: The internal SeisComP fdsnws-station service.
DEFAULT_STATION_URL: Final = "http://seiscomp-p01.vedur.is:8080/fdsnws/station/1"
#: Environment override of :data:`DEFAULT_STATION_URL`.
ENV_STATION_URL: Final = "GPS_FDSN_STATION_URL"

#: Sensor descriptions taken as 1 Hz (case-insensitive ``re.search``):
#: Lennartz LE-3D 1 s (``LE-3D.1.400``), LE-3Dlite (``LE-3DliteMkII.1.400``),
#: the ``LE1`` short name and an explicit "1 Hz"/"1s".
ONE_HZ_PATTERNS: Final = (
    r"^LE-3D(?:lite)?(?:MkII)?\.1\.",
    r"\bLE-?1\b",
    r"\b1\s*Hz\b",
    r"\bLE-3D(?:lite)?\b.*\b1\s*s\b",
)

#: Columns of :func:`parse_fdsn_channels`.
FDSN_CHANNEL_COLUMNS: Final = (
    "network",
    "station",
    "location",
    "channel",
    "latitude",
    "longitude",
    "sensor",
    "sample_rate",
    "start",
    "end",
)

#: Columns of :func:`sensor_code_report`.
SENSOR_REPORT_COLUMNS: Final = (
    "alert_day",
    "alert_id",
    "alert_time",
    "station",
    "class_code",
    "sensor",
    "one_hz",
    "issue",
)


#: The header of an fdsnws-station channel text (an empty answer, 204).
_EMPTY_TEXT: Final = (
    "#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|"
    "Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime"
)


class SensorServiceError(RuntimeError):
    """The FDSN station service failed or could not be reached."""


def parse_fdsn_channels(text: str) -> pd.DataFrame:
    """Parse ``fdsnws/station`` ``level=channel&format=text`` output.

    Station codes are lower-cased (as in the alertMap files); empty
    ``EndTime`` is an open epoch (NaT). ``sensor`` is ``SensorDescription``
    (``<NA>`` when empty).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("#Network"):
        raise ValueError("not an fdsnws-station channel text (no '#Network' header)")
    raw = pd.read_csv(
        io.StringIO("\n".join([lines[0].lstrip("#"), *lines[1:]])),
        sep="|",
        dtype=str,
        keep_default_na=False,
    )
    need = ("Network", "Station", "Location", "Channel", "SensorDescription")
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"fdsnws-station text lacks columns {missing}")

    def times(col: str) -> pd.Series:
        v = raw[col].str.strip()
        return pd.to_datetime(
            v.where(v != "", None), utc=True, format="ISO8601"
        ).astype(TIME_DTYPE)

    descr = raw["SensorDescription"].str.strip()
    sensor = descr.where(descr != "", None)
    return pd.DataFrame(
        {
            "network": raw["Network"].astype(STRING_DTYPE),
            "station": raw["Station"].str.lower().astype(STRING_DTYPE),
            "location": raw["Location"].astype(STRING_DTYPE),
            "channel": raw["Channel"].astype(STRING_DTYPE),
            "latitude": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "sensor": sensor.astype(STRING_DTYPE),
            "sample_rate": pd.to_numeric(raw["SampleRate"], errors="coerce"),
            "start": times("StartTime"),
            "end": times("EndTime"),
        }
    )[list(FDSN_CHANNEL_COLUMNS)]


def read_fdsn_channels(
    stations: Sequence[str] | None = None,
    *,
    network: str = "VI",
    base_url: str | None = None,
    session: Session | None = None,
    timeout_s: float = 60.0,
) -> pd.DataFrame:
    """Channel epochs from the FDSN station service (:func:`parse_fdsn_channels`)."""
    base = (
        base_url or os.environ.get(ENV_STATION_URL, "") or DEFAULT_STATION_URL
    ).rstrip("/")
    params = [("level", "channel"), ("format", "text"), ("network", network)]
    if stations:
        # SeisComP keeps the SIL codes lower-case; upper-case matches nothing (204).
        params.append(("station", ",".join(sorted({s.lower() for s in stations}))))
    url = f"{base}/query?{urllib.parse.urlencode(params)}"
    sess: Session = session if session is not None else UrllibSession()
    try:
        resp = sess.get(
            url, timeout=timeout_s, headers={"User-Agent": "geo_dataread-alertmap"}
        )
    except OSError as exc:
        raise SensorServiceError(
            f"FDSN station service could not be reached ({url}): {exc!r}; it "
            f"resolves only inside the IMO network (${ENV_STATION_URL} overrides)"
        ) from exc
    if resp.status_code == 204:
        return parse_fdsn_channels(_EMPTY_TEXT)
    if resp.status_code != 200:
        raise SensorServiceError(f"FDSN station service {resp.status_code} for {url}")
    return parse_fdsn_channels(resp.content.decode("utf-8", "replace"))


def _is_one_hz(sensor: str | None, patterns: Sequence[re.Pattern[str]]) -> bool | None:
    if sensor is None or not sensor.strip():
        return None
    return any(p.search(sensor) for p in patterns)


def sensors_at(
    channels: pd.DataFrame,
    time: pd.Timestamp,
    *,
    patterns: Sequence[str] = ONE_HZ_PATTERNS,
) -> pd.DataFrame:
    """Per station, the sensors of the channel epochs open at ``time``.

    Returns a frame indexed by ``station`` with ``sensor`` (distinct
    non-empty descriptions joined by ``"; "``, NA when none) and ``one_hz``
    (``True`` when any described sensor matches ``patterns``, ``False`` when
    some are described and none match, NA when none is described).
    """
    t = pd.Timestamp(time)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    open_ = (channels["start"] <= t) & (channels["end"].isna() | (channels["end"] >= t))
    rx = [re.compile(p, re.I) for p in patterns]
    rows: dict[str, tuple[str | None, bool | None]] = {}
    for sta, grp in channels[open_].groupby("station", sort=True):
        descr = sorted({str(s) for s in grp["sensor"].dropna() if str(s).strip()})
        flags = [_is_one_hz(d, rx) for d in descr]
        one: bool | None = any(bool(f) for f in flags) if descr else None
        rows[str(sta)] = ("; ".join(descr) if descr else None, one)
    idx = pd.Index(list(rows), name="station", dtype=STRING_DTYPE)
    return pd.DataFrame(
        {
            "sensor": pd.array([v[0] for v in rows.values()], dtype=STRING_DTYPE),
            "one_hz": pd.array([v[1] for v in rows.values()], dtype="boolean"),
        },
        index=idx,
    )


def sensor_code_report(
    events: Iterable[tuple[date, AlertEvent]],
    channels: pd.DataFrame,
    *,
    code: int = 22,
    patterns: Sequence[str] = ONE_HZ_PATTERNS,
) -> pd.DataFrame:
    """Code-``code`` stations and unmarked 1 Hz sensors, per detection (B4).

    One row per station of each ``(day, event)`` that has field 6 == ``code``
    or a 1 Hz sensor at the detection time (:func:`sensors_at`). ``issue``
    is one of ``code22_one_hz``, ``code22_other_sensor``,
    ``code22_unknown_sensor``, ``one_hz_without_code22`` (named for the
    default code). A station missing from ``channels`` has an NA sensor.
    """
    tag = f"code{code}"
    recs: list[dict[str, object]] = []
    for day, ev in events:
        sens = sensors_at(channels, ev.header.time, patterns=patterns)
        f = ev.frame
        for sta, cc in zip(
            f["station"].astype(str), f["class_code"].astype("Int64"), strict=True
        ):
            if sta in sens.index:
                sensor, one = sens.at[sta, "sensor"], sens.at[sta, "one_hz"]
            else:
                sensor, one = pd.NA, pd.NA
            one_b = None if one is pd.NA or one is None else bool(one)
            marked = cc is not pd.NA and int(cc) == code
            if marked:
                issue = (
                    f"{tag}_unknown_sensor"
                    if one_b is None
                    else (f"{tag}_one_hz" if one_b else f"{tag}_other_sensor")
                )
            elif one_b:
                issue = f"one_hz_without_{tag}"
            else:
                continue
            recs.append(
                {
                    "alert_day": day.isoformat(),
                    "alert_id": ev.header.alert_id,
                    "alert_time": ev.header.time,
                    "station": sta,
                    "class_code": cc,
                    "sensor": None if sensor is pd.NA else sensor,
                    "one_hz": one_b,
                    "issue": issue,
                }
            )
    out = pd.DataFrame(
        {
            "alert_day": pd.Series([r["alert_day"] for r in recs], dtype=STRING_DTYPE),
            "alert_id": pd.Series([r["alert_id"] for r in recs], dtype="int64"),
            "alert_time": pd.Series([r["alert_time"] for r in recs], dtype=TIME_DTYPE),
            "station": pd.Series([r["station"] for r in recs], dtype=STRING_DTYPE),
            "class_code": pd.Series([r["class_code"] for r in recs], dtype="Int64"),
            "sensor": pd.Series([r["sensor"] for r in recs], dtype=STRING_DTYPE),
            "one_hz": pd.Series([r["one_hz"] for r in recs], dtype="boolean"),
            "issue": pd.Series([r["issue"] for r in recs], dtype=STRING_DTYPE),
        }
    )
    if len(out):
        out = out.sort_values(["alert_time", "alert_id", "station"], kind="stable")
    out = out.reset_index(drop=True)[list(SENSOR_REPORT_COLUMNS)]
    return out
