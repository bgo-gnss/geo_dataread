"""Read one station's series the way every gpsprod product sees it.

:func:`load` is the one read path of the products: the ``.pos`` series of
a frame and session, the reference (plate removed, the stored seasonal or
background removed, or the raw frame), the view (declared blunders out,
running-window outliers out, or raw), optionally differenced against a
reference station with the baseline function gps_plot uses.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from geo_dataread.products.common import COMPONENTS, REFS, SESSIONS, TERMS, VIEWS

FloatArray = npt.NDArray[np.float64]
DAY_YEARS = 1.0 / 365.25
#: why the view dropped (or marked) an epoch, per component: bits of
#: :attr:`StationView.reasons`
REASON_OUTLIER = 1  # the cleaned view's detector
REASON_DECLARED = 2  # excluded_epochs.csv (the declared view)
REASON_ICING = 4  # antenna icing days (marked; dropped only when removed)
REASON_SNOW = 8  # snow-burial days (Up)
REASON_NETWORK = 16  # whole-network bad days
REASON_NAMES = {
    REASON_OUTLIER: "outlier",
    REASON_DECLARED: "declared",
    REASON_ICING: "icing",
    REASON_SNOW: "snow",
    REASON_NETWORK: "network",
}


@dataclasses.dataclass(frozen=True)
class Source:
    """Which series a product reads.

    Attributes:
        frame: Frame or alias (``itrf14``, ``igb14``, ``itrf2014`` …).
        session: ``24h`` or ``8h`` solutions.
        view: ``declared``, ``cleaned`` or ``raw``.
        ref: One of :data:`~geo_dataread.products.common.REFS`.
        uncert: Largest formal σ kept [mm].
        pos_dir: ``.pos`` directory (None: postprocess.cfg ``pos_dir``).
        params: Detrend parameter document (None: the frame's deployed one).
        steps: Steps catalog (None: the deployed one).
        relative_to: Difference every station against this one.
    """

    frame: str = "itrf14"
    session: str = "24h"
    view: str = "declared"
    ref: str = "plate-periodic"
    uncert: float = 10.0
    pos_dir: str | None = None
    params: str | None = None
    steps: str | None = None
    relative_to: str | None = None

    def __post_init__(self) -> None:
        if self.session not in SESSIONS:
            raise ValueError(f"session must be one of {tuple(SESSIONS)}")
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {self.view!r}")
        if self.ref not in REFS:
            raise ValueError(f"ref must be one of {REFS}, got {self.ref!r}")
        from geo_dataread.pos import pos_frame

        pos_frame(self.frame)  # raises on an unknown frame

    @property
    def pos_id(self) -> str:
        from geo_dataread.pos import pos_frame, product_id

        return product_id(pos_frame(self.frame), self.session)

    @property
    def frame_name(self) -> str:
        from geo_dataread.pos import frame_name

        return frame_name(self.frame)

    @property
    def session_years(self) -> float:
        """One session (the sampling step) in years."""
        return SESSIONS[self.session] * DAY_YEARS

    def directory(self) -> str:
        if self.pos_dir is not None:
            return str(Path(self.pos_dir).expanduser())
        from gps_parser import ConfigParser

        return str(Path(str(ConfigParser().getPostProcessDir("pos_dir"))).expanduser())

    def params_doc(self) -> Mapping[str, Any] | None:
        from geo_dataread.gps_views import default_params_path, read_detrend_params

        path = self.params or default_params_path(self.pos_id)
        if path is None or not Path(path).is_file():
            return None
        return read_detrend_params(path)


@dataclasses.dataclass(frozen=True)
class StationView:
    """One station's series as a product reads it.

    ``y``/``sigma`` are (3, N) N/E/U [mm]; ``y`` is NaN where the view
    dropped an epoch. ``ref`` is the reference actually applied (a station
    without a stored record falls back to ``plate``); ``flags`` say why.
    """

    station: str
    t: FloatArray
    y: FloatArray
    sigma: FloatArray
    ref: str
    flags: tuple[str, ...] = ()
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: the values before the view dropped any (None for a baseline)
    y_raw: FloatArray | None = None
    #: (3, N) uint8, :data:`REASON_OUTLIER` … bits per epoch (None: not known)
    reasons: npt.NDArray[np.uint8] | None = None

    def sliced(self, start: float, end: float) -> StationView:
        keep = (self.t >= start) & (self.t < end)
        return dataclasses.replace(
            self, t=self.t[keep], y=self.y[:, keep], sigma=self.sigma[:, keep],
            y_raw=None if self.y_raw is None else self.y_raw[:, keep],
            reasons=None if self.reasons is None else self.reasons[:, keep],
        )  # fmt: skip


def station_record(doc: Mapping[str, Any] | None, sta: str) -> dict[str, Any] | None:
    """The station's stored detrend record, None without one."""
    if doc is None:
        return None
    from geo_dataread.gps_views import station_detrend_record

    try:
        record, _source = station_detrend_record(doc, sta)
    except (FileNotFoundError, ValueError):
        return None
    return dict(record) if record is not None else None


def _read_one(
    sta: str,
    start: float,
    end: float,
    source: Source,
    *,
    doc: Mapping[str, Any] | None,
) -> StationView:
    from geo_dataread.gps_views import read_gps_view

    flags: list[str] = []
    ref = source.ref
    record = station_record(doc, sta) if ref in TERMS else None
    if ref in TERMS and record is None:
        ref = "plate"
        flags.append("fallback:plate")
    if ref == "background" and record is not None and record.get("seasonal_only"):
        # a regional seasonal and no background of the station's own: the
        # seasonal is all there is to remove
        ref = "plate-periodic"
        flags.append("seasonal:regional;fallback:plate-periodic")
    if record is not None and record.get("borrowed"):
        flags.append(f"borrowed:{(record['borrowed'] or {}).get('from')}")
    terms = TERMS.get(ref)
    # getData takes any ref other than plate/detrend/"itrf2008" for a plate
    # NAME: the raw frame is asked for by that literal whatever the frame
    gv_ref = "itrf2008" if ref == "itrf" else "plate"
    clean = source.view == "cleaned"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = read_gps_view(
            sta, start, end,
            view="detrended" if terms else "raw",
            ref=gv_ref,
            terms=terms or "all",
            clean=clean,
            params=doc,
            steps=source.steps,
            tType="POS", pos_id=source.pos_id, Dir=source.directory(),
            uncert=source.uncert,
        )  # fmt: skip
    attrs = dict(df.attrs.get("gps_view", {}))
    t = df["yearf"].to_numpy(dtype=np.float64)
    cols = [f"{c}_detrended" if terms else c for c in COMPONENTS]
    if any(c not in df.columns for c in cols):
        raise ValueError(
            f"{sta}: background not removed ({attrs.get('degrade_reason') or cols})"
        )
    y = np.vstack([df[c].to_numpy(dtype=np.float64) for c in cols])
    s = np.vstack([df[f"D{c}"].to_numpy(dtype=np.float64) for c in COMPONENTS])
    y_raw = y.copy()
    reasons = np.zeros(y.shape, dtype=np.uint8)
    for i, c in enumerate(COMPONENTS):
        for col, bit in (
            (f"{c}_icing", REASON_ICING),
            (f"{c}_snow", REASON_SNOW),
            (f"{c}_network", REASON_NETWORK),
        ):
            mark = df.get(col)
            if mark is not None:
                reasons[i, mark.to_numpy(dtype=bool)] |= bit
    if source.view == "declared":
        from gps_parser import outlier_catalogs as _oc

        from geo_dataread.gps_views import station_excluded_epochs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xepochs, xsource = station_excluded_epochs(sta)
        idx = list(_oc.match_declared_epochs(t, np.asarray(xepochs, float)))
        attrs["excluded_epochs_source"] = xsource
        attrs["n_flagged"] = len(idx)
        if idx:
            y[:, idx] = np.nan
            reasons[:, idx] |= REASON_DECLARED
    if clean:
        if attrs.get("outlier_abort"):
            flags.append("clean:aborted")
        else:
            for i, c in enumerate(COMPONENTS):
                mask = df.get(f"{c}_outlier")
                if mask is not None:
                    y[i, mask.to_numpy(dtype=bool)] = np.nan
                    reasons[i, mask.to_numpy(dtype=bool)] |= REASON_OUTLIER
    return StationView(sta, t, y, s, ref, tuple(flags), attrs, y_raw, reasons)


def load(
    sta: str,
    start: float,
    end: float,
    source: Source,
    *,
    doc: Mapping[str, Any] | None = None,
) -> StationView:
    """One station (or ``sta − source.relative_to``) over ``[start, end)``.

    Raises ValueError/OSError on a station-level failure (no data, the
    reference station shares no epoch); the products turn those into a
    refused record.
    """
    sta = sta.upper()
    a = _read_one(sta, start, end, source, doc=doc)
    if source.relative_to is None:
        return a
    ref_sta = source.relative_to.upper()
    b = _read_one(ref_sta, start, end, source, doc=doc)
    if a.ref != b.ref:
        raise ValueError(
            f"{sta}−{ref_sta}: the ends are in different references "
            f"({a.ref}, {b.ref}); a baseline needs both alike"
        )
    from geo_dataread.baseline import baseline

    bl = baseline(
        (a.t, a.y, a.sigma), (b.t, b.y, b.sigma),
        stations=(sta, ref_sta), tol=0.5 * source.session_years, rezero=False,
    )  # fmt: skip
    flags = tuple(dict.fromkeys(a.flags + tuple(f"{ref_sta}:{f}" for f in b.flags)))
    attrs = {"baseline": {"relative_to": ref_sta, "n_common": int(bl.yearf.size)}}
    return StationView(sta, bl.yearf, bl.data, bl.sigma, a.ref, flags, attrs)
