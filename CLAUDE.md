# geo_dataread

Geospatial data I/O — GPS displacement and coordinate time series, plus auxiliary readers
(GAS strain, SIL seismic, hydrology). **Tier 1** — processing layer; depends on Tier 0.

> **Note**: Modern import is `geo_dataread`. Older docs and the project README sometimes
> refer to it as `dread.*` — same package, legacy alias.

## Status

Stable, ~3948 LOC. Used by research workflows and `gps_plot`.

> **Refactor guard (Phase 0, plan §15.6):** `tests/goldenmaster/` pins current
> behavior of `openGlobkTimes`/`getData`/`gamittoNEU`/`gamittooneuf`/
> `read_gps_data` on frozen fixtures — run `uv run pytest tests/goldenmaster/`
> before AND after touching `gps_read.py`; never recapture goldens casually
> (see that dir's README). Production reality per the 2026-07-08 call-site
> survey: aflogun does NOT call this package anymore — it consumes the .NEU
> files that `gps-savetimes` (→ `gamittooneuf`) publishes to cdn.vedur.is.
> Live callers: `gps-savetimes`, `gps-displacemnts` (`simpleDisp`), and
> `gps_plot.timesmatplt` (`getData`). `read_gps_data` is research-only.
> Dead option `useFIT="periodic"` (was an accidental-crash pin): removed in
> refactor-B slice 4 — now raises a clean `ValueError`. `tType="08h"` got the
> same treatment in slice 4 but was REVIVED in slice 6 (JOIN, D4):
> `openGlobkTimes` reads any scheme with `mb_STA_<scheme>.dat{1,2,3}` files
> (missing scheme → clear `FileNotFoundError`), and the new `read_join(sta,
> schemes=("TOT","08h"))` holds multiple processing schemes together in one
> scheme-labeled, time-sorted DataFrame. `getData(tType="JOIN")` stays a
> legacy alias for TOT (array return, back-compat).
> Slice 7 (`refactor-B-pure-io`): the model evaluators (`line`, `periodic`,
> `xf`, `expxf`, `expf`, `dexpf`, `secondorder`) are bitwise-identical shims
> over `gps_analysis.models`, pinned by `tests/test_model_shims.py`;
> `lineperiodic` deliberately keeps its local single-expression evaluation
> (leaf association differs by ≤1 ulp; it feeds curve_fit on golden-pinned
> paths). Remaining-work ledger: `PLAN-geo_dataread-refactor-B.md`.

## Layout

```
geo_dataread/
├── src/geo_dataread/
│   ├── __init__.py
│   ├── gps_read.py       # ~1660 LOC — main GPS time-series reader (slice-4 purge + slice-6 read_join)
│   ├── gps_views.py      # apply-on-read views: raw|cleaned|detrended toggle (typed, mypy-strict)
│   ├── gps_write.py      # cleaned .NEU writer: gamittoNEU→gamittoFile, union-drop + .prov.json sidecar; steps.csv→step_epochs + protect_windows.csv (unrest lever) + outlier_overrides.csv (per-station levers incl. per-component min_outlier floor [N,E,U]); degrade → _cleaned.DEGRADED.NEU (typed, mypy-strict)
│   ├── gps_displ.py      # displacements / station-relative motion
│   ├── gps_savetimes.py  # serialise time series to disk (gps-savetimes; --clean also-writes cleaned .NEU)
│   ├── gas_read.py       # GAS (strainmeter) data
│   ├── sil_read.py       # SIL seismic data
│   └── hytro_read.py     # hydrology data
├── tests/test-gps_read.py      # legacy smoke script (not pytest-collected)
├── tests/test_gps_views.py     # view-toggle suite (raw parity, degrade, borrowing, provenance)
├── tests/test_cleaned_neu.py   # cleaned .NEU writer: byte-format identity, union drop, sidecar, steps.csv→step_epochs, protect_windows.csv (abort→clean), outlier_overrides.csv (per-station levers, precedence), .DEGRADED naming, --clean CLI
├── tests/goldenmaster/         # behavior pins for the Phase 1 refactor 📄 README
└── pyproject.toml
```

## View toggle (internal delivery, DESIGN_live_detrending §0)

`gps_views.read_gps_view(sta, view="raw"|"cleaned"|"detrended", ...)` is the
first-class raw↔cleaned↔detrended switch of the internal (direct-read) path:
raw columns are ALWAYS present and bit-identical to the legacy read; views
only ADD columns (`*_outlier`/`*_cleaned` masks from
`gps_analysis.detect_outliers`; `*_detrended` = raw − stored trajectory via
`gps_analysis.apply_detrend` — pure apply of the deployed
`detrend_params.json` record, NO re-fit on read). Provenance in
`df.attrs["gps_view"]` (`detrend_method`, frame, record version, `fitted_at`,
borrowed/UseSTA source, degrade state). Graceful degrade: any clean/detrend
failure warns (`UserWarning` + log) and serves raw — only a frame mismatch
hard-fails. Array paths revived: `getData(..., ref="detrend")` and
`gamittoNEU(..., ref="detrend")` (so `gps-savetimes --ref detrend` now
produces stored-params detrended .NEU, plate-first). The legacy
`getDetrFit`/`convconst`/`save_detrend_const` + `detrend_itrf2008.csv`
mechanism is SUPERSEDED (kept as shims for `read_gps_data` until design §8
step 5). Lint/type scope: ruff excludes + mypy per-module ignores cover the
legacy aux readers only — new code must pass `mypy --strict` (pyproject).

## Dependencies

- **In** (declared in `pyproject.toml`):
  - `geofunc` (git+https) — coordinate transforms
  - `gps_parser` (git+https) — station configuration (pinned 0.3.0: note
    `detrendFile` resolves CWD-relative there, config-dir-relative in local 0.4.x)
  - `gtimes`, `numpy`, `pandas`, `pyproj`, `scipy` — external (gtimes/numpy/
    pandas added 2026-07-08; the package could not import from a clean install
    without them)
- **Out** (used by): `gps_plot` (loads time series for plotting); research scripts.

## Console Scripts

```bash
geo-dataread          # entry: geo_dataread:main
gps-savetimes ...     # entry: geo_dataread.gps_savetimes:main
gps-displacemnts ...  # entry: geo_dataread.gps_displ:main   (sic — typo preserved verbatim from pyproject.toml)
```

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- `../geofunc/CLAUDE.md` — coordinate-math primitives this package consumes
- `../gps_parser/CLAUDE.md` — station-config provider
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-07-14 (internal-delivery slice: gps_views view toggle)*
