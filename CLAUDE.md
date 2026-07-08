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
> Known-dead branches (pinned as crashes): `useFIT="periodic"`, `tType="08h"`.

## Layout

```
geo_dataread/
├── src/geo_dataread/
│   ├── __init__.py
│   ├── gps_read.py       # 1997 LOC — main GPS time-series reader
│   ├── gps_displ.py      # displacements / station-relative motion
│   ├── gps_savetimes.py  # serialise time series to disk
│   ├── gas_read.py       # GAS (strainmeter) data
│   ├── sil_read.py       # SIL seismic data
│   └── hytro_read.py     # hydrology data
├── tests/test-gps_read.py      # legacy smoke script (not pytest-collected)
├── tests/goldenmaster/         # behavior pins for the Phase 1 refactor 📄 README
└── pyproject.toml
```

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

*Last reviewed: 2026-07-08*
