# Quakes API fixtures (ticket T9, PLAN-seismicity.md §3.1)

Recorded 2026-11-13 from host `eldey` (inside the IMO network) by
`tests/fixtures/record_api_fixtures.py`. `manifest.json` lists, for every file, the exact
URL, the HTTP status, the kept headers, the byte size, the event count and the number of
redactions. `tests/test_api_fixtures.py` reads it.

- Internal: `https://vi-api.vedur.is/quakes/internal/events` (the default per Q2).
- Public: `https://api.vedur.is/quakes/events`. OpenAPI version string `2026-08-06`.

Bodies are the **raw bytes** of the response (no pretty-printing). The only exception is the
redaction described below. Readers parse them with `json`/`csv`, so they do not depend on the
layout, but re-recording gives a clean diff.

## Requests

Every list request passes the limits explicitly:
`size_min=-3&size_max=10&depth_min=0&depth_max=50`. Without them, the server defaults are
`size_max=7` and depth 0–50.

| Window | Parameters |
|--------|------------|
| A | `start_time=2024-08-22T00:00:00Z&end_time=2024-08-22T01:00:00Z`: the first hour of the T0 SIL fixture day |
| B | `start_time=2026-09-10T00:00:00Z&end_time=2026-09-11T00:00:00Z`: the whole T0 post-February-2026 day (small, so not cut to an hour) |
| empty | `2024-08-22T00:00:00Z` to `00:02:00Z` |

| File | Request | Status | Events |
|------|---------|--------|--------|
| `internal_seiscomp_20240822T00.json` | internal, A, `system=seiscomp` | 200 | 6 |
| `internal_sil_20240822T00.json` | internal, A, `system=sil` | 200 | 7 |
| `public_seiscomp_20240822T00.json` | public, A, seiscomp | 200 | 6 |
| `public_sil_20240822T00.json` | public, A, sil | 200 | 7 |
| `internal_seiscomp_20260910.json` | internal, B, seiscomp | 200 | 48 |
| `internal_sil_20260910.json` | internal, B, sil | 200 | 19 |
| `public_seiscomp_20260910.json` | public, B, seiscomp | 200 | 37 |
| `public_sil_20260910.json` | public, B, sil | 200 | 19 |
| `internal_seiscomp_20240822T00.csv`, `internal_sil_20240822T00.csv`, `public_sil_20240822T00.csv` | as above + `format=csv` | 200 | 6, 7, 7 |
| `internal_{seiscomp,sil}_20240822T00_count.json` | `/internal/events/count`, A | 200 | body `6` / `7` |
| `internal_detail_IMO2024qnogqh.json` | `/internal/events/IMO2024qnogqh` (back-imported SIL event) | 200 | 1 |
| `public_detail_IMO2024qnogqh.json` | `https://api.vedur.is/quakes/events/IMO2024qnogqh` | 200 | 1 |
| `internal_detail_IMO2026rusibb.json` | native SeisComP event, 6 origins (5 LOCSAT + 1 SIL) | 200 | 1 |
| `internal_detail_IMO2026rutiqi.json` | native event, `manual` + `rejected` | 200 | 1 |
| `internal_empty.json`, `internal_empty.csv` | empty window | 200 | 0 |
| `empty_before_1991.json` | `1990-01-01` to `1990-01-02`, sil | **200**, `"features" : null` | 0 |
| `error_bad_system.json` | `system=bogus` | 422 (FastAPI `detail` list) | — |
| `error_sil_detail_404.json` | `/internal/events/SIL1352381` | 404 `{"message":"Event not found"}` | — |

Headers: `content-type: application/json` for JSON, and **`text/plain; charset=utf-8` for
CSV**. There is no paging, `Link` or count header. Every response carries an `x-request-id`
(not kept). No size limit was found (plan §3.1: 2 808 events in one response).

## What the readers (T10, T11) need to know

- **GeoJSON and CSV differ.** CSV `magnitude_type` is `SIL mlw` / `SIL autmag`, while GeoJSON
  has `Mlw` / `aut.mag`. CSV names the field `update_time`, GeoJSON `updated_time`. CSV
  SeisComP coordinates are padded to 6 decimals. Use GeoJSON.
- **A window before 1991 is not an error.** It returns 200 with `"features": null`, not `[]`.
- **The public SeisComP list drops fields and events.** It has no `magnitude_type`, `phases`
  or `status`. It keeps the events whose status is `reviewed`, including 2 `automatic` ones,
  and drops the 11 `rejected` ones (48 → 37 on day B). A reader with `internal=False` must
  set `magnitude_type` itself (`ML_SIL`, not verified) and leave the status empty.
- **The public and internal `system=sil` lists are byte-identical** (A and B).
- `evaluation_mode` and `status` combine freely. Day B has `manual/reviewed` 35,
  `automatic/rejected` 7, `manual/rejected` 4 and `automatic/reviewed` 2.
- Back-imported events (2024) have `type: null`. The CSV leaves it empty. Native events
  carry `earthquake`.
- In the detail, `origin` and `magnitude` are a **dict when there is one, a list when there
  are several** (IMO2024qnogqh: one origin; IMO2026rusibb: 6).
- **Horizontal uncertainty units depend on the origin.**
  - SIL origins (author `SIL`): `latitude.uncertainty` / `longitude.uncertainty` are in
    **degrees**. They are the SIL file errors (0.002 / 0.003 for fps #1), and × 111.19 km
    gives `originUncertainty` 222.39 / 333.58 m exactly. No cos(lat) is applied.
  - LOCSAT origins (`scanloc`, analyst): the uncertainties are in **km**, matching
    `horizontalUncertainty` in metres (e.g. 0.237 / 0.164 km against 288 m), with a 90 %
    `confidenceLevel` and a `confidenceEllipsoid`.
  - Depth is in km in this JSON. FDSN QuakeML gives it in metres.
- The public detail is reduced: the preferred origin and magnitude only, no ids, no quality.
- There is no `focalMechanism` in any detail (see `../fdsn/README.md` for the scan).
- `/events/SIL…` returns 404, so SIL detail comes only from the legacy API.

## Overlap with the T0 SIL fixture (`../sil/README.md`)

Window A (2024-08-22, 00–01 UTC):

| T0 row | SIL time | `system=sil` | SeisComP |
|--------|----------|--------------|----------|
| fps #1 | 00:10:04.454 | `SIL1352381` (Mlw 0.25) | `IMO2024qnogqh` (ML_SIL −0.59), same time and position to 3 decimals |
| fps #3 | 00:19:11.927 | `SIL1352384` | `IMO2024qnooms` |
| fps #4 | 00:20:10.194 | `SIL1352385` | `IMO2024qnopim` |
| cleanaut #7 (automatic only) | 00:50:29.1 | `SIL1352541` (`aut.mag` −1.9) | absent (SeisComP holds manual imports only before 2024) |
| aut.mag #1 (not in cleanaut) | 00:04:11.7 | **absent** | absent |

The API also has 00:10:37.669, 00:21:00.452 and 00:36:30.772. These are real archive events
that T0 trimmed out of its fixture.

**Hypothesis, not verified:** the SIL ingest takes automatic events from `cleanaut.mag`, not
`aut.mag`. The one aut.mag-only row in the window is in neither SIL API. T3 and T8 should
count against `cleanaut.mag`.

Window B (2026-09-10): T0 rows 1–6 (aut.mag = cleanaut.mag) are `SIL1453671`, `…72`, `…74`,
`…76`, `…77` and `…78`. In SeisComP they are `IMO2026rusibb`, `rusoag`, `ruuykc`, `ruwcid`,
`ruxyqx` and `rvhifa` (times within 0.4 s). `IMO2026rusibb` is T0 row 1 (01:40:06.2); its
detail also holds the SIL origin (author `SIL`, 64.586 / −20.316).

## Redactions

Analyst logins are replaced with `analyst-N`. N is assigned in order of first appearance
over the whole recording run, so the same person has the same token in all three directories.
The field stays present. Machine authors (`SIL`, `<program>@seiscomp-p01`) are kept.

- `creationInfo.author` of origins and magnitudes: `internal_detail_IMO2026rusibb.json`
  (3 values) and `internal_detail_IMO2026rutiqi.json` (3 values).

No list or CSV response carries an author. The mapping is never written to disk. The user
endpoints (`/internal/picks?user=`, `/internal/picks/pickers`) were not called.
