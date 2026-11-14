# Skjálftalísa (legacy SIL) API fixtures (ticket T9, PLAN-seismicity.md §3.3)

Recorded 2026-11-13 from host `eldey` (inside the IMO network) by
`tests/fixtures/record_api_fixtures.py`. `manifest.json` has the exact URLs, statuses,
headers, sizes and counts.

- Internal: `https://vi-api.vedur.is/skjalftalisa/internal/quakes`.
- Public: `https://api.vedur.is/skjalftalisa/quakes`.
- OpenAPI version `2026-04-30`.

**Only JSON is offered:** there is no `format` parameter. Every response is
`content-type: application/json`, with no paging or count header. Bodies are the raw bytes.

## Requests

| File | Request | Status | Rows |
|------|---------|--------|------|
| `internal_20240822T00.json` | `?start_time=2024-08-22T00:00:00Z&end_time=2024-08-22T01:00:00Z` | 200 | 13 rows, 7 events |
| `public_20240822T00.json` | public, the same window | 200 | 10 rows, 5 events |
| `internal_20260910.json` | `?start_time=2026-09-10T00:00:00Z&end_time=2026-09-11T00:00:00Z` | 200 | 19 (all automatic) |
| `public_20260910.json` | public, the same | 200 | 18 |
| `internal_20240822T00_count.json` | `/internal/quakes/count`, window A | 200 | `{"quake_count":7}`: it counts **events**, not rows |
| `internal_detail_1352381.json` | `/internal/quakes/1352381` (reviewed + automatic, with `sp3`) | 200 | — |
| `internal_detail_1352541.json` | automatic-only event (`sp3: null`) | 200 | — |
| `internal_detail_1453678.json` | a 2026 automatic event | 200 | — |
| `public_detail_1352381.json` | `https://api.vedur.is/skjalftalisa/quakes/1352381` | 200 | — |
| `internal_empty.json` | `2024-08-22T00:00:00Z` to `00:02:00Z` | 200 | 0 |
| `error_naive_time.json` | window A without a timezone | **422** `timezone_aware` | — |
| `error_count_out_of_bounds.json` | count, `1991-01-07T00:00:00Z` to `2026-11-13T00:00:00Z` | 422 `Start time is out of bounds.` | — |
| `error_detail_404.json` | `/internal/quakes/999999999` | 404 `{"message":"Quake not found"}` | — |

## What the reader (T12, T12b) needs to know

- **One `event_id`, two rows** (the "never merge the two" case). In
  `internal_20240822T00.json`, 1352381, …82, …84, …85, …86 and …87 each have a
  `SIL picks` row and a `SIL aut.mag` row. 1352541 has only `SIL aut.mag`. The row order
  within a pair is not fixed: 1352385 lists picks first.
- **`event_id` is an integer** here (`1352381`), and the Quakes API spells the same event
  `SIL1352381`.
- **Query times must be timezone-aware** (a naive time gives 422), but **response times
  are naive** (`2024-08-22T00:10:04.454`, UTC).
- List fields:
  - `SIL picks` rows: `m_ml`, `m_mlw`, `seismic_moment`, `depth`, `event_type` (2-letter
    code). The internal list adds `strike`, `dip`, `slip`: **one plane, Aki–Richards**, equal
    to fps plane a (341/74/170 for 1352381).
  - `SIL aut.mag` rows: `m_autmag` and `quality`.
- **The public list:**
  - drops strike/dip/slip;
  - rounds `m_mlw` to 0.1 and the reviewed coordinates to 3 decimals;
  - drops whole events. It keeps both rows or neither. Window A loses 1352387 (Mlw −0.02)
    and 1352541 (aut −1.9); day B loses 1453694 (aut −0.4). The rule looks like
    "event magnitude < 0 is dropped", but that is **not verified**: 1352381's automatic row
    at −1.2 is kept because its Mlw is 0.25.
- **Detail** (internal):
  - `origins[]` carries `latitude_error` / `longitude_error` in degrees, `depth_error` in km,
    `number_of_stations_in_location` and `quality`.
  - **`nearest_station_km` is a string** (`"3.9"`).
  - `origin_time` is naive with microseconds.
  - `row_num` on automatic origins is the **`aut.mag` row number** (1352541 → 8,
    1352381 → 2), not the cleanaut one.
  - `magnitudes[]` uses the types `Mlw`, `Ml` and `Autmag`.
  - `slungapath` is the SIL file id, the join key to the files. It is present for
    automatic-only events too.
  - `touched_aut` is 1, 0 or null.
- **`sp3` is a JSON object encoded as a string** inside the JSON: it must be decoded twice.
  It holds both planes:
  - in the raw left-hand convention: `fpa_strike/dip/slip` −19/106/10 and `fpb_*`
    74/99/164, exactly the T0 `.sp3` block;
  - **converted to Aki–Richards**: `fpa_*_AR` 341/74/170 and `fpb_*_AR` 74/81/16, equal to
    T0 `events.fps` #1.
  - It also holds the moment, corner frequency, stress drop, misfits and significance, but
    no faulting class (`faulting_mechanism` is `""`).

  T12b can take the planes from the `_AR` fields directly. `sp3` is `null` for automatic
  events.
- The public detail is reduced: one magnitude `M` 0.3, one origin, coordinates to 3
  decimals, **time truncated to whole seconds**. It keeps `slungapath`.
- The count endpoint refuses the 1991 start that the Quakes API accepts. The allowed span
  was not mapped.

## Overlap with the T0 SIL fixture (`../sil/README.md`)

Window A:

| T0 row | Legacy id | `slungapath` |
|--------|-----------|--------------|
| fps #1, 00:10:04.454 | 1352381 (picks + aut row 2) | `/2024/aug/22/00:05:00/10:05:430`, equal to the fps #1 id |
| fps #3, 00:19:11.927 | 1352384 | (list only) |
| fps #4, 00:20:10.194 | 1352385 | (list only) |
| cleanaut #7, 00:50:29.1 | 1352541 (aut only) | `/2024/aug/22/00:45:00/50:30:280`, the T0 automatic id |
| aut.mag #1, 00:04:11.7 | **absent** | — |

Day B: T0 rows 1–6 are 1453671, …72, …74, …76, …77 and …78. 1453678 (09:12:48.7, Q 17.12)
has `slungapath` `/2026/sep/10/09:10:00/12:50:312`, the T0 cleanaut #6 id.

## Redactions

None were needed. No response carries an author, user or e-mail field (checked by grep
after recording). The user endpoints of this API (`/internal/lokimp/users`,
`/internal/lokimp/picker/{username}`, `/internal/lokimp/updates`) were deliberately never
called.
