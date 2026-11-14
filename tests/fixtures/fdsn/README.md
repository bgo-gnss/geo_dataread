# FDSN event service fixtures (ticket T9, PLAN-seismicity.md §3.2)

Recorded 2026-11-13 from host `eldey` (inside the IMO network) by
`tests/fixtures/record_api_fixtures.py`. `manifest.json` has the exact URLs, statuses,
headers, sizes and counts.

- Service: `http://seiscomp-p01.vedur.is:8080/fdsnws/event/1/query`. It is plain HTTP and
  reachable from this laptop.
- `Server: SeisComP-FDSNWS/1.2.5`. The body reports service version 1.2.6.
- Content types: `text/plain; charset=utf-8` (text) and `application/xml; charset=utf-8` (QuakeML).

Bodies are the raw bytes except for the redaction below.

## Requests

| File | Query | Status | Events |
|------|-------|--------|--------|
| `text_20240822T00.txt` | `starttime=2024-08-22T00:00:00&endtime=2024-08-22T01:00:00&format=text` | 200 | 6 |
| `text_20260910.txt` | `starttime=2026-09-10T00:00:00&endtime=2026-09-11T00:00:00&format=text` | 200 | 48 |
| `quakeml_20240822T00.xml` | window A, default (QuakeML, preferred origin and magnitude) | 200 | 6 |
| `quakeml_IMO2024qnogqh_all.xml` | `eventid=…&includeallorigins=true&includeallmagnitudes=true&includefocalmechanism=true` | 200 | 1 |
| `quakeml_IMO2026rusibb_all.xml` | the same for the native 2026 event (6 origins) | 200 | 1 |
| `text_20240822T00_limit2_asc.txt` | window A `&limit=2&orderby=time-asc` | 200 | 2 |
| `error_nodata404.txt` | empty 2-minute window, `nodata=404` | 404 | — |
| `error_bad_param.txt` | window A `&bogus=1` | 400 `invalid param: bogus` | — |
| `error_updatedafter.txt` | window A `&updatedafter=2026-01-01T00:00:00` | 400 `invalid param: updatedafter` | — |
| `error_updateafter.txt` | window A `&updateafter=…` | 400 `filtering based on update time not supported` | — |

Not saved: an empty window without `nodata=404` returns **HTTP 204 with an empty body**. The
reader must treat 204 as zero events.

The day-B QuakeML list (48 events, about 200 KB) is not kept. The text list covers that
window, and the 2026 detail covers QuakeML for a native event.

## What the readers (T13, T12b, T14) need to know

- **There is no `updatedafter`.** The standard name is rejected, and the WADL's
  `updateafter` is listed but "not supported". This removes the reason the plan gave for an
  FDSN reader (§3.2, T13). T14 must re-fetch a recent window by time instead.
- **Paging exists**, unlike the Quakes API: `limit`, `offset` and `orderby` (`time-asc`, …).
- **The default order is newest first**, so the reader must sort.
- Text format: `#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType`.
  - Times have no `Z` (UTC).
  - `EventType` is empty for back-imported events, and `earthquake` or `not existing` for
    native ones. `not existing` is how rejected events show here. There is no status column.
  - Magnitudes are full floats.
- **QuakeML depth is in metres** (`2100` ± `600` for IMO2024qnogqh), while the Quakes JSON
  gives km. Lat/lon uncertainty: the same split as the Quakes detail (degrees on SIL origins,
  km on LOCSAT origins, see `../quakes/README.md`).
- The default list returns rejected events as well: the 48 events of day B equal the
  internal Quakes list.
- **No focal mechanisms in SeisComP.**
  - Both event details were requested with `includefocalmechanism=true`: no `focalMechanism`.
  - Two scans, not saved (880 KB each), found none either:
    - `starttime=2026-02-05T00:00:00&endtime=2026-11-13T00:00:00&minmagnitude=3.0&includefocalmechanism=true&includeallfocalmechanisms=true&includestationmts=true`
      (270 events);
    - `starttime=2015-01-01T00:00:00&endtime=2026-02-05T00:00:00&minmagnitude=4.0&includefocalmechanism=true&includeallfocalmechanisms=true`
      (322 events).
  - The Quakes detail of the M 4.4 `IMO2026ftypnt` has no mechanism key either.
  - For T12b: SeisComP has no mechanisms today, so the fault planes come from SIL only.
- WADL parameters (`/fdsnws/event/1/application.wadl`): the standard set plus
  `includepicks`, `includefocalmechanism`, `includeallfocalmechanisms`, `includestationmts`,
  `includecomments` and `formatted`.

## Overlap with the T0 SIL fixture

The same events as `../quakes/README.md`. Window A: fps #1, #3, #4 are `IMO2024qnogqh`,
`IMO2024qnooms`, `IMO2024qnopim`. Window B: T0 rows 1–6 are `IMO2026rusibb`, `rusoag`,
`ruuykc`, `ruwcid`, `ruxyqx`, `rvhifa`.

## Redactions

Analyst logins become `analyst-N`, consistent across all three fixture directories. The field
stays present. Machine authors (`SIL`, `<program>@seiscomp-p01`) are kept.

- `text_20260910.txt`: the `Author` and `MagAuthor` columns (83 cells, 2 distinct analysts).
- `quakeml_IMO2026rusibb_all.xml`: `creationInfo/author` (3 elements).

No login appears in a `publicID` or a comment (checked with grep after redaction).

## Added by T13 (2026-11-14)

- `quakeml_IMO2026rusibb_arrivals.xml`: `eventid=IMO2026rusibb&includearrivals=true&includepicks=true`
  (200, 45 403 bytes, 53 arrivals and 53 picks on the preferred origin). Trimmed by hand to
  5 arrivals and their 5 picks (2 S, 2 weighted P, 1 P with `timeWeight` 0 from `scautopick`).
  The station code is only in the pick (`waveformID`), so arrivals need `includepicks`.
  Redaction: the pick and origin `creationInfo/author` login → `analyst-1` (the same analyst
  as in `quakeml_IMO2026rusibb_all.xml`).
- Probed, not saved: `offset` is 1-based (`offset=0` → 400 `parameter not in domain
  [1,…]: offset`); `starttime` and `endtime` are both inclusive and take microseconds.
- 2026-11-13 (live, not saved): 2 of 64 events (automatic SILCLEAN preferred origins) carry a
  `preferredMagnitudeID` (`…/netMag/ML_SIL` on the SILCLEAN origin) that is not in the QuakeML,
  the text list (empty MagType/Magnitude) or the Quakes detail; the Quakes list still gives an
  `ML_SIL` value for them. `test_dangling_preferred_magnitude_is_na_not_replaced` is synthetic.
