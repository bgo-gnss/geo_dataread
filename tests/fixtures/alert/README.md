# alertMap / PGV fixtures (ticket T7b, PLAN-seismicity.md)

Frozen on 2026-11-14 for `tests/test_seismic_sil_alert.py`
(`geo_dataread.seismic.sil_alert`). The tests make no network requests and need no
mount. Files are byte-exact copies. The layout follows the source trees: alertMap
`<yyyy>/<mon>/<dd>/`, and SIL `sil/<yyyy>/<mon>/<dd>/<HH:MM:SS>/<MM:SS:mmm>.evb`.
File names contain `:`, as in the archive. Keep whitespace and end-of-file hooks out of
this directory: the `.evb` rows are fixed-width and keep their trailing blanks.

| File | Source | Why |
|------|--------|-----|
| `2026/nov/12/076.event` | `http://sil.vedur.is/alert/2026/nov/12/076.event` | **golden case**: 66 stations, inside the Bárðarbunga box |
| `2026/nov/12/001.event` | same day, `001.event` | 2 stations: the fit check must reject it (< 10) |
| `2026/nov/12/index.html` | same day, `index.html` | day listing, 80 detections |
| `2026/nov/12/alert.mag` | same day, `alert.mag` | alertMap day magnitudes (not the SIL alarm file) |
| `2020/jun/25/001.event` | `…/alert/2020/jun/25/001.event` | 2020 layout, `--.--` placeholders |
| `2026/nov/10/050.event`, `2026/nov/05/042.event`, `2026/nov/03/009.event`, `2026/nov/01/050.event` | same host | more Mpgv page rows (see below) |
| `sil/2026/nov/12/21:50:00/53:43:797.evb` | `/mnt_data/sildata` (read-only mount) | the golden event's evb. The archive rewrites evb files, so this is the 2026-11-14 copy |
| `sil/2026/nov/12/cleanaut.mag` | archive `cleanaut.mag`, line 35 only | the golden origin (21:53:40.5, 64.653, −17.481, Q 90.05), the one the page used |
| `sil/2024/aug/22/21:55:00/59:59:641.evb` | archive | minute and hour rollover (`21 59 59.64` → `22 00  0.02`), p and s rows |
| `sil/2020/jun/25/00:05:00/09:33:883.evb` | archive | the legacy docstring example id |
| `2016/jun/15/index.html`, `2016/jun/15/057.event` | `http://sil.vedur.is/alert/2016/jun/15/…` (T7c, 2026-11-14) | a 2016 day index (66 `NNN.html` links, no `.event` links) and one 2016 `.event` (same 11/13-field layout); the HTTP source lists days from this index |
| `fdsn/station_channels_Z.txt` | `seiscomp-p01:8080/fdsnws/station/1/query?level=channel&format=text` (T7c, 2026-11-14) | Z-channel epochs of the stations in `2026/nov/12/076.event` and `2026/nov/10/050.event`, for the B4 sensor report (`hei` LE-3D 1 Hz, `gha` LE-3D5s → Guralp 40T in 2023, `ada` no description) |
| `mpgv/2026-11.html` | `http://hraun.vedur.is/ja/Mpgv/2026/2026-11.html` | only the page's data table, 10 rows. The page header and footer were cut: the footer names people |

The HTTP files were fetched once, about a dozen small requests with a generic
User-Agent.

## Reference values (Mpgv page, golden row)

`2026-11-12 21:53:40.500`: alert sta 66, filter sta 54, mean **4.17**, median **4.15**,
alertid 76, alertfit `61/66 (92%)`. `sil_alert` gives 66 / 54 / 4.1715 / 4.1534 /
`61/66 (92%)` with the defaults: exact inverse, field 6 ≠ 22, near-field 15 km.

## What the other page rows show

All values below use the exact inverse and the trimming rule (floor(n/3) at each end,
NaN below 3 values).

- **Exact inverse.** The exact inverse reproduces the page's mean on the golden row:
  4.1715 rounds to 4.17. The far-field inverse gives 4.1646, which rounds to 4.16.
  On 11-01 #050 the exact median is 3.3052 (→ 3.31, as the page) and the far-field
  median 3.3041 (→ 3.30).
- **Trimming.** On 11-03 #009 (8 stations), dropping floor(n/3) values at each end
  gives the page's 3.01; dropping round(n/3) would give 3.10. On 11-01 #050 the page
  leaves the mean blank for 2 stations.
- **Field 6 ≠ 22 (B4) fails on two rows:**
  - 11-03 #009: the page keeps four code-22 stations (`gri`, `gil`, `hla`, `gra`).
  - 11-10 #050: `hei` has code 12 and `gha` code 14 (both 22 on 11-12). Dropping
    `hei` and `gha` by name reproduces 54 / 4.39 / 4.40. That name rule in turn fails
    on 11-05 (the page keeps `gha` there).

  The code is a property of the detection, not of the station.
  `test_code22_rule_deviations` pins these numbers.
- **Some `.event` files have grown since the page ran.** 11-01 #050 now has 27 rows
  and fits 23/27. The page shows 24 rows and `20/24`.

## T7c notes (sources, association, sensor report)

- **Day index times can be stale.** In 40 random 2021–2026 days on the hellir
  mount the index ids always equal the `NNN.event` files, but 5 of ~120 sampled
  row times differ from the `.event` header (up to 2 h 40 min; 2025-01-16 #054:
  index 19:24:15, header/`.html`/mtime 22:04). The sources use the index for the
  id list only; association reads the `.event` headers.
- **Late rewrites.** Sep–Oct 2026 `.event` files: mtime − header time median 2 min,
  99.9 % < 46 min, max 1.9 h. The HTTP cache keeps days ≥ 2 days old.
- **Sensor report (B4) on 11-12 #076 / 11-10 #050:** `hei` code 22 + LE-3D 1 Hz
  (consistent); `gha` code 22 but Guralp 40T (not 1 Hz); `ada` code 22, no FDSN
  description; 11-10 `hei` LE-3D 1 Hz **without** code 22. The code-22 rule stays
  provisional.
- **`.event` header time is not the detection time near the origin.** On 34 M ≥ 3,
  Q ≥ 50 SIL events 2016–2023 the nearest detection's header is mostly 5–70 s
  *before* the origin (golden: 4.5 s after), once −140 s, and still fits 72–95 %.
  The legacy window (evb arrival span ± eps on the header time) therefore misses
  some real detections (2016-08-29 01:47 M3.6: header −36.8 s, fit 22/41).
  Association keeps the header-time window and chooses by |header − origin|; the
  page method's fit requirement does the real discrimination. Open for T7d.
