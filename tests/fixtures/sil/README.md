# SIL fixture days (ticket T0, PLAN-seismicity.md)

These files were cut on 2026-11-13 from the SIL archive at `/mnt_data/sildata`, which is
mounted read-only. The tree keeps the archive's own layout and file names:
`<yyyy>/<mon>/<dd>/` holds the day files, and `<yyyy>/<mon>/<dd>/<HH:MM:SS>/<MM:SS:mmm>.sp3`
holds the per-event mechanism block. A reader that walks the archive can therefore walk
this tree too. File names contain `:`, as in the archive, so a Windows checkout cannot hold them.

Every kept line is a byte-exact copy of an archive line, trailing blanks included. The
original `num` column is kept, so a row can be traced back to the archive, and the
numbers have gaps. In each day, the same events are kept in `events.fps`, `events.lib`
and `lib.mag`. The automatic files keep those events' automatic rows, plus a few chosen
automatic-only rows (see Joins). The cut script lives outside the repo; it selected rows
by `num` (reviewed files) or by id (automatic files).

**Do not let whitespace or end-of-file hooks touch this directory.** Trailing blanks
are part of the fixed-width formats. `test_fixed_width_lines_are_intact` pins the width
of every row.

Kept rows, so the cut can be redone without the script. Reviewed files are selected by
`num`, automatic files by id unless noted:

- 2024-08-22: fps nums 1, 3, 4, 38, 42, 53, 64, 115. Automatic rows are those ids, plus
  `00:45:00/50:30:280`, `03:25:00/31:59:429` and `00:00:00/04:14:727`.
- 2025-05-03: fps nums 1, 8, 62, 63, 64, 97. Automatic rows are those ids, plus
  `08:15:00/20:25:008` and `07:40:00/43:38:852`.
- 2022-12-12: fps nums 1, 46, 52, 59. Automatic rows are those ids, plus `00:35:00/37:42:031`.
- 1995-06-15: fps nums 1–4. `aut.mag` archive lines 1, 2, 3, 4 and 10.
- 1993-01-29: `aut.mag` archive lines 1–5. The 0-byte reviewed files are kept as they are.
- 2026-09-10: the first 6 lines of `aut.mag` and `cleanaut.mag`, and the 0-byte `events.lib`.
- Every reviewed event's `<id>.sp3`.

## Personal data

There are no names, initials or user ids in the kept files. The test
`test_fixtures_hold_no_user_ids` checks this. Some per-event files carry a login and
were not copied, so nothing had to be redacted:

- `*.meta_lokimp` has a `user: <login>` line.
- `mail` and `touched.aut` are operational files, not needed.

The kept `.sp3` files contain only an archive-internal path (`file: /sk2/...`, `/eq1/...`).

## Days

| Day | Archive rows (fps / lib / lib.mag / aut / cleanaut) | Kept | Why |
|-----|------------------------------------------------------|------|-----|
| `2024/aug/22` | 250 / 250 / 250 / 372 / 367 | 8 / 8 / 8 / 10 / 8 | The plan's reference day (Sundhnúkur eruption onset). It covers all four sp3 faulting classes, a negative fps strike, dip = 90 planes and every join class. |
| `2025/may/03` | 210 / 210 / 210 / 248 / 224 | 6 / 6 / 6 / 7 / 7 | Event types `ql`, `sf`, `te`, `un`. Has a reviewed id that is 1 ms off its automatic twin. In one event, `aut.mag` and `cleanaut.mag` disagree. |
| `2022/dec/12` | 68 / 68 / 68 / 65 / 65 | 4 / 4 / 4 / 3 / 3 | Event types `ex`, `sm`, `un`. Has a manual id with minute 62 (`20:10:00/62:40:000`). |
| `1995/jun/15` | 26 / 26 / 26 / 31 / absent | 4 / 4 / 4 / 5 / absent | Old era. `aut.mag` is 61 characters wide with no Q and no id, and there is no `cleanaut.mag`. `events.lib` has no event type, and times are space-padded (`0316 5.701`). Ddepth 242.5 is written as 42.5 in fps. sp3 blocks are in the old layout. |
| `1993/jan/29` | 0 / 0 / 0 / 13 / absent | 0 / 0 / 0 / 5 / absent | A real day with **no reviewed events**: the three reviewed files are 0 bytes, which is the case the legacy comment names. Its `aut.mag` is not empty and uses a third layout (117 characters wide). |
| `2026/sep/10` | absent / 0 / absent / 19 / 19 | absent / 0 / absent / 6 / 6 | A real day after February 2026, **automatic only**. It has no `events.fps` and no `lib.mag`, a 0-byte `events.lib`, and identical `aut.mag` and `cleanaut.mag`. |
| `2026/sep/11` | — | 0 / 0 (`events.lib`, `aut.mag`, `cleanaut.mag`) | **SYNTHETIC** fully empty day, modelled on 2026-09-10. No real day without any automatic event was found (the search was stopped part way). The real 2026-09-11 has events. It is here to test empty days and walking a period across consecutive days. |

Note that after February 2026, reviewed files do not stop on a date: `events.fps` and
`lib.mag` still appear on a few days (feb 1–5 and 8, plus 1–3 days a month from March
to July). T3 must decide that a day is reviewed from the files present, not from a date
cutoff.

## Formats

All times are UTC. The plan checked this against the API, to the millisecond. Columns
below are 0-based and end-exclusive. A "fixed" format means slicing is mandatory:
fields can run together (`6.656-0.39`), so splitting on whitespace gives wrong fields.

### `events.fps`: reviewed events, whitespace separated, 33 fields

| # | Name (sil_read) | Meaning / unit | Status |
|---|-----------------|----------------|--------|
| 1 | num | row number in the day | ✓ |
| 2–3 | date, time | `YYYYMMDD`, `HHMMSS.sss`, zero padded, UTC | ✓ |
| 4–5 | latitude, Dlatitude | degrees, error in degrees | ✓ |
| 6–7 | longitude, Dlongitude | degrees, error in degrees | ✓ |
| 8 | depth | km | positive down; **datum not verified**. 0.135 with Ddepth 0.0 looks like a fixed depth (not verified). |
| 9 | Ddepth | km | **loses its hundreds digit**: 242.5 → `42.5`, 314.6 → `14.6` (2015-06-15). fps = lib.mag mod 100. Take Ddepth from `lib.mag`. |
| 10 | MLW | Mlw rounded to 0.1 | ✓ (0.25 → 0.20, 0.69 → 0.70) |
| 11 | id | per-event file path, the join key | ✓; opaque, not a time (minute 62 occurs) |
| 12 | seismic_moment | N·m | ✓ = sp3 moment (3 significant digits) |
| 13 | ev_mag_ATH | ? | **not verified**. Not found in sp3, lib.mag or events.lib. |
| 14 | event_type | 2-letter code: `qu ex ql sf sm te un` | ✓; `un` is not in PLAN §2.1's list |
| 15–16 | sh_min, sh_max | horizontal deviatoric stress (sp3 "lib" line) | = events.lib |
| 17 | azim_comp | azimuth of compression, degrees: the unrounded value **truncated toward zero** to an integer. events.lib has the same value rounded to 0.1, so the two differ by up to 1.0 inclusive (27.6 → 27, −0.4 → 0, 1995: 2.0 → 1). | observed |
| 18 | relative_size | sp3 "the relative size" | ✓ |
| 19 | shearwave_corner_freq | Hz (sp3, value in parentheses) | ✓ |
| 20 | fault_radius | m | ✓ |
| 21 | stress_drop | MPa | ✓ |
| 22 | peak_slip | mm | ✓ |
| 23 | source_file | always `sp3` here: extension of the mechanism file | ✓ |
| 24–29 | strike/dip/slip a, b | degrees, Aki–Richards: dip 0–90, rake −180–180. Plane b is the auxiliary of plane a (±1°). | ✓. **Strike can be negative** (`-86`) on a dip = 90 plane. |
| 30 | num_of_polarities | = sp3 "Of N first motion polarity observations" = events.lib = lib.mag | ✓ |
| 31 | num_of_amplitudes | = sp3 "N amplitudes for P and S" | ✓ |
| 32 | unknown_pacc | 1995: = sp3 "% of all mechanisms are acceptable". 2022 onwards: always `100.00`, matching nothing in sp3. | **meaning changes with era, not verified** |
| 33 | is_significant | 1/0 = sp3 "double couple solution is (not) significant" | ✓ where the sp3 has that line (1995 row 1 has none; fps says 0) |

Known rows (exact strings):

- 2024-08-22 #1: `20240822 001004.454`, 63.88752/0.002, −22.37545/0.003, depth 2.111, Ddepth 0.6, MLW 0.20,
  id `/2024/aug/22/00:05:00/10:05:430`, M0 1.77e+10, ev_mag_ATH 0.19, `qu`, planes 341/74/170 and 74/81/16,
  0 polarities, 24 amplitudes, 100.00, 1.
- 2024-08-22 #53: planes −86/90/28 and 184/62/180, MLW 1.80, 10 polarities, 60 amplitudes.
- 1995-06-15 #1: Ddepth `42.5` (lib.mag `242.5`), MLW −0.40, unknown_pacc 44.04, is_significant 0.

### `events.lib`: reviewed events, fixed width, 136 characters

`num 0–4, date 4–13, time 13–24, latitude 24–33, longitude 33–43, depth 43–50, Mlw 50–55
(0.01), event_type 55–59, id 59–92, sh_min 92–101, sh_max 101–108, azim_comp 108–115 (0.1°),
num_of_polarities 115–118`, blank to 136.

- 1995: event_type is blank, and seconds below 10 are space-padded (`0316 5.701`).
- 0-byte file on days with no reviewed events (1993-01-29) and after February 2026 (2026-09-10).

Known rows:

- 2024-08-22 #1: `001004.454 63.88752 -22.37545 2.111 0.25 qu /2024/aug/22/00:05:00/10:05:430 -0.994 0.905 27.6 0`.
- 1995-06-15 #4: time `0316 5.701` = 03:16:05.701, Mlw 0.40, type blank, azim 39.6.
- 2022-12-12 #46: `ex`, Mlw 1.25, 4 polarities.

### `lib.mag`: reviewed magnitudes, fixed width, 141 characters, no id

`num 0–4, date 4–13, time 13–24, latitude 24–33, longitude 33–43, depth 43–50, Mlw 50–56,
ML 56–62, n_st_ML 62–67, min_ML 67–73, max_ML 73–79, n_st_pha 79–82, n_P 82–85, n_S 85–88,
n_polarities 88–92, D_time 92–98 (s?), Dlat 98–104 (°), Dlon 104–110 (°), Ddepth 110–116 (km),
w_rms_res 116–123 (s?), min_dist_sta 123–130 (km?), max_azim_gap 130–136 (°), vel_model 136–138,
event_type 138–141`.

- Join to `events.fps` by row order and time (verified on 5 archive days: same order, same times).
- **Sentinel:** when no ML exists, the file writes ML −9.99, n_st_ML 0, min 9.99, max −9.99 (2024-08-22 #42).
- 1995 event_type is `qu` here, although the same rows are blank in events.lib.
- Units marked `?` and the meaning of n_st_pha against n_P/n_S are **not verified** (names follow sil_read).

Known rows:

- 2024-08-22 #1: Mlw 0.25, ML −0.50, 4 stations, −0.74/−0.20, phases 9/7/9, 0 polarities, 0.02, 0.002, 0.003, 0.6, 0.06, 3.9, 102.0, 1, `qu`.
- 2024-08-22 #42: ML −9.99, n 0, min 9.99, max −9.99.
- 1995-06-15 #1: Ddepth 242.5, gap 360.0, model 1. 2025-05-03 #8: Mlw 3.09, ML 2.32, `ql`, model 4.

### `aut.mag` / `cleanaut.mag`: automatic events, fixed width, three layouts

- **Modern** (2022 onwards here, 89 characters): `num 0–4, date 4–13, time 13–22 (0.1 s),
  latitude 22–30, longitude 30–39, depth 39–45, ML 45–51 (automatic), Q 51–57 (0–100),
  id 57–89`. `cleanaut.mag` is a subset of `aut.mag`, but **with its own row numbers**.
  For one id, the two files give different values (2025-05-03 `18:43:000`: time 081840.0
  against 081840.1, Q 90.05 against 90.04). ML can be `-0.0`. Q `10.00` is common and
  may be a floor (not verified).
- **1995** (61 characters, no Q, no id, no `cleanaut.mag`): `num 0–4, date 4–13, time 13–22,
  latitude 22–30, longitude 30–40, depth 40–48, ML 48–61` (the value sits at 58–61). The
  legacy specs mis-slice it: the longitude loses a digit, and ML lands in the Q column.
- **1993** (117 characters): `num` zero-padded (`0001`), coordinates to 0.01°, `depth 40–48`.
  It then has several **unidentified** magnitude-like columns, some of them blank, at
  48–61, 61–69, 69–81, 81–93 and 93–111. The boundaries were inferred from 5 rows only.
  **Corrected by T1 (2026-11-13):** these boundaries are wrong. Real 1992–93 lines
  have right-aligned columns ending at 61, 69, 75, … 117. `sil_files.read_aut_mag`
  reads them as `aux1`…`aux10` (meaning not verified) and leaves `ml` NA in this
  layout. The pinned row above happens to pass only because the misaligned slices
  are blank there. Also: the 117 layout is mixed with the 61 layout within 1993, and
  late 1995 to early 1996 has a further 57-character layout (Q, no id).

Known rows:

- cleanaut 2024-08-22 #7: `005029.1 63.943 -22.402 5.2 -1.9 10.00 /2024/aug/22/00:45:00/50:30:280` (automatic only).
- cleanaut 2025-05-03 #65: id `.../08:15:00/20:25:008`, Q 48.73.
- cleanaut 2026-09-10 #6: 09:12:48.7, ML `-0.0`, Q 17.12.
- aut 2024-08-22 #1: `000411.7`, ML 1.5, Q 10.00, `/2024/aug/22/00:00:00/04:14:727` (not in cleanaut, not reviewed).
- aut 1995-06-15 #10: `102244.4 64.156 -20.965 5.2 0.0`.
- aut 1993-01-29 #3: `0003 191032.7 64.24 -20.75 10.0 | 0.94 | 1.81 | — | 0.32 | 0.70`.

### `<id>.sp3`: mechanism block, one per reviewed event (text, partly fixed width)

The block holds:

- the header path;
- a date line;
- a `lib<Mlw> <sh_min> <sh_max> <azim>` line, equal to events.lib. It is fixed width, so a
  negative Mlw touches the label: `lib-0.39`. In 1995 it carries a trailing `6.00 Hz`;
- the moment (`0.177E+11 Nm`) and the local magnitude;
- the corner-frequency, fault-radius, stress-drop and peak-slip ranges;
- the P and T axes, the azimuth of compression and the relative size;
- the two fault planes;
- the normal directions;
- a faulting class: `Strike slip`, `Normal`, `Reverse` or `Dip-slip faulting dominates`;
- statistics: polarities, amplitudes, `% of all mechanisms are acceptable`, significance
  and misfit;
- in 1995 only, an extra `Low frequency limit` line, and a trailing `0` on the
  "two possible fault planes" header.

**Plane convention.** The raw sp3 planes are not Aki–Richards. Dips can exceed 90,
rakes can exceed 180, and strikes can be negative. The rule observed to reproduce the
events.fps planes (as the same double couple, and with fps plane b the auxiliary of plane a; all fixture events, and 564/564 archive
events on 5 days) is:

- if dip ≥ 90: (s, 180 − d, 180 − r);
- otherwise: (s + 180, d, r + 180).

Taken literally, sp3 behaves like a left-hand strike. At dip = 90 the writer's numbers
differ between the two branches, but the mechanism is the same. T3b must compare planes
geometrically, not by their numbers.

Fixture coverage: 13 Strike slip, 4 Normal, 4 Reverse, 1 Dip-slip (22 blocks).

Known blocks (raw values, as written):

- 2024-08-22 #1: plane a −19/106/10, plane b 74/99/164, `0.177E+11 Nm`, `lib 0.25 -0.994 0.905 27.6`,
  Strike slip, significant, 4.63 % acceptable.
- 1995-06-15 #2 (old layout): −40/123/140 and 205/123/40, Reverse, not significant, 22.50 %;
  `lib 1.20 -1.000 0.411 82.8 3.00 Hz`; has a `Low frequency limit` line.
- 2024-08-22 #53: plane a −86/90/152 (vertical; fps −86/90/28), plane b 184/118/0,
  `0.565E+12 Nm`, 10 polarities, 60 amplitudes.

## Joins (T3 targets)

| Day | reviewed ∩ cleanaut | reviewed only | cleanaut only | aut.mag only |
|-----|---------------------|---------------|---------------|--------------|
| 2024-08-22 | 6 | `14:40:00/42:55:609` (in aut.mag only), `21:20:00/26:39:991` (no automatic row) | `00:45:00/50:30:280`, `03:25:00/31:59:429` | `00:00:00/04:14:727`, `14:40:00/42:55:609` |
| 2025-05-03 | 5 | `08:15:00/20:25:007` | `07:40:00/43:38:852`, `08:15:00/20:25:008` (**same event**, 1 ms id difference) | — |
| 2022-12-12 | 2 | `00:25:00/30:22:507`, `20:10:00/62:40:000` | `00:35:00/37:42:031` | — |
| 1995-06-15 | no ids. Automatic rows 1–4 are the reviewed events 1–4 (time within seconds); row 10 matches no reviewed event. | | | |

Joining on id alone counts the 2025-05-03 event twice. The time difference is under
0.01 s, so T3 needs a time/space fallback.

## Not verified

- `ev_mag_ATH`.
- `unknown_pacc` after 1995.
- The depth datum.
- Several lib.mag units (D_time, w_rms_res, min_dist_sta) and n_st_pha.
- The meaning of the 1993 `aut.mag` columns after depth.
- Whether Q 10.00 is a floor.
- Layouts outside the sampled days.
