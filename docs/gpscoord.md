# gpscoord — coordinates in and out

One tool for coordinate tables. The maths is geofunc's (`geofunc.points`,
`geofunc.columns`); `gpscoord` picks stations, reads their `.pos` series and
writes the files. `gpsprod` stays the tool for station *motion* products.

| subcommand | what |
|---|---|
| `convert FILE` | a point table in another frame (`geofunc points`): tab/;/comma/space, decimal commas, frame from a header like `X (ISN93) [m]` or `--from` |
| `nearest FILE` | the nearest stations to each point: every station with a `.pos` series, placed at its latest solution with σ ≤ `-u` (default 10 mm) — never the approximate stations.cfg coordinates (BGÓ 2026-11-04) — or `--candidates FILE`; `-k`, `--max-km`; `--data-before D` / `--data-after D` (the station's .pos span, from the header), `--join A+B` (a moved station as one: union span, last member's place); `d_east_isn93_m`/`d_north_isn93_m` = the station less the point on the `--grid` (default isn93); `first_day`/`last_day` the station's .pos span; `--shifts` = how far the STATION moved first→last observation (5 kept days each end, cleaned, σ ≤ 10 mm; not the point — that needs its own measurement), `--since D` = from the 5 kept days nearest D (either side) to the end |
| `series STA…` | daily station coordinates with the columns you choose (below) |
| `at STA… --epoch D` | where stations are at an epoch (= `gpsprod coords`) |

## series

Positions are the `.pos` X/Y/Z with the reviewed junction offsets
(`products.positions.load_positions`, the coordinate product's read). The
view (`--view cleaned|declared|raw`, σ screen `-u`, default 10 mm) decides
which days are kept; only kept days are written unless `--with-dropped`
(every day plus a `kept` 0/1 column). Nothing is removed from the positions
(no plate, no seasonal) — they are coordinates.

Columns: `--columns` (default the `isn93` preset) then `--add`, each item
`[NAME=]KEY[@FRAME][:REL]` — time (`date datetime yearf mjd`), ECEF `x y z`,
`lat lon h` (GRS80 at the observation epoch), `east@F north@F` (isn93,
isn2004, isn2016), local `dn de du` from `--ref-pos`, σ `sx sy sz sn se su`,
`kept`. `:REL` subtracts one value from the kept days: `firstN`, `first`,
`last`, `mean`, `median`, `nearDATE[/N]` (the 5 kept days nearest DATE, either side), `A..B` (dates), a number. Presets expand anywhere:
`isn93`, `isn2004`, `isn2016`, `llh`, `xyz`, `neu`. Full list:
`gpscoord series --help`.

`east@isn93`/`north@isn93` are the station's position ON THE DAY,
projected (`--epoch-shift none`, default, BGÓ 2026-11-04): where it is now in
ISN93 coordinates. Old ISN93 coordinates (surveyed points) have drifted with
the plate since 1993.6. `--epoch-shift plate` takes each day back to the
frame's epoch (ISN93 1993.6, ISN2004 2004.6, ISN2016 2016.5) on the station's
plate (platefile + the frame's plate-motion model) to compare with such old
coordinates. The header says which was done. A shift since a date is a
`:REL` column (`dE=east@isn93:first5` from the period's start, or a date
window).

Period: `--start/--end/--days` or `--period` (dates, events.yaml names);
default the whole series. The view is always run over the whole series, so
a day is kept or dropped alike in every period. A station with nothing in
the period, or nothing passing the view and σ screen, is skipped with the
dates it has (`data from A to B (N days)`).

```bash
gpscoord series SENG                                   # isn93 preset to the screen
gpscoord series --stations-from naesta.csv --add dup=h:first5 -o out/   # out/STA.csv
gpscoord series SENG --view raw --with-dropped --add dup=h:first5
gpscoord series SENG --columns date,yearf,du,su --ref-pos first5 --start 20231101
gpscoord series --stations-from naesta.csv --start 20230525 \
    --add dE=east@isn93:first5,dN=north@isn93:first5,dup=h:first5 -o out/
```

Moved stations: LISK (2021-02-28 → 2024-10-08) → LISF (2024-10-09 →), ~300 m apart (LISF ~280 m E, ~100 m N, 41 m up): `nearest --join LISK+LISF`, and `series LISK+LISF` writes ONE series — a `mark` column, absolute columns each mark's own position, difference columns (`:REL`, dn/de/du) tied: LISF moved by LISK's last `--tie-days` (5) kept days less LISF's first 5 (no motion assumed across the 1-day gap); the tie is in the header. The tie is always taken on the days around the move (each mark's whole series), whatever `--start/--period` asks for. Marks more than 50 m apart get a `# warning` line: they may move differently (a fault between them); the tie only bridges the move (BGÓ 2026-11-04).

### nearest: shifts from series files, the place then, a fallback rank

- `--from-series DIR`: `--shifts`/`--since` from the files `gpscoord series`
  wrote to `DIR` (`DIR/STA.csv`, `DIR/A+B.csv` for a moved station): the
  shift = the mean of the last 5 kept rows of `dE/dN/dup` (tied across a
  move; else `east/north/h`) less the first 5 (or the 5 nearest `--since`);
  `first_day`/`last_day` = the file's kept rows. A station without a file
  falls back to its `.pos` (a note on stderr). So the list and the delivered
  station files always agree.
- With `--since D` the station is placed where it was at `D` (the points'
  coordinates are of about that epoch, e.g. the boreholes' ISN93 of June
  2023): `station_lon`/`station_lat` columns, and `distance_km`,
  `azimuth_deg`, `d_east`/`d_north` measured to that place, not the latest one.
- `-u MM` (default 10): the σ screen for the candidates' latest position and
  the shift days (e.g. `-u 30` for a station with noisy first days).
- `--fallback-before D`: when the nearest station started after `D`, the
  nearest one with data before `D` is added as rank 2 (with `-k 1`; e.g.
  `20230602`). gps_plot `map station --nearest-from LIST --nearest-ranks 1,2`
  draws rank 2 dashed grey.

First use 2026-11-04: the stations nearest the Reykjanes boreholes, from
2023-05-25, dE/dN/dup from each file's first 5 kept days →
`~/gps-data/data/reykjanes-borholur/` (SAFH with `-u 30`: its Up σ is
~18 mm; SUND has only 38 days of daily solutions, metre-level σ in itrf14).
