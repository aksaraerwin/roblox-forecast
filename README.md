# Up-and-Coming forecaster — roblox-up-and-coming-forecast-01

`main.py` (Python 3.13, standard library only, no network, deterministic) reads
`$MAKEFUN_INPUT_DIR/history.ndjson` + `task.json` and writes
`$MAKEFUN_OUTPUT_DIR/forecast.json`: per-place probabilities for the Roblox
Up-and-Coming top 10 across the scored window, plus an ordered top 10.

## Model

Two components, both built from the chart's rank history alone (the only fields
present in every read; votes and details are absent from archive reads):

1. **Recency-weighted occupancy.** Every history read adds weight
   `0.5 ** (age_hours / 12)` to the (game, place) cells it shows, spread over
   neighbouring places with a Gaussian kernel (σ = 3.0 places — the chart
   jitters a lot between reads). The final read is weighted ×6: the window
   opens on the chart as it stood.
2. **Trajectory extrapolation.** Each game's weighted place trajectory is
   fitted and extrapolated to the scored window's midpoint (the window is read
   from `task.json`), slope capped and shrunk, then added as a second kernel
   bump at the predicted place (weight 0.5 of the game's occupancy weight).

Each place's named rows are normalised to sum to **0.65**. The other 0.35 stays
unnamed: it is the only mass that can reach games the history never showed —
8% of the practice window's top-10 slots were such games — and deep-chart
climbers not named at a place.

## Backtests (local scorer implementing the rules exactly)

The local scorer reproduces the house baseline's published practice numbers
to 6 decimals (−43.512039, 6.4 hits, 53.5 agreement), which validates it.

- **Practice window** (train on the kit's 34 reads, score on the archive's 10
  reads of 2026-11-08 → 2026-11-10): **−33.335** vs the baseline's −43.512;
  top-10 hits 6.4 (baseline 6.4), agreement 53.6 (baseline 53.5).
- **Within-history splits** (train on the first m reads, score the next 72 h):
  this forecaster beats the baseline on every split tried (m = 14…32), by
  +4.3 to +11.3 log points, mean ≈ +9.8.

## Run

```
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
