#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for make.fun (roblox-up-and-coming-forecast-01).

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json (defaults /input) and writes
$MAKEFUN_OUTPUT_DIR/<task output> (default /output): per-place probabilities for the
chart's top places plus an ordered top 10. Python 3.13, standard library only,
no network, deterministic.

Model: recency-weighted empirical occupancy plus a trajectory extrapolation.
Each history read contributes to every (game, place) cell with weight
0.5 ** (age_hours / HALF_LIFE_H); observed places are spread over neighbouring
places with a Gaussian kernel (the chart jitters by several places between
reads); the last read gets an extra weight multiplier (the window opens on the
chart as it stood). On top of that, each game's weighted place trajectory is
extrapolated to the scored window's midpoint (window read from task.json) and
added as a second kernel bump, so climbers gain mass at higher places and
fallers lose it. Each place's named rows are normalised to sum to NAMED_MASS:
the remaining 1 - NAMED_MASS stays unnamed, the only mass that can reach games
the history never showed (8% of the practice window's top-10 slots were such
games) and deep-chart climbers the forecast does not name at a place.
"""

import json
import math
import os
from datetime import datetime, timezone

# --- tunable parameters (chosen by backtest; see README) ---
HALF_LIFE_H = 12.0        # recency half-life of a history read, hours
KERNEL_SIGMA = 3.0        # Gaussian spread of an observed place over neighbours
KERNEL_CUT = 6            # kernel support, in places
LAST_READ_BOOST = 6.0     # extra weight multiplier for the final history read
NAMED_MASS = 0.65         # per-place probability mass given to named rows
P_CAP = 0.55              # no single (game, place) row above this
MAX_ROWS_PER_PLACE = 100  # rows kept per place (rules allow 100)
TRACK_PLACES = 30         # chart places from each read that feed the model
TREND_W = 0.5             # weight of the place-trajectory extrapolation component
TREND_SLOPE_CAP = 0.15    # max |places per hour| the trend may extrapolate
TREND_SHRINK = 3.0        # slope shrinkage: beta *= W / (W + TREND_SHRINK)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chart = [row["universe_id"] for row in sorted(r["up_and_coming"], key=lambda x: x["rank"])]
            reads.append({"at": parse_ts(r["at"]), "chart": chart})
    reads.sort(key=lambda r: r["at"])
    return task, reads


def build_forecast(task, reads):
    places = int(task["places"])
    history_until = parse_ts(task["history_until"])
    window = task.get("window") or {}
    if window.get("from") and window.get("until"):
        horizon_h = (parse_ts(window["until"]) - parse_ts(window["from"])).total_seconds() / 3600.0
    else:
        horizon_h = 72.0
    mid_h = horizon_h / 2.0  # the forecast is scored across the window; aim at its middle
    last_idx = len(reads) - 1

    # raw[g][k] = weighted occupancy mass of game g at place k (k 1-based in list slots 1..places)
    raw = {}
    traj = {}  # g -> list of (hours before history_until, place, weight)
    for i, read in enumerate(reads):
        age_h = max(0.0, (history_until - read["at"]).total_seconds() / 3600.0)
        w = 0.5 ** (age_h / HALF_LIFE_H)
        if i == last_idx:
            w *= LAST_READ_BOOST
        chart = read["chart"]
        for j, g in enumerate(chart[:TRACK_PLACES], start=1):
            grow = raw.setdefault(g, [0.0] * (places + 1))
            for k in range(max(1, j - KERNEL_CUT), min(places, j + KERNEL_CUT) + 1):
                d = k - j
                grow[k] += w * math.exp(-(d * d) / (2.0 * KERNEL_SIGMA * KERNEL_SIGMA))
            traj.setdefault(g, []).append((-age_h, j, w))

    # Trend component: extrapolate each game's place trajectory to the window's
    # midpoint and add a kernel bump there, scaled by the game's total weight.
    if TREND_W > 0.0:
        for g, pts in traj.items():
            if len(pts) < 3:
                continue
            sw = sum(p[2] for p in pts)
            mx = sum(p[2] * p[0] for p in pts) / sw
            my = sum(p[2] * p[1] for p in pts) / sw
            sxx = sum(p[2] * (p[0] - mx) ** 2 for p in pts)
            if sxx <= 1e-9:
                continue
            sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts)
            beta = sxy / sxx
            beta = max(-TREND_SLOPE_CAP, min(TREND_SLOPE_CAP, beta))
            beta *= sw / (sw + TREND_SHRINK)
            pred = my + beta * (mid_h - mx)
            pred = max(1.0, min(float(TRACK_PLACES), pred))
            grow = raw.setdefault(g, [0.0] * (places + 1))
            amp = TREND_W * sw
            lo = max(1, int(math.floor(pred)) - KERNEL_CUT)
            hi = min(places, int(math.ceil(pred)) + KERNEL_CUT)
            for k in range(lo, hi + 1):
                d = k - pred
                grow[k] += amp * math.exp(-(d * d) / (2.0 * KERNEL_SIGMA * KERNEL_SIGMA))

    probabilities = []
    game_total = {}
    game_wplace = {}
    for k in range(1, places + 1):
        col = [(g, raw[g][k]) for g in raw if raw[g][k] > 0.0]
        col.sort(key=lambda x: (-x[1], x[0]))
        col = col[:MAX_ROWS_PER_PLACE]
        total = sum(v for _, v in col)
        if total <= 0.0:
            continue
        ps = {g: v / total * NAMED_MASS for g, v in col}
        # cap and redistribute the excess, a few passes
        for _ in range(8):
            over = [(g, p) for g, p in ps.items() if p > P_CAP]
            if not over:
                break
            excess = sum(p - P_CAP for _, p in over)
            for g, _p in over:
                ps[g] = P_CAP
            rest = [g for g in ps if ps[g] < P_CAP]
            rest_total = sum(ps[g] for g in rest)
            if rest_total <= 0.0:
                break
            for g in rest:
                ps[g] += excess * ps[g] / rest_total
        for g, p in ps.items():
            if p <= 0.0:
                continue
            probabilities.append({"universe_id": g, "place": k, "p": p})
            game_total[g] = game_total.get(g, 0.0) + p
            game_wplace[g] = game_wplace.get(g, 0.0) + p * k

    ranked = sorted(game_total, key=lambda g: (-game_total[g], game_wplace[g] / game_total[g], g))
    top = ranked[:places]
    top.sort(key=lambda g: (game_wplace[g] / game_total[g], -game_total[g], g))
    return {"probabilities": probabilities, "top10": top}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_history(input_dir)
    forecast = build_forecast(task, reads)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task["output"])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
