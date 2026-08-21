# fpl-ml ⚽ — ML predictions + LP optimization, playing FPL live

A Fantasy Premier League bot running a **live, public experiment**: a real FPL team managed 100% by the model for the 2026/27 season, entered before the Gameweek 1 deadline on day one of the season. Results logged in [`track_record.md`](track_record.md).

## Why this project

FPL is really **two separate problems**, and this repo treats them as swappable modules:

1. **Prediction (ML)** — how many points will each player score next gameweek?
2. **Selection (optimization)** — given predictions, pick the best legal squad: £100.0m budget, 2 GK / 5 DEF / 5 MID / 3 FWD, max 3 per club, valid formation, captain.

The optimizer (integer LP, PuLP/CBC) never changes — FPL rules are fixed. The prediction module is where all iteration happens, and every version must **beat the previous one on a held-out season** to earn its place.

## Results

**v0.2 (day one, cold-start features only)** — single holdout, full 2025/26 season:

| prediction module | MAE (pts/match) |
|---|---|
| naive (last-season pts ÷ 38) | 1.357 |
| hand-crafted heuristic v0.1 | 1.430 ❌ |
| **ML v0.2 (gradient boosting)** | **1.283 ✅** |

**v0.3 (pre-GW2: rolling form, team form from goals, walk-forward)** — retrained every 4 GWs across 2025/26, training strictly on the past. Not comparable to the v0.2 table: different features *and* a much stronger naive (rolling 5-GW mean).

| model | MAE | Spearman | precision@10 | season backtest* |
|---|---|---|---|---|
| naive (rolling 5) | 1.051 | **0.711** | 0.087 | 1900 pts |
| Ridge | 1.046 | 0.691 | 0.108 | — |
| gradient boosting | 0.992 | 0.706 | 0.113 | **2188 pts** |
| two-stage P(60+)×E(pts) | **0.911** | 0.702 | 0.113 | 2160 pts |

*fresh LP squad each GW, actual points, captain doubled, oracle transfers, no auto-subs — internal comparison only.

Honest findings so far:
- The v0.1 hand-crafted heuristic **lost to the naive rule** — always benchmark before trusting intuition.
- In v0.3 the naive rolling average is the **best ranker** (Spearman) — but ML wins where it matters: calibration (MAE), hauls (precision@10) and, decisively, **+288 pts over naive in the season backtest**.
- Best MAE ≠ best squad: two-stage wins MAE, plain gradient boosting wins the backtest — so deployment uses the backtest winner. **We optimize the metric we deploy.**
- precision@10 ≈ 0.11 for every model — FPL hauls are genuinely hard to predict; regression to the conditional mean means the model will never "promise" a 15-point haul.

Full methodology for both versions: [`README_MODEL.md`](README_MODEL.md).

## Repo layout

```
fpl_features.py       # shared: feature engineering + LP optimizer (captain in objective)
predict_gw.py         # weekly: train on all history, predict upcoming GW, pick squad
train_v03.py          # walk-forward evaluation + metrics + season backtest
select_team.py        # v0.1 — heuristic baseline (kept for the record)
train_ml.py           # v0.2 — day-one ML (kept for the record)
fetch_data.sh         # downloads current + historical data (vaastav archive)
predictions/          # UTC-timestamped squad outputs per GW
data/                 # frozen snapshot of all GW1 input data (reproducibility)
track_record.md       # live season log: model pts vs account pts, rank
.github/workflows/    # weekly automation (Fridays 06:00 UTC + manual trigger)
```

## Reproduce

```bash
pip install -r requirements.txt
bash fetch_data.sh          # or: unzip data/data_snapshot_gw1.zip for exact GW1 state
python train_v03.py         # walk-forward evaluation: full metrics table + backtest
python predict_gw.py        # squad for the upcoming GW (auto-detected; --gw N to override)
```

Weekly automation: the GitHub Actions workflow re-fetches data, retrains and commits `predictions/gw{N}.csv` every Friday 06:00 UTC (or on demand via *Run workflow*).

Data comes from the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) community archive (mirror of the official FPL API) — thanks to its maintainers.

## Chip policy

The model currently plays **no chips** — a deliberate baseline, not an omission. v0.x has no multi-gameweek planning, and chips are bets on future gameweek structure. Planned order: Triple Captain and Bench Boost first (simple double-gameweek heuristics: TC on the highest-predicted DGW captain, BB when predicted bench sum clears a threshold), then Wildcard/Free Hit, which need season simulation. Until a chip module exists and is backtested, unused chips are the honest default.

## Roadmap

- ✅ ~~Rolling form features (3/5/10 windows)~~ · ✅ ~~walk-forward validation~~ · ✅ ~~Spearman + precision@10~~ · ✅ ~~Ridge in the ladder~~ · ✅ ~~two-stage minutes model~~ · ✅ ~~captaincy in the LP objective~~ · ✅ ~~season backtest~~ · ✅ ~~weekly GitHub Actions~~
- LSTM sequence model (deliberately deferred — must beat gradient boosting to ship)
- Own xG-based opponent strength refinement; transfer module (squad continuity, max Δ-xP under 1 free transfer); chip module; hit (-4) logic
- "Model XI pts" auto-scorer for `track_record.md`; simple dashboard

## Disclaimer

Built for fun and to see how far a model can go in a live season. v0.1 and v0.2 were shipped under a real deadline (season started the same day) — limitations are documented, not hidden.
