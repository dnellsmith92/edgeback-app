# EdgeBack

EdgeBack imports historical results and closing/pregame sportsbook odds, removes proportional bookmaker vig, produces walk-forward model probabilities, selects positive-EV bets, and simulates a constrained fractional-Kelly bankroll.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python scripts/generate_sample.py
edgeback data/sample_games.csv --features rating_diff,rest_diff --min-train-rows 200 --output output
pytest
```

## Run the web app

```bash
streamlit run app.py
```

For Streamlit Community Cloud, upload the entire project to one GitHub repository and choose `app.py` as the main file. `requirements.txt` installs the necessary packages automatically.

The run writes `metrics.json`, `bets.csv`, and `predictions.csv`. ROI is profit divided by total amount staked; `bankroll_return` is profit divided by initial bankroll; maximum drawdown is peak-to-trough bankroll decline.

## Input schema

One row represents one possible market selection. Required columns:

| Column | Meaning |
|---|---|
| `event_id` | Stable game identifier |
| `event_time` | Bet cutoff/start time with timezone |
| `settled_time` | Time the result became knowable |
| `sportsbook` | Book identifier |
| `market` | e.g. moneyline |
| `selection` | e.g. home/away |
| `odds` | Price in the selected input format |
| `won` | 1 for win, 0 for loss |
| feature columns | Numeric information genuinely available before `event_time` |

Each event/book/market group must contain every mutually exclusive outcome so no-vig probabilities normalize correctly. Pushes and voids should be excluded or explicitly transformed before import. Do not include final score, postgame statistics, closing data published after the wager timestamp, or features computed over the full dataset.

## Look-ahead safeguards

- Rows are ordered by event time.
- Training outcomes must have `settled_time < event_time`; equality is excluded.
- All selections at the same timestamp are predicted as a batch.
- Model preprocessing is fitted only inside each historical training window.
- Simultaneous stakes use the same pre-event bankroll.
- `--max-bet-fraction` caps each wager and `--max-event-exposure` caps all wagers sharing a timestamp.

This is research software, not a promise of profit. Validate data quality, line availability, limits, pushes, commissions, and execution slippage before risking money.
