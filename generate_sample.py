from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = []
start = pd.Timestamp("2022-01-01", tz="UTC")
for game in range(600):
    rating_diff = rng.normal(0, 1)
    rest_diff = rng.integers(-3, 4)
    true_p = 1 / (1 + np.exp(-(0.75 * rating_diff + 0.08 * rest_diff)))
    home_win = rng.random() < true_p
    margin = 1.045
    market_p = np.clip(true_p + rng.normal(0, .06), .08, .92)
    probs = [market_p * margin, (1 - market_p) * margin]
    event_time = start + pd.Timedelta(days=game)
    for side, p, won, sign in [("home", probs[0], home_win, 1), ("away", probs[1], not home_win, -1)]:
        rows.append({"event_id": f"G{game:04}", "event_time": event_time, "settled_time": event_time + pd.Timedelta(hours=3),
                     "sportsbook": "samplebook", "market": "moneyline", "selection": side,
                     "odds": round(1 / p, 4), "won": int(won), "rating_diff": sign * rating_diff,
                     "rest_diff": sign * rest_diff})
Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("data/sample_games.csv", index=False)

