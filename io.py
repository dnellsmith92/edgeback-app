from __future__ import annotations

import pandas as pd

from .odds import add_market_probabilities, to_decimal


def load_betting_data(path, odds_format: str = "decimal") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"event_id", "event_time", "settled_time", "sportsbook", "market", "selection", "odds", "won"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["decimal_odds"] = to_decimal(df["odds"], odds_format)
    return add_market_probabilities(df)
