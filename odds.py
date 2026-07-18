"""Odds conversion and bookmaker margin removal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_decimal(odds: pd.Series, odds_format: str) -> pd.Series:
    x = pd.to_numeric(odds, errors="raise").astype(float)
    fmt = odds_format.lower()
    if fmt == "decimal":
        if (x <= 1).any():
            raise ValueError("Decimal odds must be greater than 1")
        return x
    if fmt == "american":
        if (x == 0).any():
            raise ValueError("American odds cannot be zero")
        return pd.Series(np.where(x > 0, 1 + x / 100, 1 + 100 / -x), index=x.index)
    if fmt == "fractional":
        def parse(v: object) -> float:
            a, b = str(v).split("/", maxsplit=1)
            return 1 + float(a) / float(b)
        return odds.map(parse)
    raise ValueError("odds_format must be decimal, american, or fractional")


def add_market_probabilities(
    data: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add raw implied probability, overround, and proportional no-vig probability."""
    groups = group_cols or ["event_id", "sportsbook", "market"]
    missing = set(groups + ["decimal_odds"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = data.copy()
    out["implied_prob"] = 1.0 / out["decimal_odds"]
    out["overround"] = out.groupby(groups, sort=False)["implied_prob"].transform("sum")
    out["no_vig_prob"] = out["implied_prob"] / out["overround"]
    return out

