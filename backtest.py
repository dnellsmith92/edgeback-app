"""Walk-forward prediction, value selection, and bankroll simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll: float = 1_000.0
    min_train_rows: int = 200
    retrain_every: int = 50
    min_edge: float = 0.02
    min_ev: float = 0.01
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.03
    max_event_exposure: float = 0.10
    min_bet: float = 1.0
    max_bet: float | None = None
    random_state: int = 7


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, float | int]


def _model(random_state: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2_000, random_state=random_state)),
    ])


def walk_forward_predict(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Predict chronologically; each fit uses only outcomes settled before prediction time."""
    required = {"event_time", "settled_time", "won", *feature_cols}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = data.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["settled_time"] = pd.to_datetime(df["settled_time"], utc=True)
    df = df.sort_values(["event_time", "event_id", "selection"]).reset_index(drop=True)
    df["model_prob"] = np.nan
    last_fit_at = -config.retrain_every
    fitted = None
    # Predict all rows sharing an event timestamp before considering later events.
    for event_time, idx in df.groupby("event_time", sort=True).groups.items():
        train_idx = df.index[(df["settled_time"] < event_time) & df["won"].notna()]
        if len(train_idx) < config.min_train_rows:
            continue
        if fitted is None or len(train_idx) - last_fit_at >= config.retrain_every:
            y = df.loc[train_idx, "won"].astype(int)
            if y.nunique() < 2:
                continue
            fitted = _model(config.random_state)
            fitted.fit(df.loc[train_idx, list(feature_cols)], y)
            last_fit_at = len(train_idx)
        df.loc[idx, "model_prob"] = fitted.predict_proba(df.loc[idx, list(feature_cols)])[:, 1]
    return df


def _kelly(p: pd.Series, decimal_odds: pd.Series) -> pd.Series:
    b = decimal_odds - 1.0
    return ((p * decimal_odds - 1.0) / b).clip(lower=0.0)


def run_backtest(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    pred = walk_forward_predict(data, feature_cols, cfg)
    pred["edge"] = pred["model_prob"] - pred["no_vig_prob"]
    pred["expected_value"] = pred["model_prob"] * pred["decimal_odds"] - 1.0
    pred["full_kelly"] = _kelly(pred["model_prob"], pred["decimal_odds"])
    candidates = pred[
        pred["model_prob"].notna()
        & (pred["edge"] >= cfg.min_edge)
        & (pred["expected_value"] >= cfg.min_ev)
    ].sort_values(["event_time", "event_id", "selection"]).copy()

    bankroll = cfg.initial_bankroll
    peak = bankroll
    rows: list[dict] = []
    # Stakes for simultaneous events are all based on bankroll before that timestamp.
    for event_time, group in candidates.groupby("event_time", sort=True):
        opening = bankroll
        pending: list[dict] = []
        for _, row in group.iterrows():
            fraction = min(float(row.full_kelly) * cfg.kelly_fraction, cfg.max_bet_fraction)
            stake = opening * fraction
            if cfg.max_bet is not None:
                stake = min(stake, cfg.max_bet)
            if stake < cfg.min_bet or opening <= 0:
                continue
            profit = stake * (row.decimal_odds - 1.0) if int(row.won) == 1 else -stake
            pending.append({**row.to_dict(), "stake": stake, "profit": profit, "opening_bankroll": opening})
        total_stake = sum(item["stake"] for item in pending)
        exposure_cap = opening * cfg.max_event_exposure
        if total_stake > exposure_cap and total_stake > 0:
            scale = exposure_cap / total_stake
            for item in pending:
                item["stake"] *= scale
                item["profit"] *= scale
        total_profit = sum(item["profit"] for item in pending)
        bankroll += total_profit
        peak = max(peak, bankroll)
        drawdown = (bankroll - peak) / peak if peak else 0.0
        for item in pending:
            item.update({"closing_bankroll": bankroll, "drawdown": drawdown})
            rows.append(item)

    bets = pd.DataFrame(rows)
    total_staked = float(bets["stake"].sum()) if not bets.empty else 0.0
    total_profit = float(bets["profit"].sum()) if not bets.empty else 0.0
    metrics = {
        "initial_bankroll": cfg.initial_bankroll,
        "final_bankroll": float(bankroll),
        "net_profit": total_profit,
        "total_staked": total_staked,
        "roi": total_profit / total_staked if total_staked else 0.0,
        "bankroll_return": bankroll / cfg.initial_bankroll - 1.0,
        "bets": len(bets),
        "wins": int(bets["won"].sum()) if not bets.empty else 0,
        "win_rate": float(bets["won"].mean()) if not bets.empty else 0.0,
        "max_drawdown": float(bets["drawdown"].min()) if not bets.empty else 0.0,
    }
    return BacktestResult(bets=bets, predictions=pred, metrics=metrics)
