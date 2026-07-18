from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import BacktestConfig, run_backtest
from .io import load_betting_data


def main() -> None:
    p = argparse.ArgumentParser(description="Leakage-safe sports betting backtester")
    p.add_argument("data", help="CSV containing results, odds, and pregame features")
    p.add_argument("--features", required=True, help="Comma-separated pregame feature columns")
    p.add_argument("--odds-format", choices=["decimal", "american", "fractional"], default="decimal")
    p.add_argument("--output", default="output")
    p.add_argument("--initial-bankroll", type=float, default=1000)
    p.add_argument("--min-train-rows", type=int, default=200)
    p.add_argument("--retrain-every", type=int, default=50)
    p.add_argument("--min-edge", type=float, default=.02)
    p.add_argument("--min-ev", type=float, default=.01)
    p.add_argument("--kelly-fraction", type=float, default=.25)
    p.add_argument("--max-bet-fraction", type=float, default=.03)
    p.add_argument("--max-event-exposure", type=float, default=.10,
                   help="Maximum total stake at one event timestamp as fraction of bankroll")
    p.add_argument("--min-bet", type=float, default=1)
    p.add_argument("--max-bet", type=float)
    args = p.parse_args()
    cfg = BacktestConfig(**{k: v for k, v in vars(args).items() if k in BacktestConfig.__dataclass_fields__})
    data = load_betting_data(args.data, args.odds_format)
    result = run_backtest(data, [x.strip() for x in args.features.split(",") if x.strip()], cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result.bets.to_csv(out / "bets.csv", index=False)
    result.predictions.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
