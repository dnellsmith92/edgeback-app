import pandas as pd
import pytest

from edgeback.backtest import BacktestConfig, run_backtest, walk_forward_predict
from edgeback.odds import add_market_probabilities, to_decimal


def test_odds_and_vig_removal():
    assert to_decimal(pd.Series([-110, 150]), "american").tolist() == pytest.approx([1.9090909, 2.5])
    df = pd.DataFrame({"event_id":[1,1], "sportsbook":["x","x"], "market":["ml","ml"], "decimal_odds":[1.91,1.91]})
    out = add_market_probabilities(df)
    assert out.no_vig_prob.tolist() == pytest.approx([.5,.5])
    assert out.overround.iloc[0] > 1


def _data(n=40):
    rows=[]
    for i in range(n):
        t=pd.Timestamp("2024-01-01", tz="UTC")+pd.Timedelta(days=i)
        for side in [0,1]:
            rows.append({"event_id":i,"event_time":t,"settled_time":t+pd.Timedelta(hours=2),"selection":str(side),
                         "won":int(side == i%2),"x":(1 if side else -1)*(1 if i%2 else -1),
                         "decimal_odds":2.1,"no_vig_prob":.5})
    return pd.DataFrame(rows)


def test_walk_forward_respects_settlement_cutoff():
    df=_data()
    cfg=BacktestConfig(min_train_rows=10,retrain_every=1)
    pred=walk_forward_predict(df,["x"],cfg)
    first_pred=pred[pred.model_prob.notna()].iloc[0]
    assert first_pred.event_time > df.sort_values("settled_time").iloc[9].settled_time


def test_bankroll_metrics_and_stake_cap():
    result=run_backtest(_data(),["x"],BacktestConfig(min_train_rows=10,retrain_every=1,min_edge=0,min_ev=0,
                                                     kelly_fraction=.25,max_bet_fraction=.02))
    assert set(["roi","win_rate","max_drawdown","final_bankroll"]) <= result.metrics.keys()
    if not result.bets.empty:
        assert (result.bets.stake <= result.bets.opening_bankroll*.02 + 1e-9).all()
        exposure = result.bets.groupby("event_time").agg(stake=("stake", "sum"), opening=("opening_bankroll", "first"))
        assert (exposure.stake <= exposure.opening*.10 + 1e-9).all()
