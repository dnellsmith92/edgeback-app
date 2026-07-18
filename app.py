from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from edgeback.backtest import BacktestConfig, run_backtest
from edgeback.io import load_betting_data


st.set_page_config(page_title="EdgeBack", page_icon="📈", layout="wide")
st.title("EdgeBack Sports Betting Analyzer")
st.caption("Upload historical pregame odds and results to run a leakage-safe walk-forward backtest.")

with st.sidebar:
    st.header("Strategy settings")
    odds_format = st.selectbox("Odds format", ["decimal", "american", "fractional"])
    initial_bankroll = st.number_input("Starting bankroll ($)", min_value=1.0, value=1000.0, step=100.0)
    min_train_rows = st.number_input("Minimum training rows", min_value=20, value=200, step=20)
    retrain_every = st.number_input("Retrain every N rows", min_value=1, value=50, step=10)
    min_edge = st.slider("Minimum probability edge", 0.0, 0.20, 0.02, 0.005, format="%.3f")
    min_ev = st.slider("Minimum expected value", 0.0, 0.30, 0.01, 0.005, format="%.3f")
    kelly_fraction = st.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05)
    max_bet_fraction = st.slider("Maximum stake per bet", 0.005, 0.10, 0.03, 0.005, format="%.3f")
    max_event_exposure = st.slider("Maximum simultaneous exposure", 0.01, 0.50, 0.10, 0.01)

uploaded = st.file_uploader("Upload a historical betting CSV", type="csv")

with st.expander("Required CSV format"):
    st.markdown(
        "Required columns: `event_id`, `event_time`, `settled_time`, `sportsbook`, "
        "`market`, `selection`, `odds`, and `won`. Add numeric columns containing only "
        "information known before the event; those become model features."
    )

if uploaded is None:
    st.info("Upload a CSV to begin. You can also use `data/sample_games.csv` included with the project.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"The CSV could not be read: {exc}")
    st.stop()

reserved = {
    "event_id", "event_time", "settled_time", "sportsbook", "market", "selection",
    "odds", "won", "decimal_odds", "implied_prob", "overround", "no_vig_prob",
}
numeric_features = [
    col for col in raw.columns
    if col not in reserved and pd.api.types.is_numeric_dtype(raw[col])
]

st.subheader("Data and model")
left, right = st.columns([2, 1])
with left:
    features = st.multiselect("Pregame feature columns", numeric_features, default=numeric_features[:5])
with right:
    st.metric("Rows loaded", f"{len(raw):,}")

if st.button("Run backtest", type="primary", use_container_width=True):
    if not features:
        st.error("Select at least one numeric pregame feature.")
        st.stop()
    try:
        uploaded.seek(0)
        data = load_betting_data(uploaded, odds_format)
        config = BacktestConfig(
            initial_bankroll=initial_bankroll,
            min_train_rows=int(min_train_rows),
            retrain_every=int(retrain_every),
            min_edge=min_edge,
            min_ev=min_ev,
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
            max_event_exposure=max_event_exposure,
        )
        with st.spinner("Training historical windows and simulating bets…"):
            result = run_backtest(data, features, config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

    m = result.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROI", f"{m['roi']:.2%}")
    c2.metric("Win rate", f"{m['win_rate']:.2%}")
    c3.metric("Final bankroll", f"${m['final_bankroll']:,.2f}", f"${m['net_profit']:,.2f}")
    c4.metric("Maximum drawdown", f"{m['max_drawdown']:.2%}")

    st.subheader("Bankroll performance")
    if result.bets.empty:
        st.warning("No bets met the selected edge and EV thresholds. Try lower thresholds or check the data.")
    else:
        curve = result.bets.drop_duplicates("event_time", keep="last").set_index("event_time")[["closing_bankroll"]]
        st.line_chart(curve, y_label="Bankroll ($)")
        st.subheader("Bet ledger")
        st.dataframe(result.bets, use_container_width=True, hide_index=True)

    def csv_bytes(frame: pd.DataFrame) -> bytes:
        return frame.to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)
    d1.download_button("Download bets.csv", csv_bytes(result.bets), "bets.csv", "text/csv", use_container_width=True)
    d2.download_button(
        "Download predictions.csv", csv_bytes(result.predictions), "predictions.csv", "text/csv",
        use_container_width=True,
    )

st.divider()
st.caption("For research and education only. Backtests do not guarantee future profit. Bet only what you can afford to lose.")
