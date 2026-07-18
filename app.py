from __future__ import annotations

from math import isfinite
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="WNBA Prop Lab", page_icon="🏀", layout="wide")

STAT_COLUMNS = {
    "Points": "points",
    "Rebounds": "rebounds",
    "Assists": "assists",
    "3-Pointers Made": "threes",
    "Steals": "steals",
    "Blocks": "blocks",
    "Turnovers": "turnovers",
    "Points + Rebounds": "points_rebounds",
    "Points + Assists": "points_assists",
    "Rebounds + Assists": "rebounds_assists",
    "Points + Rebounds + Assists": "pra",
}
BASE_REQUIRED = {
    "game_date", "player", "team", "opponent", "home_away", "minutes",
    "points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers",
}


def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def american_implied(odds: int) -> float:
    return 1 / american_to_decimal(odds)


def fair_american(probability: float) -> str:
    p = min(max(probability, 0.0001), 0.9999)
    value = -100 * p / (1 - p) if p >= .5 else 100 * (1 - p) / p
    return f"{value:+.0f}"


def add_combo_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["points_rebounds"] = out["points"] + out["rebounds"]
    out["points_assists"] = out["points"] + out["assists"]
    out["rebounds_assists"] = out["rebounds"] + out["assists"]
    out["pra"] = out["points"] + out["rebounds"] + out["assists"]
    return out


def validate_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = BASE_REQUIRED - set(out.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if out["game_date"].isna().any():
        raise ValueError("Some game_date values are invalid")
    numeric = ["minutes", "points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[numeric].isna().any().any():
        raise ValueError("One or more statistic columns contain non-numeric values")
    out["home_away"] = out["home_away"].astype(str).str.upper().str[0]
    if not out["home_away"].isin(["H", "A"]).all():
        raise ValueError("home_away must contain H/Home or A/Away")
    return add_combo_stats(out).sort_values("game_date")


@st.cache_data(ttl=21600, show_spinner=False)
def load_wehoop_season(season: int, regular_season_only: bool = True) -> pd.DataFrame:
    """Load a public SportsDataverse wehoop WNBA player-boxscore season."""
    url = (
        "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
        f"espn_wnba_player_boxscores/player_box_{season}.parquet"
    )
    raw = pd.read_parquet(url)
    if regular_season_only and "season_type" in raw:
        raw = raw[raw["season_type"] == 2]
    if "did_not_play" in raw:
        raw = raw[~raw["did_not_play"].fillna(False)]
    mapped = pd.DataFrame({
        "game_date": raw["game_date"],
        "player": raw["athlete_display_name"],
        "team": raw["team_abbreviation"],
        "opponent": raw["opponent_team_abbreviation"],
        "home_away": raw["home_away"],
        "minutes": raw["minutes"],
        "points": raw["points"],
        "rebounds": raw["rebounds"],
        "assists": raw["assists"],
        "threes": raw["three_point_field_goals_made"],
        "steals": raw["steals"],
        "blocks": raw["blocks"],
        "turnovers": raw["turnovers"],
        "season": season,
    })
    mapped = mapped[pd.to_numeric(mapped["minutes"], errors="coerce").fillna(0) > 0]
    return validate_logs(mapped)


def load_wehoop_seasons(seasons: list[int], regular_season_only: bool = True) -> pd.DataFrame:
    if not seasons:
        raise ValueError("Select at least one season")
    return pd.concat(
        [load_wehoop_season(year, regular_season_only) for year in seasons],
        ignore_index=True,
    ).sort_values("game_date")


def demo_logs() -> pd.DataFrame:
    rng = np.random.default_rng(24)
    rows = []
    opponents = ["NYL", "CON", "ATL", "CHI", "IND", "MIN"]
    for i in range(32):
        minutes = float(np.clip(rng.normal(32, 3), 22, 38))
        home = i % 2 == 0
        rows.append({
            "game_date": pd.Timestamp("2026-05-15") + pd.Timedelta(days=i * 2),
            "player": "Demo Player", "team": "WAS", "opponent": opponents[i % len(opponents)],
            "home_away": "H" if home else "A", "minutes": round(minutes, 1),
            "points": max(2, int(rng.normal(19 + (1.5 if home else 0), 5))),
            "rebounds": max(0, int(rng.normal(7, 2.5))), "assists": max(0, int(rng.normal(4, 2))),
            "threes": max(0, int(rng.normal(1.8, 1.2))), "steals": max(0, int(rng.normal(1.2, .8))),
            "blocks": max(0, int(rng.normal(.8, .7))), "turnovers": max(0, int(rng.normal(2.2, 1))),
        })
    return validate_logs(pd.DataFrame(rows))


def hit_rate(values: pd.Series, line: float, side: str) -> float:
    if values.empty:
        return float("nan")
    return float((values > line).mean() if side == "Over" else (values < line).mean())


def project_stat(games: pd.DataFrame, stat: str, venue: str, opponent: str) -> tuple[float, list[str]]:
    """Transparent recency/split projection; all inputs precede the analyzed game."""
    pieces: list[tuple[float, float, str]] = []
    values = games[stat]
    pieces.append((float(values.tail(5).mean()), .40, "last 5"))
    pieces.append((float(values.tail(10).mean()), .30, "last 10"))
    pieces.append((float(values.mean()), .15, "season"))
    venue_values = games.loc[games.home_away == venue, stat]
    if len(venue_values) >= 3:
        pieces.append((float(venue_values.mean()), .10, "venue"))
    opp_values = games.loc[games.opponent == opponent, stat]
    if len(opp_values) >= 2:
        pieces.append((float(opp_values.mean()), .05, "opponent"))
    total_weight = sum(weight for _, weight, _ in pieces)
    projection = sum(value * weight for value, weight, _ in pieces) / total_weight
    return projection, [label for _, _, label in pieces]


def no_vig_probs(over_odds: int, under_odds: int) -> tuple[float, float]:
    over_raw, under_raw = american_implied(over_odds), american_implied(under_odds)
    total = over_raw + under_raw
    return over_raw / total, under_raw / total


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    return max(0.0, (probability * decimal_odds - 1) / b)


st.title("🏀 WNBA Historical Prop Lab")
st.caption("Compare a sportsbook prop with a player's historical game logs—no live-statistics subscription required.")

with st.sidebar:
    st.header("Bankroll controls")
    bankroll = st.number_input("Bankroll ($)", min_value=1.0, value=1000.0, step=100.0)
    kelly_multiplier = st.slider("Kelly multiplier", .05, .50, .25, .05)
    max_stake_pct = st.slider("Maximum stake", .005, .05, .02, .005, format="%.3f")
    min_ev = st.slider("Minimum EV to flag", 0.0, .15, .02, .005, format="%.3f")

source = st.radio(
    "Historical data source",
    ["Automatic WNBA history", "Upload my CSV", "Demo data"],
    horizontal=True,
)
template = pd.DataFrame(columns=sorted(BASE_REQUIRED))
upload = None
selected_seasons: list[int] = []
regular_only = True

if source == "Automatic WNBA history":
    source1, source2 = st.columns([3, 1])
    with source1:
        selected_seasons = st.multiselect(
            "Seasons",
            [2023, 2024, 2025, 2026],
            default=[2025, 2026],
            help="Using recent seasons helps reflect current player roles.",
        )
    with source2:
        regular_only = st.toggle("Regular season only", value=True)
elif source == "Upload my CSV":
    upload_col, template_col = st.columns([3, 1])
    with upload_col:
        upload = st.file_uploader("Upload WNBA player game logs", type="csv")
    with template_col:
        st.download_button("Download CSV template", template.to_csv(index=False), "wnba_game_logs_template.csv", "text/csv")
    if upload is None:
        st.info("Upload a game-log CSV to begin.")
        with st.expander("Required columns"):
            st.code(", ".join(sorted(BASE_REQUIRED)))
        st.stop()

try:
    if source == "Automatic WNBA history":
        with st.spinner("Loading WNBA player game logs…"):
            logs = load_wehoop_seasons(selected_seasons, regular_only)
        st.caption(f"Loaded {len(logs):,} player-game records from SportsDataverse wehoop.")
    elif source == "Demo data":
        logs = demo_logs()
    else:
        logs = validate_logs(pd.read_csv(upload))
except Exception as exc:
    st.error(f"Could not load game logs: {exc}")
    if source == "Automatic WNBA history":
        st.info("Try again shortly or choose 'Upload my CSV' as a fallback.")
    st.stop()

players = sorted(logs.player.astype(str).unique())
player = st.selectbox("Player", players)
player_games = logs[logs.player.astype(str) == player].sort_values("game_date").copy()

setup1, setup2, setup3, setup4 = st.columns(4)
with setup1:
    market_label = st.selectbox("Prop", list(STAT_COLUMNS))
with setup2:
    line = st.number_input("Sportsbook line", min_value=0.0, value=19.5, step=.5)
with setup3:
    over_odds = st.number_input("Over odds (American)", value=-110, step=5)
with setup4:
    under_odds = st.number_input("Under odds (American)", value=-110, step=5)

context1, context2 = st.columns(2)
with context1:
    venue = st.radio("Upcoming venue", ["H", "A"], format_func=lambda x: "Home" if x == "H" else "Away", horizontal=True)
with context2:
    opponent_options = sorted(player_games.opponent.astype(str).unique())
    opponent = st.selectbox("Upcoming opponent", opponent_options)

stat = STAT_COLUMNS[market_label]
values = player_games[stat]
projection, components = project_stat(player_games, stat, venue, opponent)
spread = float(values.tail(20).std(ddof=1)) if len(values) > 1 else 1.0
spread = max(spread, .75)
dist = NormalDist(mu=projection, sigma=spread)
model_over = 1 - dist.cdf(line)
model_under = dist.cdf(line)
market_over, market_under = no_vig_probs(int(over_odds), int(under_odds))
over_decimal, under_decimal = american_to_decimal(int(over_odds)), american_to_decimal(int(under_odds))
over_ev = model_over * over_decimal - 1
under_ev = model_under * under_decimal - 1
best_side = "Over" if over_ev >= under_ev else "Under"
best_ev = max(over_ev, under_ev)
best_prob = model_over if best_side == "Over" else model_under
best_decimal = over_decimal if best_side == "Over" else under_decimal
stake_fraction = min(kelly_fraction(best_prob, best_decimal) * kelly_multiplier, max_stake_pct)
stake = bankroll * stake_fraction if best_ev >= min_ev else 0.0

st.subheader(f"{player} — {market_label} {line:g}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Projection", f"{projection:.1f}")
m2.metric("Last 5 average", f"{values.tail(5).mean():.1f}")
m3.metric("Last 10 average", f"{values.tail(10).mean():.1f}")
m4.metric("Season average", f"{values.mean():.1f}")
m5.metric("Average minutes", f"{player_games.minutes.tail(10).mean():.1f}")

summary = pd.DataFrame([
    {"Side": "Over", "Model probability": model_over, "No-vig market probability": market_over,
     "Edge": model_over - market_over, "EV": over_ev, "Fair odds": fair_american(model_over),
     "Overall hit rate": hit_rate(values, line, "Over")},
    {"Side": "Under", "Model probability": model_under, "No-vig market probability": market_under,
     "Edge": model_under - market_under, "EV": under_ev, "Fair odds": fair_american(model_under),
     "Overall hit rate": hit_rate(values, line, "Under")},
])
st.dataframe(summary.style.format({
    "Model probability": "{:.1%}", "No-vig market probability": "{:.1%}", "Edge": "{:+.1%}",
    "EV": "{:+.1%}", "Overall hit rate": "{:.1%}",
}), use_container_width=True, hide_index=True)

if best_ev >= min_ev:
    st.success(f"Research signal: {best_side} {line:g} | EV {best_ev:+.1%} | capped fractional-Kelly stake ${stake:,.2f}")
else:
    st.warning(f"No side clears the selected {min_ev:.1%} EV threshold. Best estimate: {best_side} at {best_ev:+.1%} EV.")

split_rows = []
for label, subset in [
    ("All games", player_games), ("Last 5", player_games.tail(5)), ("Last 10", player_games.tail(10)),
    ("Last 20", player_games.tail(20)), ("Home", player_games[player_games.home_away == "H"]),
    ("Away", player_games[player_games.home_away == "A"]),
    (f"vs {opponent}", player_games[player_games.opponent == opponent]),
]:
    split_rows.append({"Split": label, "Games": len(subset), "Average": subset[stat].mean(),
                       "Over hit rate": hit_rate(subset[stat], line, "Over"),
                       "Under hit rate": hit_rate(subset[stat], line, "Under")})
splits = pd.DataFrame(split_rows)

chart_col, split_col = st.columns([3, 2])
with chart_col:
    st.subheader("Game-log trend")
    chart = player_games.set_index("game_date")[[stat]].tail(20).rename(columns={stat: market_label})
    chart["Prop line"] = line
    st.line_chart(chart)
with split_col:
    st.subheader("Historical splits")
    st.dataframe(splits.style.format({"Average": "{:.1f}", "Over hit rate": "{:.1%}", "Under hit rate": "{:.1%}"}),
                 use_container_width=True, hide_index=True)

st.subheader("Recent game logs")
display_cols = ["game_date", "opponent", "home_away", "minutes", "points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers"]
st.dataframe(player_games.sort_values("game_date", ascending=False)[display_cols].head(20), use_container_width=True, hide_index=True)
st.caption("Projection components: " + ", ".join(components) + ". Research only; historical performance does not guarantee future results.")
