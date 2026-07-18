from __future__ import annotations

from math import isfinite
from statistics import NormalDist
import json
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

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
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DK_MARKETS = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
}
DK_MARKET_LABELS = {value: key for key, value in DK_MARKETS.items()}


def normalize_player_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def odds_api_key() -> str:
    try:
        return str(st.secrets.get("THE_ODDS_API_KEY", "")).strip()
    except Exception:
        return ""


def get_json(url: str, params: dict[str, str]) -> object:
    with urlopen(f"{url}?{urlencode(params)}", timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_draftkings_props(payload: dict, market_keys: set[str]) -> list[dict]:
    rows: dict[tuple[str, str, str, float], dict] = {}
    game = f"{payload.get('away_team', '')} @ {payload.get('home_team', '')}".strip(" @")
    for bookmaker in payload.get("bookmakers", []):
        if bookmaker.get("key") != "draftkings":
            continue
        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            if market_key not in market_keys:
                continue
            for outcome in market.get("outcomes", []):
                player = str(outcome.get("description", "")).strip()
                point = outcome.get("point")
                side = str(outcome.get("name", "")).title()
                if not player or point is None or side not in {"Over", "Under"}:
                    continue
                key = (str(payload.get("id", "")), str(market_key), player, float(point))
                row = rows.setdefault(key, {
                    "Player": player, "Prop": DK_MARKET_LABELS.get(str(market_key), str(market_key)),
                    "Line": float(point), "Over Odds": np.nan,
                    "Under Odds": np.nan, "Game": game,
                    "Start Time": payload.get("commence_time", ""),
                    "Last Update": market.get("last_update", bookmaker.get("last_update", "")),
                })
                row[f"{side} Odds"] = outcome.get("price", np.nan)
    return list(rows.values())


@st.cache_data(ttl=900, show_spinner=False)
def fetch_draftkings_props(api_key: str, market_keys_csv: str) -> pd.DataFrame:
    market_keys = set(market_keys_csv.split(","))
    events = get_json(
        f"{ODDS_API_BASE}/sports/basketball_wnba/events",
        {"apiKey": api_key},
    )
    rows: list[dict] = []
    for event in events:
        payload = get_json(
            f"{ODDS_API_BASE}/sports/basketball_wnba/events/{event['id']}/odds",
            {
                "apiKey": api_key,
                "bookmakers": "draftkings",
                "markets": market_keys_csv,
                "oddsFormat": "american",
            },
        )
        rows.extend(parse_draftkings_props(payload, market_keys))
    return pd.DataFrame(rows)


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


def make_prop_board(logs: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    eligible = logs[logs["team"].isin(teams)] if teams else logs
    latest = eligible.sort_values("game_date").groupby("player", as_index=False).tail(1)
    return latest[["player", "team"]].sort_values(["team", "player"]).assign(
        line=np.nan,
        over_odds=-110,
        under_odds=-110,
    ).rename(columns={
        "player": "Player", "team": "Team", "line": "Line",
        "over_odds": "Over Odds", "under_odds": "Under Odds",
    })


def all_player_trends(
    logs: pd.DataFrame,
    prop_board: pd.DataFrame,
    stat: str,
    window: int,
) -> pd.DataFrame:
    rows = []
    for _, prop in prop_board.iterrows():
        if pd.isna(prop.get("Line")):
            continue
        player_games = logs[logs["player"] == prop["Player"]].sort_values("game_date").tail(window)
        if player_games.empty:
            continue
        line = float(prop["Line"])
        values = player_games[stat]
        over_hits = values > line
        under_hits = values < line
        pushes = values == line
        sequence = " ".join(
            "O" if value > line else "U" if value < line else "P"
            for value in values.tolist()
        )
        rows.append({
            "Player": prop["Player"], "Team": prop.get("Team", ""), "Line": line,
            "Over Odds": int(prop.get("Over Odds", -110)), "Under Odds": int(prop.get("Under Odds", -110)),
            "Game": prop.get("Game", ""), "Start Time": prop.get("Start Time", ""),
            "Games": len(values), "Average": float(values.mean()),
            "Over Hit Rate": float(over_hits.mean()), "Under Hit Rate": float(under_hits.mean()),
            "Overs": int(over_hits.sum()), "Unders": int(under_hits.sum()), "Pushes": int(pushes.sum()),
            "Oldest → Newest": sequence,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Over Hit Rate", "Average"], ascending=False)


def prop_feed(
    logs: pd.DataFrame,
    prop_board: pd.DataFrame,
    fallback_prop: str,
    rank_window: int,
    side_filter: str,
) -> pd.DataFrame:
    """Build an app-style feed with L5/L10/L20 rates and one selectable pick per prop."""
    rows = []
    for _, prop in prop_board.iterrows():
        prop_label = str(prop.get("Prop", fallback_prop))
        if prop_label not in STAT_COLUMNS or pd.isna(prop.get("Line")):
            continue
        games = logs[logs["player"] == prop["Player"]].sort_values("game_date")
        if games.empty:
            continue
        values = games[STAT_COLUMNS[prop_label]]
        line = float(prop["Line"])
        rates = {}
        for window in (5, 10, 20):
            recent = values.tail(window)
            rates[("Over", window)] = hit_rate(recent, line, "Over")
            rates[("Under", window)] = hit_rate(recent, line, "Under")
        if side_filter == "Best side":
            side = "Over" if rates[("Over", rank_window)] >= rates[("Under", rank_window)] else "Under"
        else:
            side = side_filter
        odds = int(prop.get(f"{side} Odds", -110))
        rows.append({
            "Player": prop["Player"], "Team": prop.get("Team", ""), "Prop": prop_label,
            "Pick": f"{side} {line:g}", "Odds": odds, "Line": line,
            "Over Odds": int(prop.get("Over Odds", -110)),
            "Under Odds": int(prop.get("Under Odds", -110)),
            "L5": rates[(side, 5)], "L10": rates[(side, 10)], "L20": rates[(side, 20)],
            "Average": float(values.tail(rank_window).mean()),
            "Game": prop.get("Game", ""), "Start Time": prop.get("Start Time", ""),
        })
    if not rows:
        return pd.DataFrame()
    rank_col = f"L{rank_window}"
    return pd.DataFrame(rows).sort_values([rank_col, "Average"], ascending=False).reset_index(drop=True)


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

with st.expander("📊 All-Player Prop Trends", expanded=True):
    st.caption("Filter the DraftKings prop feed, compare L5/L10/L20 hit rates, and click a player for full details.")
    trend1, trend2, trend3 = st.columns(3)
    with trend1:
        trend_stat_label = st.radio(
            "Prop category", ["All Props", "Points", "Rebounds", "Assists"], horizontal=True
        )
    with trend2:
        trend_window_label = st.radio("Rank by", ["Last 5", "Last 10", "Last 20"], horizontal=True)
    with trend3:
        side_filter = st.radio("Side", ["Best side", "Over", "Under"], horizontal=True)

    prop_template = make_prop_board(logs, [])
    odds_source = st.radio(
        "Sportsbook odds source",
        ["DraftKings automatic", "Enter or upload manually"],
        horizontal=True,
    )
    board = pd.DataFrame()
    if odds_source == "DraftKings automatic":
        api_key = odds_api_key()
        if not api_key:
            st.warning("DraftKings automatic odds need a private Odds API key. Add it in Manage app → Settings → Secrets.")
            with st.expander("One-time API-key setup"):
                st.markdown(
                    "1. Create a key at [The Odds API](https://the-odds-api.com/).\n"
                    "2. Open **Manage app → Settings → Secrets**.\n"
                    "3. Add the line below, replacing the sample text with your key.\n"
                    "4. Save the secret and restart the app."
                )
                st.code('THE_ODDS_API_KEY = "paste-your-key-here"')
        else:
            try:
                with st.spinner("Loading current DraftKings player props…"):
                    requested_markets = (
                        ",".join(DK_MARKETS.values())
                        if trend_stat_label == "All Props"
                        else DK_MARKETS[trend_stat_label]
                    )
                    board = fetch_draftkings_props(api_key, requested_markets)
                if board.empty:
                    st.info("DraftKings has no WNBA props posted for this market right now. Try again closer to game time.")
                else:
                    latest_players = logs.sort_values("game_date").groupby("player", as_index=False).tail(1)
                    name_lookup = dict(zip(latest_players["player"].map(normalize_player_name), latest_players["player"]))
                    team_lookup = dict(zip(latest_players["player"], latest_players["team"]))
                    board["Sportsbook Player"] = board["Player"]
                    board["Player"] = board["Player"].map(
                        lambda value: name_lookup.get(normalize_player_name(value), value)
                    )
                    board["Team"] = board["Player"].map(team_lookup).fillna("")
                    board = board.dropna(subset=["Over Odds", "Under Odds"])
                    category_text = "player" if trend_stat_label == "All Props" else trend_stat_label.lower()
                    st.success(f"Loaded {len(board)} current DraftKings {category_text} props. Odds refresh every 15 minutes.")
                    st.dataframe(
                        board[["Player", "Team", "Prop", "Game", "Start Time", "Line", "Over Odds", "Under Odds"]],
                        use_container_width=True,
                        hide_index=True,
                    )
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exc:
                st.error(f"Could not load DraftKings odds: {exc}")
                st.info("Check the API key and account quota, or use manual entry below.")
    else:
        upload_board = st.file_uploader(
            "Optional: upload a prop board CSV",
            type="csv",
            key="prop_board_upload",
            help="Columns: Player, Team, Line, Over Odds, Under Odds",
        )
        st.download_button(
            "Download prop-board template",
            prop_template.to_csv(index=False),
            "wnba_prop_board.csv",
            "text/csv",
        )

        if upload_board is not None:
            try:
                board = pd.read_csv(upload_board)
                needed = {"Player", "Line", "Over Odds", "Under Odds"}
                missing = needed - set(board.columns)
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
                if "Team" not in board:
                    board["Team"] = ""
            except Exception as exc:
                st.error(f"Could not load prop board: {exc}")
                board = prop_template
        else:
            st.caption("Enter lines and odds directly below. Leave players without a current prop blank.")
            board = st.data_editor(
                prop_template,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key="prop_editor_all_players",
                column_config={
                    "Player": st.column_config.TextColumn(disabled=True),
                    "Team": st.column_config.TextColumn(disabled=True),
                    "Line": st.column_config.NumberColumn(min_value=0.0, step=.5),
                    "Over Odds": st.column_config.NumberColumn(step=5),
                    "Under Odds": st.column_config.NumberColumn(step=5),
                },
            )

    if board.empty:
        board = prop_template.iloc[0:0]

    fallback_prop = "Points" if trend_stat_label == "All Props" else trend_stat_label
    trend_stat = STAT_COLUMNS[fallback_prop]
    trend_window = int(trend_window_label.split()[-1])
    trend_table = prop_feed(logs, board, fallback_prop, trend_window, side_filter)
    if trend_table.empty:
        st.info("Enter at least one prop line above to generate the all-player hit-rate table.")
    else:
        trend_table.insert(0, "Rank", np.arange(1, len(trend_table) + 1))
        st.caption("Click any player row to open that player in the Individual Player Analyzer below.")
        ranking_event = st.dataframe(
            trend_table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"prop_rankings_{trend_stat_label}_{trend_window}_{side_filter}",
            column_order=[
                "Rank", "Player", "Team", "Prop", "Pick", "Odds", "Game",
                "Start Time", "L5", "L10", "L20", "Average",
            ],
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Line": st.column_config.NumberColumn(format="%.1f"),
                "Average": st.column_config.NumberColumn(format="%.1f"),
                "L5": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0
                ),
                "L10": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0
                ),
                "L20": st.column_config.ProgressColumn(
                    format="percent", min_value=0.0, max_value=1.0
                ),
            },
        )
        if ranking_event.selection.rows:
            selected_prop = trend_table.iloc[ranking_event.selection.rows[0]]
            selection_token = (
                selected_prop["Player"], selected_prop["Prop"], float(selected_prop["Line"]),
                int(selected_prop["Over Odds"]), int(selected_prop["Under Odds"]),
            )
            if st.session_state.get("prop_selection_token") != selection_token:
                st.session_state["prop_selection_token"] = selection_token
                st.session_state["individual_player_select"] = selected_prop["Player"]
                st.session_state["individual_market_select"] = selected_prop["Prop"]
                st.session_state["individual_line"] = float(selected_prop["Line"])
                st.session_state["individual_over_odds"] = int(selected_prop["Over Odds"])
                st.session_state["individual_under_odds"] = int(selected_prop["Under Odds"])
        st.download_button(
            "Download trend results",
            trend_table.to_csv(index=False),
            f"wnba_{trend_stat}_{trend_window}_game_trends.csv",
            "text/csv",
        )

st.divider()
st.header("🔎 Individual Player Analyzer")
st.caption(
    "Select one player to view their sportsbook line and odds, projection, expected value, "
    "Last 5 and Last 10 averages, historical splits, chart, and recent game logs."
)
players = sorted(logs.player.astype(str).unique())
if st.session_state.get("individual_player_select") not in players:
    st.session_state["individual_player_select"] = players[0]
player = st.selectbox("Player", players, key="individual_player_select")
player_games = logs[logs.player.astype(str) == player].sort_values("game_date").copy()

setup1, setup2, setup3, setup4 = st.columns(4)
with setup1:
    market_label = st.selectbox("Prop", list(STAT_COLUMNS), key="individual_market_select")
with setup2:
    line = st.number_input("Sportsbook line", min_value=0.0, value=19.5, step=.5, key="individual_line")
with setup3:
    over_odds = st.number_input("Over odds (American)", value=-110, step=5, key="individual_over_odds")
with setup4:
    under_odds = st.number_input("Under odds (American)", value=-110, step=5, key="individual_under_odds")

context1, context2 = st.columns(2)
with context1:
    venue = st.radio("Upcoming venue", ["H", "A"], format_func=lambda x: "Home" if x == "H" else "Away", horizontal=True)
with context2:
    opponent_options = sorted(player_games.opponent.astype(str).unique())
    opponent = st.selectbox("Upcoming opponent", opponent_options)

detail_window_label = st.radio(
    "Player view",
    ["Last 5", "Last 10", "Last 20", "Season", "Head-to-head"],
    horizontal=True,
    help="Head-to-head uses this player's previous games against the selected opponent.",
)

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

if detail_window_label == "Season":
    detail_games = player_games
elif detail_window_label == "Head-to-head":
    detail_games = player_games[player_games.opponent == opponent]
else:
    detail_games = player_games.tail(int(detail_window_label.split()[-1]))
detail_values = detail_games[stat]
latest_team = str(player_games.iloc[-1]["team"])

st.subheader(f"{player} ({latest_team})")
st.caption(f"{market_label} player prop • selected matchup: {latest_team} vs {opponent}")
price1, price2, price3, price4 = st.columns(4)
price1.metric("DraftKings line", f"{line:g}")
price2.metric("Over price", f"{int(over_odds):+d}")
price3.metric("Under price", f"{int(under_odds):+d}")
price4.metric(
    f"{detail_window_label} over hit rate",
    "N/A" if detail_values.empty else f"{hit_rate(detail_values, line, 'Over'):.0%}",
)

st.subheader("Projection and form")
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
    chart_source = detail_games if not detail_games.empty else player_games.tail(20)
    chart = chart_source.set_index("game_date")[[stat]].rename(columns={stat: market_label})
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
