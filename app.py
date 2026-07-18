from __future__ import annotations

from math import isfinite
from statistics import NormalDist
import html
import json
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="WNBA Prop Labs", page_icon="🏀", layout="wide")

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
APP_URL = "https://edgeback-app-gnvtxztutnsbhmjp5cftvg.streamlit.app/"
DK_MARKETS = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
}
DK_MARKET_LABELS = {value: key for key, value in DK_MARKETS.items()}
SPORTSBOOKS = {
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "BetOnline": "betonlineag",
}
WNBA_TEAM_ABBREVIATIONS = {
    "Atlanta Dream": "ATL", "Chicago Sky": "CHI", "Connecticut Sun": "CON",
    "Dallas Wings": "DAL", "Golden State Valkyries": "GSV", "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA", "Los Angeles Sparks": "LAS", "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL", "Phoenix Mercury": "PHX", "Seattle Storm": "SEA",
    "Washington Mystics": "WAS", "Portland Fire": "POR", "Toronto Tempo": "TOR",
}
WNBA_TEAM_ALIASES = {
    "NY": "NYL", "GS": "GSV", "LV": "LVA", "LA": "LAS", "WSH": "WAS",
    "PHO": "PHX",
}


def normalize_player_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def matchup_opponent(game: object, player_team: object) -> str:
    teams = [WNBA_TEAM_ABBREVIATIONS.get(name.strip(), "") for name in str(game).split(" @ ")]
    normalized_player_team = WNBA_TEAM_ALIASES.get(str(player_team), str(player_team))
    return next((team for team in teams if team and team != normalized_player_team), "")


def matchup_label(game: object) -> str:
    """Display an Odds API matchup as compact team abbreviations."""
    teams = [
        WNBA_TEAM_ABBREVIATIONS.get(name.strip(), name.strip())
        for name in str(game).split(" @ ")
        if name.strip()
    ]
    return " vs ".join(teams) if len(teams) == 2 else str(game)


def odds_api_key() -> str:
    try:
        return str(st.secrets.get("THE_ODDS_API_KEY", "")).strip()
    except Exception:
        return ""


def get_json(url: str, params: dict[str, str]) -> object:
    with urlopen(f"{url}?{urlencode(params)}", timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_sportsbook_props(
    payload: dict,
    market_keys: set[str],
    bookmaker_key: str,
) -> list[dict]:
    rows: dict[tuple[str, str, str, float], dict] = {}
    game = f"{payload.get('away_team', '')} @ {payload.get('home_team', '')}".strip(" @")
    for bookmaker in payload.get("bookmakers", []):
        if bookmaker.get("key") != bookmaker_key:
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
def fetch_sportsbook_props(
    api_key: str,
    market_keys_csv: str,
    bookmaker_key: str,
) -> pd.DataFrame:
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
                "bookmakers": bookmaker_key,
                "markets": market_keys_csv,
                "oddsFormat": "american",
            },
        )
        rows.extend(parse_sportsbook_props(payload, market_keys, bookmaker_key))
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
        opponent = str(prop.get("Opponent", ""))
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
        h2h_values = games.loc[games["opponent"] == opponent, STAT_COLUMNS[prop_label]]
        h2h_rate = hit_rate(h2h_values, line, side)
        projection = (
            .50 * float(values.tail(5).mean())
            + .30 * float(values.tail(10).mean())
            + .20 * float(values.tail(20).mean())
        )
        spread = max(float(values.tail(20).std(ddof=1)), .75) if len(values) > 1 else 1.0
        over_probability = 1 - NormalDist(mu=projection, sigma=spread).cdf(line)
        under_probability = 1 - over_probability
        market_over, market_under = no_vig_probs(
            int(prop.get("Over Odds", -110)),
            int(prop.get("Under Odds", -110)),
        )
        model_probability = over_probability if side == "Over" else 1 - over_probability
        estimated_ev = model_probability * american_to_decimal(odds) - 1
        player_link = APP_URL + "?" + urlencode({
            "player": str(prop["Player"]), "prop": prop_label, "line": line,
            "over": int(prop.get("Over Odds", -110)), "under": int(prop.get("Under Odds", -110)),
            "opponent": opponent, "sportsbook": str(prop.get("Sportsbook", "DraftKings")),
        }) + f"#{prop['Player']}"
        rows.append({
            "Player": prop["Player"], "Team": prop.get("Team", ""), "Prop": prop_label,
            "Player Link": player_link,
            "Pick": f"{side} {line:g}", "Odds": odds, "Line": line,
            "Over Odds": int(prop.get("Over Odds", -110)),
            "Under Odds": int(prop.get("Under Odds", -110)),
            "L5": rates[(side, 5)], "L10": rates[(side, 10)], "L20": rates[(side, 20)],
            "H2H": h2h_rate, "H2H Games": int(len(h2h_values)),
            "Average": float(values.tail(rank_window).mean()),
            "EV+": "EV+" if estimated_ev > 0 else "—",
            "Estimated EV": estimated_ev, "Fair Odds": fair_american(model_probability),
            "Over %": over_probability, "Under %": under_probability,
            "Edge (Over)": over_probability - market_over,
            "Edge (Under)": under_probability - market_under,
            "Game": prop.get("Game", ""), "Opponent": opponent,
        })
    if not rows:
        return pd.DataFrame()
    rank_col = f"L{rank_window}"
    return pd.DataFrame(rows).sort_values([rank_col, "Average"], ascending=False).reset_index(drop=True)


def render_prop_table(df: pd.DataFrame, ev_only: bool = False) -> None:
    """Render sortable prop tables with traffic-light percentage bars."""
    if ev_only:
        columns = [
            "Player", "Team", "VS", "Prop", "Line", "Average", "Pick", "Odds",
            "Fair Odds", "Estimated EV", "Over %", "Under %",
            "Edge (Over)", "Edge (Under)",
        ]
    else:
        columns = [
            "Player", "Team", "Prop", "Line", "Pick", "Odds", "VS",
            "Average", "L5", "L10", "L20", "H2H",
        ]
    rate_columns = {"L5", "L10", "L20", "H2H", "Over %", "Under %"}
    delta_columns = {"Estimated EV", "Edge (Over)", "Edge (Under)"}
    table_id = "ev-prop-table" if ev_only else "main-prop-table"
    header_cells = []
    for index, column in enumerate(columns):
        header_cells.append(
            f'<th><button class="prop-sort-link" onclick="sortPropTable(\'{table_id}\',{index},this)">'
            f'{html.escape(column)}<span class="sort-arrow"></span></button></th>'
        )
    header = "".join(header_cells)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            if column == "Player":
                url = html.escape(str(row.get("Player Link", "")), quote=True)
                name = html.escape(str(row.get("Player", "")))
                raw_sort = str(row.get("Player", ""))
                value = f'<a class="prop-player-link" href="{url}" target="_blank" rel="noopener">{name}</a>'
            elif column == "VS":
                raw_sort = str(row.get("Opponent", ""))
                value = html.escape(str(row.get("Opponent", "")))
            elif column in rate_columns:
                raw = row.get(column, np.nan)
                raw_sort = "" if pd.isna(raw) else str(float(raw))
                if pd.isna(raw):
                    value = "—"
                else:
                    pct = float(raw) * 100
                    bar_width = min(max(pct, 0.0), 100.0)
                    color_class = "rate-green" if pct >= 70 else "rate-red" if pct <= 50 else "rate-yellow"
                    value = (
                        f'<div class="prop-percent {color_class}">'
                        f'<div class="prop-bar"><span style="width:{bar_width:.1f}%"></span></div>'
                        f'<b>{pct:.0f}%</b></div>'
                    )
            elif column in delta_columns:
                raw = row.get(column, np.nan)
                raw_sort = "" if pd.isna(raw) else str(float(raw))
                if pd.isna(raw):
                    value = "—"
                else:
                    raw = float(raw)
                    delta_class = "delta-positive" if raw > 0 else "delta-negative" if raw < 0 else "delta-neutral"
                    value = f'<b class="{delta_class}">{raw:+.1%}</b>'
            elif column in {"Line", "Average"}:
                raw_sort = str(float(row.get(column, 0)))
                value = f"{float(row.get(column, 0)):.1f}"
            else:
                raw_sort = str(row.get(column, ""))
                value = html.escape(str(row.get(column, "")))
            cells.append(f'<td data-sort="{html.escape(raw_sort, quote=True)}">{value}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        """
        <html><head><meta name="color-scheme" content="dark"><style>
        html, body {margin:0; padding:0; background:#0e1117; color:#f3f5f7; font-family:Arial,sans-serif;}
        .prop-table-wrap {overflow-x:auto; border:1px solid #343943; border-radius:10px;}
        .prop-table {width:100%; border-collapse:collapse; white-space:nowrap; font-size:.9rem;}
        .prop-table th {background:#171a20; color:#d9dee7; text-align:left; padding:10px; position:sticky; top:0;}
        .prop-table td {padding:9px 10px; border-top:1px solid #30343d; color:#f3f5f7;}
        .prop-table tr:hover td {background:#20252c;}
        .prop-player-link, .prop-player-link:visited {color:#ffffff !important; font-weight:700; text-decoration:none !important;}
        .prop-player-link:hover {color:#71ff9a !important; text-decoration:none !important;}
        .prop-sort-link {color:#d9dee7; font-weight:700; text-decoration:none; border:0; background:transparent; padding:0; cursor:pointer; font-size:inherit; white-space:nowrap;}
        .prop-sort-link:hover {color:#71ff9a;}
        .sort-arrow {color:#71ff9a; margin-left:4px;}
        .prop-percent {display:flex; align-items:center; gap:8px; min-width:110px;}
        .prop-bar {width:72px; height:8px; overflow:hidden; border-radius:10px; background:#30363d;}
        .prop-bar span {display:block; height:100%; border-radius:10px; background:currentColor; box-shadow:0 0 8px currentColor;}
        .rate-green {color:#39ff7a;}
        .rate-yellow {color:#ffe04b;}
        .rate-red {color:#ff4b4b;}
        .delta-positive {color:#39ff7a;}
        .delta-negative {color:#ff4b4b;}
        .delta-neutral {color:#ffe04b;}
        </style></head><body>
        """
        + f'<div class="prop-table-wrap"><table id="{table_id}" class="prop-table"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + """</tbody></table></div>
        <script>
        const propDirections = {};
        function sortPropTable(tableId, columnIndex, button) {
          const table = document.getElementById(tableId);
          const tbody = table.tBodies[0];
          const key = tableId + ':' + columnIndex;
          const ascending = !(propDirections[key] === true);
          propDirections[key] = ascending;
          const rows = Array.from(tbody.rows);
          rows.sort((a, b) => {
            const av = a.cells[columnIndex].dataset.sort || '';
            const bv = b.cells[columnIndex].dataset.sort || '';
            const an = Number(av), bn = Number(bv);
            const bothNumeric = av !== '' && bv !== '' && !Number.isNaN(an) && !Number.isNaN(bn);
            const result = bothNumeric ? an - bn : av.localeCompare(bv, undefined, {numeric:true, sensitivity:'base'});
            return ascending ? result : -result;
          });
          rows.forEach(row => tbody.appendChild(row));
          table.querySelectorAll('.sort-arrow').forEach(el => el.textContent = '');
          button.querySelector('.sort-arrow').textContent = ascending ? '▲' : '▼';
        }
        </script></body></html>"""
    )
    components.html(table_html, height=min(650, 55 + len(df) * 43), scrolling=True)


def render_player_detail_page(
    logs: pd.DataFrame,
    bankroll: float,
    kelly_multiplier: float,
    max_stake_pct: float,
    min_ev: float,
) -> None:
    if st.button("← Back to Props", type="primary"):
        st.session_state["app_view"] = "props"
        st.query_params.clear()
        st.rerun()

    players = sorted(logs.player.astype(str).unique())
    selected = st.session_state.get("individual_player_select", players[0])
    if selected not in players:
        selected = players[0]
    player = selected
    games = logs[logs.player.astype(str) == player].sort_values("game_date").copy()
    team = str(games.iloc[-1]["team"])

    st.title(player)
    st.caption(f"{team} • WNBA")

    current_season = int(games["game_date"].dt.year.max())
    season_games = games[games["game_date"].dt.year == current_season]
    st.subheader(f"{current_season} Season Stats")
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("PTS", f"{season_games['points'].mean():.1f}")
    stat2.metric("REB", f"{season_games['rebounds'].mean():.1f}")
    stat3.metric("AST", f"{season_games['assists'].mean():.1f}")
    stat4.metric("MIN", f"{season_games['minutes'].mean():.1f}")

    st.divider()
    st.subheader("Trends")

    prop_options = list(STAT_COLUMNS)
    selected_prop = st.session_state.get("individual_market_select", "Points")
    if selected_prop not in prop_options:
        selected_prop = "Points"
    c1, c2, c3 = st.columns(3)
    with c1:
        window_label = st.selectbox("History", ["L5", "L10", "L20", "Season", "H2H"], index=1, key="page_window")
    with c2:
        prop_label = st.selectbox("Prop", prop_options, index=prop_options.index(selected_prop), key="page_prop")
    with c3:
        trend_side = st.selectbox("Side", ["Over", "Under"], key="page_side")

    opponents = sorted(games.opponent.astype(str).unique())
    selected_opponent = st.session_state.get("selected_opponent", opponents[0])
    if selected_opponent not in opponents:
        selected_opponent = opponents[0]
    sportsbook = str(st.session_state.get("selected_sportsbook", "DraftKings"))
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        opponent = st.selectbox("Matchup", opponents, index=opponents.index(selected_opponent), key="page_opponent")
    with c5:
        line = st.number_input(f"{sportsbook} line", min_value=0.0, value=float(st.session_state.get("individual_line", 19.5)), step=.5, key="page_line")
    with c6:
        over_odds = st.number_input("Over odds", value=int(st.session_state.get("individual_over_odds", -110)), step=5, key="page_over")
    with c7:
        under_odds = st.number_input("Under odds", value=int(st.session_state.get("individual_under_odds", -110)), step=5, key="page_under")

    stat = STAT_COLUMNS[prop_label]
    values = games[stat]
    if window_label == "Season":
        view_games = games
    elif window_label == "H2H":
        view_games = games[games.opponent == opponent]
    else:
        view_games = games.tail(int(window_label[1:]))
    view_values = view_games[stat]
    projection, projection_components = project_stat(games, stat, "H", opponent)
    spread = max(float(values.tail(20).std(ddof=1)), .75) if len(values) > 1 else 1.0
    model_over = 1 - NormalDist(mu=projection, sigma=spread).cdf(line)
    model_under = 1 - model_over
    over_ev = model_over * american_to_decimal(int(over_odds)) - 1
    under_ev = model_under * american_to_decimal(int(under_odds)) - 1
    best_side = "Over" if over_ev >= under_ev else "Under"
    best_ev = max(over_ev, under_ev)
    best_prob = model_over if best_side == "Over" else model_under
    best_decimal = american_to_decimal(int(over_odds if best_side == "Over" else under_odds))
    stake_pct = min(kelly_fraction(best_prob, best_decimal) * kelly_multiplier, max_stake_pct)

    selected_hit_rate = hit_rate(view_values, line, trend_side)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Line", f"{line:g}")
    m2.metric("Odds", f"{int(over_odds if trend_side == 'Over' else under_odds):+d}")
    m3.metric("Average", "N/A" if view_values.empty else f"{view_values.mean():.1f}")
    m4.metric("Hit Rate", "N/A" if view_values.empty else f"{selected_hit_rate:.0%}")

    if best_ev > 0:
        stake = bankroll * stake_pct if best_ev >= min_ev else 0.0
        st.success(f"EV+ estimate: {best_side} {line:g} at {best_ev:+.1%} • fractional-Kelly stake ${stake:,.2f}")
    else:
        st.warning(f"No positive estimated value at the current prices. Best side: {best_side} {best_ev:+.1%}.")

    chart_games = view_games.tail(20).copy()
    max_value = max(float(chart_games[stat].max()) if not chart_games.empty else line, line, 1.0) * 1.15
    bars = []
    for _, game in chart_games.iterrows():
        actual = float(game[stat])
        covered = actual > line if trend_side == "Over" else actual < line
        color = "#00c968" if covered else "#ff2638"
        height = max(8.0, actual / max_value * 230)
        bars.append(
            '<div class="trend-col">'
            f'<div class="trend-bar" style="height:{height:.1f}px;background:{color};"><b>{actual:g}</b></div>'
            f'<span>{html.escape(str(game["opponent"]))}</span>'
            f'<small>{pd.Timestamp(game["game_date"]).strftime("%m/%d")}</small>'
            '</div>'
        )
    line_bottom = 42 + line / max_value * 230
    chart_html = f"""
    <html><head><style>
    html,body{{margin:0;background:#0e1117;color:#fff;font-family:Arial,sans-serif;}}
    .chart{{height:310px;position:relative;display:flex;align-items:flex-end;gap:8px;padding:10px 8px 42px;box-sizing:border-box;}}
    .prop-line{{position:absolute;left:0;right:0;bottom:{line_bottom:.1f}px;border-top:2px solid #ffae00;z-index:2;}}
    .prop-line b{{background:#ffae00;color:#111;padding:4px 7px;border-radius:6px;position:absolute;left:0;top:-14px;}}
    .trend-col{{height:250px;flex:1;min-width:38px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;position:relative;}}
    .trend-bar{{width:100%;max-width:70px;border-radius:8px 8px 2px 2px;text-align:center;box-sizing:border-box;padding-top:7px;}}
    .trend-col span{{font-weight:700;margin-top:6px;font-size:.75rem;}}
    .trend-col small{{color:#9da3ae;font-size:.7rem;margin-top:2px;}}
    </style></head><body><div class="chart"><div class="prop-line"><b>{line:g}</b></div>{''.join(bars)}</div></body></html>
    """
    components.html(chart_html, height=320, scrolling=True)

    result_rows = []
    for _, game in chart_games.sort_values("game_date", ascending=False).iterrows():
        actual = float(game[stat])
        covered = actual > line if trend_side == "Over" else actual < line
        result_rows.append({
            "Result": "✅ Covered" if covered else "❌ Missed",
            "Date": pd.Timestamp(game["game_date"]).date(),
            "VS": game["opponent"],
            "Venue": "Home" if game["home_away"] == "H" else "Away",
            prop_label: actual,
            "Pick": f"{trend_side} {line:g}",
        })
    st.dataframe(pd.DataFrame(result_rows), hide_index=True, use_container_width=True)

    split_rows = []
    for label, subset in [
        ("Last 5", games.tail(5)), ("Last 10", games.tail(10)), ("Last 20", games.tail(20)),
        ("Season", games), (f"H2H vs {opponent}", games[games.opponent == opponent]),
        ("Home", games[games.home_away == "H"]), ("Away", games[games.home_away == "A"]),
    ]:
        split_rows.append({
            "Split": label, "Games": len(subset), "Average": subset[stat].mean(),
            "Over": hit_rate(subset[stat], line, "Over"), "Under": hit_rate(subset[stat], line, "Under"),
        })
    with st.expander("Historical splits"):
        st.dataframe(
            pd.DataFrame(split_rows), hide_index=True, use_container_width=True,
            column_config={
                "Average": st.column_config.NumberColumn(format="%.1f"),
                "Over": st.column_config.NumberColumn(format="percent"),
                "Under": st.column_config.NumberColumn(format="percent"),
            },
        )
    st.caption("Projection components: " + ", ".join(projection_components) + ". Research only; estimates are not guarantees.")


st.title("🏀 WNBA Prop Labs")
st.caption("Compare a sportsbook prop with a player's historical game logs—no live-statistics subscription required.")

# Internal risk defaults keep EV and staking estimates consistent without a sidebar.
bankroll = 100.0
kelly_multiplier = .25
max_stake_pct = .02
min_ev = .02

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

linked_player = st.query_params.get("player")
if linked_player:
    st.session_state["individual_player_select"] = linked_player
    st.session_state["individual_market_select"] = st.query_params.get("prop", "Points")
    try:
        st.session_state["individual_line"] = float(st.query_params.get("line", 19.5))
        st.session_state["individual_over_odds"] = int(st.query_params.get("over", -110))
        st.session_state["individual_under_odds"] = int(st.query_params.get("under", -110))
    except (TypeError, ValueError):
        pass
    st.session_state["selected_opponent"] = st.query_params.get("opponent", "")
    st.session_state["selected_sportsbook"] = st.query_params.get("sportsbook", "DraftKings")
    st.session_state["app_view"] = "player"

if st.session_state.get("app_view") == "player":
    render_player_detail_page(logs, bankroll, kelly_multiplier, max_stake_pct, min_ev)
    st.stop()

with st.expander("📊 All-Player Prop Trends", expanded=True):
    st.caption("Filter the sportsbook prop feed, compare L5/L10/L20 hit rates, and click a player for full details.")
    trend1, trend2, trend3 = st.columns(3)
    with trend1:
        trend_stat_label = st.selectbox(
            "Prop category", ["All Props", "Points", "Rebounds", "Assists"]
        )
        odds_source = st.selectbox(
            "Sportsbook odds source",
            ["DraftKings", "FanDuel", "BetOnline"],
        )
    with trend2:
        team_options = ["All Teams"] + sorted(
            team for team in logs["team"].dropna().astype(str).unique() if team
        )
        selected_team = st.selectbox("Team", team_options)
        game_filter_slot = st.empty()
    with trend3:
        side_filter = st.selectbox("Side", ["Best side", "Over", "Under"])

    prop_template = make_prop_board(logs, [])
    board = pd.DataFrame()
    selected_sportsbook = odds_source
    api_key = odds_api_key()
    if not api_key:
        st.warning("Sportsbook odds need a private Odds API key. Add it in Manage app → Settings → Secrets.")
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
            with st.spinner(f"Loading current {odds_source} player props…"):
                requested_markets = (
                    ",".join(DK_MARKETS.values())
                    if trend_stat_label == "All Props"
                    else DK_MARKETS[trend_stat_label]
                )
                board = fetch_sportsbook_props(
                    api_key,
                    requested_markets,
                    SPORTSBOOKS[odds_source],
                )
            if board.empty:
                st.info(f"{odds_source} has no WNBA props posted for this market right now. Try again closer to game time.")
            else:
                latest_players = logs.sort_values("game_date").groupby("player", as_index=False).tail(1)
                name_lookup = dict(zip(latest_players["player"].map(normalize_player_name), latest_players["player"]))
                team_lookup = dict(zip(latest_players["player"], latest_players["team"]))
                board["Sportsbook Player"] = board["Player"]
                board["Player"] = board["Player"].map(
                    lambda value: name_lookup.get(normalize_player_name(value), value)
                )
                board["Team"] = board["Player"].map(team_lookup).fillna("")
                board["Sportsbook"] = odds_source
                board["Opponent"] = board.apply(
                    lambda row: matchup_opponent(row.get("Game", ""), row.get("Team", "")), axis=1
                )
                board = board.dropna(subset=["Over Odds", "Under Odds"])
                category_text = "player" if trend_stat_label == "All Props" else trend_stat_label.lower()
                st.success(f"Loaded {len(board)} current {odds_source} {category_text} props. Odds refresh every 15 minutes.")
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exc:
            st.error(f"Could not load {odds_source} odds: {exc}")
            st.info("Check the API key and account quota, then try again.")

    if board.empty:
        board = prop_template.iloc[0:0]
    team_board = board
    if selected_team != "All Teams":
        team_board = board[board["Team"].astype(str) == selected_team].copy()
    game_options = ["All Games"] + sorted(team_board.get("Game", pd.Series(dtype=str)).dropna().astype(str).unique())
    selected_game = game_filter_slot.selectbox(
        "Game",
        game_options,
        format_func=lambda value: value if value == "All Games" else matchup_label(value),
    )
    board = team_board
    if selected_game != "All Games":
        board = board[board["Game"].astype(str) == selected_game].copy()

    fallback_prop = "Points" if trend_stat_label == "All Props" else trend_stat_label
    trend_stat = STAT_COLUMNS[fallback_prop]
    trend_window = 5
    trend_table = prop_feed(logs, board, fallback_prop, trend_window, side_filter)
    if trend_table.empty:
        st.info("Enter at least one prop line above to generate the all-player hit-rate table.")
    else:
        st.caption("Click a player name to open that player's separate analysis page.")
        render_prop_table(trend_table)

        st.subheader("EV+ Betting Candidates")
        st.caption(
            f"Estimated from the historical projection and current {selected_sportsbook} price. "
            "EV+ is a research signal, not a guarantee of profit."
        )
        ev_table = trend_table[trend_table["Estimated EV"] > 0].sort_values(
            "Estimated EV", ascending=False
        )
        if ev_table.empty:
            st.info("No displayed props currently have positive estimated value.")
        else:
            render_prop_table(ev_table, ev_only=True)
        st.download_button(
            "Download trend results",
            trend_table.to_csv(index=False),
            f"wnba_{trend_stat}_{trend_window}_game_trends.csv",
            "text/csv",
        )

# The detailed analyzer is rendered as its own view after a prop-row click.
# Keep the main page focused on the prop feed and EV+ candidates.
st.stop()

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
