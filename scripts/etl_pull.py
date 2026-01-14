"""Pull raw data from nba_api and Basketball-Reference when needed."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguestandings,
    commonplayoffseries,
    leaguegamefinder,
)
from nba_api.stats.static import teams as nba_teams

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
NBA_DIR = RAW_DIR / "nba_api"
BREF_DIR = RAW_DIR / "bref"
DEFAULT_START_YEAR = 1996
DEFAULT_SLEEP = 0.6


def season_label(start_year: int) -> str:
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def current_start_year() -> int:
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def build_seasons(start_year: int, end_year: int) -> list[str]:
    return [season_label(year) for year in range(start_year, end_year + 1)]


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fetch_team_list() -> pd.DataFrame:
    teams = nba_teams.get_teams()
    return pd.DataFrame(teams)


def fetch_team_stats(season: str, season_type: str) -> pd.DataFrame:
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
    )
    return stats.get_data_frames()[0]


def fetch_player_stats(season: str, season_type: str) -> pd.DataFrame:
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
    )
    return stats.get_data_frames()[0]


def fetch_standings(season: str) -> pd.DataFrame:
    standings = leaguestandings.LeagueStandings(season=season)
    return standings.get_data_frames()[0]


def playoff_round_from_series_id(series_id: str | int) -> int:
    series_str = str(series_id)
    if len(series_str) < 3:
        return 0
    suffix = int(series_str[-3:])
    if 10 <= suffix <= 17:
        return 1
    if 20 <= suffix <= 23:
        return 2
    if 30 <= suffix <= 31:
        return 3
    if suffix == 40:
        return 4
    return 0


def fetch_playoff_series_results(season: str) -> pd.DataFrame:
    series_games = commonplayoffseries.CommonPlayoffSeries(season=season).get_data_frames()[0]
    team_games = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Playoffs",
        player_or_team_abbreviation="T",
    ).get_data_frames()[0]

    winners = team_games[team_games["WL"] == "W"][["GAME_ID", "TEAM_ID"]]
    winners = winners.rename(columns={"TEAM_ID": "WINNER_TEAM_ID"})
    series_games = series_games.merge(winners, on="GAME_ID", how="left")
    series_games["round"] = series_games["SERIES_ID"].apply(playoff_round_from_series_id)

    series_games["HOME_TEAM_ID"] = series_games["HOME_TEAM_ID"].astype(int)
    series_games["VISITOR_TEAM_ID"] = series_games["VISITOR_TEAM_ID"].astype(int)

    wins_by_team = (
        series_games.groupby(["SERIES_ID", "WINNER_TEAM_ID"], as_index=False)
        .size()
        .rename(columns={"size": "team_wins", "WINNER_TEAM_ID": "TEAM_ID"})
    )

    participants = (
        series_games.groupby("SERIES_ID", as_index=False)[
            ["HOME_TEAM_ID", "VISITOR_TEAM_ID", "round"]
        ]
        .first()
        .rename(columns={"HOME_TEAM_ID": "TEAM_A", "VISITOR_TEAM_ID": "TEAM_B"})
    )

    records = []
    for _, row in participants.iterrows():
        series_id = row["SERIES_ID"]
        team_a = int(row["TEAM_A"])
        team_b = int(row["TEAM_B"])
        round_num = int(row["round"])

        wins_a = wins_by_team.loc[
            (wins_by_team["SERIES_ID"] == series_id) & (wins_by_team["TEAM_ID"] == team_a),
            "team_wins",
        ].sum()
        wins_b = wins_by_team.loc[
            (wins_by_team["SERIES_ID"] == series_id) & (wins_by_team["TEAM_ID"] == team_b),
            "team_wins",
        ].sum()

        for team_id, opp_id, wins, opp_wins in [
            (team_a, team_b, wins_a, wins_b),
            (team_b, team_a, wins_b, wins_a),
        ]:
            records.append(
                {
                    "season": season,
                    "series_id": series_id,
                    "round": round_num,
                    "team_id": team_id,
                    "opponent_id": opp_id,
                    "team_wins": int(wins),
                    "opponent_wins": int(opp_wins),
                    "is_winner": wins > opp_wins,
                }
            )

    return pd.DataFrame(records)


def fetch_rosters(team_ids: Iterable[int], season: str, sleep_seconds: float) -> pd.DataFrame:
    frames = []
    for team_id in team_ids:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
        df["TEAM_ID"] = team_id
        frames.append(df)
        time.sleep(sleep_seconds)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _extract_bref_table(html: str, table_id: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html), attrs={"id": table_id})
        if tables:
            return tables[0]
    except ValueError:
        pass

    soup = BeautifulSoup(html, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if table_id in comment:
            try:
                tables = pd.read_html(StringIO(comment), attrs={"id": table_id})
                if tables:
                    return tables[0]
            except ValueError:
                continue

    return pd.DataFrame()


def fetch_bref_playoff_series(end_year: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/playoffs/NBA_{end_year}.html"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    series_table = _extract_bref_table(response.text, "playoff_series")
    if series_table.empty:
        return series_table

    series_table = series_table.rename(
        columns={
            "Series": "round",
            "Winner": "winner",
            "Loser": "loser",
            "W": "winner_wins",
            "L": "loser_wins",
        }
    )
    series_table["season_end_year"] = end_year
    return series_table[
        ["season_end_year", "round", "winner", "winner_wins", "loser", "loser_wins"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull raw NBA data.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=current_start_year())
    parser.add_argument("--seasons", nargs="*", help="Season labels like 2023-24")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--include-bref", action="store_true")
    parser.add_argument("--only-standings", action="store_true")
    parser.add_argument("--only-bref", action="store_true")
    parser.add_argument("--only-series", action="store_true")
    args = parser.parse_args()

    NBA_DIR.mkdir(parents=True, exist_ok=True)
    BREF_DIR.mkdir(parents=True, exist_ok=True)

    if args.seasons:
        seasons = args.seasons
    else:
        seasons = build_seasons(args.start_year, args.end_year)

    teams_df = fetch_team_list()
    write_dataframe(teams_df, NBA_DIR / "teams.csv")
    config_path = NBA_DIR / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        previous = set(existing.get("seasons", []))
    else:
        previous = set()
    merged = sorted(previous.union(seasons))
    config_path.write_text(
        json.dumps({"seasons": merged, "generated_at": date.today().isoformat()}, indent=2)
    )

    team_ids = teams_df["id"].tolist()

    for season in seasons:
        season_dir = NBA_DIR / season
        season_dir.mkdir(parents=True, exist_ok=True)

        if not args.only_standings and not args.only_bref and not args.only_series:
            team_regular = fetch_team_stats(season, "Regular Season")
            write_dataframe(team_regular, season_dir / "team_regular.csv")
            time.sleep(args.sleep)

            team_playoffs = fetch_team_stats(season, "Playoffs")
            write_dataframe(team_playoffs, season_dir / "team_playoffs.csv")
            time.sleep(args.sleep)

            player_regular = fetch_player_stats(season, "Regular Season")
            write_dataframe(player_regular, season_dir / "player_regular.csv")
            time.sleep(args.sleep)

            player_playoffs = fetch_player_stats(season, "Playoffs")
            write_dataframe(player_playoffs, season_dir / "player_playoffs.csv")
            time.sleep(args.sleep)

        if not args.only_bref and not args.only_series:
            standings = fetch_standings(season)
            write_dataframe(standings, season_dir / "standings.csv")
            time.sleep(args.sleep)

        if not args.only_standings and not args.only_bref and not args.only_series:
            rosters = fetch_rosters(team_ids, season, args.sleep)
            write_dataframe(rosters, season_dir / "rosters.csv")

        if not args.only_standings and not args.only_bref:
            series_results = fetch_playoff_series_results(season)
            if not series_results.empty:
                write_dataframe(series_results, season_dir / "series_results.csv")
                time.sleep(args.sleep)

        if (args.include_bref or args.only_bref) and not args.only_standings:
            end_year = int(season.split("-")[0]) + 1
            series_df = fetch_bref_playoff_series(end_year)
            if not series_df.empty:
                write_dataframe(series_df, BREF_DIR / f"playoff_series_{season}.csv")


if __name__ == "__main__":
    main()
