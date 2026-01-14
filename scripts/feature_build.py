"""Build season-level team features for experience modeling."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
NBA_DIR = RAW_DIR / "nba_api"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for col in df.columns:
        col = re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()
        cleaned.append(col)
    df.columns = cleaned
    return df


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def list_seasons() -> list[str]:
    seasons = []
    if NBA_DIR.exists():
        for path in NBA_DIR.iterdir():
            if path.is_dir() and re.match(r"^\d{4}-\d{2}$", path.name):
                seasons.append(path.name)
    config_path = NBA_DIR / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        seasons.extend(config.get("seasons", []))
    return sorted(set(seasons))


def load_season_csv(season: str, name: str) -> pd.DataFrame:
    path = NBA_DIR / season / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = season
    df = normalize_columns(df)
    if "teamid" in df.columns and "team_id" not in df.columns:
        df = df.rename(columns={"teamid": "team_id"})
    if "playerid" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"playerid": "player_id"})
    return df




def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    seasons = list_seasons()
    if not seasons:
        print("No seasons found under data/raw/nba_api.")
        return

    teams_path = NBA_DIR / "teams.csv"
    teams_df = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame()
    if not teams_df.empty:
        teams_df = normalize_columns(teams_df)
        team_id_set = set(teams_df["id"].astype(int).tolist())
    else:
        team_id_set = set()

    reg_frames = [load_season_csv(season, "player_regular.csv") for season in seasons]
    playoff_frames = [
        load_season_csv(season, "player_playoffs.csv") for season in seasons
    ]
    team_frames = [load_season_csv(season, "team_regular.csv") for season in seasons]
    team_playoff_frames = [
        load_season_csv(season, "team_playoffs.csv") for season in seasons
    ]
    standings_frames = [load_season_csv(season, "standings.csv") for season in seasons]
    series_frames = [load_season_csv(season, "series_results.csv") for season in seasons]

    reg_all = pd.concat([df for df in reg_frames if not df.empty], ignore_index=True)
    playoff_all = pd.concat(
        [df for df in playoff_frames if not df.empty], ignore_index=True
    )
    team_all = pd.concat([df for df in team_frames if not df.empty], ignore_index=True)
    team_playoff_all = pd.concat(
        [df for df in team_playoff_frames if not df.empty], ignore_index=True
    )
    standings_all = pd.concat(
        [df for df in standings_frames if not df.empty], ignore_index=True
    )
    series_all = pd.concat([df for df in series_frames if not df.empty], ignore_index=True)

    if team_id_set:
        if "team_id" in team_all.columns:
            team_all = team_all[team_all["team_id"].isin(team_id_set)]
        if "team_id" in team_playoff_all.columns:
            team_playoff_all = team_playoff_all[team_playoff_all["team_id"].isin(team_id_set)]
        if "team_id" in standings_all.columns:
            standings_all = standings_all[standings_all["team_id"].isin(team_id_set)]

    if not team_all.empty:
        team_all = team_all.sort_values(["season", "team_id"]).drop_duplicates(
            subset=["season", "team_id"]
        )
    if not team_playoff_all.empty:
        team_playoff_all = team_playoff_all.sort_values(["season", "team_id"]).drop_duplicates(
            subset=["season", "team_id"]
        )
    if not standings_all.empty:
        standings_all = standings_all.sort_values(["season", "team_id"]).drop_duplicates(
            subset=["season", "team_id"]
        )

    reg_all["season_start"] = reg_all["season"].map(season_start_year)
    reg_all = reg_all.sort_values(["player_id", "season_start"])
    reg_all["seasons_in_league"] = reg_all.groupby("player_id").cumcount()

    playoff_all["season_start"] = playoff_all["season"].map(season_start_year)
    playoff_all = playoff_all.sort_values(["player_id", "season_start"])
    if "gp" in playoff_all.columns:
        playoff_all["playoff_games_prior"] = (
            playoff_all.groupby("player_id")["gp"].cumsum() - playoff_all["gp"]
        )
    else:
        playoff_all["playoff_games_prior"] = 0

    if "w" in playoff_all.columns:
        playoff_all["playoff_wins_prior"] = (
            playoff_all.groupby("player_id")["w"].cumsum() - playoff_all["w"]
        )
    else:
        playoff_all["playoff_wins_prior"] = 0

    playoff_prior = playoff_all[
        ["player_id", "season", "playoff_games_prior", "playoff_wins_prior"]
    ]

    reg_all = reg_all.merge(
        playoff_prior, on=["player_id", "season"], how="left"
    ).fillna({"playoff_games_prior": 0, "playoff_wins_prior": 0})

    if "min" not in reg_all.columns:
        reg_all["min"] = 0
    if "gp" not in reg_all.columns:
        reg_all["gp"] = 0

    team_minutes = (
        reg_all.groupby(["season", "team_id"], as_index=False)["min"].sum().rename(
            columns={"min": "team_minutes"}
        )
    )
    reg_all = reg_all.merge(team_minutes, on=["season", "team_id"], how="left")
    reg_all["weight"] = reg_all["min"] / reg_all["team_minutes"].replace(0, 1)

    team_games = team_all[["season", "team_id", "gp"]].rename(
        columns={"gp": "team_gp"}
    )
    reg_all = reg_all.merge(team_games, on=["season", "team_id"], how="left")
    reg_all["team_gp"] = reg_all["team_gp"].fillna(82)
    reg_all["games_missed"] = (reg_all["team_gp"] - reg_all["gp"]).clip(lower=0)
    prev_roster = reg_all[["player_id", "team_id", "season_start"]].copy()
    prev_roster["season_start"] = prev_roster["season_start"] + 1
    prev_roster["was_on_team_prev"] = True
    reg_all = reg_all.merge(
        prev_roster, on=["player_id", "team_id", "season_start"], how="left"
    )
    reg_all["was_on_team_prev"] = reg_all["was_on_team_prev"].fillna(False)

    def weighted_avg(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float((df[col] * df["weight"]).sum())

    team_features = reg_all.groupby(["season", "team_id"]).apply(
        lambda df: pd.Series(
            {
                "total_minutes": df["min"].sum(),
                "roster_size": df["player_id"].nunique(),
                "avg_age": weighted_avg(df, "age"),
                "avg_seasons_in_league": weighted_avg(df, "seasons_in_league"),
                "avg_playoff_games_prior": weighted_avg(df, "playoff_games_prior"),
                "avg_playoff_wins_prior": weighted_avg(df, "playoff_wins_prior"),
                "injury_games_missed": df["games_missed"].sum(),
                "roster_continuity": (
                    df.loc[df["was_on_team_prev"], "min"].sum()
                    / max(df["min"].sum(), 1)
                ),
            }
        )
    )
    team_features = team_features.reset_index()

    if not team_all.empty:
        team_cols = [
            col
            for col in ["season", "team_id", "w", "l", "w_pct", "net_rating", "pace"]
            if col in team_all.columns
        ]
        team_features = team_features.merge(
            team_all[team_cols],
            on=["season", "team_id"],
            how="left",
            suffixes=("", "_team"),
        )
        team_features = team_features.rename(
            columns={"w": "team_wins", "l": "team_losses", "w_pct": "team_win_pct"}
        )

    if not standings_all.empty:
        standings_cols = [
            col
            for col in [
                "season",
                "team_id",
                "conference",
                "playoff_rank",
                "playoffrank",
                "conf_rank",
                "conference_rank",
            ]
            if col in standings_all.columns
        ]
        standings_trimmed = standings_all[standings_cols].copy()
        if "team_id" in standings_trimmed.columns:
            standings_trimmed["team_id"] = pd.to_numeric(
                standings_trimmed["team_id"], errors="coerce"
            ).astype("Int64")
        seed = pd.Series([pd.NA] * len(standings_trimmed))
        for col in ["playoff_rank", "playoffrank", "conf_rank", "conference_rank"]:
            if col in standings_trimmed.columns:
                seed = seed.fillna(standings_trimmed[col])
        standings_trimmed["seed"] = seed
        standings_trimmed["seed"] = pd.to_numeric(
            standings_trimmed["seed"], errors="coerce"
        )
        standings_trimmed = standings_trimmed.drop(
            columns=[
                col
                for col in [
                    "playoff_rank",
                    "playoffrank",
                    "conf_rank",
                    "conference_rank",
                ]
                if col in standings_trimmed.columns
            ]
        )
        team_features = team_features.merge(
            standings_trimmed,
            on=["season", "team_id"],
            how="left",
        )

    if not team_playoff_all.empty:
        playoff_cols = [
            col
            for col in ["season", "team_id", "w", "l", "w_pct"]
            if col in team_playoff_all.columns
        ]
        playoff_data = team_playoff_all[playoff_cols].rename(
            columns={
                "w": "playoff_wins",
                "l": "playoff_losses",
                "w_pct": "playoff_win_pct",
            }
        )
        team_features = team_features.merge(
            playoff_data,
            on=["season", "team_id"],
            how="left",
        )
        team_features["made_playoffs"] = team_features["playoff_wins"].notna()

        def estimate_round(wins: float) -> int:
            if pd.isna(wins):
                return 0
            if wins >= 12:
                return 4
            if wins >= 8:
                return 3
            if wins >= 4:
                return 2
            return 1

        team_features["playoff_round_reached_est"] = team_features[
            "playoff_wins"
        ].apply(estimate_round)

    if not teams_df.empty:
        team_features = team_features.merge(
            teams_df[["id", "full_name", "abbreviation"]],
            left_on="team_id",
            right_on="id",
            how="left",
        ).drop(columns=["id"])

    if not series_all.empty:
        series_rounds = (
            series_all.groupby(["season", "team_id"], as_index=False)["round"]
            .max()
            .rename(columns={"round": "playoff_round_reached"})
        )
        team_features = team_features.merge(
            series_rounds, on=["season", "team_id"], how="left"
        )
        team_features["playoff_round_reached"] = team_features[
            "playoff_round_reached"
        ].fillna(0)
        team_features["playoff_round_reached_source"] = "nba_api_series"
        team_features = team_features.sort_values(["team_id", "season"])
        team_features["playoff_rounds_prior_total"] = (
            team_features.groupby("team_id")["playoff_round_reached"].cumsum()
            - team_features["playoff_round_reached"]
        )
        series_all.to_csv(PROCESSED_DIR / "playoff_series.csv", index=False)

    if "playoff_round_reached" not in team_features.columns:
        team_features["playoff_round_reached"] = team_features.get(
            "playoff_round_reached_est", 0
        )
        team_features["playoff_round_reached_source"] = "wins_estimate"

    reg_all.to_csv(PROCESSED_DIR / "player_experience.csv", index=False)
    team_features.to_csv(PROCESSED_DIR / "team_features.csv", index=False)

    print(f"Wrote team features to {PROCESSED_DIR / 'team_features.csv'}")


if __name__ == "__main__":
    main()
