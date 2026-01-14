"""Build season-level team features for experience modeling."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
NBA_DIR = RAW_DIR / "nba_api"
BREF_DIR = RAW_DIR / "bref"
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
    config_path = NBA_DIR / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return list(config.get("seasons", []))
    return sorted([path.name for path in NBA_DIR.iterdir() if path.is_dir()])


def load_season_csv(season: str, name: str) -> pd.DataFrame:
    path = NBA_DIR / season / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = season
    return normalize_columns(df)


def load_bref_series() -> pd.DataFrame:
    frames = []
    for path in BREF_DIR.glob("playoff_series_*.csv"):
        df = pd.read_csv(path)
        frames.append(normalize_columns(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def round_rank(round_name: str) -> int:
    label = round_name.lower()
    if "finals" in label and "conference" not in label and "division" not in label:
        return 4
    if "conference finals" in label or "division finals" in label:
        return 3
    if "semifinals" in label:
        return 2
    if "first round" in label or "quarterfinals" in label:
        return 1
    return 0


def normalize_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def compute_team_rounds(series_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    if series_df.empty:
        return series_df

    team_map = {
        normalize_team_name(row["full_name"]): row["id"]
        for _, row in teams_df.iterrows()
    }
    for _, row in teams_df.iterrows():
        team_map[normalize_team_name(row["nickname"])] = row["id"]

    series_df["season"] = series_df["season_end_year"].apply(
        lambda year: f"{year - 1}-{str(year)[-2:]}"
    )
    series_df["round_rank"] = series_df["round"].apply(round_rank)
    series_df["winner_id"] = series_df["winner"].map(
        lambda name: team_map.get(normalize_team_name(name))
    )
    series_df["loser_id"] = series_df["loser"].map(
        lambda name: team_map.get(normalize_team_name(name))
    )

    melted = []
    for side in ("winner", "loser"):
        melted.append(
            series_df[
                [
                    "season",
                    f"{side}_id",
                    "round_rank",
                ]
            ].rename(columns={f"{side}_id": "team_id"})
        )
    team_rounds = pd.concat(melted, ignore_index=True)
    team_rounds = team_rounds.dropna(subset=["team_id"])
    team_rounds["team_id"] = team_rounds["team_id"].astype(int)
    return (
        team_rounds.groupby(["season", "team_id"], as_index=False)["round_rank"]
        .max()
        .rename(columns={"round_rank": "playoff_round_reached"})
    )


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

    reg_frames = [load_season_csv(season, "player_regular.csv") for season in seasons]
    playoff_frames = [
        load_season_csv(season, "player_playoffs.csv") for season in seasons
    ]
    team_frames = [load_season_csv(season, "team_regular.csv") for season in seasons]
    team_playoff_frames = [
        load_season_csv(season, "team_playoffs.csv") for season in seasons
    ]

    reg_all = pd.concat([df for df in reg_frames if not df.empty], ignore_index=True)
    playoff_all = pd.concat(
        [df for df in playoff_frames if not df.empty], ignore_index=True
    )
    team_all = pd.concat([df for df in team_frames if not df.empty], ignore_index=True)
    team_playoff_all = pd.concat(
        [df for df in team_playoff_frames if not df.empty], ignore_index=True
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

    if not teams_df.empty:
        team_features = team_features.merge(
            teams_df[["id", "full_name", "abbreviation"]],
            left_on="team_id",
            right_on="id",
            how="left",
        ).drop(columns=["id"])

    series_df = load_bref_series()
    if not series_df.empty and not teams_df.empty:
        series_processed = compute_team_rounds(series_df, teams_df)
        team_features = team_features.merge(
            series_processed, on=["season", "team_id"], how="left"
        )
        team_features["playoff_round_reached"] = team_features[
            "playoff_round_reached"
        ].fillna(0)
        team_features = team_features.sort_values(["team_id", "season"])
        team_features["playoff_rounds_prior_total"] = (
            team_features.groupby("team_id")["playoff_round_reached"].cumsum()
            - team_features["playoff_round_reached"]
        )

        series_df.to_csv(PROCESSED_DIR / "playoff_series.csv", index=False)

    reg_all.to_csv(PROCESSED_DIR / "player_experience.csv", index=False)
    team_features.to_csv(PROCESSED_DIR / "team_features.csv", index=False)

    print(f"Wrote team features to {PROCESSED_DIR / 'team_features.csv'}")


if __name__ == "__main__":
    main()
