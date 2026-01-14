from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
FEATURES_PATH = DATA_DIR / "team_features.csv"
_FEATURES_CACHE: pd.DataFrame | None = None


def load_features() -> pd.DataFrame | None:
    global _FEATURES_CACHE
    if _FEATURES_CACHE is not None:
        return _FEATURES_CACHE
    if not FEATURES_PATH.exists():
        return None
    _FEATURES_CACHE = pd.read_csv(FEATURES_PATH)
    return _FEATURES_CACHE


def build_feature_breakdown(row: pd.Series) -> list[dict]:
    mapping = [
        ("avg_age", "Avg age"),
        ("avg_seasons_in_league", "Avg seasons in league"),
        ("avg_playoff_games_prior", "Avg playoff games"),
        ("avg_playoff_wins_prior", "Avg playoff wins"),
        ("injury_games_missed", "Injury games missed"),
        ("team_win_pct", "Team win%"),
        ("net_rating", "Net rating"),
    ]
    breakdown = []
    for key, label in mapping:
        if key in row:
            value = row[key]
            if pd.isna(value):
                continue
            breakdown.append({"key": key, "label": label, "value": float(value)})
    return breakdown


@router.get("/{team_id}/season/{season}")
async def get_team_prediction(team_id: int, season: str) -> dict:
    if len(season) != 9 or season[4] != "-":
        raise HTTPException(status_code=400, detail="Season must look like 2023-24")

    features_df = load_features()
    if features_df is not None:
        match = features_df[
            (features_df["team_id"] == team_id) & (features_df["season"] == season)
        ]
        if not match.empty:
            row = match.iloc[0]
            return {
                "team_id": team_id,
                "season": season,
                "prediction": {
                    "playoff_round": "TBD",
                    "confidence": 0.0,
                },
                "experience_breakdown": build_feature_breakdown(row),
                "notes": "Model pipeline not wired yet.",
            }

    return {
        "team_id": team_id,
        "season": season,
        "prediction": {
            "playoff_round": "TBD",
            "confidence": 0.0,
        },
        "experience_breakdown": [
            {"key": "avg_age", "label": "Avg age", "value": 27.4},
            {"key": "avg_seasons", "label": "Avg seasons in league", "value": 5.6},
            {"key": "avg_playoff_games", "label": "Avg playoff games", "value": 42.0},
            {"key": "injury_games", "label": "Injury games missed", "value": 118.0},
        ],
        "notes": "Model pipeline not wired yet.",
    }
