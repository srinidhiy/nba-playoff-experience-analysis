from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
FEATURES_PATH = DATA_DIR / "team_features.csv"
MODEL_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "playoff_round_model.joblib"
_FEATURES_CACHE: pd.DataFrame | None = None
_MODEL_CACHE: dict | None = None

ROUND_LABELS = {
    0: "No Playoffs",
    1: "First Round",
    2: "Second Round",
    3: "Conference Finals",
    4: "Finals",
}


def load_features() -> pd.DataFrame | None:
    global _FEATURES_CACHE
    if _FEATURES_CACHE is not None:
        return _FEATURES_CACHE
    if not FEATURES_PATH.exists():
        return None
    _FEATURES_CACHE = pd.read_csv(FEATURES_PATH)
    return _FEATURES_CACHE


def load_model_bundle() -> dict | None:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        return None
    _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


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
            model_bundle = load_model_bundle()
            if model_bundle is None:
                raise HTTPException(
                    status_code=503,
                    detail="Model not trained. Run scripts/train_model.py.",
                )

            model = model_bundle["model"]
            feature_columns = model_bundle["feature_columns"]
            classes = [int(label) for label in model.classes_]
            label_map = {label: ROUND_LABELS.get(label, f"Round {label}") for label in classes}

            x = row[feature_columns].to_frame().T
            predicted_round = int(model.predict(x)[0])
            probabilities = model.predict_proba(x)[0].tolist()

            probability_output = [
                {
                    "round": label,
                    "label": label_map[label],
                    "probability": float(prob),
                }
                for label, prob in zip(classes, probabilities)
            ]

            return {
                "team_id": team_id,
                "season": season,
                "prediction": {
                    "round": predicted_round,
                    "label": ROUND_LABELS.get(predicted_round, f"Round {predicted_round}"),
                },
                "probabilities": probability_output,
                "features_used": {
                    key: float(row[key]) for key in feature_columns if key in row
                },
                "experience_breakdown": build_feature_breakdown(row),
                "notes": "Prediction generated from trained model.",
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
