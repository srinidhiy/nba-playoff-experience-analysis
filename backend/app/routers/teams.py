from pathlib import Path

import joblib
import pandas as pd
import re

from fastapi import APIRouter, HTTPException

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
FEATURES_PATH = DATA_DIR / "team_features.csv"
MODEL_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "playoff_round_model.joblib"
ERA_MODEL_PATHS = {
    "2015-2018": Path(__file__).resolve().parents[3]
    / "artifacts"
    / "playoff_round_model_2015-2018.joblib",
    "2019-2024": Path(__file__).resolve().parents[3]
    / "artifacts"
    / "playoff_round_model_2019-2024.joblib",
}
RAW_TEAMS_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "nba_api" / "teams.csv"
_FEATURES_CACHE: pd.DataFrame | None = None
_MODEL_CACHE: dict | None = None
_ERA_MODEL_CACHE: dict[str, dict] = {}

ROUND_LABELS = {
    0: "No Playoffs",
    1: "First Round",
    2: "Second Round",
    3: "Conference Finals",
    4: "Finals",
}


def load_features(force_reload: bool = False) -> pd.DataFrame | None:
    global _FEATURES_CACHE
    if _FEATURES_CACHE is not None and not force_reload:
        return _FEATURES_CACHE
    if not FEATURES_PATH.exists():
        return None
    _FEATURES_CACHE = pd.read_csv(FEATURES_PATH)
    return _FEATURES_CACHE


@router.post("/reload")
async def reload_data() -> dict:
    """Reload features from disk (clears cache)."""
    global _FEATURES_CACHE, _MODEL_CACHE, _ERA_MODEL_CACHE
    _FEATURES_CACHE = None
    _MODEL_CACHE = None
    _ERA_MODEL_CACHE = {}
    load_features(force_reload=True)
    return {"status": "reloaded"}


def load_model_bundle() -> dict | None:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        return None
    _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def era_for_season(season: str) -> str:
    season_start = int(season[:4])
    return "2019-2024" if season_start >= 2019 else "2015-2018"


def load_era_model_bundle(era: str) -> dict | None:
    if era in _ERA_MODEL_CACHE:
        return _ERA_MODEL_CACHE[era]
    path = ERA_MODEL_PATHS.get(era)
    if path is None or not path.exists():
        return None
    _ERA_MODEL_CACHE[era] = joblib.load(path)
    return _ERA_MODEL_CACHE[era]


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


@router.get("/meta")
async def get_metadata() -> dict:
    features_df = load_features()
    if features_df is None or features_df.empty:
        raise HTTPException(status_code=503, detail="Feature data not available.")

    seasons = sorted(features_df["season"].dropna().unique().tolist())
    teams = []
    if RAW_TEAMS_PATH.exists():
        teams_df = pd.read_csv(RAW_TEAMS_PATH)
        teams = (
            teams_df[["id", "full_name", "abbreviation"]]
            .dropna()
            .sort_values("full_name")
            .to_dict(orient="records")
        )
    else:
        team_meta = (
            features_df[["team_id", "full_name", "abbreviation"]]
            .dropna(subset=["team_id", "full_name"])
            .drop_duplicates(subset=["team_id"])
            .sort_values("full_name")
        )
        teams = team_meta.rename(columns={"team_id": "id"}).to_dict(orient="records")

    return {"seasons": seasons, "teams": teams}


@router.get("/{team_id}/season/{season}")
async def get_team_prediction(team_id: int, season: str) -> dict:
    season = season.strip()
    if re.match(r"^\d{4}-\d{4}$", season):
        season = f"{season[:4]}-{season[-2:]}"
    if not re.match(r"^\d{4}-\d{2}$", season):
        raise HTTPException(status_code=400, detail="Season must look like 2023-24")

    features_df = load_features()
    if features_df is not None:
        match = features_df[
            (features_df["team_id"] == team_id) & (features_df["season"] == season)
        ]
        if not match.empty:
            row = match.iloc[0]
            era = era_for_season(season)
            model_bundle = load_era_model_bundle(era)
            if model_bundle is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Era model not trained. "
                        "Run scripts/train_model_era.py."
                    ),
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


@router.get("/compare/{team1_id}/{team2_id}/season/{season}")
async def compare_teams(team1_id: int, team2_id: int, season: str) -> dict:
    """Compare two teams head-to-head for a given season."""
    season = season.strip()
    if re.match(r"^\d{4}-\d{4}$", season):
        season = f"{season[:4]}-{season[-2:]}"
    if not re.match(r"^\d{4}-\d{2}$", season):
        raise HTTPException(status_code=400, detail="Season must look like 2023-24")

    features_df = load_features()
    if features_df is None or features_df.empty:
        raise HTTPException(status_code=503, detail="Feature data not available.")

    team1_match = features_df[
        (features_df["team_id"] == team1_id) & (features_df["season"] == season)
    ]
    team2_match = features_df[
        (features_df["team_id"] == team2_id) & (features_df["season"] == season)
    ]

    if team1_match.empty:
        raise HTTPException(
            status_code=404, detail=f"Team {team1_id} not found for season {season}"
        )
    if team2_match.empty:
        raise HTTPException(
            status_code=404, detail=f"Team {team2_id} not found for season {season}"
        )

    row1 = team1_match.iloc[0]
    row2 = team2_match.iloc[0]

    era = era_for_season(season)
    model_bundle = load_era_model_bundle(era)
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Era model not trained. Run scripts/train_model_era.py.",
        )

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    classes = [int(label) for label in model.classes_]
    label_map = {label: ROUND_LABELS.get(label, f"Round {label}") for label in classes}

    x1 = row1[feature_columns].to_frame().T
    x2 = row2[feature_columns].to_frame().T

    pred1 = int(model.predict(x1)[0])
    pred2 = int(model.predict(x2)[0])
    probs1 = model.predict_proba(x1)[0].tolist()
    probs2 = model.predict_proba(x2)[0].tolist()

    # Feature comparison
    comparison_features = [
        ("avg_age", "Avg age"),
        ("avg_seasons_in_league", "Avg seasons"),
        ("avg_playoff_games_prior", "Avg playoff games"),
        ("avg_playoff_wins_prior", "Avg playoff wins"),
        ("max_playoff_games_top3", "Star playoff games"),
        ("team_win_pct", "Win %"),
        ("net_rating", "Net rating"),
        ("seed", "Seed"),
        ("roster_continuity", "Roster continuity"),
    ]

    feature_comparison = []
    for key, label in comparison_features:
        if key in row1 and key in row2:
            val1 = row1[key]
            val2 = row2[key]
            if pd.notna(val1) and pd.notna(val2):
                # Determine which team has advantage
                # For seed, lower is better; for others, context-dependent
                if key == "seed":
                    advantage = "team1" if val1 < val2 else ("team2" if val2 < val1 else "tie")
                else:
                    advantage = "team1" if val1 > val2 else ("team2" if val2 > val1 else "tie")

                feature_comparison.append({
                    "key": key,
                    "label": label,
                    "team1_value": float(val1),
                    "team2_value": float(val2),
                    "advantage": advantage,
                })

    # Calculate expected round advantage
    expected_rounds_1 = sum(r * p for r, p in zip(classes, probs1))
    expected_rounds_2 = sum(r * p for r, p in zip(classes, probs2))

    return {
        "season": season,
        "team1": {
            "team_id": team1_id,
            "full_name": row1.get("full_name", f"Team {team1_id}"),
            "abbreviation": row1.get("abbreviation", ""),
            "prediction": {
                "round": pred1,
                "label": label_map.get(pred1, f"Round {pred1}"),
            },
            "probabilities": [
                {"round": r, "label": label_map[r], "probability": p}
                for r, p in zip(classes, probs1)
            ],
            "expected_round": float(expected_rounds_1),
        },
        "team2": {
            "team_id": team2_id,
            "full_name": row2.get("full_name", f"Team {team2_id}"),
            "abbreviation": row2.get("abbreviation", ""),
            "prediction": {
                "round": pred2,
                "label": label_map.get(pred2, f"Round {pred2}"),
            },
            "probabilities": [
                {"round": r, "label": label_map[r], "probability": p}
                for r, p in zip(classes, probs2)
            ],
            "expected_round": float(expected_rounds_2),
        },
        "feature_comparison": feature_comparison,
        "advantage": "team1" if expected_rounds_1 > expected_rounds_2 else (
            "team2" if expected_rounds_2 > expected_rounds_1 else "tie"
        ),
    }
