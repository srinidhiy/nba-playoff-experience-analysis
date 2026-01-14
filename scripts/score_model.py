"""Score a single team-season using the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nba_api"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a team-season.")
    parser.add_argument("--team-id", type=int, required=True)
    parser.add_argument("--season", required=True, help="Season label like 2023-24")
    args = parser.parse_args()

    model_path = ARTIFACTS_DIR / "playoff_round_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train the model with train_model.py first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    features_path = PROCESSED_DIR / "team_features.csv"
    if not features_path.exists():
        raise FileNotFoundError("Run feature_build.py before scoring.")

    df = pd.read_csv(features_path)
    match = df[(df["team_id"] == args.team_id) & (df["season"] == args.season)]
    if match.empty:
        raise ValueError("No matching team-season found in team_features.csv")

    row = match.iloc[0]
    x = row[feature_columns].to_frame().T
    prediction = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0].tolist()
    classes = [int(label) for label in model.classes_]

    team_name = None
    teams_path = RAW_DIR / "teams.csv"
    if teams_path.exists():
        teams_df = pd.read_csv(teams_path)
        match_team = teams_df[teams_df["id"] == args.team_id]
        if not match_team.empty:
            team_name = match_team.iloc[0].get("full_name")

    print(
        {
            "team_id": args.team_id,
            "team_name": team_name,
            "season": args.season,
            "predicted_round": prediction,
            "class_labels": classes,
            "probabilities": proba,
        }
    )


if __name__ == "__main__":
    main()
