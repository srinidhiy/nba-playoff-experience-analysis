"""Compute confusion matrix and per-round recall for the last 10 seasons."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "playoff_round_model.joblib"
EVAL_WINDOW_YEARS = 10

ROUND_LABELS = {
    0: "No Playoffs",
    1: "First Round",
    2: "Second Round",
    3: "Conference Finals",
    4: "Finals",
}


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run feature_build.py before error analysis.")
    return pd.read_csv(path)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model with train_model.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = load_features()
    df = df[df["playoff_round_reached_source"] == "nba_api_series"].copy()
    df["season_start"] = df["season"].str[:4].astype(int)

    latest_season = df["season_start"].max()
    eval_start = latest_season - (EVAL_WINDOW_YEARS - 1)
    eval_df = df[df["season_start"] >= eval_start]
    eval_df = eval_df.dropna(subset=feature_columns)

    if eval_df.empty:
        raise ValueError("No evaluation data found for last 10 seasons.")

    x = eval_df[feature_columns]
    y_true = eval_df["playoff_round_reached"].astype(int)
    y_pred = model.predict(x).astype(int)

    labels = [int(label) for label in model.classes_]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(matrix, index=labels, columns=labels)
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"

    recall = {}
    for label in labels:
        total = int((y_true == label).sum())
        correct = int(((y_true == label) & (y_pred == label)).sum())
        recall[label] = {
            "label": ROUND_LABELS.get(label, f"Round {label}"),
            "recall": 0.0 if total == 0 else correct / total,
            "support": total,
        }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(ARTIFACTS_DIR / "confusion_matrix.csv")
    pd.DataFrame.from_dict(recall, orient="index").to_csv(
        ARTIFACTS_DIR / "round_recall.csv", index=False
    )

    (ARTIFACTS_DIR / "error_analysis.json").write_text(
        json.dumps(
            {
                "eval_start": int(eval_start),
                "eval_window_years": int(EVAL_WINDOW_YEARS),
                "confusion_matrix": cm_df.to_dict(),
                "round_recall": recall,
            },
            indent=2,
        )
    )

    print("Confusion matrix saved to artifacts/confusion_matrix.csv")
    print("Round recall saved to artifacts/round_recall.csv")


if __name__ == "__main__":
    main()
