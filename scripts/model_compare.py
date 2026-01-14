"""Compare experience-only vs confounders-only vs full models."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

CANDIDATE_FEATURES = [
    "avg_age",
    "avg_seasons_in_league",
    "avg_playoff_games_prior",
    "avg_playoff_wins_prior",
    "injury_games_missed",
    "roster_continuity",
    "team_win_pct",
    "seed",
    "playoff_rounds_prior_total",
]

EXPERIENCE_FEATURES = [
    "avg_age",
    "avg_seasons_in_league",
    "avg_playoff_games_prior",
    "avg_playoff_wins_prior",
    "playoff_rounds_prior_total",
]

CONFOUNDER_FEATURES = [
    "injury_games_missed",
    "roster_continuity",
    "team_win_pct",
    "seed",
]

TARGET_COLUMN = "playoff_round_reached"
SOURCE_COLUMN = "playoff_round_reached_source"
EVAL_WINDOW_YEARS = 10


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run feature_build.py before training the model.")
    return pd.read_csv(path)


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df[TARGET_COLUMN].notna()]
    df = df[df[SOURCE_COLUMN] == "nba_api_series"]
    df["season_start"] = df["season"].str[:4].astype(int)
    return df


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), feature_columns)], remainder="drop"
    )
    classifier = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000,
    )
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def evaluate_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> dict:
    model.fit(x_train, y_train)
    preds = model.predict(x_holdout)
    accuracy = float((preds == y_holdout).mean())
    within_one = float((abs(preds - y_holdout) <= 1).mean())
    report = classification_report(y_holdout, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy,
        "within_one_round": within_one,
        "report": report,
    }


def main() -> None:
    df = load_features()
    df = prepare_dataset(df)

    latest_season = df["season_start"].max()
    eval_start = latest_season - (EVAL_WINDOW_YEARS - 1)
    train_df = df[df["season_start"] < eval_start]
    holdout_df = df[df["season_start"] >= eval_start]

    if holdout_df.empty or train_df.empty:
        raise ValueError("Not enough data to run comparison.")

    configs = {
        "experience_only": available_features(df, EXPERIENCE_FEATURES),
        "confounders_only": available_features(df, CONFOUNDER_FEATURES),
        "full": available_features(df, CANDIDATE_FEATURES),
    }

    results = {}
    rows = []
    for name, features in configs.items():
        if not features:
            continue
        filtered_train = train_df.dropna(subset=features)
        filtered_holdout = holdout_df.dropna(subset=features)
        if filtered_train.empty or filtered_holdout.empty:
            continue
        model = build_pipeline(features)
        outcome = evaluate_model(
            model,
            filtered_train[features],
            filtered_train[TARGET_COLUMN],
            filtered_holdout[features],
            filtered_holdout[TARGET_COLUMN],
        )
        results[name] = {
            "features": features,
            **outcome,
        }
        rows.append(
            {
                "model": name,
                "features": len(features),
                "accuracy": outcome["accuracy"],
                "within_one_round": outcome["within_one_round"],
            }
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "trained_at": date.today().isoformat(),
        "eval_start": int(eval_start),
        "eval_window_years": int(EVAL_WINDOW_YEARS),
        "results": results,
    }
    (ARTIFACTS_DIR / "model_comparison.json").write_text(json.dumps(metrics, indent=2))

    table = pd.DataFrame(rows).sort_values("model")
    table["accuracy"] = table["accuracy"].map(lambda x: f"{x:.3f}")
    table["within_one_round"] = table["within_one_round"].map(lambda x: f"{x:.3f}")
    (ARTIFACTS_DIR / "model_comparison.csv").write_text(table.to_csv(index=False))

    print("Model comparison (last 10 seasons):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
