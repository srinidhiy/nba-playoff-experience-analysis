"""Train and evaluate models separately for pre/post 2019 eras."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

FEATURES = [
    "avg_age",
    "avg_seasons_in_league",
    "avg_playoff_games_prior",
    "avg_playoff_wins_prior",
    "injury_games_missed",
    "roster_continuity",
    "team_win_pct",
    "seed",
    "playoff_rounds_prior_total",
    "net_rating",
    "pace",
    "off_rating",
    "def_rating",
]

TARGET_COLUMN = "playoff_round_reached"
SOURCE_COLUMN = "playoff_round_reached_source"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run feature_build.py before training.")
    return pd.read_csv(path)


def available_features(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURES if col in df.columns]


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), feature_columns)], remainder="drop"
    )
    classifier = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000,
        class_weight="balanced",
    )
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def evaluate(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, feature_columns: list[str]
) -> dict:
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_holdout = holdout_df[feature_columns]
    y_holdout = holdout_df[TARGET_COLUMN]

    model = build_pipeline(feature_columns)
    model.fit(x_train, y_train)
    pred = model.predict(x_holdout)
    report = classification_report(y_holdout, pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float((pred == y_holdout).mean()),
        "within_one_round": float((abs(pred - y_holdout) <= 1).mean()),
        "report": report,
    }


def main() -> None:
    df = load_features()
    df = df[df[SOURCE_COLUMN] == "nba_api_series"].copy()
    df["season_start"] = df["season"].str[:4].astype(int)

    feature_columns = available_features(df)
    if not feature_columns:
        raise ValueError("No available features found.")

    results = {}
    eras = {
        "2015-2018": (2015, 2018),
        "2019-2024": (2019, 2024),
    }
    for label, (start, end) in eras.items():
        subset = df[df["season_start"].between(start, end)].dropna(
            subset=feature_columns
        )
        train_df = subset[subset["season_start"] < end]
        holdout_df = subset[subset["season_start"] == end]
        if train_df.empty or holdout_df.empty:
            continue
        results[label] = {
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "holdout_season": int(end),
            "features": feature_columns,
            **evaluate(train_df, holdout_df, feature_columns),
        }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "trained_at": date.today().isoformat(),
        "results": results,
    }
    (ARTIFACTS_DIR / "model_compare_era.json").write_text(
        json.dumps(metrics, indent=2)
    )

    rows = []
    for era, result in results.items():
        rows.append(
            {
                "era": era,
                "holdout_season": result.get("holdout_season"),
                "accuracy": result["accuracy"],
                "within_one_round": result["within_one_round"],
                "train_rows": result.get("train_rows", 0),
                "holdout_rows": result.get("holdout_rows", 0),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["accuracy"] = table["accuracy"].map(lambda x: f"{x:.3f}")
        table["within_one_round"] = table["within_one_round"].map(lambda x: f"{x:.3f}")
        (ARTIFACTS_DIR / "model_compare_era.csv").write_text(table.to_csv(index=False))
        print("Era-specific model performance:")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
