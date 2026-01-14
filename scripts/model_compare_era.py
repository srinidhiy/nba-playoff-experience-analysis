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


def evaluate(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    x = df[feature_columns]
    y = df[TARGET_COLUMN]
    model = build_pipeline(feature_columns)
    model.fit(x, y)
    pred = model.predict(x)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float((pred == y).mean()),
        "within_one_round": float((abs(pred - y) <= 1).mean()),
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
    for label, mask in {
        "2015-2018": df["season_start"].between(2015, 2018),
        "2019-2024": df["season_start"].between(2019, 2024),
    }.items():
        subset = df[mask].dropna(subset=feature_columns)
        if subset.empty:
            continue
        results[label] = {
            "rows": int(len(subset)),
            "features": feature_columns,
            **evaluate(subset, feature_columns),
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
                "accuracy": result["accuracy"],
                "within_one_round": result["within_one_round"],
                "rows": result["rows"],
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
