"""Train era-specific models for playoff rounds."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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


def train_era(df: pd.DataFrame, feature_columns: list[str], era: str) -> None:
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    model = build_pipeline(feature_columns)
    model.fit(x, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS_DIR / f"playoff_round_model_{era}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "trained_at": date.today().isoformat(),
            "era": era,
        },
        output,
    )
    print(f"Saved {output}")


def main() -> None:
    df = load_features()
    df = df[df[SOURCE_COLUMN] == "nba_api_series"].copy()
    df["season_start"] = df["season"].str[:4].astype(int)

    feature_columns = available_features(df)
    if not feature_columns:
        raise ValueError("No available features found.")

    eras = {
        "2015-2018": df["season_start"].between(2015, 2018),
        "2019-2024": df["season_start"].between(2019, 2024),
    }

    for era, mask in eras.items():
        subset = df[mask].dropna(subset=feature_columns)
        if subset.empty:
            print(f"No data for era {era}, skipping.")
            continue
        train_era(subset, feature_columns, era)


if __name__ == "__main__":
    main()
