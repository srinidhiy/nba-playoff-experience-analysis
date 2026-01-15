"""Train playoff outcome models using team features."""

from __future__ import annotations

import os
import json
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
    "net_rating",
    "pace",
    "seed",
    "era_post_2019",
    # Star player experience features
    "max_playoff_games_top3",
    "max_playoff_wins_top3",
    "sum_playoff_games_top3",
    "max_seasons_top3",
]

TARGET_COLUMN = "playoff_round_reached"
SOURCE_COLUMN = "playoff_round_reached_source"
EVAL_WINDOW_YEARS = 10


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run feature_build.py before training the model.")
    return pd.read_csv(path)


def available_features(df: pd.DataFrame) -> list[str]:
    return [col for col in CANDIDATE_FEATURES if col in df.columns]


def prepare_dataset(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["season_start"] = df["season"].str[:4].astype(int)
    df["era_post_2019"] = (df["season_start"] >= 2019).astype(int)
    df = df[df[feature_columns].notna().all(axis=1)]
    df = df[df[TARGET_COLUMN].notna()]
    df = df[df[SOURCE_COLUMN] == "nba_api_series"]
    return df


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


def main() -> None:
    df = load_features()
    feature_columns = available_features(df)
    if not feature_columns:
        raise ValueError("No available features found in team_features.csv")

    df = prepare_dataset(df, feature_columns)

    latest_season = df["season_start"].max()
    eval_start = latest_season - (EVAL_WINDOW_YEARS - 1)
    train_df = df[df["season_start"] < eval_start]
    holdout_df = df[df["season_start"] >= eval_start]

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    model = build_pipeline(feature_columns)
    model.fit(x_train, y_train)

    report = {}
    season_reports = {}
    season_summary = {}
    if not holdout_df.empty:
        x_holdout = holdout_df[feature_columns]
        y_holdout = holdout_df[TARGET_COLUMN]
        holdout_pred = model.predict(x_holdout)
        report = classification_report(
            y_holdout,
            holdout_pred,
            output_dict=True,
            zero_division=0,
        )

        for season, season_df in holdout_df.groupby("season_start"):
            x_season = season_df[feature_columns]
            y_season = season_df[TARGET_COLUMN]
            season_pred = model.predict(x_season)
            season_reports[int(season)] = classification_report(
                y_season,
                season_pred,
                output_dict=True,
                zero_division=0,
            )
            season_summary[int(season)] = {
                "accuracy": float((season_pred == y_season).mean()),
                "within_one_round": float((abs(season_pred - y_season) <= 1).mean()),
                "support": int(len(y_season)),
            }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "trained_at": date.today().isoformat(),
        },
        ARTIFACTS_DIR / "playoff_round_model.joblib",
    )

    metrics = {
        "latest_season": int(latest_season),
        "eval_window_years": int(EVAL_WINDOW_YEARS),
        "eval_start": int(eval_start),
        "rows_train": int(len(train_df)),
        "rows_holdout": int(len(holdout_df)),
        "features": feature_columns,
        "classification_report": report,
        "season_reports": season_reports,
        "season_summary": season_summary,
    }
    (ARTIFACTS_DIR / "playoff_round_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    if season_summary:
        summary_rows = [
            {"season_start": season, **values} for season, values in season_summary.items()
        ]
        summary_df = pd.DataFrame(summary_rows).sort_values("season_start")
        summary_df["accuracy"] = summary_df["accuracy"].map(lambda x: f"{x:.3f}")
        summary_df["within_one_round"] = summary_df["within_one_round"].map(
            lambda x: f"{x:.3f}"
        )
        (ARTIFACTS_DIR / "season_accuracy.csv").write_text(
            summary_df.to_csv(index=False)
        )
        print("\nPer-season accuracy:")
        print(summary_df.to_string(index=False))

    print("Model trained. Metrics saved to artifacts/playoff_round_metrics.json")


if __name__ == "__main__":
    main()
