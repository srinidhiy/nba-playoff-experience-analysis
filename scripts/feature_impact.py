"""Report feature impact from a trained multinomial logistic model."""

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

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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
    "net_rating",
    "pace",
    "off_rating",
    "def_rating",
    "era_post_2019",
]

TARGET_COLUMN = "playoff_round_reached"
SOURCE_COLUMN = "playoff_round_reached_source"


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
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    model = build_pipeline(feature_columns)
    model.fit(x, y)

    classifier = model.named_steps["clf"]
    coefs = pd.DataFrame(
        classifier.coef_,
        columns=feature_columns,
        index=[int(label) for label in classifier.classes_],
    )

    impact = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_coef": coefs.abs().mean(axis=0),
        }
    ).sort_values("mean_abs_coef", ascending=False)

    for label in coefs.index:
        impact[f"coef_round_{label}"] = coefs.loc[label].values

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    impact.to_csv(ARTIFACTS_DIR / "feature_impact.csv", index=False)
    (ARTIFACTS_DIR / "feature_impact.json").write_text(
        json.dumps(
            {
                "trained_at": date.today().isoformat(),
                "features": feature_columns,
                "impact": impact.to_dict(orient="records"),
            },
            indent=2,
        )
    )

    display = impact.copy()
    display["mean_abs_coef"] = display["mean_abs_coef"].map(lambda x: f"{x:.3f}")
    print("Feature impact (mean |coef| across rounds):")
    print(display[["feature", "mean_abs_coef"]].to_string(index=False))


if __name__ == "__main__":
    main()
