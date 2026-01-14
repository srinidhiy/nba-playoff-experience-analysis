"""Compute feature impact separately for each era."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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


def compute_impact(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
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

    return impact


def main() -> None:
    df = load_features()
    df = df[df[SOURCE_COLUMN] == "nba_api_series"].copy()
    df["season_start"] = df["season"].str[:4].astype(int)

    feature_columns = available_features(df)
    if not feature_columns:
        raise ValueError("No available features found.")

    outputs = {}
    for label, mask in {
        "2015-2018": df["season_start"].between(2015, 2018),
        "2019-2024": df["season_start"].between(2019, 2024),
    }.items():
        subset = df[mask].dropna(subset=feature_columns)
        if subset.empty:
            continue
        impact = compute_impact(subset, feature_columns)
        outputs[label] = impact.to_dict(orient="records")

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS_DIR / f"feature_impact_{label.replace('-', '_')}.csv"
        impact.to_csv(out_path, index=False)

    (ARTIFACTS_DIR / "feature_impact_era.json").write_text(
        json.dumps(
            {
                "trained_at": date.today().isoformat(),
                "features": feature_columns,
                "impact": outputs,
            },
            indent=2,
        )
    )

    for label, rows in outputs.items():
        top = pd.DataFrame(rows).head(10)
        top["mean_abs_coef"] = top["mean_abs_coef"].map(lambda x: f"{x:.3f}")
        print(f"\nTop features for {label}:")
        print(top[["feature", "mean_abs_coef"]].to_string(index=False))


if __name__ == "__main__":
    main()
