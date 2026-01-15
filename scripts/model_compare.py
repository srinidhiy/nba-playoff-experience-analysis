"""Compare experience-only vs confounders-only vs full models."""

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
    # Star player experience features
    "max_playoff_games_top3",
    "max_playoff_wins_top3",
    "sum_playoff_games_top3",
    "max_seasons_top3",
]

EXPERIENCE_FEATURES = [
    "avg_age",
    "avg_seasons_in_league",
    "avg_playoff_games_prior",
    "avg_playoff_wins_prior",
    "playoff_rounds_prior_total",
    "era_post_2019",
    # Star player experience features
    "max_playoff_games_top3",
    "max_playoff_wins_top3",
    "sum_playoff_games_top3",
    "max_seasons_top3",
]

CONFOUNDER_FEATURES = [
    "injury_games_missed",
    "roster_continuity",
    "team_win_pct",
    "seed",
    "net_rating",
    "pace",
    "off_rating",
    "def_rating",
    "era_post_2019",
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
    df["era_post_2019"] = (df["season_start"] >= 2019).astype(int)
    return df


def build_pipeline(
    feature_columns: list[str], model_type: str = "logistic"
) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), feature_columns)], remainder="drop"
    )
    if model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        classifier = LogisticRegression(
            multi_class="multinomial",
            max_iter=2000,
            class_weight="balanced",
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
    cm = confusion_matrix(y_holdout, preds, labels=sorted(y_holdout.unique()))
    return {
        "accuracy": accuracy,
        "within_one_round": within_one,
        "report": report,
        "confusion_matrix": cm.tolist(),
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

    feature_configs = {
        "experience_only": available_features(df, EXPERIENCE_FEATURES),
        "confounders_only": available_features(df, CONFOUNDER_FEATURES),
        "full": available_features(df, CANDIDATE_FEATURES),
    }

    model_types = ["logistic", "gradient_boosting"]

    results = {}
    rows = []
    for feature_name, features in feature_configs.items():
        if not features:
            continue
        filtered_train = train_df.dropna(subset=features)
        filtered_holdout = holdout_df.dropna(subset=features)
        if filtered_train.empty or filtered_holdout.empty:
            continue

        for model_type in model_types:
            config_name = f"{feature_name}_{model_type}"
            model = build_pipeline(features, model_type=model_type)
            outcome = evaluate_model(
                model,
                filtered_train[features],
                filtered_train[TARGET_COLUMN],
                filtered_holdout[features],
                filtered_holdout[TARGET_COLUMN],
            )
            results[config_name] = {
                "features": features,
                "model_type": model_type,
                **outcome,
            }
            rows.append(
                {
                    "model": config_name,
                    "model_type": model_type,
                    "feature_set": feature_name,
                    "num_features": len(features),
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

    table = pd.DataFrame(rows).sort_values(["feature_set", "model_type"])
    table_display = table.copy()
    table_display["accuracy"] = table_display["accuracy"].map(lambda x: f"{x:.3f}")
    table_display["within_one_round"] = table_display["within_one_round"].map(
        lambda x: f"{x:.3f}"
    )
    (ARTIFACTS_DIR / "model_comparison.csv").write_text(table.to_csv(index=False))

    print("Model comparison (last 10 seasons):")
    print(table_display.to_string(index=False))

    # Print summary comparison
    print("\n--- Summary: Best model per feature set ---")
    best_per_set = table.loc[table.groupby("feature_set")["accuracy"].idxmax()]
    for _, row in best_per_set.iterrows():
        print(
            f"  {row['feature_set']}: {row['model_type']} "
            f"(accuracy={row['accuracy']:.3f}, within_one={row['within_one_round']:.3f})"
        )


if __name__ == "__main__":
    main()
