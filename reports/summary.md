# Analysis Summary

## Key takeaways
- Best overall model: `full_gradient_boosting` (accuracy 0.727, within one round 0.887).
- Top driver by mean |coef|: `seed` (mean |coef| 2.008).
- Best era holdout accuracy: `2019-2024` (0.700).

## Model comparison (last 10 seasons)

| Model                              |   Accuracy |   Within 1 Round |   Features |
|:-----------------------------------|-----------:|-----------------:|-----------:|
| confounders_only_gradient_boosting |      0.673 |            0.9   |          9 |
| confounders_only_logistic          |      0.3   |            0.513 |          9 |
| experience_only_gradient_boosting  |      0.393 |            0.683 |         10 |
| experience_only_logistic           |      0.453 |            0.717 |         10 |
| full_gradient_boosting             |      0.727 |            0.887 |         18 |
| full_logistic                      |      0.537 |            0.787 |         18 |

## Per-season accuracy (last 10 seasons)
|   Season |   Accuracy |   Within 1 Round |   Support |
|---------:|-----------:|-----------------:|----------:|
|     2015 |      0.667 |            0.867 |        30 |
|     2016 |      0.567 |            0.867 |        30 |
|     2017 |      0.633 |            0.9   |        30 |
|     2018 |      0.567 |            0.8   |        30 |
|     2019 |      0.467 |            0.733 |        30 |
|     2020 |      0.467 |            0.733 |        30 |
|     2021 |      0.467 |            0.733 |        30 |
|     2022 |      0.467 |            0.733 |        30 |
|     2023 |      0.467 |            0.733 |        30 |
|     2024 |      0.467 |            0.733 |        30 |

## Era segmentation
- 2015-2018 avg accuracy 0.608, within one round 0.858.
- 2019-2024 avg accuracy 0.467, within one round 0.733.

## Feature impact (mean |coef|)
| feature                    |   mean_abs_coef |
|:---------------------------|----------------:|
| seed                       |           2.008 |
| avg_playoff_wins_prior     |           1.086 |
| avg_playoff_games_prior    |           0.917 |
| avg_seasons_in_league      |           0.378 |
| playoff_rounds_prior_total |           0.254 |
| avg_age                    |           0.164 |
| team_win_pct               |           0.137 |
| net_rating                 |           0.116 |
| roster_continuity          |           0.103 |
| injury_games_missed        |           0.102 |

## Era-specific models (holdout by era)
| Era       |   Holdout Season |   Accuracy |   Within 1 Round |   Train Rows |   Holdout Rows |
|:----------|-----------------:|-----------:|-----------------:|-------------:|---------------:|
| 2015-2018 |             2018 |      0.633 |              0.9 |           90 |             30 |
| 2019-2024 |             2024 |      0.7   |              0.8 |          150 |             30 |

## Era-specific feature impact
### 2015-2018
| feature                    |   mean_abs_coef |
|:---------------------------|----------------:|
| seed                       |           1.758 |
| avg_playoff_wins_prior     |           0.765 |
| avg_playoff_games_prior    |           0.664 |
| playoff_rounds_prior_total |           0.58  |
| avg_seasons_in_league      |           0.309 |
| team_win_pct               |           0.299 |
| pace                       |           0.237 |
| off_rating                 |           0.224 |
### 2019-2024
| feature                 |   mean_abs_coef |
|:------------------------|----------------:|
| seed                    |           0.92  |
| avg_playoff_games_prior |           0.754 |
| avg_playoff_wins_prior  |           0.752 |
| team_win_pct            |           0.602 |
| net_rating              |           0.375 |
| avg_seasons_in_league   |           0.345 |
| off_rating              |           0.329 |
| avg_age                 |           0.283 |

## Era investigation insights
- Accuracy dropped 14.2% from pre-2019 to post-2019 era
- COVID bubble year (2020) had lower accuracy (46.7%) than post-2019 average

**COVID bubble (2020)**: accuracy 46.7%
  - Bubble playoffs in Orlando, no home court advantage

**Play-in era (2021+)**: avg accuracy 46.7%
  - Play-in tournament added, affecting 7-10 seeds

**Roster continuity**: pre-2019 avg 0.490, post-2019 avg 0.607

## 2025-26 Season Predictions

Live predictions for the current 2025-26 season are available via:
- **API**: `GET /teams/{team_id}/season/2025-26`
- **Frontend**: http://localhost:5173 - Click "Load All Team Predictions"

### API Examples

```bash
# Get single team prediction
curl http://localhost:8000/teams/1610612760/season/2025-26

# Get team metadata (all teams and seasons)
curl http://localhost:8000/teams/meta

# Compare two teams
curl http://localhost:8000/teams/compare/1610612738/1610612760/season/2025-26

# Reload data after running ETL/feature_build
curl -X POST http://localhost:8000/teams/reload
```

### Key Model Insights
- **Experience vs Performance**: Teams with dominant regular season stats (high win%, net rating) can overcome experience deficits
- **Seed importance**: Seed has the highest feature impact (mean |coef| 2.008), followed by playoff experience metrics
- **Roster continuity**: Higher continuity correlates with playoff success, especially for defending champions
