# Project 9 — AI-Enhanced Employee Attrition Prediction & Explainability System

## Final Model Report

> This portfolio project uses synthetic workforce data only. Model outputs are predictive diagnostics, not causal conclusions or employment recommendations.

## 1. Final Model Specification

- Model: **Gradient Boosting**
- Estimator: `GradientBoostingClassifier`
- Prediction horizon: **90 days**
- Target: `attrition_within_90_days`
- Model features: **19**
- Operating threshold: **0.125**
- Threshold selected using: **validation data**
- Calibration method: **None**
- Random seed: **42**
- Hyperparameter tuning performed: **False**
- Final test used for model selection: **False**

## 2. Chronological Modeling Design

| Split | Date range | Rows | Purpose |
|---|---|---:|---|
| Train | 2025-01-01 00:00:00 to 2025-12-01 00:00:00 | 1,997 | Model fitting |
| Validation | 2026-01-01 00:00:00 to 2026-03-01 00:00:00 | 496 | Model/threshold development |
| Final test | 2026-04-01 00:00:00 to 2026-06-01 00:00:00 | 507 | Untouched final evaluation |

The final test period was not used to select the model or operating threshold.

## 3. Final Untouched Test Performance

| Metric | Final test result |
|---|---:|
| Target prevalence | 9.9% |
| ROC-AUC | 0.616 |
| PR-AUC | 0.199 |
| Brier score | 0.0872 |
| Precision | 0.161 |
| Recall | 0.380 |
| F1 | 0.226 |
| Predicted-positive rate | 23.3% |

## 4. Final Test Confusion Matrix

| | Predicted No Attrition | Predicted Attrition |
|---|---:|---:|
| Actual No Attrition | 358 | 99 |
| Actual Attrition | 31 | 19 |

## 5. Final Test Ranking Performance

| Risk group | Population | Attrition captured | Observed attrition | Lift |
|---|---:|---:|---:|---:|
| Top 5% | 5.1% | 10.0% | 19.2% | 1.95x |
| Top 10% | 10.1% | 20.0% | 19.6% | 1.99x |
| Top 20% | 20.1% | 34.0% | 16.7% | 1.69x |

## 6. Validation Error Analysis

- True Positive: **19**
- True Negative: **378**
- False Positive: **70**
- False Negative: **29**

The error analysis contains both threshold-boundary errors and high-confidence errors. Therefore, changing the operating threshold alone would not eliminate all model mistakes.

## 7. Permutation Feature Importance

Validation permutation importance was measured using **PR-AUC / average precision**.

| Rank | Feature | Mean PR-AUC decrease |
|---:|---|---:|
| 1 | `manager_changes_12m` | 0.0473 |
| 2 | `tenure_months` | 0.0331 |
| 3 | `salary_change_pct_12m` | 0.0126 |
| 4 | `commute_minutes` | 0.0098 |
| 5 | `engagement_score` | 0.0077 |
| 6 | `absences_90d` | 0.0071 |
| 7 | `overtime_hours_90d` | 0.0064 |
| 8 | `age` | 0.0063 |
| 9 | `job_level` | 0.0023 |
| 10 | `department` | 0.0018 |

Permutation importance describes predictive dependence. It does not establish causal effects.

## 8. SHAP Explainability

| Rank | Feature | Mean absolute SHAP | Importance share |
|---:|---|---:|---:|
| 1 | `manager_changes_12m` | 0.2290 | 13.7% |
| 2 | `engagement_score` | 0.2098 | 12.6% |
| 3 | `tenure_months` | 0.1550 | 9.3% |
| 4 | `performance_score` | 0.1352 | 8.1% |
| 5 | `absences_90d` | 0.1226 | 7.3% |
| 6 | `monthly_salary_annualized` | 0.1116 | 6.7% |
| 7 | `overtime_hours_90d` | 0.1089 | 6.5% |
| 8 | `salary_change_pct_12m` | 0.1020 | 6.1% |
| 9 | `age` | 0.0956 | 5.7% |
| 10 | `commute_minutes` | 0.0870 | 5.2% |

SHAP values explain how the fitted model generated its scores. Positive and negative SHAP values describe model contributions and should not be interpreted as causal relationships.

## 9. Synthetic Subgroup Diagnostics

Subgroup metrics were calculated only for groups meeting the configured minimum sample-size requirement.

### department

| Group | N | Prevalence | Flag rate | ROC-AUC | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 61 | 8.2% | 18.0% | 0.786 | 0.800 | 0.125 | 0.200 |
| Finance | 57 | 12.3% | 17.5% | 0.591 | 0.286 | 0.160 | 0.714 |
| Operations | 94 | 10.6% | 17.0% | 0.638 | 0.300 | 0.155 | 0.700 |
| People | 68 | 7.4% | 17.6% | 0.629 | 0.000 | 0.190 | 1.000 |
| Programs | 157 | 7.6% | 18.5% | 0.728 | 0.667 | 0.145 | 0.333 |
| Research | 59 | 15.3% | 18.6% | 0.611 | 0.222 | 0.180 | 0.778 |

### job_family

| Group | N | Prevalence | Flag rate | ROC-AUC | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Data | 79 | 12.7% | 34.2% | 0.549 | 0.500 | 0.319 | 0.500 |
| Finance | 82 | 7.3% | 14.6% | 0.739 | 0.500 | 0.118 | 0.500 |
| Fundraising | 98 | 11.2% | 15.3% | 0.603 | 0.273 | 0.138 | 0.727 |
| HR | 84 | 7.1% | 17.9% | 0.564 | 0.333 | 0.167 | 0.667 |
| Operations | 74 | 12.2% | 14.9% | 0.841 | 0.556 | 0.092 | 0.444 |
| Program Delivery | 79 | 7.6% | 11.4% | 0.717 | 0.167 | 0.110 | 0.833 |

### job_level

| Group | N | Prevalence | Flag rate | ROC-AUC | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 170 | 12.9% | 14.7% | 0.586 | 0.182 | 0.142 | 0.818 |
| L2 | 167 | 7.2% | 23.4% | 0.666 | 0.667 | 0.200 | 0.333 |
| L3 | 115 | 9.6% | 15.7% | 0.814 | 0.455 | 0.125 | 0.545 |

### location

| Group | N | Prevalence | Flag rate | ROC-AUC | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chicago-Downtown | 179 | 10.1% | 19.0% | 0.616 | 0.222 | 0.186 | 0.778 |
| Chicago-North | 98 | 10.2% | 16.3% | 0.768 | 0.600 | 0.114 | 0.400 |
| Chicago-South | 92 | 10.9% | 16.3% | 0.728 | 0.400 | 0.134 | 0.600 |
| Remote-US | 127 | 7.9% | 18.9% | 0.559 | 0.500 | 0.162 | 0.500 |

### employment_type

| Group | N | Prevalence | Flag rate | ROC-AUC | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-Time | 418 | 10.0% | 18.9% | 0.678 | 0.429 | 0.162 | 0.571 |
| Part-Time | 78 | 7.7% | 12.8% | 0.500 | 0.167 | 0.125 | 0.833 |

Observed differences across synthetic subgroups are diagnostics of model performance. They do not establish that the model is fair or unfair.

## 10. Reproducibility

- Serialized pipeline: `artifacts\final_model_pipeline.joblib`
- Model SHA-256: `12f9c529ecaffe70f8c1ad401ab398e25b135d8a5ecab6edd71584d65a126755`
- Dataset: `data/raw/employee_attrition_snapshots_synthetic.csv`
- Python: `3.10.11`
- scikit-learn: `1.7.2`
- pandas: `1.5.3`
- NumPy: `1.26.4`

## 11. Limitations

- The dataset is synthetic and does not represent a real employer population.
- Predictive associations do not establish causality.
- The operating threshold was selected statistically using validation F1 rather than real operational costs or intervention capacity.
- Some validation subgroup metrics vary materially across synthetic groups.
- Final holdout performance was lower than validation on some metrics, illustrating temporal generalization risk.
- This system should not be used to make adverse employment decisions.

## 12. AI Boundary

The predictive model—not an LLM—produces the attrition probability. All metrics, thresholds, ranking diagnostics, SHAP values, and subgroup calculations are produced by deterministic Python/ML code. Any later AI component may only summarize these verified analytical outputs.
