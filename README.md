# AI-Enhanced Employee Attrition Prediction & Explainability System

An end-to-end machine-learning portfolio project that predicts whether a synthetic employee will leave within 90 days and combines predictive modeling, temporal validation, threshold analysis, explainability, subgroup diagnostics, model persistence, and a controlled AI interpretation layer.

> **Important:** This project uses synthetic workforce data only. Model outputs are predictive diagnostics and are not intended for real employment decisions.

---

## Project Overview

This project demonstrates a production-style data science workflow for an imbalanced employee attrition problem.

The system includes:

- chronological train / validation / test splitting
- leakage-aware feature controls
- logistic regression baseline
- Random Forest comparison
- Gradient Boosting model
- ROC-AUC and PR-AUC evaluation
- Brier score and calibration diagnostics
- threshold analysis
- confusion matrix analysis
- lift and cumulative gains
- permutation feature importance
- SHAP global and local explainability
- subgroup performance diagnostics
- model serialization and metadata
- deterministic reporting
- controlled OpenAI interpretation of verified analytical outputs
- automated pytest coverage

The machine-learning pipeline—not an LLM—produces all probabilities, predictions, metrics, thresholds, feature importance values, and SHAP explanations.

---

## Final Model

| Item | Result |
|---|---|
| Model | Gradient Boosting |
| Estimator | `GradientBoostingClassifier` |
| Prediction target | Attrition within 90 days |
| Features | 19 |
| Operating threshold | 0.125 |
| Threshold selection | Maximum validation F1 |
| Calibration | None |
| Hyperparameter tuning | None |
| Final test used for model selection | No |
| Data | Synthetic only |

The model and threshold were locked before the final test period was evaluated.

---

## Chronological Modeling Design

| Split | Period | Rows | Purpose |
|---|---|---:|---|
| Train | Jan 2025 – Dec 2025 | 1,997 | Fit preprocessing and models |
| Validation | Jan 2026 – Mar 2026 | 496 | Model comparison and threshold development |
| Final Test | Apr 2026 – Jun 2026 | 507 | Untouched temporal holdout |

The final test period was not used to choose the model or threshold.

---

## Final Untouched Test Results

| Metric | Result |
|---|---:|
| Target prevalence | 9.862% |
| ROC-AUC | 0.616 |
| PR-AUC | 0.199 |
| Brier score | 0.0872 |
| Precision | 0.161 |
| Recall | 0.380 |
| F1 | 0.226 |
| Predicted-positive rate | 23.274% |

### Final Test Confusion Matrix

| | Predicted No Attrition | Predicted Attrition |
|---|---:|---:|
| Actual No Attrition | 358 | 99 |
| Actual Attrition | 31 | 19 |

The holdout results are intentionally reported as observed rather than tuning the model after seeing them.

---

## Ranking Performance

The model retained meaningful ranking signal on the final temporal holdout.

| Risk Group | Attrition Captured | Lift |
|---|---:|---:|
| Top 5% | 10.0% | 1.95x |
| Top 10% | 20.0% | 1.99x |
| Top 20% | 34.0% | 1.69x |

This means higher-scored groups contained attrition cases at a higher rate than the overall final-test population.

---

## Model Development Results

### Baseline

The initial unweighted logistic regression improved substantially over a naive prevalence model:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Naive prevalence | 0.500 | 0.097 |
| Logistic regression | 0.674 | 0.171 |

At a default threshold of 0.50, the logistic model predicted no positive cases. This demonstrated why an arbitrary 0.50 threshold was inappropriate for this imbalanced problem.

### Candidate Models

| Model | Validation ROC-AUC | Validation PR-AUC | Brier |
|---|---:|---:|---:|
| Logistic Regression | 0.674 | 0.171 | 0.085 |
| Random Forest | 0.651 | 0.175 | 0.085 |
| Gradient Boosting | 0.659 | 0.195 | 0.085 |

Gradient Boosting was retained as the development candidate because it provided the strongest validation PR-AUC and the highest validation F1 at its selected threshold.

---

## Threshold Analysis

The operating threshold was selected using validation data only.

### Logistic Regression

- selected threshold: `0.100`
- validation F1: `0.253`
- recall: `0.625`
- flagged population: `38.1%`

### Gradient Boosting

- selected threshold: `0.125`
- validation F1: `0.277`
- recall: `0.396`
- flagged population: `17.9%`

The Gradient Boosting threshold was selected using a deterministic maximum-F1 rule.

This is a statistical portfolio-development rule, not a recommended real-world HR operating policy.

---

## Explainability

### Permutation Importance

Validation permutation importance was measured using PR-AUC / average precision.

Leading features included:

1. `manager_changes_12m`
2. `tenure_months`
3. `salary_change_pct_12m`
4. `commute_minutes`
5. `engagement_score`
6. `absences_90d`
7. `overtime_hours_90d`
8. `age`

Permutation importance measures predictive dependence. It does not establish causation.

### SHAP

SHAP explanations were generated for the Gradient Boosting model.

The implementation:

- calculates Tree SHAP values
- aggregates one-hot encoded values back to the original workforce features
- produces global feature importance
- produces local explanations for selected synthetic records
- distinguishes upward versus downward model-score contributions

SHAP values describe the fitted model's behavior and should not be interpreted as evidence that a feature causes attrition.

---

## Error Analysis

At the selected validation threshold of `0.125`:

| Classification | Count |
|---|---:|
| True Positive | 19 |
| True Negative | 378 |
| False Positive | 70 |
| False Negative | 29 |

The analysis identified both:

- threshold-boundary errors
- high-confidence errors

This shows that changing the threshold alone would not eliminate all model mistakes.

---

## Subgroup Diagnostics

Performance was examined across synthetic:

- department
- job family
- job level
- location
- employment type

Metrics included:

- subgroup prevalence
- predicted-positive rate
- ROC-AUC
- PR-AUC
- precision
- recall
- false-positive rate
- false-negative rate
- Brier score

Only groups meeting the configured minimum sample size were reported.

The results showed meaningful variation across some synthetic groups. These results are performance diagnostics only and do not establish that the model is fair or unfair.

---

## Controlled AI Interpretation

OpenAI is used only after all predictive calculations are completed.

The AI layer receives aggregate, verified outputs such as:

- final model metrics
- ranking / lift results
- permutation importance
- SHAP summary statistics
- subgroup diagnostics

It does **not** receive employee-level rows.

The AI is explicitly prohibited from:

- generating predictions
- changing probabilities
- changing thresholds
- calculating new metrics
- describing associations as causal
- recommending hiring, firing, discipline, promotion, compensation, or other employment actions

Generated AI artifacts are stored separately from deterministic model outputs.

---

## Portfolio Visualizations

Generated charts are available in:

```text
outputs/charts/