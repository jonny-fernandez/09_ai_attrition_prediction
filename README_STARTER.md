# AI Attrition Prediction & Explainability

Project 9 of the advanced AI/data-science portfolio.

## Goal

Build a leakage-aware supervised classification system for synthetic employee attrition risk, with calibration, threshold analysis, explainability, subgroup diagnostics, and an optional AI interpretation layer.

## Starter Pack V2

The repository begins with a richer synthetic dataset, documented configuration, data dictionary, validation rules, a conservative baseline, and automated tests. The starter is not a pre-completed project; it is a stable foundation for step-by-step development.

## Dataset

One row represents one synthetic employee as of a documented snapshot date. The target is future attrition within 90 days.

## Engineering principles

- Chronological splitting before advanced modeling.
- Feature policy explicitly excludes identifiers and post-outcome leakage.
- Deterministic metrics first; AI only interprets verified outputs later.
- Prediction is not causation and must not be framed as an employment decision.

## Planned outputs

- `model_metrics.json`
- `model_comparison.csv`
- `threshold_analysis.csv`
- `subgroup_metrics.csv`
- `feature_importance.csv`
- `shap_summary.csv`
- `error_analysis.csv`
- `model_report.md`

## First commands

```powershell
python -m pytest tests -q
python -m src.inspect_data
```

See `docs/Build_Guide.docx` before advancing beyond the inspection checkpoint.
