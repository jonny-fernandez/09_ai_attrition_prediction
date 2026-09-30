from pathlib import Path


def format_percentage(value):
    """
    Convert a proportion into a percentage string.
    """

    return f"{float(value):.1%}"


def format_metric(value, decimals=3):
    """
    Format a numeric model metric.
    """

    return f"{float(value):.{decimals}f}"


def generate_model_report(
    *,
    metadata,
    model_lock,
    final_metrics,
    top_risk,
    permutation_importance,
    shap_summary,
    subgroup_metrics,
    error_analysis,
):
    """
    Generate a deterministic Markdown model report.

    All analytical values supplied to this function must already
    have been calculated by the Python modeling pipeline.

    This function performs reporting only.

    It does NOT:
    - train a model,
    - alter predictions,
    - select a threshold,
    - recalculate the final test results,
    - call an LLM,
    - or make employment recommendations.
    """

    lines = []

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------
    lines.append(
        "# Project 9 — AI-Enhanced Employee Attrition "
        "Prediction & Explainability System"
    )

    lines.append("")

    lines.append(
        "## Final Model Report"
    )

    lines.append("")

    lines.append(
        "> This portfolio project uses synthetic workforce "
        "data only. Model outputs are predictive diagnostics, "
        "not causal conclusions or employment recommendations."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Model specification
    # ---------------------------------------------------------
    lines.append(
        "## 1. Final Model Specification"
    )

    lines.append("")

    lines.append(
        f"- Model: **{metadata['model_label']}**"
    )

    lines.append(
        f"- Estimator: `{metadata['model_name']}`"
    )

    lines.append(
        f"- Prediction horizon: "
        f"**{metadata['prediction_horizon_days']} days**"
    )

    lines.append(
        f"- Target: `{metadata['target']}`"
    )

    lines.append(
        f"- Model features: "
        f"**{metadata['feature_count']}**"
    )

    lines.append(
        f"- Operating threshold: "
        f"**{metadata['operating_threshold']:.3f}**"
    )

    lines.append(
        "- Threshold selected using: "
        f"**{model_lock['threshold_selected_on']} data**"
    )

    lines.append(
        f"- Calibration method: "
        f"**{model_lock['calibration_method'] or 'None'}**"
    )

    lines.append(
        f"- Random seed: "
        f"**{metadata['random_seed']}**"
    )

    lines.append(
        "- Hyperparameter tuning performed: "
        f"**{model_lock['hyperparameter_tuning_performed']}**"
    )

    lines.append(
        "- Final test used for model selection: "
        f"**{metadata['test_used_for_model_selection']}**"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Temporal design
    # ---------------------------------------------------------
    lines.append(
        "## 2. Chronological Modeling Design"
    )

    lines.append("")

    train = metadata[
        "splits"
    ][
        "train"
    ]

    validation = metadata[
        "splits"
    ][
        "validation"
    ]

    test = metadata[
        "splits"
    ][
        "test"
    ]

    lines.append(
        "| Split | Date range | Rows | Purpose |"
    )

    lines.append(
        "|---|---|---:|---|"
    )

    lines.append(
        f"| Train | {train['start']} to {train['end']} | "
        f"{train['rows']:,} | Model fitting |"
    )

    lines.append(
        f"| Validation | {validation['start']} to "
        f"{validation['end']} | "
        f"{validation['rows']:,} | Model/threshold development |"
    )

    lines.append(
        f"| Final test | {test['start']} to {test['end']} | "
        f"{test['rows']:,} | Untouched final evaluation |"
    )

    lines.append("")

    lines.append(
        "The final test period was not used to select the model "
        "or operating threshold."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Final test metrics
    # ---------------------------------------------------------
    lines.append(
        "## 3. Final Untouched Test Performance"
    )

    lines.append("")

    lines.append(
        "| Metric | Final test result |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Target prevalence | "
        f"{format_percentage(final_metrics['target_prevalence'])} |"
    )

    lines.append(
        f"| ROC-AUC | "
        f"{format_metric(final_metrics['roc_auc'])} |"
    )

    lines.append(
        f"| PR-AUC | "
        f"{format_metric(final_metrics['pr_auc'])} |"
    )

    lines.append(
        f"| Brier score | "
        f"{format_metric(final_metrics['brier_score'], 4)} |"
    )

    lines.append(
        f"| Precision | "
        f"{format_metric(final_metrics['precision'])} |"
    )

    lines.append(
        f"| Recall | "
        f"{format_metric(final_metrics['recall'])} |"
    )

    lines.append(
        f"| F1 | "
        f"{format_metric(final_metrics['f1'])} |"
    )

    lines.append(
        f"| Predicted-positive rate | "
        f"{format_percentage(final_metrics['predicted_positive_rate'])} |"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------
    lines.append(
        "## 4. Final Test Confusion Matrix"
    )

    lines.append("")

    lines.append(
        "| | Predicted No Attrition | Predicted Attrition |"
    )

    lines.append(
        "|---|---:|---:|"
    )

    lines.append(
        f"| Actual No Attrition | "
        f"{final_metrics['true_negative']:,} | "
        f"{final_metrics['false_positive']:,} |"
    )

    lines.append(
        f"| Actual Attrition | "
        f"{final_metrics['false_negative']:,} | "
        f"{final_metrics['true_positive']:,} |"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Ranking / lift
    # ---------------------------------------------------------
    lines.append(
        "## 5. Final Test Ranking Performance"
    )

    lines.append("")

    lines.append(
        "| Risk group | Population | Attrition captured | "
        "Observed attrition | Lift |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for _, row in top_risk.iterrows():
        lines.append(
            f"| Top {row['requested_top_percentage']:.0%} | "
            f"{format_percentage(row['actual_population_rate'])} | "
            f"{format_percentage(row['attrition_capture_rate'])} | "
            f"{format_percentage(row['observed_attrition_rate'])} | "
            f"{row['lift']:.2f}x |"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Validation error analysis
    # ---------------------------------------------------------
    lines.append(
        "## 6. Validation Error Analysis"
    )

    lines.append("")

    category_counts = (
        error_analysis[
            "error_category"
        ]
        .value_counts()
    )

    for category in [
        "TRUE_POSITIVE",
        "TRUE_NEGATIVE",
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
    ]:
        lines.append(
            f"- {category.replace('_', ' ').title()}: "
            f"**{int(category_counts.get(category, 0)):,}**"
        )

    lines.append("")

    lines.append(
        "The error analysis contains both threshold-boundary "
        "errors and high-confidence errors. Therefore, changing "
        "the operating threshold alone would not eliminate all "
        "model mistakes."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Permutation importance
    # ---------------------------------------------------------
    lines.append(
        "## 7. Permutation Feature Importance"
    )

    lines.append("")

    lines.append(
        "Validation permutation importance was measured using "
        "**PR-AUC / average precision**."
    )

    lines.append("")

    lines.append(
        "| Rank | Feature | Mean PR-AUC decrease |"
    )

    lines.append(
        "|---:|---|---:|"
    )

    for _, row in (
        permutation_importance
        .head(
            10
        )
        .iterrows()
    ):
        lines.append(
            f"| {int(row['rank'])} | "
            f"`{row['feature']}` | "
            f"{row['importance_mean']:.4f} |"
        )

    lines.append("")

    lines.append(
        "Permutation importance describes predictive dependence. "
        "It does not establish causal effects."
    )

    lines.append("")

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------
    lines.append(
        "## 8. SHAP Explainability"
    )

    lines.append("")

    lines.append(
        "| Rank | Feature | Mean absolute SHAP | "
        "Importance share |"
    )

    lines.append(
        "|---:|---|---:|---:|"
    )

    for _, row in (
        shap_summary
        .head(
            10
        )
        .iterrows()
    ):
        lines.append(
            f"| {int(row['rank'])} | "
            f"`{row['feature']}` | "
            f"{row['mean_abs_shap']:.4f} | "
            f"{row['importance_share']:.1%} |"
        )

    lines.append("")

    lines.append(
        "SHAP values explain how the fitted model generated its "
        "scores. Positive and negative SHAP values describe model "
        "contributions and should not be interpreted as causal "
        "relationships."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Subgroups
    # ---------------------------------------------------------
    lines.append(
        "## 9. Synthetic Subgroup Diagnostics"
    )

    lines.append("")

    lines.append(
        "Subgroup metrics were calculated only for groups meeting "
        "the configured minimum sample-size requirement."
    )

    lines.append("")

    for group_column in (
        subgroup_metrics[
            "group_column"
        ]
        .drop_duplicates()
        .tolist()
    ):
        group = subgroup_metrics[
            subgroup_metrics[
                "group_column"
            ]
            == group_column
        ]

        lines.append(
            f"### {group_column}"
        )

        lines.append("")

        lines.append(
            "| Group | N | Prevalence | Flag rate | ROC-AUC | "
            "Recall | FPR | FNR |"
        )

        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )

        for _, row in group.iterrows():
            lines.append(
                f"| {row['group_value']} | "
                f"{int(row['record_count'])} | "
                f"{row['attrition_prevalence']:.1%} | "
                f"{row['predicted_positive_rate']:.1%} | "
                f"{row['roc_auc']:.3f} | "
                f"{row['recall']:.3f} | "
                f"{row['false_positive_rate']:.3f} | "
                f"{row['false_negative_rate']:.3f} |"
            )

        lines.append("")

    lines.append(
        "Observed differences across synthetic subgroups are "
        "diagnostics of model performance. They do not establish "
        "that the model is fair or unfair."
    )

    lines.append("")

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    lines.append(
        "## 10. Reproducibility"
    )

    lines.append("")

    lines.append(
        f"- Serialized pipeline: `{metadata['model_file']}`"
    )

    lines.append(
        f"- Model SHA-256: `{metadata['model_sha256']}`"
    )

    lines.append(
        f"- Dataset: `{metadata['dataset_file']}`"
    )

    lines.append(
        f"- Python: `{metadata['library_versions']['python']}`"
    )

    lines.append(
        "- scikit-learn: "
        f"`{metadata['library_versions']['scikit_learn']}`"
    )

    lines.append(
        f"- pandas: `{metadata['library_versions']['pandas']}`"
    )

    lines.append(
        f"- NumPy: `{metadata['library_versions']['numpy']}`"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Limitations
    # ---------------------------------------------------------
    lines.append(
        "## 11. Limitations"
    )

    lines.append("")

    lines.append(
        "- The dataset is synthetic and does not represent a real "
        "employer population."
    )

    lines.append(
        "- Predictive associations do not establish causality."
    )

    lines.append(
        "- The operating threshold was selected statistically "
        "using validation F1 rather than real operational costs "
        "or intervention capacity."
    )

    lines.append(
        "- Some validation subgroup metrics vary materially across "
        "synthetic groups."
    )

    lines.append(
        "- Final holdout performance was lower than validation on "
        "some metrics, illustrating temporal generalization risk."
    )

    lines.append(
        "- This system should not be used to make adverse "
        "employment decisions."
    )

    lines.append("")

    lines.append(
        "## 12. AI Boundary"
    )

    lines.append("")

    lines.append(
        "The predictive model—not an LLM—produces the attrition "
        "probability. All metrics, thresholds, ranking diagnostics, "
        "SHAP values, and subgroup calculations are produced by "
        "deterministic Python/ML code. Any later AI component may "
        "only summarize these verified analytical outputs."
    )

    lines.append("")

    return "\n".join(
        lines
    )


def save_model_report(
    report,
    output_path,
):
    """
    Save Markdown model report.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )