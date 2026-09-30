import json
from pathlib import Path


AI_OUTPUT_KEYS = [
    "executive_summary",
    "performance_interpretation",
    "ranking_interpretation",
    "model_driver_notes",
    "subgroup_diagnostic_notes",
    "limitations",
    "governance_note",
]


def _to_builtin(value):
    """
    Convert pandas / NumPy values into normal Python values.
    """

    if hasattr(value, "item"):
        return value.item()

    return value


def dataframe_records(
    frame,
    columns,
    limit=None,
):
    """
    Convert selected DataFrame columns into JSON-safe records.
    """

    working = frame[
        columns
    ].copy()

    if limit is not None:
        working = working.head(
            limit
        )

    records = []

    for _, row in working.iterrows():
        record = {}

        for column in columns:
            record[column] = _to_builtin(
                row[column]
            )

        records.append(
            record
        )

    return records


def build_ai_fact_package(
    *,
    metadata,
    final_metrics,
    top_risk,
    permutation_importance,
    shap_summary,
    subgroup_metrics,
):
    """
    Build the deterministic aggregate fact package supplied to AI.

    No employee-level rows or employee identifiers are included.
    """

    # ---------------------------------------------------------
    # Model specification
    # ---------------------------------------------------------
    model = {
        "model_label": metadata[
            "model_label"
        ],
        "estimator": metadata[
            "model_name"
        ],
        "prediction_horizon_days": metadata[
            "prediction_horizon_days"
        ],
        "operating_threshold": metadata[
            "operating_threshold"
        ],
        "feature_count": metadata[
            "feature_count"
        ],
        "synthetic_data_only": metadata[
            "synthetic_data_only"
        ],
        "test_used_for_model_selection": metadata[
            "test_used_for_model_selection"
        ],
    }

    # ---------------------------------------------------------
    # Final untouched-test results
    # ---------------------------------------------------------
    final_test = {
        "row_count": final_metrics[
            "row_count"
        ],
        "target_prevalence": final_metrics[
            "target_prevalence"
        ],
        "roc_auc": final_metrics[
            "roc_auc"
        ],
        "pr_auc": final_metrics[
            "pr_auc"
        ],
        "brier_score": final_metrics[
            "brier_score"
        ],
        "precision": final_metrics[
            "precision"
        ],
        "recall": final_metrics[
            "recall"
        ],
        "f1": final_metrics[
            "f1"
        ],
        "predicted_positive_rate": final_metrics[
            "predicted_positive_rate"
        ],
        "true_positive": final_metrics[
            "true_positive"
        ],
        "true_negative": final_metrics[
            "true_negative"
        ],
        "false_positive": final_metrics[
            "false_positive"
        ],
        "false_negative": final_metrics[
            "false_negative"
        ],
    }

    # ---------------------------------------------------------
    # Ranking / lift
    # ---------------------------------------------------------
    ranking = dataframe_records(
        top_risk,
        columns=[
            "requested_top_percentage",
            "actual_population_rate",
            "attrition_cases",
            "attrition_capture_rate",
            "observed_attrition_rate",
            "lift",
        ],
    )

    # ---------------------------------------------------------
    # Global model dependence
    # ---------------------------------------------------------
    permutation = dataframe_records(
        permutation_importance,
        columns=[
            "rank",
            "feature",
            "importance_mean",
            "importance_std",
        ],
        limit=8,
    )

    shap = dataframe_records(
        shap_summary,
        columns=[
            "rank",
            "feature",
            "mean_abs_shap",
            "importance_share",
            "mean_signed_shap",
        ],
        limit=8,
    )

    # ---------------------------------------------------------
    # Subgroup diagnostic extremes
    #
    # These are supplied only as model-performance diagnostics.
    # ---------------------------------------------------------
    highest_recall = (
        subgroup_metrics
        .sort_values(
            by="recall",
            ascending=False,
        )
        .head(3)
    )

    lowest_recall = (
        subgroup_metrics
        .sort_values(
            by="recall",
            ascending=True,
        )
        .head(3)
    )

    highest_fpr = (
        subgroup_metrics
        .sort_values(
            by="false_positive_rate",
            ascending=False,
        )
        .head(3)
    )

    subgroup_columns = [
        "group_column",
        "group_value",
        "record_count",
        "attrition_prevalence",
        "predicted_positive_rate",
        "roc_auc",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
    ]

    subgroup_diagnostics = {
        "highest_recall_groups": dataframe_records(
            highest_recall,
            subgroup_columns,
        ),
        "lowest_recall_groups": dataframe_records(
            lowest_recall,
            subgroup_columns,
        ),
        "highest_false_positive_rate_groups": (
            dataframe_records(
                highest_fpr,
                subgroup_columns,
            )
        ),
    }

    return {
        "project_context": {
            "project": (
                "AI-Enhanced Employee Attrition "
                "Prediction & Explainability System"
            ),
            "data_type": (
                "synthetic workforce data only"
            ),
            "ai_role": (
                "explanation of verified analytical "
                "outputs only"
            ),
        },
        "model": model,
        "final_test_metrics": final_test,
        "top_risk_ranking": ranking,
        "permutation_importance": permutation,
        "shap_summary": shap,
        "subgroup_diagnostics": subgroup_diagnostics,
        "required_boundaries": [
            (
                "Do not calculate or change any model "
                "probability, metric, or threshold."
            ),
            (
                "Do not describe predictive associations "
                "as causal relationships."
            ),
            (
                "Do not recommend hiring, firing, promotion, "
                "discipline, compensation, or other employment "
                "actions."
            ),
            (
                "Do not claim subgroup diagnostics prove "
                "that the model is fair or unfair."
            ),
            (
                "Do not introduce numerical claims that are "
                "not present in this fact package."
            ),
        ],
    }


def build_ai_instructions():
    """
    System-level instructions for the controlled AI layer.
    """

    return """
You are the explanation layer for a synthetic machine-learning
portfolio project.

Your role is limited to translating verified deterministic model
outputs into clear professional language.

Rules:
1. Use only facts supplied in the input.
2. Do not calculate, modify, estimate, or invent probabilities,
   metrics, thresholds, counts, or rankings.
3. Do not make predictions.
4. Do not describe model associations as causal relationships.
5. Do not recommend employment actions involving any person or group.
6. Treat all subgroup results as model-performance diagnostics only.
7. Do not claim that subgroup results prove fairness or unfairness.
8. Explicitly distinguish model behavior from real-world causation.
9. Emphasize that the dataset is synthetic.
10. If the supplied facts do not support a conclusion, say that the
    evidence is insufficient.
11. Keep the explanation concise and suitable for a technical
    portfolio README or model report.
""".strip()


def build_ai_schema():
    """
    Structured Outputs schema for the AI explanation.
    """

    return {
        "type": "json_schema",
        "name": "attrition_model_interpretation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                },
                "performance_interpretation": {
                    "type": "string",
                },
                "ranking_interpretation": {
                    "type": "string",
                },
                "model_driver_notes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "subgroup_diagnostic_notes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "limitations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "governance_note": {
                    "type": "string",
                },
            },
            "required": AI_OUTPUT_KEYS,
            "additionalProperties": False,
        },
    }


def validate_ai_output(
    payload,
):
    """
    Validate the parsed AI result before saving it.
    """

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "AI output must be a dictionary."
        )

    missing = [
        key
        for key in AI_OUTPUT_KEYS
        if key not in payload
    ]

    if missing:
        raise ValueError(
            "AI output is missing required keys: "
            f"{missing}"
        )

    extras = (
        set(payload)
        - set(AI_OUTPUT_KEYS)
    )

    if extras:
        raise ValueError(
            "AI output contains unexpected keys: "
            f"{sorted(extras)}"
        )

    return True


def render_ai_markdown(
    payload,
    model_name,
):
    """
    Convert validated structured AI output into Markdown.
    """

    validate_ai_output(
        payload
    )

    lines = [
        "# AI Interpretation of Verified Model Results",
        "",
        (
            "> AI-generated explanation based only on verified "
            "deterministic analytical outputs. The AI did not "
            "generate predictions, metrics, or thresholds."
        ),
        "",
        f"API model: `{model_name}`",
        "",
        "## Executive Summary",
        "",
        payload[
            "executive_summary"
        ],
        "",
        "## Performance Interpretation",
        "",
        payload[
            "performance_interpretation"
        ],
        "",
        "## Ranking Interpretation",
        "",
        payload[
            "ranking_interpretation"
        ],
        "",
        "## Model Driver Notes",
        "",
    ]

    for item in payload[
        "model_driver_notes"
    ]:
        lines.append(
            f"- {item}"
        )

    lines.extend(
        [
            "",
            "## Subgroup Diagnostic Notes",
            "",
        ]
    )

    for item in payload[
        "subgroup_diagnostic_notes"
    ]:
        lines.append(
            f"- {item}"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )

    for item in payload[
        "limitations"
    ]:
        lines.append(
            f"- {item}"
        )

    lines.extend(
        [
            "",
            "## Governance Note",
            "",
            payload[
                "governance_note"
            ],
            "",
        ]
    )

    return "\n".join(
        lines
    )


def save_text(
    text,
    output_path,
):
    """
    Save UTF-8 text.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )


def save_json(
    payload,
    output_path,
):
    """
    Save formatted JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )