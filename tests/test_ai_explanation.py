import json

import pandas as pd
import pytest

from src.ai_explanation import (
    AI_OUTPUT_KEYS,
    build_ai_fact_package,
    build_ai_instructions,
    build_ai_schema,
    render_ai_markdown,
    validate_ai_output,
)


def make_ai_inputs():
    metadata = {
        "model_label": "Gradient Boosting",
        "model_name": "GradientBoostingClassifier",
        "prediction_horizon_days": 90,
        "operating_threshold": 0.125,
        "feature_count": 19,
        "synthetic_data_only": True,
        "test_used_for_model_selection": False,
    }

    final_metrics = {
        "row_count": 100,
        "target_prevalence": 0.10,
        "roc_auc": 0.65,
        "pr_auc": 0.20,
        "brier_score": 0.09,
        "precision": 0.20,
        "recall": 0.40,
        "f1": 0.27,
        "predicted_positive_rate": 0.20,
        "true_positive": 4,
        "true_negative": 74,
        "false_positive": 16,
        "false_negative": 6,
    }

    top_risk = pd.DataFrame(
        [
            {
                "requested_top_percentage": 0.10,
                "actual_population_rate": 0.10,
                "attrition_cases": 3,
                "attrition_capture_rate": 0.30,
                "observed_attrition_rate": 0.30,
                "lift": 3.0,
            }
        ]
    )

    permutation = pd.DataFrame(
        [
            {
                "rank": 1,
                "feature": "tenure_months",
                "importance_mean": 0.05,
                "importance_std": 0.01,
            }
        ]
    )

    shap = pd.DataFrame(
        [
            {
                "rank": 1,
                "feature": "tenure_months",
                "mean_abs_shap": 0.50,
                "importance_share": 0.30,
                "mean_signed_shap": 0.01,
            }
        ]
    )

    subgroup = pd.DataFrame(
        [
            {
                "group_column": "department",
                "group_value": "A",
                "record_count": 60,
                "attrition_prevalence": 0.10,
                "predicted_positive_rate": 0.20,
                "roc_auc": 0.65,
                "precision": 0.20,
                "recall": 0.40,
                "false_positive_rate": 0.15,
                "false_negative_rate": 0.60,
            }
        ]
    )

    return (
        metadata,
        final_metrics,
        top_risk,
        permutation,
        shap,
        subgroup,
    )


def make_valid_ai_output():
    return {
        "executive_summary": (
            "Synthetic model summary."
        ),
        "performance_interpretation": (
            "The supplied metrics show predictive signal."
        ),
        "ranking_interpretation": (
            "Higher-risk groups contain more positive cases."
        ),
        "model_driver_notes": [
            "Feature importance explains model dependence."
        ],
        "subgroup_diagnostic_notes": [
            "Subgroup metrics vary."
        ],
        "limitations": [
            "The dataset is synthetic."
        ],
        "governance_note": (
            "Outputs should not drive employment actions."
        ),
    }


def test_fact_package_excludes_employee_identifiers():
    inputs = make_ai_inputs()

    package = build_ai_fact_package(
        metadata=inputs[0],
        final_metrics=inputs[1],
        top_risk=inputs[2],
        permutation_importance=inputs[3],
        shap_summary=inputs[4],
        subgroup_metrics=inputs[5],
    )

    serialized = json.dumps(
        package
    )

    assert (
        "employee_id"
        not in serialized
    )


def test_ai_instructions_include_core_boundaries():
    instructions = (
        build_ai_instructions()
    )

    assert (
        "Do not make predictions"
        in instructions
    )

    assert (
        "Do not recommend employment actions"
        in instructions
    )

    assert (
        "synthetic"
        in instructions.lower()
    )


def test_ai_schema_is_strict():
    schema = build_ai_schema()

    assert (
        schema["strict"]
        is True
    )

    assert (
        schema[
            "schema"
        ][
            "additionalProperties"
        ]
        is False
    )

    assert (
        schema[
            "schema"
        ][
            "required"
        ]
        == AI_OUTPUT_KEYS
    )


def test_ai_output_validation_rejects_missing_key():
    payload = (
        make_valid_ai_output()
    )

    payload.pop(
        "governance_note"
    )

    with pytest.raises(
        ValueError,
        match="missing required keys",
    ):
        validate_ai_output(
            payload
        )


def test_ai_markdown_contains_disclosure():
    payload = (
        make_valid_ai_output()
    )

    report = render_ai_markdown(
        payload=payload,
        model_name="test-model",
    )

    assert (
        "AI-generated explanation"
        in report
    )

    assert (
        "did not generate predictions"
        in report
    )