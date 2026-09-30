import numpy as np
import pandas as pd
import pytest

from src.shap_explainability import (
    aggregate_shap_to_original_features,
    build_local_explanations,
    build_shap_summary,
    normalize_shap_values,
)


def test_normalize_shap_values_accepts_2d_matrix():
    values = np.array(
        [
            [
                0.1,
                0.2,
            ],
            [
                0.3,
                0.4,
            ],
        ]
    )

    normalized = (
        normalize_shap_values(
            values
        )
    )

    assert (
        normalized.shape
        == (
            2,
            2,
        )
    )


def test_aggregate_shap_returns_original_features():
    shap_values = np.array(
        [
            [
                0.10,
                0.20,
                -0.05,
            ],
            [
                0.30,
                -0.10,
                0.15,
            ],
        ]
    )

    result = (
        aggregate_shap_to_original_features(
            shap_values=shap_values,
            original_feature_mapping=[
                "age",
                "department",
                "department",
            ],
            original_features=[
                "age",
                "department",
            ],
        )
    )

    assert list(
        result.columns
    ) == [
        "age",
        "department",
    ]

    assert (
        result.iloc[0][
            "age"
        ]
        == pytest.approx(
            0.10
        )
    )

    assert (
        result.iloc[0][
            "department"
        ]
        == pytest.approx(
            0.15
        )
    )


def test_shap_summary_ranks_larger_contribution_first():
    aggregated = pd.DataFrame(
        {
            "strong_feature": [
                0.50,
                -0.40,
                0.60,
            ],
            "weak_feature": [
                0.01,
                -0.02,
                0.01,
            ],
        }
    )

    summary = build_shap_summary(
        aggregated
    )

    assert (
        summary.iloc[0][
            "feature"
        ]
        == "strong_feature"
    )

    assert (
        summary.iloc[0][
            "rank"
        ]
        == 1
    )


def test_local_explanation_returns_top_contributors():
    aggregated = pd.DataFrame(
        {
            "feature_a": [
                0.50,
            ],
            "feature_b": [
                -0.20,
            ],
            "feature_c": [
                0.05,
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "employee_id": [
                "EMP-1",
            ],
            "actual_target": [
                1,
            ],
            "predicted_probability": [
                0.70,
            ],
        }
    )

    result = build_local_explanations(
        aggregated_shap=aggregated,
        metadata=metadata,
        row_indices=[
            0,
        ],
        top_n=2,
    )

    assert len(
        result
    ) == 2

    assert (
        result.iloc[0][
            "feature"
        ]
        == "feature_a"
    )

    assert (
        result.iloc[1][
            "feature"
        ]
        == "feature_b"
    )