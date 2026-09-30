import json

import joblib
from sklearn.linear_model import LogisticRegression

from src.model_persistence import (
    build_model_metadata,
    save_metadata,
    save_model_artifact,
)


def test_model_artifact_can_be_saved_and_loaded(
    tmp_path,
):
    model = LogisticRegression()

    path = (
        tmp_path
        / "model.joblib"
    )

    model_hash = save_model_artifact(
        model=model,
        output_path=path,
    )

    loaded = joblib.load(
        path
    )

    assert isinstance(
        loaded,
        LogisticRegression,
    )

    assert len(
        model_hash
    ) == 64


def test_metadata_contains_reproducibility_fields():
    metadata = build_model_metadata(
        model_name=(
            "GradientBoostingClassifier"
        ),
        model_label=(
            "Gradient Boosting"
        ),
        model_parameters={
            "random_state": 42,
        },
        threshold=0.125,
        feature_list=[
            "feature_a",
            "feature_b",
        ],
        excluded_feature_list=[
            "employee_id",
        ],
        target="target",
        prediction_horizon_days=90,
        random_seed=42,
        train_start="2025-01-01",
        train_end="2025-12-31",
        validation_start="2026-01-01",
        validation_end="2026-03-31",
        test_start="2026-04-01",
        test_end="2026-06-30",
        train_rows=100,
        validation_rows=20,
        test_rows=20,
        final_test_metrics={
            "roc_auc": 0.70,
        },
        dataset_file="data.csv",
        model_file="model.joblib",
        model_sha256="abc123",
    )

    assert (
        metadata[
            "operating_threshold"
        ]
        == 0.125
    )

    assert (
        metadata[
            "prediction_horizon_days"
        ]
        == 90
    )

    assert (
        metadata[
            "feature_count"
        ]
        == 2
    )

    assert (
        metadata[
            "test_used_for_model_selection"
        ]
        is False
    )


def test_metadata_can_be_saved(
    tmp_path,
):
    metadata = {
        "model": "test",
        "threshold": 0.125,
    }

    path = (
        tmp_path
        / "metadata.json"
    )

    save_metadata(
        metadata=metadata,
        output_path=path,
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        loaded = json.load(
            f
        )

    assert loaded == metadata