import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


def save_model_artifact(
    model,
    output_path,
):
    """
    Serialize a fitted model pipeline and return its SHA-256 hash.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )

    sha256 = hashlib.sha256()

    with open(
        output_path,
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def build_model_metadata(
    *,
    model_name,
    model_label,
    model_parameters,
    threshold,
    feature_list,
    excluded_feature_list,
    target,
    prediction_horizon_days,
    random_seed,
    train_start,
    train_end,
    validation_start,
    validation_end,
    test_start,
    test_end,
    train_rows,
    validation_rows,
    test_rows,
    final_test_metrics,
    dataset_file,
    model_file,
    model_sha256,
):
    """
    Build machine-readable reproducibility metadata.
    """

    return {
        "project": (
            "09_ai_attrition_prediction"
        ),
        "artifact_type": (
            "final_evaluated_model"
        ),
        "model_name": (
            model_name
        ),
        "model_label": (
            model_label
        ),
        "model_parameters": (
            model_parameters
        ),
        "operating_threshold": float(
            threshold
        ),
        "threshold_selected_on": (
            "validation"
        ),
        "target": (
            target
        ),
        "prediction_horizon_days": int(
            prediction_horizon_days
        ),
        "random_seed": int(
            random_seed
        ),
        "feature_count": int(
            len(
                feature_list
            )
        ),
        "feature_list": list(
            feature_list
        ),
        "excluded_feature_count": int(
            len(
                excluded_feature_list
            )
        ),
        "excluded_feature_list": list(
            excluded_feature_list
        ),
        "splits": {
            "train": {
                "start": train_start,
                "end": train_end,
                "rows": int(
                    train_rows
                ),
            },
            "validation": {
                "start": validation_start,
                "end": validation_end,
                "rows": int(
                    validation_rows
                ),
            },
            "test": {
                "start": test_start,
                "end": test_end,
                "rows": int(
                    test_rows
                ),
            },
        },
        "dataset_file": (
            dataset_file
        ),
        "model_file": (
            model_file
        ),
        "model_sha256": (
            model_sha256
        ),
        "final_test_metrics": (
            final_test_metrics
        ),
        "library_versions": {
            "python": (
                platform.python_version()
            ),
            "numpy": (
                np.__version__
            ),
            "pandas": (
                pd.__version__
            ),
            "scikit_learn": (
                sklearn.__version__
            ),
            "joblib": (
                joblib.__version__
            ),
        },
        "metadata_created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "test_used_for_model_selection": (
            False
        ),
        "synthetic_data_only": (
            True
        ),
    }


def save_metadata(
    metadata,
    output_path,
):
    """
    Persist metadata as formatted JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )