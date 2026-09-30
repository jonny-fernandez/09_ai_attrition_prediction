import json
from pathlib import Path

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
)
from src.model_persistence import (
    build_model_metadata,
    save_metadata,
    save_model_artifact,
)


def main():
    # ---------------------------------------------------------
    # Load locked project decisions
    # ---------------------------------------------------------
    with open(
        "config/project_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)

    with open(
        "artifacts/development_model_lock.json",
        "r",
        encoding="utf-8",
    ) as f:
        model_lock = json.load(f)

    with open(
        "outputs/final_test_metrics.json",
        "r",
        encoding="utf-8",
    ) as f:
        final_metrics = json.load(f)

    # ---------------------------------------------------------
    # Validate frozen development decision
    # ---------------------------------------------------------
    if (
        model_lock.get(
            "status"
        )
        != "LOCKED_BEFORE_FINAL_TEST"
    ):
        raise ValueError(
            "Development model was not properly locked."
        )

    if (
        model_lock.get(
            "test_used_for_selection"
        )
        is not False
    ):
        raise ValueError(
            "Test data must not have been used "
            "for model selection."
        )

    threshold = float(
        model_lock[
            "operating_threshold"
        ]
    )

    # ---------------------------------------------------------
    # Load dataset and reproduce exact final-evaluation fit
    # ---------------------------------------------------------
    df = load_data()

    target = cfg[
        "target"
    ]

    date_column = cfg[
        "date_column"
    ]

    train = df[
        df[
            date_column
        ]
        <= cfg[
            "train_end"
        ]
    ].copy()

    validation = df[
        (
            df[
                date_column
            ]
            >= cfg[
                "validation_start"
            ]
        )
        & (
            df[
                date_column
            ]
            <= cfg[
                "validation_end"
            ]
        )
    ].copy()

    test = df[
        (
            df[
                date_column
            ]
            >= cfg[
                "test_start"
            ]
        )
        & (
            df[
                date_column
            ]
            <= cfg[
                "test_end"
            ]
        )
    ].copy()

    features = [
        column
        for column in df.columns
        if column
        not in PROHIBITED_FEATURES
    ]

    excluded_features = [
        column
        for column in df.columns
        if column
        in PROHIBITED_FEATURES
    ]

    X_train = train[
        features
    ]

    y_train = train[
        target
    ]

    categorical_features = (
        X_train
        .select_dtypes(
            include="object"
        )
        .columns
        .tolist()
    )

    numeric_features = [
        column
        for column in features
        if column
        not in categorical_features
    ]

    # ---------------------------------------------------------
    # Exact same model specification used in final evaluation
    # ---------------------------------------------------------
    model = (
        build_gradient_boosting_pipeline(
            numeric_features=(
                numeric_features
            ),
            categorical_features=(
                categorical_features
            ),
            random_seed=cfg[
                "random_seed"
            ],
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Persist fitted pipeline
    # ---------------------------------------------------------
    model_path = Path(
        "artifacts/"
        "final_model_pipeline.joblib"
    )

    model_sha256 = (
        save_model_artifact(
            model=model,
            output_path=model_path,
        )
    )

    estimator = (
        model
        .named_steps[
            "model"
        ]
    )

    # ---------------------------------------------------------
    # Build reproducibility metadata
    # ---------------------------------------------------------
    metadata = (
        build_model_metadata(
            model_name=(
                type(
                    estimator
                ).__name__
            ),
            model_label=(
                model_lock[
                    "model_label"
                ]
            ),
            model_parameters=(
                estimator.get_params()
            ),
            threshold=(
                threshold
            ),
            feature_list=(
                features
            ),
            excluded_feature_list=(
                excluded_features
            ),
            target=(
                target
            ),
            prediction_horizon_days=(
                cfg[
                    "prediction_horizon_days"
                ]
            ),
            random_seed=(
                cfg[
                    "random_seed"
                ]
            ),
            train_start=(
                str(
                    train[
                        date_column
                    ].min()
                )
            ),
            train_end=(
                str(
                    train[
                        date_column
                    ].max()
                )
            ),
            validation_start=(
                str(
                    validation[
                        date_column
                    ].min()
                )
            ),
            validation_end=(
                str(
                    validation[
                        date_column
                    ].max()
                )
            ),
            test_start=(
                str(
                    test[
                        date_column
                    ].min()
                )
            ),
            test_end=(
                str(
                    test[
                        date_column
                    ].max()
                )
            ),
            train_rows=(
                len(
                    train
                )
            ),
            validation_rows=(
                len(
                    validation
                )
            ),
            test_rows=(
                len(
                    test
                )
            ),
            final_test_metrics=(
                final_metrics
            ),
            dataset_file=(
                "data/raw/"
                "employee_attrition_snapshots_synthetic.csv"
            ),
            model_file=(
                str(
                    model_path
                )
            ),
            model_sha256=(
                model_sha256
            ),
        )
    )

    metadata_path = Path(
        "artifacts/"
        "model_metadata.json"
    )

    save_metadata(
        metadata=metadata,
        output_path=metadata_path,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Final Model Persistence"
    )
    print(
        "=" * 35
    )

    print(
        "Persisted the exact model specification "
        "used for final test evaluation."
    )

    print()

    print(
        f"Model:       "
        f"{metadata['model_name']}"
    )

    print(
        f"Threshold:   "
        f"{threshold:.3f}"
    )

    print(
        f"Features:    "
        f"{len(features)}"
    )

    print(
        f"Train rows:  "
        f"{len(train):,}"
    )

    print()

    print(
        f"Model artifact: "
        f"{model_path}"
    )

    print(
        f"Metadata:       "
        f"{metadata_path}"
    )

    print(
        f"SHA-256:        "
        f"{model_sha256}"
    )

    print()

    print(
        "Validation and test data were not added "
        "to the persisted model's training set."
    )

    print(
        "This preserves the exact model represented "
        "by the reported final test results."
    )


if __name__ == "__main__":
    main()