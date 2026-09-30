import json

import numpy as np

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
    build_logistic_pipeline,
)
from src.thresholds import analyze_thresholds


def print_threshold_table(
    model_name,
    table,
):
    """
    Print a compact validation threshold table.
    """

    print()
    print(
        f"{model_name} - Threshold Analysis"
    )
    print(
        "-" * (
            len(model_name)
            + 21
        )
    )

    print(
        f"{'Threshold':<11}"
        f"{'Flagged':<10}"
        f"{'Flag %':<10}"
        f"{'Precision':<11}"
        f"{'Recall':<10}"
        f"{'F1':<9}"
        f"{'TP':<6}"
        f"{'FP':<6}"
        f"{'FN':<6}"
        f"{'Capture':<10}"
    )

    print(
        "-" * 89
    )

    for _, row in table.iterrows():
        print(
            f"{row['threshold']:<11.3f}"
            f"{int(row['predicted_positive_count']):<10}"
            f"{row['predicted_positive_rate']:<10.1%}"
            f"{row['precision']:<11.3f}"
            f"{row['recall']:<10.3f}"
            f"{row['f1']:<9.3f}"
            f"{int(row['true_positive']):<6}"
            f"{int(row['false_positive']):<6}"
            f"{int(row['false_negative']):<6}"
            f"{row['attrition_capture_rate']:<10.1%}"
        )


def main():
    # ---------------------------------------------------------
    # Configuration and validated data
    # ---------------------------------------------------------
    with open(
        "config/project_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)

    df = load_data()

    target = cfg["target"]

    # ---------------------------------------------------------
    # Development windows
    #
    # Final TEST is deliberately not accessed.
    # ---------------------------------------------------------
    train = df[
        df["snapshot_date"]
        <= cfg["train_end"]
    ].copy()

    validation = df[
        (
            df["snapshot_date"]
            >= cfg["validation_start"]
        )
        & (
            df["snapshot_date"]
            <= cfg["validation_end"]
        )
    ].copy()

    if train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty."
        )

    # ---------------------------------------------------------
    # Model feature set
    # ---------------------------------------------------------
    features = [
        column
        for column in df.columns
        if column not in PROHIBITED_FEATURES
    ]

    X_train = train[features]
    y_train = train[target]

    X_validation = validation[features]
    y_validation = validation[target]

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
        if column not in categorical_features
    ]

    # ---------------------------------------------------------
    # Leading models
    # ---------------------------------------------------------
    logistic_model = build_logistic_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    gradient_boosting_model = (
        build_gradient_boosting_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_seed=cfg["random_seed"],
        )
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    gradient_boosting_model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # VALIDATION probabilities only
    # ---------------------------------------------------------
    logistic_probabilities = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    gradient_probabilities = (
        gradient_boosting_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Thresholds being inspected
    #
    # These are diagnostic checkpoints only.
    # No threshold is selected by this script.
    # ---------------------------------------------------------
    thresholds = np.array(
        [
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.175,
            0.20,
            0.225,
            0.25,
            0.30,
            0.40,
            0.50,
        ]
    )

    logistic_table = analyze_thresholds(
        y_true=y_validation,
        y_probability=logistic_probabilities,
        thresholds=thresholds,
    )

    gradient_table = analyze_thresholds(
        y_true=y_validation,
        y_probability=gradient_probabilities,
        thresholds=thresholds,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation Threshold Diagnostics"
    )
    print(
        "=" * 44
    )

    print(
        "Final test set is NOT accessed."
    )

    print(
        "Thresholds are evaluated, not selected."
    )

    print(
        "No optimization is performed."
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Actual attrition cases: "
        f"{int(y_validation.sum()):,}"
    )

    print_threshold_table(
        model_name="Logistic Regression",
        table=logistic_table,
    )

    print_threshold_table(
        model_name="Gradient Boosting",
        table=gradient_table,
    )

    print()

    print(
        "The next checkpoint will use these tradeoffs "
        "to define an operating-threshold rule rather "
        "than assuming 0.50."
    )


if __name__ == "__main__":
    main()