import json
from pathlib import Path

import numpy as np

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.error_analysis import (
    build_error_analysis_table,
    get_representative_errors,
    summarize_error_analysis,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
)
from src.thresholds import (
    analyze_thresholds,
    select_threshold_by_f1,
)


def print_examples(
    title,
    frame,
):
    """
    Print compact representative error records.
    """

    print()
    print(title)
    print("-" * len(title))

    if frame.empty:
        print(
            "No records."
        )
        return

    columns = [
        "employee_id",
        "predicted_probability",
        "actual_target",
        "predicted_class",
        "error_category",
        "department",
        "job_family",
        "job_level",
        "engagement_score",
        "overtime_hours_90d",
        "absences_90d",
    ]

    print(
        frame[
            columns
        ].to_string(
            index=False
        )
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
    id_column = cfg["id_column"]
    date_column = cfg["date_column"]

    # ---------------------------------------------------------
    # TRAIN and VALIDATION only
    #
    # Final TEST remains untouched.
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
    # Allowed model features
    # ---------------------------------------------------------
    features = [
        column
        for column in df.columns
        if column not in PROHIBITED_FEATURES
    ]

    X_train = train[
        features
    ]

    y_train = train[
        target
    ]

    X_validation = validation[
        features
    ]

    y_validation = validation[
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
        if column not in categorical_features
    ]

    # ---------------------------------------------------------
    # Current development model
    # ---------------------------------------------------------
    model = (
        build_gradient_boosting_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_seed=cfg["random_seed"],
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Reproduce validation threshold-selection rule
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

    threshold_table = analyze_thresholds(
        y_true=y_validation,
        y_probability=validation_probabilities,
        thresholds=thresholds,
    )

    selected = (
        select_threshold_by_f1(
            threshold_table
        )
    )

    selected_threshold = float(
        selected["threshold"]
    )

    # ---------------------------------------------------------
    # Build row-level error analysis
    # ---------------------------------------------------------
    context_columns = [
        "department",
        "job_family",
        "job_level",
        "location",
        "employment_type",
        "age",
        "tenure_months",
        "monthly_salary_annualized",
        "overtime_hours_90d",
        "absences_90d",
        "performance_score",
        "engagement_score",
        "manager_changes_12m",
        "months_since_promotion",
        "commute_minutes",
        "training_hours_12m",
        "remote_work_ratio",
        "salary_change_pct_12m",
        "internal_moves_24m",
    ]

    error_table = (
        build_error_analysis_table(
            source_frame=validation,
            target_column=target,
            predicted_probability=validation_probabilities,
            threshold=selected_threshold,
            id_column=id_column,
            date_column=date_column,
            context_columns=context_columns,
        )
    )

    summary = (
        summarize_error_analysis(
            error_table
        )
    )

    representative = (
        get_representative_errors(
            error_table,
            n=5,
        )
    )

    # ---------------------------------------------------------
    # Save incremental validation artifact
    # ---------------------------------------------------------
    output_path = Path(
        "outputs/validation_error_analysis.csv"
    )

    error_table.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation Error Analysis"
    )
    print(
        "=" * 37
    )

    print(
        "Final test set is NOT accessed."
    )

    print(
        "Current development model: "
        "Gradient Boosting"
    )

    print(
        f"Validation threshold: "
        f"{selected_threshold:.3f}"
    )

    print()

    print(
        f"Validation rows:      "
        f"{summary['row_count']:,}"
    )

    print(
        f"True positives:       "
        f"{summary['true_positive']:,}"
    )

    print(
        f"True negatives:       "
        f"{summary['true_negative']:,}"
    )

    print(
        f"False positives:      "
        f"{summary['false_positive']:,}"
    )

    print(
        f"False negatives:      "
        f"{summary['false_negative']:,}"
    )

    print(
        f"Total errors:         "
        f"{summary['error_count']:,}"
    )

    print_examples(
        title="High-Confidence False Positives",
        frame=representative[
            "high_confidence_false_positives"
        ],
    )

    print_examples(
        title="High-Confidence False Negatives",
        frame=representative[
            "high_confidence_false_negatives"
        ],
    )

    print_examples(
        title="Threshold-Boundary Errors",
        frame=representative[
            "boundary_errors"
        ],
    )

    print()

    print(
        f"Saved artifact: "
        f"{output_path}"
    )

    print()

    print(
        "These records describe model errors in "
        "synthetic data. They do not establish causal "
        "relationships or employment recommendations."
    )


if __name__ == "__main__":
    main()