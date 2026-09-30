import json
from pathlib import Path

import numpy as np

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
)
from src.subgroup_analysis import (
    calculate_multiple_subgroups,
)
from src.thresholds import (
    analyze_thresholds,
    select_threshold_by_f1,
)


def print_group_table(
    subgroup_table,
    group_column,
):
    """
    Print diagnostics for one subgroup field.
    """

    group = subgroup_table[
        subgroup_table[
            "group_column"
        ]
        == group_column
    ].copy()

    if group.empty:
        print()
        print(
            f"{group_column}: "
            "No groups met the minimum size."
        )
        return

    group = group.sort_values(
        by="group_value"
    )

    print()
    print(
        f"{group_column} diagnostics"
    )
    print(
        "-" * (
            len(group_column)
            + 12
        )
    )

    print(
        f"{'Group':<20}"
        f"{'N':<7}"
        f"{'Prev':<9}"
        f"{'Avg Pred':<10}"
        f"{'Flag %':<9}"
        f"{'ROC':<8}"
        f"{'PR':<8}"
        f"{'Prec':<8}"
        f"{'Recall':<8}"
        f"{'FPR':<8}"
        f"{'FNR':<8}"
    )

    print(
        "-" * 103
    )

    for _, row in group.iterrows():
        roc_text = (
            f"{row['roc_auc']:.3f}"
            if not np.isnan(
                row["roc_auc"]
            )
            else "N/A"
        )

        pr_text = (
            f"{row['pr_auc']:.3f}"
            if not np.isnan(
                row["pr_auc"]
            )
            else "N/A"
        )

        fpr_text = (
            f"{row['false_positive_rate']:.3f}"
            if not np.isnan(
                row[
                    "false_positive_rate"
                ]
            )
            else "N/A"
        )

        fnr_text = (
            f"{row['false_negative_rate']:.3f}"
            if not np.isnan(
                row[
                    "false_negative_rate"
                ]
            )
            else "N/A"
        )

        print(
            f"{row['group_value']:<20}"
            f"{int(row['record_count']):<7}"
            f"{row['attrition_prevalence']:<9.1%}"
            f"{row['average_predicted_probability']:<10.1%}"
            f"{row['predicted_positive_rate']:<9.1%}"
            f"{roc_text:<8}"
            f"{pr_text:<8}"
            f"{row['precision']:<8.3f}"
            f"{row['recall']:<8.3f}"
            f"{fpr_text:<8}"
            f"{fnr_text:<8}"
        )


def main():
    # ---------------------------------------------------------
    # Configuration and validated dataset
    # ---------------------------------------------------------
    with open(
        "config/project_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)

    df = load_data()

    target = cfg["target"]

    minimum_group_size = cfg[
        "minimum_subgroup_size"
    ]

    # ---------------------------------------------------------
    # Development data only
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
    # Allowed feature set
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
    # Reproduce the development threshold
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

    selected = select_threshold_by_f1(
        threshold_table
    )

    selected_threshold = float(
        selected["threshold"]
    )

    # ---------------------------------------------------------
    # Core subgroup fields
    # ---------------------------------------------------------
    subgroup_columns = [
        "department",
        "job_family",
        "job_level",
        "location",
        "employment_type",
    ]

    subgroup_metrics = (
        calculate_multiple_subgroups(
            source_frame=validation,
            group_columns=subgroup_columns,
            target_column=target,
            predicted_probability=validation_probabilities,
            threshold=selected_threshold,
            minimum_group_size=minimum_group_size,
        )
    )

    # ---------------------------------------------------------
    # Save incremental artifact
    # ---------------------------------------------------------
    output_path = Path(
        "outputs/validation_subgroup_metrics.csv"
    )

    subgroup_metrics.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation Subgroup Diagnostics"
    )
    print(
        "=" * 43
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

    print(
        f"Minimum subgroup size: "
        f"{minimum_group_size}"
    )

    print()

    print(
        "These are synthetic model-performance "
        "diagnostics. They are not a claim that "
        "the model is fair."
    )

    for group_column in subgroup_columns:
        print_group_table(
            subgroup_table=subgroup_metrics,
            group_column=group_column,
        )

    print()

    print(
        f"Saved artifact: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()