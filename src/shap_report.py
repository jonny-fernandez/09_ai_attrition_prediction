import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
)
from src.shap_explainability import (
    build_local_explanations,
    build_shap_summary,
    compute_tree_shap,
)
from src.thresholds import (
    analyze_thresholds,
    select_threshold_by_f1,
)


def print_local_record(
    title,
    local_table,
):
    """
    Print local SHAP contributors for selected records.
    """

    print()
    print(title)
    print("-" * len(title))

    if local_table.empty:
        print(
            "No records."
        )
        return

    for employee_id, group in (
        local_table.groupby(
            "employee_id",
            sort=False,
        )
    ):
        first = (
            group.iloc[0]
        )

        print()
        print(
            f"Employee: {employee_id}"
        )

        print(
            f"Actual target: "
            f"{int(first['actual_target'])}"
        )

        print(
            f"Predicted probability: "
            f"{first['predicted_probability']:.3%}"
        )

        print(
            f"Predicted class: "
            f"{int(first['predicted_class'])}"
        )

        print(
            f"Error category: "
            f"{first['error_category']}"
        )

        print(
            "Top SHAP contributors:"
        )

        for _, row in group.iterrows():
            direction = (
                "higher"
                if row[
                    "shap_value"
                ] > 0
                else "lower"
            )

            print(
                f"  {int(row['local_rank'])}. "
                f"{row['feature']}: "
                f"{row['shap_value']:+.4f} "
                f"(pushed model score {direction})"
            )


def main():
    # ---------------------------------------------------------
    # Load configuration and validated synthetic data
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

    # Reset index so row positions remain aligned with
    # SHAP arrays and local-example selection.
    validation = (
        validation
        .reset_index(
            drop=True
        )
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
    # Current development candidate
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
    # Reproduce selected validation threshold
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

    threshold_table = (
        analyze_thresholds(
            y_true=y_validation,
            y_probability=validation_probabilities,
            thresholds=thresholds,
        )
    )

    selected = (
        select_threshold_by_f1(
            threshold_table
        )
    )

    selected_threshold = float(
        selected[
            "threshold"
        ]
    )

    predicted_class = (
        validation_probabilities
        >= selected_threshold
    ).astype(int)

    actual = (
        y_validation
        .to_numpy()
    )

    error_category = np.select(
        [
            (
                (actual == 1)
                & (predicted_class == 1)
            ),
            (
                (actual == 0)
                & (predicted_class == 0)
            ),
            (
                (actual == 0)
                & (predicted_class == 1)
            ),
            (
                (actual == 1)
                & (predicted_class == 0)
            ),
        ],
        [
            "TRUE_POSITIVE",
            "TRUE_NEGATIVE",
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ],
        default="UNKNOWN",
    )

    # ---------------------------------------------------------
    # Calculate SHAP
    # ---------------------------------------------------------
    shap_result = (
        compute_tree_shap(
            fitted_pipeline=model,
            X=X_validation,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            original_features=features,
        )
    )

    aggregated_shap = (
        shap_result[
            "aggregated_shap"
        ]
    )

    shap_summary = (
        build_shap_summary(
            aggregated_shap
        )
    )

    # ---------------------------------------------------------
    # Add metadata to row-level SHAP artifact
    # ---------------------------------------------------------
    shap_values_output = (
        aggregated_shap
        .copy()
    )

    shap_values_output.insert(
        0,
        "error_category",
        error_category,
    )

    shap_values_output.insert(
        0,
        "predicted_class",
        predicted_class,
    )

    shap_values_output.insert(
        0,
        "predicted_probability",
        validation_probabilities,
    )

    shap_values_output.insert(
        0,
        "actual_target",
        actual,
    )

    shap_values_output.insert(
        0,
        id_column,
        validation[
            id_column
        ].to_numpy(),
    )

    # ---------------------------------------------------------
    # Select representative local examples
    #
    # 2 highest-risk records
    # 2 highest-confidence false positives
    # 2 highest-confidence false negatives
    # ---------------------------------------------------------
    highest_risk_indices = (
        np.argsort(
            -validation_probabilities
        )[
            :2
        ]
        .tolist()
    )

    false_positive_indices = np.where(
        error_category
        == "FALSE_POSITIVE"
    )[0]

    false_negative_indices = np.where(
        error_category
        == "FALSE_NEGATIVE"
    )[0]

    highest_fp_indices = (
        false_positive_indices[
            np.argsort(
                -validation_probabilities[
                    false_positive_indices
                ]
            )
        ][
            :2
        ]
        .tolist()
    )

    strongest_fn_indices = (
        false_negative_indices[
            np.argsort(
                validation_probabilities[
                    false_negative_indices
                ]
            )
        ][
            :2
        ]
        .tolist()
    )

    metadata = pd.DataFrame(
        {
            "employee_id": validation[
                id_column
            ],
            "actual_target": actual,
            "predicted_probability": (
                validation_probabilities
            ),
            "predicted_class": (
                predicted_class
            ),
            "error_category": (
                error_category
            ),
        }
    )

    high_risk_local = (
        build_local_explanations(
            aggregated_shap=aggregated_shap,
            metadata=metadata,
            row_indices=highest_risk_indices,
            top_n=5,
        )
    )

    high_fp_local = (
        build_local_explanations(
            aggregated_shap=aggregated_shap,
            metadata=metadata,
            row_indices=highest_fp_indices,
            top_n=5,
        )
    )

    strong_fn_local = (
        build_local_explanations(
            aggregated_shap=aggregated_shap,
            metadata=metadata,
            row_indices=strongest_fn_indices,
            top_n=5,
        )
    )

    high_risk_local[
        "example_type"
    ] = "HIGHEST_RISK"

    high_fp_local[
        "example_type"
    ] = (
        "HIGH_CONFIDENCE_FALSE_POSITIVE"
    )

    strong_fn_local[
        "example_type"
    ] = (
        "HIGH_CONFIDENCE_FALSE_NEGATIVE"
    )

    local_examples = pd.concat(
        [
            high_risk_local,
            high_fp_local,
            strong_fn_local,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Save incremental artifacts
    # ---------------------------------------------------------
    summary_path = Path(
        "outputs/"
        "validation_shap_summary.csv"
    )

    values_path = Path(
        "outputs/"
        "validation_shap_values.csv"
    )

    local_path = Path(
        "outputs/"
        "validation_shap_local_examples.csv"
    )

    shap_summary.to_csv(
        summary_path,
        index=False,
    )

    shap_values_output.to_csv(
        values_path,
        index=False,
    )

    local_examples.to_csv(
        local_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation SHAP Explainability"
    )
    print(
        "=" * 41
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
        "Global SHAP Importance"
    )
    print(
        "----------------------"
    )

    print(
        f"{'Rank':<7}"
        f"{'Feature':<32}"
        f"{'Mean |SHAP|':<15}"
        f"{'Share':<10}"
        f"{'Mean SHAP':<12}"
    )

    print(
        "-" * 76
    )

    for _, row in (
        shap_summary
        .head(
            15
        )
        .iterrows()
    ):
        print(
            f"{int(row['rank']):<7}"
            f"{row['feature']:<32}"
            f"{row['mean_abs_shap']:<15.4f}"
            f"{row['importance_share']:<10.1%}"
            f"{row['mean_signed_shap']:<+12.4f}"
        )

    print_local_record(
        title="Highest-Risk Local Explanations",
        local_table=high_risk_local,
    )

    print_local_record(
        title="High-Confidence False Positive Explanations",
        local_table=high_fp_local,
    )

    print_local_record(
        title="High-Confidence False Negative Explanations",
        local_table=strong_fn_local,
    )

    print()

    print(
        f"Saved artifact: "
        f"{summary_path}"
    )

    print(
        f"Saved artifact: "
        f"{values_path}"
    )

    print(
        f"Saved artifact: "
        f"{local_path}"
    )

    print()

    print(
        "Positive SHAP values mean the fitted model "
        "pushed that synthetic record's model score upward."
    )

    print(
        "Negative SHAP values mean the fitted model "
        "pushed that synthetic record's model score downward."
    )

    print(
        "SHAP explains model behavior. It does not "
        "establish that a feature caused attrition."
    )


if __name__ == "__main__":
    main()