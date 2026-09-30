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
from src.thresholds import (
    analyze_thresholds,
    select_threshold_by_f1,
)


def print_selection(
    model_name,
    selected,
):
    """
    Print the selected validation threshold and its metrics.
    """

    print(model_name)
    print("-" * len(model_name))

    print(
        f"Selected threshold:       "
        f"{selected['threshold']:.3f}"
    )

    print(
        f"Flagged records:          "
        f"{int(selected['predicted_positive_count']):,}"
    )

    print(
        f"Flagged population:       "
        f"{selected['predicted_positive_rate']:.1%}"
    )

    print(
        f"Precision:                "
        f"{selected['precision']:.3f}"
    )

    print(
        f"Recall:                   "
        f"{selected['recall']:.3f}"
    )

    print(
        f"F1:                       "
        f"{selected['f1']:.3f}"
    )

    print(
        f"True positives:           "
        f"{int(selected['true_positive']):,}"
    )

    print(
        f"False positives:          "
        f"{int(selected['false_positive']):,}"
    )

    print(
        f"False negatives:          "
        f"{int(selected['false_negative']):,}"
    )

    print(
        f"Attrition cases captured: "
        f"{selected['attrition_capture_rate']:.1%}"
    )

    print()


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
    # Final test period is deliberately not accessed.
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
    # Build leading models
    # ---------------------------------------------------------
    logistic_model = build_logistic_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    gradient_model = (
        build_gradient_boosting_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_seed=cfg["random_seed"],
        )
    )

    # ---------------------------------------------------------
    # Fit using TRAIN only
    # ---------------------------------------------------------
    logistic_model.fit(
        X_train,
        y_train,
    )

    gradient_model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Validation probabilities
    # ---------------------------------------------------------
    logistic_probabilities = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    gradient_probabilities = (
        gradient_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Threshold grid
    #
    # Same diagnostic grid used in Step 9.
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
    # Select maximum-F1 threshold using validation only
    # ---------------------------------------------------------
    logistic_selected = (
        select_threshold_by_f1(
            logistic_table
        )
    )

    gradient_selected = (
        select_threshold_by_f1(
            gradient_table
        )
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Development Threshold Selection"
    )
    print(
        "=" * 43
    )

    print(
        "Final test set is NOT accessed."
    )

    print(
        "Selection rule: maximum validation F1."
    )

    print(
        "This is a synthetic development rule, "
        "not a real-world employment policy."
    )

    print()

    print_selection(
        model_name="Logistic Regression",
        selected=logistic_selected,
    )

    print_selection(
        model_name="Gradient Boosting",
        selected=gradient_selected,
    )

    # ---------------------------------------------------------
    # Current development candidate
    #
    # We compare the models at their individually selected
    # validation thresholds.
    # ---------------------------------------------------------
    if (
        gradient_selected["f1"]
        > logistic_selected["f1"]
    ):
        development_model = (
            "Gradient Boosting"
        )

        development_threshold = float(
            gradient_selected["threshold"]
        )

        development_f1 = float(
            gradient_selected["f1"]
        )

    else:
        development_model = (
            "Logistic Regression"
        )

        development_threshold = float(
            logistic_selected["threshold"]
        )

        development_f1 = float(
            logistic_selected["f1"]
        )

    print(
        "Current development candidate"
    )
    print(
        "-----------------------------"
    )

    print(
        f"Model:      "
        f"{development_model}"
    )

    print(
        f"Threshold:  "
        f"{development_threshold:.3f}"
    )

    print(
        f"Validation F1: "
        f"{development_f1:.3f}"
    )

    print()

    print(
        "This development choice remains subject "
        "to error analysis, lift analysis, subgroup "
        "diagnostics, explainability, and final "
        "untouched test evaluation."
    )


if __name__ == "__main__":
    main()