import json

from src.calibration import calibration_diagnostics
from src.data_contract import PROHIBITED_FEATURES, load_data
from src.model_comparison import (
    build_gradient_boosting_pipeline,
    build_logistic_pipeline,
)


def print_calibration_summary(
    model_name,
    diagnostics,
):
    """
    Print model-level calibration diagnostics.
    """

    print(model_name)
    print("-" * len(model_name))

    print(
        f"Rows:                           "
        f"{diagnostics['row_count']:,}"
    )

    print(
        f"Observed attrition rate:        "
        f"{diagnostics['target_prevalence']:.3%}"
    )

    print(
        f"Average predicted probability:  "
        f"{diagnostics['average_predicted_probability']:.3%}"
    )

    print(
        f"Brier score:                    "
        f"{diagnostics['brier_score']:.4f}"
    )

    print(
        f"Weighted abs calibration error: "
        f"{diagnostics['weighted_absolute_calibration_error']:.3%}"
    )

    print()


def print_calibration_table(
    model_name,
    diagnostics,
):
    """
    Print predicted versus observed probability bands.
    """

    table = diagnostics[
        "calibration_table"
    ]

    print(
        f"{model_name} - Calibration Bands"
    )
    print(
        "-" * (
            len(model_name)
            + 20
        )
    )

    print(
        f"{'Band':<8}"
        f"{'N':<8}"
        f"{'Avg Pred':<14}"
        f"{'Observed':<14}"
        f"{'Gap':<14}"
    )

    print(
        "-" * 58
    )

    for _, row in table.iterrows():
        print(
            f"{int(row['calibration_bin']):<8}"
            f"{int(row['record_count']):<8}"
            f"{row['average_predicted_probability']:<14.3%}"
            f"{row['observed_attrition_rate']:<14.3%}"
            f"{row['calibration_gap']:<+14.3%}"
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
    # Chronological development windows
    #
    # Final TEST data is deliberately not created/accessed.
    # ---------------------------------------------------------
    train = df[
        df["snapshot_date"] <= cfg["train_end"]
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
    # Feature selection
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
    # Leading development models
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

    # ---------------------------------------------------------
    # Fit on TRAIN only
    # ---------------------------------------------------------
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

    gradient_boosting_probabilities = (
        gradient_boosting_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Calibration diagnostics
    #
    # No probability transformation or recalibration occurs.
    # ---------------------------------------------------------
    logistic_diagnostics = (
        calibration_diagnostics(
            y_true=y_validation,
            y_probability=logistic_probabilities,
            n_bins=10,
        )
    )

    gradient_boosting_diagnostics = (
        calibration_diagnostics(
            y_true=y_validation,
            y_probability=gradient_boosting_probabilities,
            n_bins=10,
        )
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation Calibration Diagnostics"
    )
    print(
        "=" * 46
    )

    print(
        "Final test set is NOT accessed."
    )

    print(
        "No calibration method is fitted."
    )

    print(
        "No threshold is selected or optimized."
    )

    print()

    print_calibration_summary(
        model_name="Logistic Regression",
        diagnostics=logistic_diagnostics,
    )

    print_calibration_summary(
        model_name="Gradient Boosting",
        diagnostics=gradient_boosting_diagnostics,
    )

    print_calibration_table(
        model_name="Logistic Regression",
        diagnostics=logistic_diagnostics,
    )

    print_calibration_table(
        model_name="Gradient Boosting",
        diagnostics=gradient_boosting_diagnostics,
    )

    print(
        "Calibration is being measured before "
        "deciding whether probability calibration "
        "is justified."
    )


if __name__ == "__main__":
    main()