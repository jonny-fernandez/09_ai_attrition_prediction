import json
from pathlib import Path

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
)
from src.risk_analysis import (
    analyze_risk_bands,
    analyze_top_risk_groups,
)


def print_top_risk_table(
    table,
):
    """
    Print top-risk population performance.
    """

    print(
        "Top-Risk Capture"
    )
    print(
        "----------------"
    )

    print(
        f"{'Group':<12}"
        f"{'N':<7}"
        f"{'Pop %':<10}"
        f"{'Cases':<8}"
        f"{'Capture':<11}"
        f"{'Observed':<11}"
        f"{'Lift':<8}"
    )

    print(
        "-" * 67
    )

    for _, row in table.iterrows():
        print(
            f"Top "
            f"{row['requested_top_percentage']:.0%}"
            f"{'':<5}"
            f"{int(row['record_count']):<7}"
            f"{row['actual_population_rate']:<10.1%}"
            f"{int(row['attrition_cases']):<8}"
            f"{row['attrition_capture_rate']:<11.1%}"
            f"{row['observed_attrition_rate']:<11.1%}"
            f"{row['lift']:<8.2f}"
        )

    print()


def print_risk_band_table(
    table,
):
    """
    Print decile-level lift and cumulative gains.
    """

    print(
        "Risk Deciles / Gains"
    )
    print(
        "--------------------"
    )

    print(
        f"{'Band':<6}"
        f"{'N':<6}"
        f"{'Avg Pred':<11}"
        f"{'Observed':<11}"
        f"{'Cases':<7}"
        f"{'Lift':<8}"
        f"{'Cum Pop':<10}"
        f"{'Cum Capture':<13}"
        f"{'Cum Lift':<9}"
    )

    print(
        "-" * 81
    )

    for _, row in table.iterrows():
        print(
            f"{int(row['risk_band']):<6}"
            f"{int(row['record_count']):<6}"
            f"{row['average_predicted_probability']:<11.1%}"
            f"{row['observed_attrition_rate']:<11.1%}"
            f"{int(row['attrition_cases']):<7}"
            f"{row['lift']:<8.2f}"
            f"{row['cumulative_population_rate']:<10.1%}"
            f"{row['cumulative_capture_rate']:<13.1%}"
            f"{row['cumulative_lift']:<9.2f}"
        )

    print()


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
    # Ranking diagnostics
    # ---------------------------------------------------------
    risk_bands = (
        analyze_risk_bands(
            y_true=y_validation,
            y_probability=validation_probabilities,
            n_bands=10,
        )
    )

    top_risk = (
        analyze_top_risk_groups(
            y_true=y_validation,
            y_probability=validation_probabilities,
            percentages=(
                0.05,
                0.10,
                0.20,
            ),
        )
    )

    # ---------------------------------------------------------
    # Incremental artifact
    # ---------------------------------------------------------
    output_path = Path(
        "outputs/validation_risk_deciles.csv"
    )

    risk_bands.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation Risk / Lift Analysis"
    )
    print(
        "=" * 42
    )

    print(
        "Final test set is NOT accessed."
    )

    print(
        "Current development model: "
        "Gradient Boosting"
    )

    print(
        "This analysis evaluates ranking, "
        "not an operating threshold."
    )

    print()

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Actual attrition cases: "
        f"{int(y_validation.sum()):,}"
    )

    print(
        f"Overall prevalence: "
        f"{y_validation.mean():.3%}"
    )

    print()

    print_top_risk_table(
        top_risk
    )

    print_risk_band_table(
        risk_bands
    )

    print(
        f"Saved artifact: "
        f"{output_path}"
    )

    print()

    print(
        "Lift above 1.0 means the selected risk "
        "group contains attrition cases at a higher "
        "rate than the validation population overall."
    )

    print(
        "These are synthetic ranking diagnostics, "
        "not recommendations for employment action."
    )


if __name__ == "__main__":
    main()