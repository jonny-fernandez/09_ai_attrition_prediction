import json
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.feature_importance import (
    calculate_permutation_importance,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
)


def main():
    # ---------------------------------------------------------
    # Configuration and validated synthetic dataset
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
    # TRAIN and VALIDATION only.
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

    baseline_pr_auc = (
        average_precision_score(
            y_validation,
            validation_probabilities,
        )
    )

    baseline_roc_auc = (
        roc_auc_score(
            y_validation,
            validation_probabilities,
        )
    )

    # ---------------------------------------------------------
    # Permutation importance
    #
    # PR-AUC / average precision is the scoring metric.
    # Positive importance means model performance decreased
    # when that feature was shuffled.
    # ---------------------------------------------------------
    importance_table = (
        calculate_permutation_importance(
            model=model,
            X=X_validation,
            y=y_validation,
            feature_names=features,
            random_seed=cfg["random_seed"],
            n_repeats=20,
            scoring="average_precision",
        )
    )

    # ---------------------------------------------------------
    # Save incremental artifact
    # ---------------------------------------------------------
    output_path = Path(
        "outputs/"
        "validation_permutation_importance.csv"
    )

    importance_table.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation Permutation Feature Importance"
    )
    print(
        "=" * 51
    )

    print(
        "Final test set is NOT accessed."
    )

    print(
        "Current development model: "
        "Gradient Boosting"
    )

    print(
        "Importance scoring metric: "
        "PR-AUC / average precision"
    )

    print()

    print(
        f"Validation ROC-AUC: "
        f"{baseline_roc_auc:.3f}"
    )

    print(
        f"Validation PR-AUC:  "
        f"{baseline_pr_auc:.3f}"
    )

    print()

    print(
        f"{'Rank':<7}"
        f"{'Feature':<32}"
        f"{'Mean PR-AUC decrease':<23}"
        f"{'Std':<12}"
    )

    print(
        "-" * 74
    )

    for _, row in importance_table.iterrows():
        print(
            f"{int(row['rank']):<7}"
            f"{row['feature']:<32}"
            f"{row['importance_mean']:<23.4f}"
            f"{row['importance_std']:<12.4f}"
        )

    print()

    print(
        f"Saved artifact: "
        f"{output_path}"
    )

    print()

    print(
        "A larger positive value means validation "
        "PR-AUC declined more when that feature was "
        "permuted."
    )

    print(
        "Negative or near-zero importance means "
        "shuffling the feature did not reduce validation "
        "PR-AUC in this analysis."
    )

    print(
        "These values describe model behavior and "
        "predictive dependence, not causal effects."
    )


if __name__ == "__main__":
    main()