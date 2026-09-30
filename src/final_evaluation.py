import json
from pathlib import Path

import pandas as pd

from src.data_contract import (
    PROHIBITED_FEATURES,
    load_data,
)
from src.evaluation import (
    evaluate_binary_classifier,
)
from src.model_comparison import (
    build_gradient_boosting_pipeline,
)
from src.risk_analysis import (
    analyze_risk_bands,
    analyze_top_risk_groups,
)


def main():
    # ---------------------------------------------------------
    # Load configuration and locked development decision
    # ---------------------------------------------------------
    with open(
        "config/project_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)

    lock_path = Path(
        "artifacts/development_model_lock.json"
    )

    if not lock_path.exists():
        raise FileNotFoundError(
            "Model lock file does not exist. "
            "Final test evaluation cannot proceed."
        )

    with open(
        lock_path,
        "r",
        encoding="utf-8",
    ) as f:
        model_lock = json.load(f)

    if (
        model_lock.get("status")
        != "LOCKED_BEFORE_FINAL_TEST"
    ):
        raise ValueError(
            "Model lock status is invalid."
        )

    if (
        model_lock.get(
            "test_used_for_selection"
        )
        is not False
    ):
        raise ValueError(
            "Model lock must confirm that test data "
            "was not used for selection."
        )

    threshold = float(
        model_lock[
            "operating_threshold"
        ]
    )

    # ---------------------------------------------------------
    # Load validated synthetic dataset
    # ---------------------------------------------------------
    df = load_data()

    target = cfg["target"]
    id_column = cfg["id_column"]
    date_column = cfg["date_column"]

    # ---------------------------------------------------------
    # Training data
    #
    # Keep the same TRAIN window used throughout development.
    #
    # Validation was used for model / threshold decisions.
    # It is NOT added back into training before this final
    # evaluation so the exact development model specification
    # remains comparable.
    # ---------------------------------------------------------
    train = df[
        df[date_column]
        <= cfg["train_end"]
    ].copy()

    # ---------------------------------------------------------
    # FINAL TEST
    #
    # This is the first intended access to the test period.
    # No model or threshold decisions may be changed based on
    # these results.
    # ---------------------------------------------------------
    test = df[
        (
            df[date_column]
            >= cfg["test_start"]
        )
        & (
            df[date_column]
            <= cfg["test_end"]
        )
    ].copy()

    if train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if test.empty:
        raise ValueError(
            "Final test split is empty."
        )

    if (
        train[date_column].max()
        >= test[date_column].min()
    ):
        raise ValueError(
            "Training and final test periods "
            "are not chronologically separated."
        )

    # ---------------------------------------------------------
    # Feature policy enforcement
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

    X_test = test[
        features
    ]

    y_test = test[
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
    # Rebuild the already-selected Gradient Boosting model
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

    # ---------------------------------------------------------
    # FINAL TEST probabilities
    # ---------------------------------------------------------
    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Locked-threshold evaluation
    # ---------------------------------------------------------
    metrics = (
        evaluate_binary_classifier(
            y_true=y_test,
            y_probability=test_probabilities,
            threshold=threshold,
        )
    )

    # ---------------------------------------------------------
    # Row-level final predictions
    # ---------------------------------------------------------
    predictions = test[
        [
            id_column,
            date_column,
        ]
    ].copy()

    predictions[
        "actual_target"
    ] = (
        y_test.to_numpy()
    )

    predictions[
        "predicted_probability"
    ] = (
        test_probabilities
    )

    predictions[
        "operating_threshold"
    ] = threshold

    predictions[
        "predicted_class"
    ] = (
        test_probabilities
        >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Final confusion matrix artifact
    # ---------------------------------------------------------
    confusion = pd.DataFrame(
        [
            {
                "actual": 0,
                "predicted": 0,
                "count": metrics[
                    "true_negative"
                ],
            },
            {
                "actual": 0,
                "predicted": 1,
                "count": metrics[
                    "false_positive"
                ],
            },
            {
                "actual": 1,
                "predicted": 0,
                "count": metrics[
                    "false_negative"
                ],
            },
            {
                "actual": 1,
                "predicted": 1,
                "count": metrics[
                    "true_positive"
                ],
            },
        ]
    )

    # ---------------------------------------------------------
    # Final ranking diagnostics
    # ---------------------------------------------------------
    risk_deciles = (
        analyze_risk_bands(
            y_true=y_test,
            y_probability=test_probabilities,
            n_bands=10,
        )
    )

    top_risk = (
        analyze_top_risk_groups(
            y_true=y_test,
            y_probability=test_probabilities,
            percentages=(
                0.05,
                0.10,
                0.20,
            ),
        )
    )

    # ---------------------------------------------------------
    # Persist FINAL TEST artifacts
    # ---------------------------------------------------------
    metrics_path = Path(
        "outputs/final_test_metrics.json"
    )

    predictions_path = Path(
        "outputs/final_test_predictions.csv"
    )

    confusion_path = Path(
        "outputs/final_test_confusion_matrix.csv"
    )

    risk_path = Path(
        "outputs/final_test_risk_deciles.csv"
    )

    top_risk_path = Path(
        "outputs/final_test_top_risk.csv"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    confusion.to_csv(
        confusion_path,
        index=False,
    )

    risk_deciles.to_csv(
        risk_path,
        index=False,
    )

    top_risk.to_csv(
        top_risk_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - FINAL Untouched Test Evaluation"
    )
    print(
        "=" * 44
    )

    print(
        "Development decisions were locked before "
        "this test evaluation."
    )

    print(
        f"Locked model: "
        f"{model_lock['model_label']}"
    )

    print(
        f"Locked threshold: "
        f"{threshold:.3f}"
    )

    print()

    print(
        f"Test rows:                  "
        f"{metrics['row_count']:,}"
    )

    print(
        f"Target prevalence:          "
        f"{metrics['target_prevalence']:.3%}"
    )

    print(
        f"Predicted positive count:   "
        f"{metrics['predicted_positive_count']:,}"
    )

    print(
        f"Predicted positive rate:    "
        f"{metrics['predicted_positive_rate']:.3%}"
    )

    print()

    print(
        f"ROC-AUC:                    "
        f"{metrics['roc_auc']:.3f}"
    )

    print(
        f"PR-AUC:                     "
        f"{metrics['pr_auc']:.3f}"
    )

    print(
        f"Brier score:                "
        f"{metrics['brier_score']:.4f}"
    )

    print()

    print(
        f"Precision:                  "
        f"{metrics['precision']:.3f}"
    )

    print(
        f"Recall:                     "
        f"{metrics['recall']:.3f}"
    )

    print(
        f"F1:                         "
        f"{metrics['f1']:.3f}"
    )

    print()

    print(
        f"True negatives:             "
        f"{metrics['true_negative']:,}"
    )

    print(
        f"False positives:            "
        f"{metrics['false_positive']:,}"
    )

    print(
        f"False negatives:            "
        f"{metrics['false_negative']:,}"
    )

    print(
        f"True positives:             "
        f"{metrics['true_positive']:,}"
    )

    print()

    print(
        "Top-risk test capture"
    )
    print(
        "---------------------"
    )

    for _, row in top_risk.iterrows():
        print(
            f"Top "
            f"{row['requested_top_percentage']:.0%}: "
            f"{row['attrition_capture_rate']:.1%} "
            f"of attrition cases captured; "
            f"lift {row['lift']:.2f}x"
        )

    print()

    print(
        "FINAL TEST ARTIFACTS"
    )
    print(
        "--------------------"
    )

    print(
        metrics_path
    )

    print(
        predictions_path
    )

    print(
        confusion_path
    )

    print(
        risk_path
    )

    print(
        top_risk_path
    )

    print()

    print(
        "Do not retune the model or threshold based "
        "on these final test results."
    )


if __name__ == "__main__":
    main()