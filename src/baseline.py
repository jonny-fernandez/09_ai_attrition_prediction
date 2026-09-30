import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_contract import PROHIBITED_FEATURES, load_data
from src.evaluation import evaluate_binary_classifier


def print_metrics(title, metrics):
    """
    Print a consistent human-readable metric summary.
    """

    print(title)
    print("-" * len(title))

    print(
        f"Rows:                       "
        f"{metrics['row_count']:,}"
    )

    print(
        f"Target prevalence:          "
        f"{metrics['target_prevalence']:.3%}"
    )

    print(
        f"Evaluation threshold:       "
        f"{metrics['threshold']:.3f}"
    )

    print(
        f"Predicted positive count:   "
        f"{metrics['predicted_positive_count']:,}"
    )

    print(
        f"Predicted positive rate:    "
        f"{metrics['predicted_positive_rate']:.3%}"
    )

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
        f"{metrics['brier_score']:.3f}"
    )

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


def main():
    # ---------------------------------------------------------
    # Load project configuration
    # ---------------------------------------------------------
    with open(
        "config/project_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)

    # ---------------------------------------------------------
    # Load validated synthetic data
    # ---------------------------------------------------------
    df = load_data()

    target = cfg["target"]

    # This is ONLY the temporary baseline evaluation threshold.
    # Threshold optimization happens later using validation data.
    baseline_threshold = cfg[
        "default_threshold_for_baseline_only"
    ]

    # ---------------------------------------------------------
    # Chronological development splits
    #
    # TRAIN:
    # Used for preprocessing and model fitting.
    #
    # VALIDATION:
    # Used for development-stage evaluation.
    #
    # TEST:
    # Deliberately not created or accessed here.
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

    # ---------------------------------------------------------
    # Split validation
    # ---------------------------------------------------------
    if train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty."
        )

    if (
        train["snapshot_date"].max()
        >= validation["snapshot_date"].min()
    ):
        raise ValueError(
            "Training and validation windows "
            "are not chronologically separated."
        )

    # ---------------------------------------------------------
    # Feature selection
    #
    # PROHIBITED_FEATURES includes:
    # - identifier
    # - snapshot date
    # - target
    # - known leakage / post-outcome fields
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

    # ---------------------------------------------------------
    # Separate categorical and numeric fields
    # ---------------------------------------------------------
    categorical_features = (
        X_train
        .select_dtypes(include="object")
        .columns
        .tolist()
    )

    numeric_features = [
        column
        for column in features
        if column not in categorical_features
    ]

    # ---------------------------------------------------------
    # Numeric preprocessing
    # ---------------------------------------------------------
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # ---------------------------------------------------------
    # Categorical preprocessing
    # ---------------------------------------------------------
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # Combined preprocessing
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Unweighted logistic-regression baseline
    # ---------------------------------------------------------
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # Naive prevalence baseline
    #
    # Every validation record receives the training-set
    # prevalence as its probability.
    # ---------------------------------------------------------
    train_prevalence = float(
        y_train.mean()
    )

    naive_probabilities = np.full(
        shape=len(y_validation),
        fill_value=train_prevalence,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Train logistic model on TRAIN only
    # ---------------------------------------------------------
    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Produce validation probabilities
    # ---------------------------------------------------------
    logistic_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Validation probability distribution
    #
    # Diagnostic only.
    #
    # We are inspecting the model's score distribution before
    # performing any threshold optimization.
    # ---------------------------------------------------------
    probability_percentiles = np.percentile(
        logistic_probabilities,
        [
            0,
            25,
            50,
            75,
            90,
            95,
            99,
            100,
        ],
    )

    actual_positive_probabilities = (
        logistic_probabilities[
            y_validation.to_numpy() == 1
        ]
    )

    actual_negative_probabilities = (
        logistic_probabilities[
            y_validation.to_numpy() == 0
        ]
    )

    # ---------------------------------------------------------
    # Deterministic evaluation
    #
    # evaluate_binary_classifier does not select or optimize
    # the threshold. It evaluates the supplied threshold only.
    # ---------------------------------------------------------
    naive_metrics = evaluate_binary_classifier(
        y_true=y_validation,
        y_probability=naive_probabilities,
        threshold=baseline_threshold,
    )

    logistic_metrics = evaluate_binary_classifier(
        y_true=y_validation,
        y_probability=logistic_probabilities,
        threshold=baseline_threshold,
    )

    # ---------------------------------------------------------
    # Main report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Baseline Model Evaluation"
    )
    print(
        "=" * 37
    )

    print(
        "Final test set is NOT accessed "
        "during this checkpoint."
    )

    print(
        "The 0.50 threshold is for baseline "
        "evaluation only."
    )

    print()

    print(
        f"Training rows:               "
        f"{len(train):,}"
    )

    print(
        f"Validation rows:             "
        f"{len(validation):,}"
    )

    print(
        f"Model features:              "
        f"{len(features)}"
    )

    print(
        f"Numeric features:            "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features:        "
        f"{len(categorical_features)}"
    )

    print(
        f"Training prevalence:         "
        f"{y_train.mean():.3%}"
    )

    print(
        f"Validation prevalence:       "
        f"{y_validation.mean():.3%}"
    )

    print()

    # ---------------------------------------------------------
    # Baseline metric reports
    # ---------------------------------------------------------
    print_metrics(
        title="Naive Prevalence Baseline",
        metrics=naive_metrics,
    )

    print_metrics(
        title="Unweighted Logistic Regression",
        metrics=logistic_metrics,
    )

    # ---------------------------------------------------------
    # Logistic probability distribution
    #
    # This helps explain why a 0.50 threshold produces no
    # positive predictions, without selecting a new threshold.
    # ---------------------------------------------------------
    print(
        "Logistic Probability Distribution"
    )
    print(
        "---------------------------------"
    )

    percentile_labels = [
        "Minimum",
        "25th percentile",
        "Median",
        "75th percentile",
        "90th percentile",
        "95th percentile",
        "99th percentile",
        "Maximum",
    ]

    for label, value in zip(
        percentile_labels,
        probability_percentiles,
    ):
        print(
            f"{label:<24}"
            f"{value:.3%}"
        )

    print()

    print(
        f"Average probability - "
        f"actual non-attrition: "
        f"{actual_negative_probabilities.mean():.3%}"
    )

    print(
        f"Average probability - "
        f"actual attrition:     "
        f"{actual_positive_probabilities.mean():.3%}"
    )

    print()

    print(
        "No threshold was selected or optimized "
        "during this checkpoint."
    )


if __name__ == "__main__":
    main()