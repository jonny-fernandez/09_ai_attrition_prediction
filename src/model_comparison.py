import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_contract import PROHIBITED_FEATURES, load_data
from src.evaluation import evaluate_binary_classifier


def build_logistic_pipeline(
    numeric_features,
    categorical_features,
):
    """
    Build the unweighted logistic-regression baseline.
    """

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

    return Pipeline(
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


def build_random_forest_pipeline(
    numeric_features,
    categorical_features,
    random_seed,
):
    """
    Build the first nonlinear tree-based candidate.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

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

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_seed,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline(
    numeric_features,
    categorical_features,
    random_seed,
):
    """
    Build a conservative gradient-boosting candidate.

    Default GradientBoostingClassifier behavior is retained
    except for the reproducibility seed.

    No hyperparameter tuning is performed.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

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
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

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
        ],
        sparse_threshold=0,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=random_seed,
                ),
            ),
        ]
    )


def print_comparison_row(
    name,
    metrics,
):
    """
    Print threshold-independent validation metrics.
    """

    print(
        f"{name:<32}"
        f"{metrics['roc_auc']:<12.3f}"
        f"{metrics['pr_auc']:<12.3f}"
        f"{metrics['brier_score']:<12.3f}"
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

    baseline_threshold = cfg[
        "default_threshold_for_baseline_only"
    ]

    # ---------------------------------------------------------
    # Chronological development windows
    #
    # The final TEST period is deliberately not created
    # or accessed during model development.
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
    # Naive prevalence comparator
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
    # Build candidate models
    # ---------------------------------------------------------
    logistic_model = build_logistic_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    random_forest_model = build_random_forest_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_seed=cfg["random_seed"],
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

    random_forest_model.fit(
        X_train,
        y_train,
    )

    gradient_boosting_model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Generate VALIDATION probabilities
    # ---------------------------------------------------------
    logistic_probabilities = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    random_forest_probabilities = (
        random_forest_model.predict_proba(
            X_validation
        )[:, 1]
    )

    gradient_boosting_probabilities = (
        gradient_boosting_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ---------------------------------------------------------
    # Deterministic evaluation
    #
    # 0.50 remains only a baseline evaluation threshold.
    # The metrics compared below are ROC-AUC, PR-AUC,
    # and Brier score.
    #
    # No operating threshold is selected here.
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

    random_forest_metrics = evaluate_binary_classifier(
        y_true=y_validation,
        y_probability=random_forest_probabilities,
        threshold=baseline_threshold,
    )

    gradient_boosting_metrics = (
        evaluate_binary_classifier(
            y_true=y_validation,
            y_probability=gradient_boosting_probabilities,
            threshold=baseline_threshold,
        )
    )

    # ---------------------------------------------------------
    # Model comparison report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Validation Model Comparison"
    )
    print(
        "=" * 39
    )

    print(
        "Final test set is NOT accessed."
    )

    print(
        "No threshold was selected or optimized."
    )

    print(
        "No hyperparameter tuning was performed."
    )

    print()

    print(
        f"Training rows:      {len(train):,}"
    )

    print(
        f"Validation rows:    {len(validation):,}"
    )

    print(
        f"Model features:     {len(features)}"
    )

    print()

    print(
        f"{'Model':<32}"
        f"{'ROC-AUC':<12}"
        f"{'PR-AUC':<12}"
        f"{'Brier':<12}"
    )

    print(
        "-" * 68
    )

    print_comparison_row(
        "Naive prevalence",
        naive_metrics,
    )

    print_comparison_row(
        "Logistic regression",
        logistic_metrics,
    )

    print_comparison_row(
        "Random forest",
        random_forest_metrics,
    )

    print_comparison_row(
        "Gradient boosting",
        gradient_boosting_metrics,
    )

    # ---------------------------------------------------------
    # Probability-range diagnostics
    # ---------------------------------------------------------
    print()

    print(
        "Probability ranges"
    )
    print(
        "------------------"
    )

    print(
        f"Logistic maximum:          "
        f"{logistic_probabilities.max():.3%}"
    )

    print(
        f"Random forest maximum:     "
        f"{random_forest_probabilities.max():.3%}"
    )

    print(
        f"Gradient boosting maximum: "
        f"{gradient_boosting_probabilities.max():.3%}"
    )

    print()

    print(
        "This checkpoint tests whether gradient "
        "boosting provides a material validation "
        "improvement before any model tuning."
    )


if __name__ == "__main__":
    main()