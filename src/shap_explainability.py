import numpy as np
import pandas as pd


def normalize_shap_values(
    shap_values,
):
    """
    Normalize SHAP output into a 2-D matrix:

        rows x transformed features

    SHAP versions / classifiers may return slightly different
    output shapes. This helper isolates those differences.
    """

    # SHAP Explanation object
    if hasattr(
        shap_values,
        "values",
    ):
        shap_values = (
            shap_values.values
        )

    # Older SHAP APIs may return one array per class.
    if isinstance(
        shap_values,
        list,
    ):
        if len(shap_values) == 0:
            raise ValueError(
                "SHAP returned an empty list."
            )

        # For binary classification, the final element
        # generally represents the positive class.
        shap_values = (
            shap_values[-1]
        )

    values = np.asarray(
        shap_values,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Standard regression / binary-classification format
    # ---------------------------------------------------------
    if values.ndim == 2:
        return values

    # ---------------------------------------------------------
    # Some SHAP outputs include an explicit output dimension.
    #
    # Common form:
    # rows x features x classes
    #
    # For binary classification we explain the positive class.
    # ---------------------------------------------------------
    if values.ndim == 3:
        if values.shape[2] == 2:
            return values[
                :,
                :,
                1,
            ]

        if values.shape[2] == 1:
            return values[
                :,
                :,
                0,
            ]

    raise ValueError(
        "Unsupported SHAP value shape: "
        f"{values.shape}"
    )


def build_original_feature_mapping(
    fitted_preprocessor,
    numeric_features,
    categorical_features,
):
    """
    Map transformed preprocessing columns back to the original
    workforce feature that produced them.

    Numeric features usually map one-to-one.

    Categorical features may map to multiple one-hot encoded
    transformed columns.
    """

    transformed_names = list(
        fitted_preprocessor
        .get_feature_names_out()
    )

    mapping = [
        None
    ] * len(
        transformed_names
    )

    # ---------------------------------------------------------
    # Numeric transformed columns
    # ---------------------------------------------------------
    numeric_slice = (
        fitted_preprocessor
        .output_indices_[
            "num"
        ]
    )

    numeric_positions = list(
        range(
            numeric_slice.start,
            numeric_slice.stop,
        )
    )

    if (
        len(numeric_positions)
        != len(numeric_features)
    ):
        raise ValueError(
            "Numeric transformed-column count does not "
            "match numeric feature count."
        )

    for position, feature in zip(
        numeric_positions,
        numeric_features,
    ):
        mapping[
            position
        ] = feature

    # ---------------------------------------------------------
    # Categorical transformed columns
    # ---------------------------------------------------------
    categorical_slice = (
        fitted_preprocessor
        .output_indices_[
            "cat"
        ]
    )

    categorical_positions = list(
        range(
            categorical_slice.start,
            categorical_slice.stop,
        )
    )

    categorical_pipeline = (
        fitted_preprocessor
        .named_transformers_[
            "cat"
        ]
    )

    encoder = (
        categorical_pipeline
        .named_steps[
            "onehot"
        ]
    )

    categorical_mapping = []

    for feature, categories in zip(
        categorical_features,
        encoder.categories_,
    ):
        categorical_mapping.extend(
            [
                feature
            ] * len(categories)
        )

    if (
        len(categorical_positions)
        != len(categorical_mapping)
    ):
        raise ValueError(
            "Categorical transformed-column count does not "
            "match one-hot encoding output."
        )

    for position, feature in zip(
        categorical_positions,
        categorical_mapping,
    ):
        mapping[
            position
        ] = feature

    if any(
        feature is None
        for feature in mapping
    ):
        raise ValueError(
            "Not every transformed column could be mapped "
            "to an original feature."
        )

    return (
        transformed_names,
        mapping,
    )


def aggregate_shap_to_original_features(
    shap_values,
    original_feature_mapping,
    original_features,
):
    """
    Aggregate transformed SHAP contributions back to original
    feature-level contributions.

    Example
    -------
    Several one-hot department columns are summed back into:

        department
    """

    values = np.asarray(
        shap_values,
        dtype=float,
    )

    if values.ndim != 2:
        raise ValueError(
            "shap_values must be a two-dimensional matrix."
        )

    if (
        values.shape[1]
        != len(
            original_feature_mapping
        )
    ):
        raise ValueError(
            "SHAP column count does not match the "
            "feature mapping."
        )

    original_features = list(
        original_features
    )

    feature_index = {
        feature: index
        for index, feature
        in enumerate(
            original_features
        )
    }

    aggregated = np.zeros(
        (
            values.shape[0],
            len(original_features),
        ),
        dtype=float,
    )

    for transformed_index, original_feature in enumerate(
        original_feature_mapping
    ):
        if (
            original_feature
            not in feature_index
        ):
            raise ValueError(
                "Unknown original feature in mapping: "
                f"{original_feature}"
            )

        original_index = (
            feature_index[
                original_feature
            ]
        )

        aggregated[
            :,
            original_index,
        ] += values[
            :,
            transformed_index,
        ]

    return pd.DataFrame(
        aggregated,
        columns=original_features,
    )


def build_shap_summary(
    aggregated_shap,
):
    """
    Build global original-feature SHAP importance.

    mean_abs_shap:
        Average absolute contribution magnitude.

    mean_signed_shap:
        Average signed model contribution.

    importance_share:
        Feature's share of total mean absolute SHAP magnitude.
    """

    if aggregated_shap.empty:
        raise ValueError(
            "aggregated_shap cannot be empty."
        )

    rows = []

    for feature in aggregated_shap.columns:
        values = (
            aggregated_shap[
                feature
            ].to_numpy()
        )

        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(
                    np.mean(
                        np.abs(
                            values
                        )
                    )
                ),
                "mean_signed_shap": float(
                    np.mean(
                        values
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    total_importance = float(
        summary[
            "mean_abs_shap"
        ].sum()
    )

    if total_importance > 0:
        summary[
            "importance_share"
        ] = (
            summary[
                "mean_abs_shap"
            ]
            / total_importance
        )
    else:
        summary[
            "importance_share"
        ] = 0.0

    summary = (
        summary
        .sort_values(
            by=[
                "mean_abs_shap",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    summary.insert(
        0,
        "rank",
        range(
            1,
            len(summary) + 1,
        ),
    )

    return summary


def build_local_explanations(
    aggregated_shap,
    metadata,
    row_indices,
    top_n=5,
):
    """
    Convert selected validation records into a long-format table
    containing their strongest feature contributions.

    Positive SHAP values pushed the model score upward.
    Negative SHAP values pushed the model score downward.

    These describe model behavior, not causal relationships.
    """

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    rows = []

    for row_index in row_indices:
        contributions = (
            aggregated_shap
            .iloc[
                row_index
            ]
        )

        ranked_features = (
            contributions
            .abs()
            .sort_values(
                ascending=False
            )
            .head(
                top_n
            )
            .index
        )

        record_metadata = (
            metadata
            .iloc[
                row_index
            ]
        )

        for rank, feature in enumerate(
            ranked_features,
            start=1,
        ):
            shap_value = float(
                contributions[
                    feature
                ]
            )

            row = {
                "local_rank": rank,
                "feature": feature,
                "shap_value": (
                    shap_value
                ),
                "absolute_shap_value": (
                    abs(
                        shap_value
                    )
                ),
            }

            for column in metadata.columns:
                row[
                    column
                ] = (
                    record_metadata[
                        column
                    ]
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def compute_tree_shap(
    fitted_pipeline,
    X,
    numeric_features,
    categorical_features,
    original_features,
):
    """
    Calculate Tree SHAP values for the fitted Gradient Boosting
    pipeline and aggregate them back to original features.

    SHAP is imported inside this function so the rest of the
    project's deterministic modules can still be imported
    without requiring the optional SHAP dependency.
    """

    import shap

    preprocessor = (
        fitted_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    estimator = (
        fitted_pipeline
        .named_steps[
            "model"
        ]
    )

    transformed_X = (
        preprocessor.transform(
            X
        )
    )

    (
        transformed_feature_names,
        original_feature_mapping,
    ) = build_original_feature_mapping(
        fitted_preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    explainer = (
        shap.TreeExplainer(
            estimator
        )
    )

    explanation = explainer(
        transformed_X
    )

    normalized_values = (
        normalize_shap_values(
            explanation
        )
    )

    if (
        normalized_values.shape[1]
        != len(
            transformed_feature_names
        )
    ):
        raise ValueError(
            "SHAP feature count does not match "
            "the transformed feature count."
        )

    aggregated_shap = (
        aggregate_shap_to_original_features(
            shap_values=normalized_values,
            original_feature_mapping=original_feature_mapping,
            original_features=original_features,
        )
    )

    return {
        "transformed_feature_names": (
            transformed_feature_names
        ),
        "original_feature_mapping": (
            original_feature_mapping
        ),
        "aggregated_shap": (
            aggregated_shap
        ),
    }