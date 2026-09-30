import numpy as np
import pandas as pd


def build_error_analysis_table(
    source_frame,
    target_column,
    predicted_probability,
    threshold,
    id_column,
    date_column,
    context_columns=None,
):
    """
    Build a deterministic row-level classification error table.

    Parameters
    ----------
    source_frame:
        DataFrame containing the evaluation records.

    target_column:
        Name of the binary observed target.

    predicted_probability:
        Positive-class probabilities aligned to source_frame.

    threshold:
        Already-selected classification threshold.

    id_column:
        Identifier retained for traceability only.

    date_column:
        Prediction snapshot date column.

    context_columns:
        Optional list of descriptive feature columns to preserve.

    Returns
    -------
    pandas.DataFrame
        Row-level prediction/error analysis.

    Notes
    -----
    This function does NOT:
    - train a model,
    - select a threshold,
    - change probabilities,
    - calculate SHAP values,
    - or make employment decisions.
    """

    if context_columns is None:
        context_columns = []

    probabilities = np.asarray(
        predicted_probability,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------
    required_columns = {
        target_column,
        id_column,
        date_column,
        *context_columns,
    }

    missing_columns = (
        required_columns
        - set(source_frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Source frame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Validate probabilities
    # ---------------------------------------------------------
    if probabilities.ndim != 1:
        raise ValueError(
            "predicted_probability must be one-dimensional."
        )

    if len(probabilities) != len(source_frame):
        raise ValueError(
            "Predicted probabilities must have the same "
            "row count as source_frame."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Predicted probabilities contain NaN "
            "or infinite values."
        )

    if (
        (probabilities < 0)
        | (probabilities > 1)
    ).any():
        raise ValueError(
            "Predicted probabilities must be "
            "between 0 and 1."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    # ---------------------------------------------------------
    # Validate target
    # ---------------------------------------------------------
    actual = source_frame[
        target_column
    ].to_numpy()

    unique_targets = set(
        np.unique(actual).tolist()
    )

    if not unique_targets.issubset(
        {0, 1}
    ):
        raise ValueError(
            "Target must contain only binary "
            "values 0 and 1."
        )

    # ---------------------------------------------------------
    # Apply already-selected threshold
    # ---------------------------------------------------------
    predicted_class = (
        probabilities >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Deterministic error category
    # ---------------------------------------------------------
    error_category = np.select(
        condlist=[
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
        choicelist=[
            "TRUE_POSITIVE",
            "TRUE_NEGATIVE",
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ],
        default="UNKNOWN",
    )

    # ---------------------------------------------------------
    # Build output table
    # ---------------------------------------------------------
    output_columns = [
        id_column,
        date_column,
    ] + context_columns

    result = source_frame[
        output_columns
    ].copy()

    result[
        "actual_target"
    ] = actual.astype(int)

    result[
        "predicted_probability"
    ] = probabilities

    result[
        "predicted_class"
    ] = predicted_class

    result[
        "operating_threshold"
    ] = float(
        threshold
    )

    result[
        "distance_from_threshold"
    ] = np.abs(
        probabilities - threshold
    )

    result[
        "prediction_correct"
    ] = (
        actual == predicted_class
    )

    result[
        "error_category"
    ] = error_category

    return result


def summarize_error_analysis(
    error_table,
):
    """
    Return deterministic counts from an error-analysis table.
    """

    required_columns = {
        "actual_target",
        "predicted_class",
        "error_category",
    }

    missing_columns = (
        required_columns
        - set(error_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Error table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    counts = (
        error_table[
            "error_category"
        ]
        .value_counts()
        .to_dict()
    )

    true_positive = int(
        counts.get(
            "TRUE_POSITIVE",
            0,
        )
    )

    true_negative = int(
        counts.get(
            "TRUE_NEGATIVE",
            0,
        )
    )

    false_positive = int(
        counts.get(
            "FALSE_POSITIVE",
            0,
        )
    )

    false_negative = int(
        counts.get(
            "FALSE_NEGATIVE",
            0,
        )
    )

    return {
        "row_count": int(
            len(error_table)
        ),
        "true_positive": (
            true_positive
        ),
        "true_negative": (
            true_negative
        ),
        "false_positive": (
            false_positive
        ),
        "false_negative": (
            false_negative
        ),
        "error_count": int(
            false_positive
            + false_negative
        ),
    }


def get_representative_errors(
    error_table,
    n=5,
):
    """
    Return representative model errors.

    High-confidence false positives:
        False positives with the highest probabilities.

    High-confidence false negatives:
        False negatives with the lowest probabilities.

    Boundary errors:
        False positives/negatives closest to the threshold.
    """

    if n < 1:
        raise ValueError(
            "n must be at least 1."
        )

    false_positives = error_table[
        error_table[
            "error_category"
        ]
        == "FALSE_POSITIVE"
    ].copy()

    false_negatives = error_table[
        error_table[
            "error_category"
        ]
        == "FALSE_NEGATIVE"
    ].copy()

    all_errors = error_table[
        error_table[
            "error_category"
        ].isin(
            [
                "FALSE_POSITIVE",
                "FALSE_NEGATIVE",
            ]
        )
    ].copy()

    high_confidence_false_positives = (
        false_positives
        .sort_values(
            by="predicted_probability",
            ascending=False,
        )
        .head(n)
    )

    high_confidence_false_negatives = (
        false_negatives
        .sort_values(
            by="predicted_probability",
            ascending=True,
        )
        .head(n)
    )

    boundary_errors = (
        all_errors
        .sort_values(
            by="distance_from_threshold",
            ascending=True,
        )
        .head(n)
    )

    return {
        "high_confidence_false_positives": (
            high_confidence_false_positives
        ),
        "high_confidence_false_negatives": (
            high_confidence_false_negatives
        ),
        "boundary_errors": (
            boundary_errors
        ),
    }