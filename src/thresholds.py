import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


def analyze_thresholds(
    y_true,
    y_probability,
    thresholds,
):
    """
    Evaluate model behavior across supplied classification thresholds.

    Parameters
    ----------
    y_true:
        Actual binary outcomes.

    y_probability:
        Predicted positive-class probabilities.

    thresholds:
        Iterable of thresholds to evaluate.

    Returns
    -------
    pandas.DataFrame
        One row per threshold.

    Notes
    -----
    This function does NOT select an operating threshold.

    It only reports deterministic performance at thresholds
    supplied by the caller.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------
    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if y_probability.ndim != 1:
        raise ValueError(
            "y_probability must be one-dimensional."
        )

    if len(y_true) != len(y_probability):
        raise ValueError(
            "y_true and y_probability must have "
            "the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Threshold-analysis inputs cannot be empty."
        )

    unique_targets = set(
        np.unique(y_true).tolist()
    )

    if not unique_targets.issubset(
        {0, 1}
    ):
        raise ValueError(
            "y_true must contain only binary "
            "values 0 and 1."
        )

    if not np.isfinite(
        y_probability
    ).all():
        raise ValueError(
            "y_probability contains NaN "
            "or infinite values."
        )

    if (
        (y_probability < 0)
        | (y_probability > 1)
    ).any():
        raise ValueError(
            "Predicted probabilities must "
            "be between 0 and 1."
        )

    if thresholds.ndim != 1:
        raise ValueError(
            "thresholds must be one-dimensional."
        )

    if len(thresholds) == 0:
        raise ValueError(
            "At least one threshold is required."
        )

    if (
        (thresholds < 0)
        | (thresholds > 1)
    ).any():
        raise ValueError(
            "Thresholds must be between 0 and 1."
        )

    total_actual_positives = int(
        np.sum(y_true == 1)
    )

    rows = []

    # ---------------------------------------------------------
    # Deterministically evaluate each supplied threshold
    # ---------------------------------------------------------
    for threshold in thresholds:
        y_predicted = (
            y_probability >= threshold
        ).astype(int)

        true_positive = int(
            np.sum(
                (y_true == 1)
                & (y_predicted == 1)
            )
        )

        false_positive = int(
            np.sum(
                (y_true == 0)
                & (y_predicted == 1)
            )
        )

        true_negative = int(
            np.sum(
                (y_true == 0)
                & (y_predicted == 0)
            )
        )

        false_negative = int(
            np.sum(
                (y_true == 1)
                & (y_predicted == 0)
            )
        )

        predicted_positive_count = int(
            np.sum(y_predicted)
        )

        predicted_positive_rate = (
            predicted_positive_count
            / len(y_true)
        )

        if total_actual_positives > 0:
            attrition_capture_rate = (
                true_positive
                / total_actual_positives
            )
        else:
            attrition_capture_rate = 0.0

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "predicted_positive_count": (
                    predicted_positive_count
                ),
                "predicted_positive_rate": float(
                    predicted_positive_rate
                ),
                "precision": float(
                    precision_score(
                        y_true,
                        y_predicted,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true,
                        y_predicted,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_true,
                        y_predicted,
                        zero_division=0,
                    )
                ),
                "true_positive": (
                    true_positive
                ),
                "false_positive": (
                    false_positive
                ),
                "true_negative": (
                    true_negative
                ),
                "false_negative": (
                    false_negative
                ),
                "attrition_capture_rate": float(
                    attrition_capture_rate
                ),
            }
        )

    return pd.DataFrame(rows)

def select_threshold_by_f1(
    threshold_table,
):
    """
    Select a development threshold using maximum F1.

    Tie-breaking rules
    ------------------
    If multiple thresholds have the same F1:

    1. Prefer higher precision.
    2. Then prefer higher recall.
    3. Then prefer the higher threshold.

    Parameters
    ----------
    threshold_table:
        DataFrame returned by analyze_thresholds().

    Returns
    -------
    pandas.Series
        The selected threshold-analysis row.

    Notes
    -----
    This is a reproducible development-stage statistical rule.

    It is NOT a claim that the selected threshold is universally
    optimal for real employment decisions.

    The final test set must not be used to select the threshold.
    """

    required_columns = {
        "threshold",
        "precision",
        "recall",
        "f1",
    }

    missing_columns = (
        required_columns
        - set(threshold_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Threshold table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if threshold_table.empty:
        raise ValueError(
            "Threshold table cannot be empty."
        )

    ranked = threshold_table.sort_values(
        by=[
            "f1",
            "precision",
            "recall",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        kind="mergesort",
    )

    return ranked.iloc[0].copy()