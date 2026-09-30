import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calibration_diagnostics(
    y_true,
    y_probability,
    n_bins=10,
):
    """
    Calculate deterministic probability-calibration diagnostics.

    Parameters
    ----------
    y_true:
        Observed binary outcomes containing only 0 and 1.

    y_probability:
        Predicted probability of the positive class.

    n_bins:
        Number of approximately equal-sized probability bands.

    Returns
    -------
    dict
        Includes:
        - row_count
        - target_prevalence
        - average_predicted_probability
        - brier_score
        - weighted_absolute_calibration_error
        - bin_count
        - calibration_table

    Notes
    -----
    This function measures calibration only.

    It does NOT:
    - train a model,
    - modify probabilities,
    - calibrate a model,
    - choose a threshold,
    - or access any dataset split itself.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(
        y_probability,
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
            "Calibration inputs cannot be empty."
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

    if not isinstance(
        n_bins,
        int,
    ):
        raise ValueError(
            "n_bins must be an integer."
        )

    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2."
        )

    # There cannot be more populated bins
    # than observations.
    effective_bins = min(
        n_bins,
        len(y_true),
    )

    # ---------------------------------------------------------
    # Assign approximately equal-sized probability bands
    #
    # Records are sorted by predicted probability and then
    # assigned to sequential equal-count groups.
    #
    # This avoids empty fixed-width bins when most predicted
    # probabilities are concentrated in a narrow range.
    # ---------------------------------------------------------
    order = np.argsort(
        y_probability,
        kind="mergesort",
    )

    sorted_positions = np.arange(
        len(y_probability)
    )

    sorted_bin_ids = (
        (
            sorted_positions
            * effective_bins
        )
        // len(y_probability)
    ) + 1

    bin_ids = np.empty(
        len(y_probability),
        dtype=int,
    )

    bin_ids[order] = sorted_bin_ids

    # ---------------------------------------------------------
    # Build calibration table
    # ---------------------------------------------------------
    frame = pd.DataFrame(
        {
            "calibration_bin": bin_ids,
            "actual": y_true.astype(int),
            "predicted_probability": (
                y_probability
            ),
        }
    )

    calibration_table = (
        frame
        .groupby(
            "calibration_bin",
            as_index=False,
        )
        .agg(
            record_count=(
                "actual",
                "size",
            ),
            average_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_attrition_rate=(
                "actual",
                "mean",
            ),
        )
    )

    calibration_table[
        "calibration_gap"
    ] = (
        calibration_table[
            "observed_attrition_rate"
        ]
        - calibration_table[
            "average_predicted_probability"
        ]
    )

    calibration_table[
        "absolute_calibration_gap"
    ] = (
        calibration_table[
            "calibration_gap"
        ].abs()
    )

    # ---------------------------------------------------------
    # Weighted absolute calibration error
    #
    # This is a descriptive diagnostic:
    # the average absolute difference between predicted and
    # observed rates across the calibration bands, weighted by
    # band size.
    # ---------------------------------------------------------
    weighted_absolute_calibration_error = (
        (
            calibration_table[
                "absolute_calibration_gap"
            ]
            * calibration_table[
                "record_count"
            ]
        ).sum()
        / len(y_true)
    )

    return {
        "row_count": int(
            len(y_true)
        ),
        "target_prevalence": float(
            np.mean(y_true)
        ),
        "average_predicted_probability": float(
            np.mean(y_probability)
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_probability,
            )
        ),
        "weighted_absolute_calibration_error": float(
            weighted_absolute_calibration_error
        ),
        "bin_count": int(
            len(calibration_table)
        ),
        "calibration_table": calibration_table,
    }