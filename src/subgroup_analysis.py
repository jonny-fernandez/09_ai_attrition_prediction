import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_subgroup_metrics(
    source_frame,
    group_column,
    target_column,
    predicted_probability,
    threshold,
    minimum_group_size,
):
    """
    Calculate deterministic model diagnostics by subgroup.

    Parameters
    ----------
    source_frame:
        Evaluation DataFrame.

    group_column:
        Column defining the subgroup.

    target_column:
        Binary observed target column.

    predicted_probability:
        Positive-class probabilities aligned with source_frame.

    threshold:
        Already-selected operating threshold.

    minimum_group_size:
        Minimum number of records required before a subgroup
        is included in the diagnostic output.

    Returns
    -------
    pandas.DataFrame
        One row per eligible subgroup.

    Notes
    -----
    These are model-performance diagnostics.

    They do NOT establish that a model is legally, ethically,
    or statistically "fair".
    """

    probabilities = np.asarray(
        predicted_probability,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Required-column validation
    # ---------------------------------------------------------
    required_columns = {
        group_column,
        target_column,
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
    # General input validation
    # ---------------------------------------------------------
    if len(source_frame) == 0:
        raise ValueError(
            "Source frame cannot be empty."
        )

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

    if minimum_group_size < 1:
        raise ValueError(
            "minimum_group_size must be at least 1."
        )

    # ---------------------------------------------------------
    # Validate binary target
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
    # Attach deterministic model outputs to a working frame
    # ---------------------------------------------------------
    working = source_frame[
        [
            group_column,
            target_column,
        ]
    ].copy()

    working[
        "predicted_probability"
    ] = probabilities

    working[
        "predicted_class"
    ] = (
        probabilities >= threshold
    ).astype(int)

    rows = []

    # ---------------------------------------------------------
    # Calculate metrics for each eligible subgroup
    # ---------------------------------------------------------
    for group_value, group in working.groupby(
        group_column,
        dropna=False,
    ):
        record_count = len(
            group
        )

        if record_count < minimum_group_size:
            continue

        y_true = group[
            target_column
        ].to_numpy()

        y_probability = group[
            "predicted_probability"
        ].to_numpy()

        y_predicted = group[
            "predicted_class"
        ].to_numpy()

        # ---------------------------------------------
        # Confusion components
        # ---------------------------------------------
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

        negative_count = (
            true_negative
            + false_positive
        )

        positive_count = (
            true_positive
            + false_negative
        )

        # ---------------------------------------------
        # ROC-AUC / PR-AUC require both classes
        # ---------------------------------------------
        subgroup_classes = set(
            np.unique(
                y_true
            ).tolist()
        )

        if subgroup_classes == {0, 1}:
            roc_auc = float(
                roc_auc_score(
                    y_true,
                    y_probability,
                )
            )

            pr_auc = float(
                average_precision_score(
                    y_true,
                    y_probability,
                )
            )
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        # ---------------------------------------------
        # Error rates
        # ---------------------------------------------
        if negative_count > 0:
            false_positive_rate = (
                false_positive
                / negative_count
            )
        else:
            false_positive_rate = np.nan

        if positive_count > 0:
            false_negative_rate = (
                false_negative
                / positive_count
            )
        else:
            false_negative_rate = np.nan

        rows.append(
            {
                "group_column": (
                    group_column
                ),
                "group_value": (
                    str(group_value)
                ),
                "record_count": int(
                    record_count
                ),
                "attrition_cases": int(
                    np.sum(y_true)
                ),
                "attrition_prevalence": float(
                    np.mean(y_true)
                ),
                "average_predicted_probability": float(
                    np.mean(
                        y_probability
                    )
                ),
                "predicted_positive_count": int(
                    np.sum(
                        y_predicted
                    )
                ),
                "predicted_positive_rate": float(
                    np.mean(
                        y_predicted
                    )
                ),
                "roc_auc": (
                    roc_auc
                ),
                "pr_auc": (
                    pr_auc
                ),
                "brier_score": float(
                    brier_score_loss(
                        y_true,
                        y_probability,
                    )
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
                "false_positive_rate": float(
                    false_positive_rate
                ),
                "false_negative_rate": float(
                    false_negative_rate
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
            }
        )

    return pd.DataFrame(
        rows
    )


def calculate_multiple_subgroups(
    source_frame,
    group_columns,
    target_column,
    predicted_probability,
    threshold,
    minimum_group_size,
):
    """
    Calculate subgroup diagnostics for multiple grouping fields.
    """

    tables = []

    for group_column in group_columns:
        table = (
            calculate_subgroup_metrics(
                source_frame=source_frame,
                group_column=group_column,
                target_column=target_column,
                predicted_probability=predicted_probability,
                threshold=threshold,
                minimum_group_size=minimum_group_size,
            )
        )

        if not table.empty:
            tables.append(
                table
            )

    if not tables:
        return pd.DataFrame()

    return pd.concat(
        tables,
        ignore_index=True,
    )