import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(y_true, y_probability, threshold):
    """
    Calculate deterministic binary-classification metrics.

    Parameters
    ----------
    y_true:
        Observed binary target values containing only 0 and 1.

    y_probability:
        Predicted probability for the positive class.

    threshold:
        Explicit classification threshold between 0 and 1.

    Returns
    -------
    dict
        Probability-based and threshold-based evaluation metrics.

    Notes
    -----
    This function evaluates already-computed model probabilities.

    It does NOT:
    - train a model,
    - select a threshold,
    - optimize a threshold,
    - modify predictions,
    - access train/validation/test datasets,
    - or call AI.
    """

    # Convert inputs into predictable NumPy arrays.
    y_true = np.asarray(y_true)
    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------
    if y_true.ndim != 1 or y_probability.ndim != 1:
        raise ValueError(
            "y_true and y_probability must be one-dimensional."
        )

    if len(y_true) != len(y_probability):
        raise ValueError(
            "y_true and y_probability must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Evaluation inputs cannot be empty."
        )

    unique_targets = set(
        np.unique(y_true).tolist()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            "y_true must contain only binary values 0 and 1."
        )

    # ROC-AUC and PR-AUC are not meaningful when only
    # one class exists in the evaluation dataset.
    if len(unique_targets) < 2:
        raise ValueError(
            "ROC-AUC and PR-AUC require both target "
            "classes to be present."
        )

    if not np.isfinite(y_probability).all():
        raise ValueError(
            "y_probability contains NaN or infinite values."
        )

    if (
        (y_probability < 0)
        | (y_probability > 1)
    ).any():
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    # ---------------------------------------------------------
    # Convert probabilities into class predictions
    #
    # Important:
    # The threshold is supplied by the caller.
    # This function does not decide what threshold is best.
    # ---------------------------------------------------------
    y_predicted = (
        y_probability >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Confusion matrix
    #
    # Explicit labels ensure a stable:
    # TN, FP, FN, TP
    # ordering.
    # ---------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1],
    ).ravel()

    # ---------------------------------------------------------
    # Return normalized metric schema
    # ---------------------------------------------------------
    return {
        "row_count": int(
            len(y_true)
        ),
        "target_prevalence": float(
            np.mean(y_true)
        ),
        "threshold": float(
            threshold
        ),
        "predicted_positive_count": int(
            np.sum(y_predicted)
        ),
        "predicted_positive_rate": float(
            np.mean(y_predicted)
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
            )
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
        "f1": float(
            f1_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "true_negative": int(
            tn
        ),
        "false_positive": int(
            fp
        ),
        "false_negative": int(
            fn
        ),
        "true_positive": int(
            tp
        ),
    }