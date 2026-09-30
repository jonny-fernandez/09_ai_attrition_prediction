import numpy as np
import pandas as pd


def _validate_risk_inputs(
    y_true,
    y_probability,
):
    """
    Validate common risk-analysis inputs.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

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
            "Risk-analysis inputs cannot be empty."
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

    return (
        y_true,
        y_probability,
    )


def analyze_risk_bands(
    y_true,
    y_probability,
    n_bands=10,
):
    """
    Calculate risk-band, lift, and cumulative-gains metrics.

    Risk band 1 represents the highest predicted-risk group.

    This function evaluates ranking behavior only.

    It does NOT:
    - select a classification threshold,
    - modify probabilities,
    - train a model,
    - or make employment recommendations.
    """

    (
        y_true,
        y_probability,
    ) = _validate_risk_inputs(
        y_true,
        y_probability,
    )

    if not isinstance(
        n_bands,
        int,
    ):
        raise ValueError(
            "n_bands must be an integer."
        )

    if n_bands < 2:
        raise ValueError(
            "n_bands must be at least 2."
        )

    effective_bands = min(
        n_bands,
        len(y_true),
    )

    total_records = len(
        y_true
    )

    total_attrition = int(
        np.sum(y_true)
    )

    overall_prevalence = float(
        np.mean(y_true)
    )

    # ---------------------------------------------------------
    # Sort highest predicted risk first.
    #
    # mergesort preserves deterministic ordering for ties.
    # ---------------------------------------------------------
    order = np.argsort(
        -y_probability,
        kind="mergesort",
    )

    sorted_actual = (
        y_true[order]
    )

    sorted_probability = (
        y_probability[order]
    )

    sorted_positions = np.arange(
        total_records
    )

    risk_band = (
        (
            sorted_positions
            * effective_bands
        )
        // total_records
    ) + 1

    frame = pd.DataFrame(
        {
            "risk_band": risk_band,
            "actual_target": (
                sorted_actual.astype(int)
            ),
            "predicted_probability": (
                sorted_probability
            ),
        }
    )

    # ---------------------------------------------------------
    # Aggregate by risk band
    # ---------------------------------------------------------
    table = (
        frame
        .groupby(
            "risk_band",
            as_index=False,
        )
        .agg(
            record_count=(
                "actual_target",
                "size",
            ),
            attrition_cases=(
                "actual_target",
                "sum",
            ),
            average_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_attrition_rate=(
                "actual_target",
                "mean",
            ),
        )
    )

    table[
        "population_share"
    ] = (
        table["record_count"]
        / total_records
    )

    if total_attrition > 0:
        table[
            "band_attrition_capture_rate"
        ] = (
            table[
                "attrition_cases"
            ]
            / total_attrition
        )
    else:
        table[
            "band_attrition_capture_rate"
        ] = 0.0

    if overall_prevalence > 0:
        table[
            "lift"
        ] = (
            table[
                "observed_attrition_rate"
            ]
            / overall_prevalence
        )
    else:
        table[
            "lift"
        ] = 0.0

    # ---------------------------------------------------------
    # Cumulative gains
    # ---------------------------------------------------------
    table[
        "cumulative_record_count"
    ] = (
        table[
            "record_count"
        ].cumsum()
    )

    table[
        "cumulative_population_rate"
    ] = (
        table[
            "cumulative_record_count"
        ]
        / total_records
    )

    table[
        "cumulative_attrition_cases"
    ] = (
        table[
            "attrition_cases"
        ].cumsum()
    )

    if total_attrition > 0:
        table[
            "cumulative_capture_rate"
        ] = (
            table[
                "cumulative_attrition_cases"
            ]
            / total_attrition
        )
    else:
        table[
            "cumulative_capture_rate"
        ] = 0.0

    table[
        "cumulative_lift"
    ] = (
        table[
            "cumulative_capture_rate"
        ]
        / table[
            "cumulative_population_rate"
        ]
    )

    return table


def analyze_top_risk_groups(
    y_true,
    y_probability,
    percentages=(
        0.05,
        0.10,
        0.20,
    ),
):
    """
    Calculate ranking performance for top-risk population groups.
    """

    (
        y_true,
        y_probability,
    ) = _validate_risk_inputs(
        y_true,
        y_probability,
    )

    total_records = len(
        y_true
    )

    total_attrition = int(
        np.sum(y_true)
    )

    overall_prevalence = float(
        np.mean(y_true)
    )

    order = np.argsort(
        -y_probability,
        kind="mergesort",
    )

    sorted_actual = (
        y_true[order]
    )

    sorted_probability = (
        y_probability[order]
    )

    rows = []

    for percentage in percentages:
        if (
            percentage <= 0
            or percentage > 1
        ):
            raise ValueError(
                "Risk-group percentages must "
                "be greater than 0 and no more than 1."
            )

        record_count = max(
            1,
            int(
                np.ceil(
                    total_records
                    * percentage
                )
            ),
        )

        group_actual = (
            sorted_actual[
                :record_count
            ]
        )

        group_probability = (
            sorted_probability[
                :record_count
            ]
        )

        attrition_cases = int(
            np.sum(
                group_actual
            )
        )

        observed_rate = float(
            np.mean(
                group_actual
            )
        )

        actual_population_rate = (
            record_count
            / total_records
        )

        if total_attrition > 0:
            capture_rate = (
                attrition_cases
                / total_attrition
            )
        else:
            capture_rate = 0.0

        if overall_prevalence > 0:
            lift = (
                observed_rate
                / overall_prevalence
            )
        else:
            lift = 0.0

        rows.append(
            {
                "requested_top_percentage": float(
                    percentage
                ),
                "record_count": int(
                    record_count
                ),
                "actual_population_rate": float(
                    actual_population_rate
                ),
                "attrition_cases": (
                    attrition_cases
                ),
                "attrition_capture_rate": float(
                    capture_rate
                ),
                "observed_attrition_rate": float(
                    observed_rate
                ),
                "average_predicted_probability": float(
                    np.mean(
                        group_probability
                    )
                ),
                "lift": float(
                    lift
                ),
            }
        )

    return pd.DataFrame(
        rows
    )