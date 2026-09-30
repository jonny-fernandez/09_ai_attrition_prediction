from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import calibration_diagnostics
from src.thresholds import analyze_thresholds


CHART_DIR = Path(
    "outputs/charts"
)


def save_figure(
    output_path,
):
    """
    Save the current matplotlib figure with consistent settings.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def chart_threshold_tradeoff(
    error_analysis,
):
    """
    Plot validation precision, recall, and F1 across the same
    development threshold grid used during threshold analysis.
    """

    thresholds = np.array(
        [
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.175,
            0.20,
            0.225,
            0.25,
            0.30,
            0.40,
            0.50,
        ]
    )

    table = analyze_thresholds(
        y_true=error_analysis[
            "actual_target"
        ],
        y_probability=error_analysis[
            "predicted_probability"
        ],
        thresholds=thresholds,
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        table[
            "threshold"
        ],
        table[
            "precision"
        ],
        marker="o",
        label="Precision",
    )

    plt.plot(
        table[
            "threshold"
        ],
        table[
            "recall"
        ],
        marker="o",
        label="Recall",
    )

    plt.plot(
        table[
            "threshold"
        ],
        table[
            "f1"
        ],
        marker="o",
        label="F1",
    )

    plt.axvline(
        x=0.125,
        linestyle="--",
        label="Selected threshold = 0.125",
    )

    plt.title(
        "Validation Threshold Tradeoff"
    )

    plt.xlabel(
        "Classification Threshold"
    )

    plt.ylabel(
        "Metric"
    )

    plt.ylim(
        0,
        1,
    )

    plt.legend()

    save_figure(
        CHART_DIR
        / "validation_threshold_tradeoff.png"
    )


def chart_calibration(
    error_analysis,
):
    """
    Plot validation predicted probability versus observed rate
    using equal-count calibration bands.
    """

    diagnostics = calibration_diagnostics(
        y_true=error_analysis[
            "actual_target"
        ],
        y_probability=error_analysis[
            "predicted_probability"
        ],
        n_bins=10,
    )

    table = diagnostics[
        "calibration_table"
    ]

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        table[
            "average_predicted_probability"
        ],
        table[
            "observed_attrition_rate"
        ],
        marker="o",
        label="Gradient Boosting",
    )

    plt.plot(
        [
            0,
            0.30,
        ],
        [
            0,
            0.30,
        ],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.title(
        "Validation Calibration Diagnostic"
    )

    plt.xlabel(
        "Average Predicted Probability"
    )

    plt.ylabel(
        "Observed Attrition Rate"
    )

    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: (
                f"{value:.0%}"
            )
        )
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: (
                f"{value:.0%}"
            )
        )
    )

    plt.legend()

    save_figure(
        CHART_DIR
        / "validation_calibration_curve.png"
    )


def chart_lift_curve(
    risk_deciles,
):
    """
    Plot cumulative lift across validation risk bands.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        risk_deciles[
            "cumulative_population_rate"
        ],
        risk_deciles[
            "cumulative_lift"
        ],
        marker="o",
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
    )

    plt.title(
        "Validation Cumulative Lift"
    )

    plt.xlabel(
        "Cumulative Population"
    )

    plt.ylabel(
        "Cumulative Lift"
    )

    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: (
                f"{value:.0%}"
            )
        )
    )

    save_figure(
        CHART_DIR
        / "validation_lift_curve.png"
    )


def chart_gains_curve(
    risk_deciles,
):
    """
    Plot cumulative attrition capture versus cumulative population.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        risk_deciles[
            "cumulative_population_rate"
        ],
        risk_deciles[
            "cumulative_capture_rate"
        ],
        marker="o",
        label="Model",
    )

    plt.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        label="Random ranking",
    )

    plt.title(
        "Validation Cumulative Gains"
    )

    plt.xlabel(
        "Cumulative Population"
    )

    plt.ylabel(
        "Cumulative Attrition Captured"
    )

    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: (
                f"{value:.0%}"
            )
        )
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: (
                f"{value:.0%}"
            )
        )
    )

    plt.legend()

    save_figure(
        CHART_DIR
        / "validation_gains_curve.png"
    )


def chart_permutation_importance(
    permutation,
):
    """
    Plot the ten strongest positive validation permutation
    importance values.
    """

    top = (
        permutation[
            permutation[
                "importance_mean"
            ]
            > 0
        ]
        .head(
            10
        )
        .sort_values(
            by="importance_mean",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        top[
            "feature"
        ],
        top[
            "importance_mean"
        ],
    )

    plt.title(
        "Validation Permutation Importance"
    )

    plt.xlabel(
        "Mean PR-AUC Decrease After Permutation"
    )

    plt.ylabel(
        "Feature"
    )

    save_figure(
        CHART_DIR
        / "validation_permutation_importance.png"
    )


def chart_shap_importance(
    shap_summary,
):
    """
    Plot the ten highest global SHAP features.
    """

    top = (
        shap_summary
        .head(
            10
        )
        .sort_values(
            by="mean_abs_shap",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        top[
            "feature"
        ],
        top[
            "mean_abs_shap"
        ],
    )

    plt.title(
        "Validation Global SHAP Importance"
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    save_figure(
        CHART_DIR
        / "validation_shap_importance.png"
    )


def build_subgroup_labels(
    subgroup_metrics,
):
    """
    Create readable labels such as:

        department: Programs
        location: Chicago-North
    """

    return (
        subgroup_metrics[
            "group_column"
        ].astype(
            str
        )
        + ": "
        + subgroup_metrics[
            "group_value"
        ].astype(
            str
        )
    )


def chart_subgroup_recall(
    subgroup_metrics,
):
    """
    Compare recall across all eligible synthetic subgroups.
    """

    working = subgroup_metrics.copy()

    working[
        "label"
    ] = build_subgroup_labels(
        working
    )

    working = (
        working
        .sort_values(
            by="recall",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 9)
    )

    plt.barh(
        working[
            "label"
        ],
        working[
            "recall"
        ],
    )

    plt.title(
        "Validation Recall Across Synthetic Subgroups"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Synthetic Subgroup"
    )

    plt.xlim(
        0,
        1,
    )

    save_figure(
        CHART_DIR
        / "validation_subgroup_recall.png"
    )


def chart_subgroup_false_positive_rate(
    subgroup_metrics,
):
    """
    Compare false-positive rates across eligible synthetic
    subgroups.
    """

    working = subgroup_metrics.copy()

    working[
        "label"
    ] = build_subgroup_labels(
        working
    )

    working = (
        working
        .sort_values(
            by="false_positive_rate",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 9)
    )

    plt.barh(
        working[
            "label"
        ],
        working[
            "false_positive_rate"
        ],
    )

    plt.title(
        "Validation False-Positive Rate Across Synthetic Subgroups"
    )

    plt.xlabel(
        "False-Positive Rate"
    )

    plt.ylabel(
        "Synthetic Subgroup"
    )

    plt.xlim(
        0,
        1,
    )

    save_figure(
        CHART_DIR
        / "validation_subgroup_false_positive_rate.png"
    )


def main():
    # ---------------------------------------------------------
    # Load previously generated deterministic artifacts only.
    #
    # No model training or rescoring occurs here.
    # ---------------------------------------------------------
    error_analysis = pd.read_csv(
        "outputs/"
        "validation_error_analysis.csv"
    )

    risk_deciles = pd.read_csv(
        "outputs/"
        "validation_risk_deciles.csv"
    )

    permutation = pd.read_csv(
        "outputs/"
        "validation_permutation_importance.csv"
    )

    shap_summary = pd.read_csv(
        "outputs/"
        "validation_shap_summary.csv"
    )

    subgroup_metrics = pd.read_csv(
        "outputs/"
        "validation_subgroup_metrics.csv"
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Generate portfolio diagnostics
    # ---------------------------------------------------------
    chart_threshold_tradeoff(
        error_analysis
    )

    chart_calibration(
        error_analysis
    )

    chart_lift_curve(
        risk_deciles
    )

    chart_gains_curve(
        risk_deciles
    )

    chart_permutation_importance(
        permutation
    )

    chart_shap_importance(
        shap_summary
    )

    chart_subgroup_recall(
        subgroup_metrics
    )

    chart_subgroup_false_positive_rate(
        subgroup_metrics
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Diagnostic Portfolio Charts"
    )
    print(
        "=" * 39
    )

    print(
        "No model training was performed."
    )

    print(
        "No predictions or thresholds were changed."
    )

    print()

    chart_files = [
        "validation_threshold_tradeoff.png",
        "validation_calibration_curve.png",
        "validation_lift_curve.png",
        "validation_gains_curve.png",
        "validation_permutation_importance.png",
        "validation_shap_importance.png",
        "validation_subgroup_recall.png",
        "validation_subgroup_false_positive_rate.png",
    ]

    for filename in chart_files:
        print(
            CHART_DIR
            / filename
        )


if __name__ == "__main__":
    main()