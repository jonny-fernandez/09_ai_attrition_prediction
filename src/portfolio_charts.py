import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.data_contract import load_data


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


def chart_target_distribution(
    df,
    target_column,
):
    """
    Plot overall synthetic target counts.
    """

    counts = (
        df[
            target_column
        ]
        .value_counts()
        .sort_index()
    )

    labels = [
        "No Attrition",
        "Attrition",
    ]

    plt.figure(
        figsize=(7, 5)
    )

    bars = plt.bar(
        labels,
        [
            counts.get(
                0,
                0,
            ),
            counts.get(
                1,
                0,
            ),
        ],
    )

    plt.title(
        "Synthetic Attrition Target Distribution"
    )

    plt.ylabel(
        "Employee Snapshots"
    )

    for bar in bars:
        height = (
            bar.get_height()
        )

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
        )

    save_figure(
        CHART_DIR
        / "target_distribution.png"
    )


def chart_chronological_split(
    df,
    date_column,
    cfg,
):
    """
    Visualize monthly row counts by chronological split.
    """

    working = df[
        [
            date_column,
        ]
    ].copy()

    working[
        date_column
    ] = pd.to_datetime(
        working[
            date_column
        ]
    )

    working[
        "split"
    ] = "UNASSIGNED"

    working.loc[
        working[
            date_column
        ]
        <= pd.Timestamp(
            cfg[
                "train_end"
            ]
        ),
        "split",
    ] = "Train"

    working.loc[
        (
            working[
                date_column
            ]
            >= pd.Timestamp(
                cfg[
                    "validation_start"
                ]
            )
        )
        & (
            working[
                date_column
            ]
            <= pd.Timestamp(
                cfg[
                    "validation_end"
                ]
            )
        ),
        "split",
    ] = "Validation"

    working.loc[
        (
            working[
                date_column
            ]
            >= pd.Timestamp(
                cfg[
                    "test_start"
                ]
            )
        )
        & (
            working[
                date_column
            ]
            <= pd.Timestamp(
                cfg[
                    "test_end"
                ]
            )
        ),
        "split",
    ] = "Final Test"

    working[
        "month"
    ] = (
        working[
            date_column
        ]
        .dt
        .to_period(
            "M"
        )
        .dt
        .to_timestamp()
    )

    monthly = (
        working
        .groupby(
            [
                "month",
                "split",
            ],
            as_index=False,
        )
        .size()
    )

    plt.figure(
        figsize=(11, 5)
    )

    for split_name in [
        "Train",
        "Validation",
        "Final Test",
    ]:
        split_data = monthly[
            monthly[
                "split"
            ]
            == split_name
        ]

        plt.plot(
            split_data[
                "month"
            ],
            split_data[
                "size"
            ],
            marker="o",
            label=split_name,
        )

    plt.title(
        "Chronological Train / Validation / Final Test Design"
    )

    plt.xlabel(
        "Snapshot Month"
    )

    plt.ylabel(
        "Employee Snapshots"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    save_figure(
        CHART_DIR
        / "chronological_split.png"
    )


def chart_final_test_roc(
    predictions,
):
    """
    Plot final untouched-test ROC curve.
    """

    plt.figure(
        figsize=(7, 6)
    )

    RocCurveDisplay.from_predictions(
        predictions[
            "actual_target"
        ],
        predictions[
            "predicted_probability"
        ],
    )

    plt.title(
        "Final Test ROC Curve"
    )

    save_figure(
        CHART_DIR
        / "final_test_roc_curve.png"
    )


def chart_final_test_pr(
    predictions,
):
    """
    Plot final untouched-test precision-recall curve.
    """

    plt.figure(
        figsize=(7, 6)
    )

    PrecisionRecallDisplay.from_predictions(
        predictions[
            "actual_target"
        ],
        predictions[
            "predicted_probability"
        ],
    )

    plt.title(
        "Final Test Precision-Recall Curve"
    )

    save_figure(
        CHART_DIR
        / "final_test_pr_curve.png"
    )


def chart_confusion_matrix(
    confusion,
):
    """
    Plot final-test confusion matrix from the saved artifact.
    """

    lookup = {
        (
            int(
                row[
                    "actual"
                ]
            ),
            int(
                row[
                    "predicted"
                ]
            ),
        ): int(
            row[
                "count"
            ]
        )
        for _, row
        in confusion.iterrows()
    }

    matrix = [
        [
            lookup.get(
                (
                    0,
                    0,
                ),
                0,
            ),
            lookup.get(
                (
                    0,
                    1,
                ),
                0,
            ),
        ],
        [
            lookup.get(
                (
                    1,
                    0,
                ),
                0,
            ),
            lookup.get(
                (
                    1,
                    1,
                ),
                0,
            ),
        ],
    ]

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        "Final Test Confusion Matrix"
    )

    plt.xticks(
        [
            0,
            1,
        ],
        [
            "Predicted No",
            "Predicted Yes",
        ],
    )

    plt.yticks(
        [
            0,
            1,
        ],
        [
            "Actual No",
            "Actual Yes",
        ],
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    for row_index in range(
        2
    ):
        for column_index in range(
            2
        ):
            plt.text(
                column_index,
                row_index,
                f"{matrix[row_index][column_index]:,}",
                ha="center",
                va="center",
            )

    save_figure(
        CHART_DIR
        / "final_test_confusion_matrix.png"
    )


def chart_risk_deciles(
    risk_deciles,
):
    """
    Plot observed attrition rate by final-test risk decile.

    Band 1 is the highest predicted-risk group.
    """

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        risk_deciles[
            "risk_band"
        ].astype(
            str
        ),
        risk_deciles[
            "observed_attrition_rate"
        ],
    )

    plt.title(
        "Final Test Attrition Rate by Predicted Risk Decile"
    )

    plt.xlabel(
        "Risk Decile (1 = Highest Predicted Risk)"
    )

    plt.ylabel(
        "Observed Attrition Rate"
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: (
                f"{value:.0%}"
            )
        )
    )

    save_figure(
        CHART_DIR
        / "final_test_risk_deciles.png"
    )


def main():
    # ---------------------------------------------------------
    # Load existing deterministic project data/artifacts.
    #
    # No model fitting occurs in this script.
    # ---------------------------------------------------------
    with open(
        "config/project_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)

    df = load_data()

    predictions = pd.read_csv(
        "outputs/"
        "final_test_predictions.csv"
    )

    confusion = pd.read_csv(
        "outputs/"
        "final_test_confusion_matrix.csv"
    )

    risk_deciles = pd.read_csv(
        "outputs/"
        "final_test_risk_deciles.csv"
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Generate charts
    # ---------------------------------------------------------
    chart_target_distribution(
        df=df,
        target_column=cfg[
            "target"
        ],
    )

    chart_chronological_split(
        df=df,
        date_column=cfg[
            "date_column"
        ],
        cfg=cfg,
    )

    chart_final_test_roc(
        predictions=predictions,
    )

    chart_final_test_pr(
        predictions=predictions,
    )

    chart_confusion_matrix(
        confusion=confusion,
    )

    chart_risk_deciles(
        risk_deciles=risk_deciles,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Core Portfolio Charts"
    )
    print(
        "=" * 33
    )

    print(
        "No model training was performed."
    )

    print(
        "No model or threshold decisions were changed."
    )

    print()

    chart_files = [
        "target_distribution.png",
        "chronological_split.png",
        "final_test_roc_curve.png",
        "final_test_pr_curve.png",
        "final_test_confusion_matrix.png",
        "final_test_risk_deciles.png",
    ]

    for filename in chart_files:
        print(
            CHART_DIR
            / filename
        )


if __name__ == "__main__":
    main()