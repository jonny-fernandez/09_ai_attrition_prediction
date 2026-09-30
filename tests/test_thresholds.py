import pandas as pd
import pytest

from src.thresholds import (
    analyze_thresholds,
    select_threshold_by_f1,
)


def test_threshold_analysis_returns_expected_rows():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_probability = [
        0.10,
        0.20,
        0.60,
        0.90,
    ]

    result = analyze_thresholds(
        y_true=y_true,
        y_probability=y_probability,
        thresholds=[
            0.25,
            0.50,
            0.75,
        ],
    )

    assert len(result) == 3


def test_threshold_metrics_are_correct():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_probability = [
        0.10,
        0.40,
        0.60,
        0.90,
    ]

    result = analyze_thresholds(
        y_true=y_true,
        y_probability=y_probability,
        thresholds=[
            0.50,
        ],
    )

    row = result.iloc[0]

    assert (
        row["predicted_positive_count"]
        == 2
    )

    assert (
        row["predicted_positive_rate"]
        == pytest.approx(0.50)
    )

    assert (
        row["precision"]
        == pytest.approx(1.0)
    )

    assert (
        row["recall"]
        == pytest.approx(1.0)
    )

    assert (
        row["true_positive"]
        == 2
    )

    assert (
        row["false_positive"]
        == 0
    )

    assert (
        row["false_negative"]
        == 0
    )

    assert (
        row["attrition_capture_rate"]
        == pytest.approx(1.0)
    )


def test_higher_threshold_changes_recall():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_probability = [
        0.10,
        0.20,
        0.60,
        0.90,
    ]

    result = analyze_thresholds(
        y_true=y_true,
        y_probability=y_probability,
        thresholds=[
            0.80,
        ],
    )

    row = result.iloc[0]

    assert (
        row["true_positive"]
        == 1
    )

    assert (
        row["false_negative"]
        == 1
    )

    assert (
        row["recall"]
        == pytest.approx(0.50)
    )


def test_threshold_analysis_rejects_invalid_threshold():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        analyze_thresholds(
            y_true=[
                0,
                1,
            ],
            y_probability=[
                0.20,
                0.80,
            ],
            thresholds=[
                1.20,
            ],
        )


def test_select_threshold_by_f1_returns_highest_f1():
    table = analyze_thresholds(
        y_true=[
            0,
            0,
            1,
            1,
        ],
        y_probability=[
            0.10,
            0.40,
            0.60,
            0.90,
        ],
        thresholds=[
            0.30,
            0.50,
            0.80,
        ],
    )

    selected = select_threshold_by_f1(
        table
    )

    assert (
        selected["threshold"]
        == pytest.approx(0.50)
    )

    assert (
        selected["f1"]
        == pytest.approx(1.0)
    )


def test_threshold_selector_tie_breaks_deterministically():
    table = pd.DataFrame(
        [
            {
                "threshold": 0.10,
                "precision": 0.20,
                "recall": 0.50,
                "f1": 0.30,
            },
            {
                "threshold": 0.20,
                "precision": 0.25,
                "recall": 0.40,
                "f1": 0.30,
            },
        ]
    )

    selected = select_threshold_by_f1(
        table
    )

    assert (
        selected["threshold"]
        == pytest.approx(0.20)
    )


def test_threshold_selector_rejects_empty_table():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        select_threshold_by_f1(
            pd.DataFrame(
                columns=[
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                ]
            )
        )