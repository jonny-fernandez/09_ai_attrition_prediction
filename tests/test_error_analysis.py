import pandas as pd
import pytest

from src.error_analysis import (
    build_error_analysis_table,
    get_representative_errors,
    summarize_error_analysis,
)


def make_test_frame():
    return pd.DataFrame(
        {
            "employee_id": [
                "E1",
                "E2",
                "E3",
                "E4",
            ],
            "snapshot_date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
            ],
            "department": [
                "A",
                "A",
                "B",
                "B",
            ],
            "target": [
                0,
                0,
                1,
                1,
            ],
        }
    )


def test_error_analysis_assigns_all_four_categories():
    frame = make_test_frame()

    result = build_error_analysis_table(
        source_frame=frame,
        target_column="target",
        predicted_probability=[
            0.10,
            0.80,
            0.20,
            0.90,
        ],
        threshold=0.50,
        id_column="employee_id",
        date_column="snapshot_date",
        context_columns=[
            "department",
        ],
    )

    assert list(
        result["error_category"]
    ) == [
        "TRUE_NEGATIVE",
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
        "TRUE_POSITIVE",
    ]


def test_error_summary_counts_are_correct():
    frame = make_test_frame()

    result = build_error_analysis_table(
        source_frame=frame,
        target_column="target",
        predicted_probability=[
            0.10,
            0.80,
            0.20,
            0.90,
        ],
        threshold=0.50,
        id_column="employee_id",
        date_column="snapshot_date",
    )

    summary = summarize_error_analysis(
        result
    )

    assert summary["row_count"] == 4
    assert summary["true_positive"] == 1
    assert summary["true_negative"] == 1
    assert summary["false_positive"] == 1
    assert summary["false_negative"] == 1
    assert summary["error_count"] == 2


def test_error_analysis_rejects_invalid_probability():
    frame = make_test_frame()

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        build_error_analysis_table(
            source_frame=frame,
            target_column="target",
            predicted_probability=[
                0.10,
                0.20,
                0.30,
                1.20,
            ],
            threshold=0.50,
            id_column="employee_id",
            date_column="snapshot_date",
        )


def test_representative_errors_are_sorted_correctly():
    frame = make_test_frame()

    result = build_error_analysis_table(
        source_frame=frame,
        target_column="target",
        predicted_probability=[
            0.10,
            0.80,
            0.20,
            0.90,
        ],
        threshold=0.50,
        id_column="employee_id",
        date_column="snapshot_date",
    )

    examples = get_representative_errors(
        result,
        n=1,
    )

    assert (
        examples[
            "high_confidence_false_positives"
        ].iloc[0]["employee_id"]
        == "E2"
    )

    assert (
        examples[
            "high_confidence_false_negatives"
        ].iloc[0]["employee_id"]
        == "E3"
    )