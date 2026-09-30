import pandas as pd
import pytest

from src.subgroup_analysis import (
    calculate_multiple_subgroups,
    calculate_subgroup_metrics,
)


def make_subgroup_frame():
    return pd.DataFrame(
        {
            "department": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
            ],
            "location": [
                "X",
                "X",
                "Y",
                "Y",
                "X",
                "X",
                "Y",
                "Y",
            ],
            "target": [
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )


def test_subgroup_metrics_preserve_eligible_groups():
    frame = make_subgroup_frame()

    result = calculate_subgroup_metrics(
        source_frame=frame,
        group_column="department",
        target_column="target",
        predicted_probability=[
            0.10,
            0.20,
            0.80,
            0.90,
            0.10,
            0.70,
            0.30,
            0.60,
        ],
        threshold=0.50,
        minimum_group_size=2,
    )

    assert len(result) == 2

    assert set(
        result["group_value"]
    ) == {
        "A",
        "B",
    }


def test_subgroup_metrics_respect_minimum_size():
    frame = make_subgroup_frame()

    result = calculate_subgroup_metrics(
        source_frame=frame,
        group_column="location",
        target_column="target",
        predicted_probability=[
            0.10,
            0.20,
            0.80,
            0.90,
            0.10,
            0.70,
            0.30,
            0.60,
        ],
        threshold=0.50,
        minimum_group_size=5,
    )

    assert result.empty


def test_subgroup_metrics_calculate_expected_counts():
    frame = make_subgroup_frame()

    result = calculate_subgroup_metrics(
        source_frame=frame,
        group_column="department",
        target_column="target",
        predicted_probability=[
            0.10,
            0.20,
            0.80,
            0.90,
            0.10,
            0.70,
            0.30,
            0.60,
        ],
        threshold=0.50,
        minimum_group_size=2,
    )

    department_a = result[
        result["group_value"]
        == "A"
    ].iloc[0]

    assert (
        department_a[
            "record_count"
        ]
        == 4
    )

    assert (
        department_a[
            "attrition_prevalence"
        ]
        == pytest.approx(0.50)
    )

    assert (
        department_a[
            "precision"
        ]
        == pytest.approx(1.0)
    )

    assert (
        department_a[
            "recall"
        ]
        == pytest.approx(1.0)
    )


def test_multiple_subgroups_returns_group_identifiers():
    frame = make_subgroup_frame()

    result = calculate_multiple_subgroups(
        source_frame=frame,
        group_columns=[
            "department",
            "location",
        ],
        target_column="target",
        predicted_probability=[
            0.10,
            0.20,
            0.80,
            0.90,
            0.10,
            0.70,
            0.30,
            0.60,
        ],
        threshold=0.50,
        minimum_group_size=2,
    )

    assert set(
        result["group_column"]
    ) == {
        "department",
        "location",
    }