import pytest

from src.risk_analysis import (
    analyze_risk_bands,
    analyze_top_risk_groups,
)


def test_risk_bands_preserve_all_rows():
    y_true = [
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]

    y_probability = [
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
        0.05,
    ]

    table = analyze_risk_bands(
        y_true=y_true,
        y_probability=y_probability,
        n_bands=5,
    )

    assert (
        table[
            "record_count"
        ].sum()
        == 10
    )

    assert (
        table[
            "attrition_cases"
        ].sum()
        == 3
    )


def test_highest_risk_band_is_band_one():
    table = analyze_risk_bands(
        y_true=[
            1,
            1,
            0,
            0,
        ],
        y_probability=[
            0.90,
            0.80,
            0.20,
            0.10,
        ],
        n_bands=2,
    )

    highest = table.iloc[0]

    assert (
        highest["risk_band"]
        == 1
    )

    assert (
        highest[
            "observed_attrition_rate"
        ]
        == pytest.approx(1.0)
    )


def test_top_half_captures_all_positive_cases_when_ranking_is_perfect():
    result = analyze_top_risk_groups(
        y_true=[
            1,
            1,
            0,
            0,
        ],
        y_probability=[
            0.90,
            0.80,
            0.20,
            0.10,
        ],
        percentages=[
            0.50,
        ],
    )

    row = result.iloc[0]

    assert (
        row[
            "record_count"
        ]
        == 2
    )

    assert (
        row[
            "attrition_cases"
        ]
        == 2
    )

    assert (
        row[
            "attrition_capture_rate"
        ]
        == pytest.approx(1.0)
    )

    assert (
        row[
            "lift"
        ]
        == pytest.approx(2.0)
    )


def test_risk_analysis_rejects_invalid_probability():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        analyze_risk_bands(
            y_true=[
                0,
                1,
            ],
            y_probability=[
                0.20,
                1.10,
            ],
            n_bands=2,
        )