import pytest

from src.calibration import calibration_diagnostics


def test_calibration_preserves_all_rows():
    y_true = [
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
    ]

    y_probability = [
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.25,
        0.30,
        0.45,
        0.60,
    ]

    result = calibration_diagnostics(
        y_true=y_true,
        y_probability=y_probability,
        n_bins=5,
    )

    table = result[
        "calibration_table"
    ]

    assert result["row_count"] == 10

    assert result["bin_count"] == 5

    assert (
        table["record_count"].sum()
        == 10
    )


def test_calibration_summary_values_are_valid():
    y_true = [
        0,
        0,
        0,
        1,
        1,
        1,
    ]

    y_probability = [
        0.05,
        0.10,
        0.20,
        0.60,
        0.75,
        0.90,
    ]

    result = calibration_diagnostics(
        y_true=y_true,
        y_probability=y_probability,
        n_bins=3,
    )

    assert (
        result["target_prevalence"]
        == pytest.approx(0.5)
    )

    assert (
        0
        <= result["brier_score"]
        <= 1
    )

    assert (
        result[
            "weighted_absolute_calibration_error"
        ]
        >= 0
    )


def test_calibration_rejects_invalid_probability():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        calibration_diagnostics(
            y_true=[
                0,
                1,
            ],
            y_probability=[
                0.10,
                1.10,
            ],
            n_bins=2,
        )