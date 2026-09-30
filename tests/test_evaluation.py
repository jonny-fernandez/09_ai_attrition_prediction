import pytest

from src.evaluation import evaluate_binary_classifier


def test_evaluation_metrics_for_perfect_classification():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_probability = [
        0.10,
        0.20,
        0.80,
        0.90,
    ]

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        y_probability=y_probability,
        threshold=0.50,
    )

    assert metrics["row_count"] == 4

    assert metrics[
        "target_prevalence"
    ] == pytest.approx(0.50)

    assert metrics[
        "roc_auc"
    ] == pytest.approx(1.0)

    assert metrics[
        "pr_auc"
    ] == pytest.approx(1.0)

    assert metrics[
        "brier_score"
    ] == pytest.approx(0.025)

    assert metrics[
        "precision"
    ] == pytest.approx(1.0)

    assert metrics[
        "recall"
    ] == pytest.approx(1.0)

    assert metrics[
        "f1"
    ] == pytest.approx(1.0)

    assert metrics[
        "true_negative"
    ] == 2

    assert metrics[
        "false_positive"
    ] == 0

    assert metrics[
        "false_negative"
    ] == 0

    assert metrics[
        "true_positive"
    ] == 2


def test_threshold_is_applied_without_being_optimized():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_probability = [
        0.10,
        0.20,
        0.80,
        0.90,
    ]

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        y_probability=y_probability,
        threshold=0.85,
    )

    assert metrics[
        "threshold"
    ] == pytest.approx(0.85)

    assert metrics[
        "predicted_positive_count"
    ] == 1

    assert metrics[
        "predicted_positive_rate"
    ] == pytest.approx(0.25)

    assert metrics[
        "precision"
    ] == pytest.approx(1.0)

    assert metrics[
        "recall"
    ] == pytest.approx(0.50)

    assert metrics[
        "false_negative"
    ] == 1

    assert metrics[
        "true_positive"
    ] == 1


def test_evaluation_rejects_invalid_probabilities():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_binary_classifier(
            y_true=[
                0,
                1,
            ],
            y_probability=[
                0.20,
                1.20,
            ],
            threshold=0.50,
        )


def test_evaluation_rejects_non_binary_targets():
    with pytest.raises(
        ValueError,
        match="binary values",
    ):
        evaluate_binary_classifier(
            y_true=[
                0,
                2,
                1,
            ],
            y_probability=[
                0.10,
                0.50,
                0.90,
            ],
            threshold=0.50,
        )