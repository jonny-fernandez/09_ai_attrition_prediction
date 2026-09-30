import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.feature_importance import (
    calculate_permutation_importance,
)


def make_feature_importance_data():
    X = pd.DataFrame(
        {
            "signal": (
                [0] * 50
                + [1] * 50
            ),
            "noise": [
                index % 2
                for index in range(100)
            ],
        }
    )

    y = np.array(
        [0] * 50
        + [1] * 50
    )

    return X, y


def test_permutation_importance_returns_all_features():
    X, y = make_feature_importance_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        y,
    )

    result = calculate_permutation_importance(
        model=model,
        X=X,
        y=y,
        feature_names=X.columns,
        random_seed=42,
        n_repeats=5,
    )

    assert len(result) == 2

    assert set(
        result["feature"]
    ) == {
        "signal",
        "noise",
    }

    assert list(
        result["rank"]
    ) == [
        1,
        2,
    ]


def test_signal_feature_ranks_above_noise():
    X, y = make_feature_importance_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        y,
    )

    result = calculate_permutation_importance(
        model=model,
        X=X,
        y=y,
        feature_names=X.columns,
        random_seed=42,
        n_repeats=5,
    )

    assert (
        result.iloc[0][
            "feature"
        ]
        == "signal"
    )


def test_feature_importance_rejects_wrong_feature_names():
    X, y = make_feature_importance_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        y,
    )

    with pytest.raises(
        ValueError,
        match="number of columns",
    ):
        calculate_permutation_importance(
            model=model,
            X=X,
            y=y,
            feature_names=[
                "signal",
            ],
            random_seed=42,
            n_repeats=5,
        )