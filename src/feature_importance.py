import pandas as pd
from sklearn.inspection import permutation_importance


def calculate_permutation_importance(
    model,
    X,
    y,
    feature_names,
    random_seed,
    n_repeats=20,
    scoring="average_precision",
):
    """
    Calculate model-agnostic permutation feature importance.

    Importance represents the average decrease in the supplied
    validation scoring metric after one feature is shuffled.

    Parameters
    ----------
    model:
        Already-fitted sklearn-compatible model or pipeline.

    X:
        Evaluation features.

    y:
        Observed target.

    feature_names:
        Original feature names corresponding to X.

    random_seed:
        Reproducibility seed.

    n_repeats:
        Number of independent permutations per feature.

    scoring:
        sklearn scoring metric.

    Returns
    -------
    pandas.DataFrame
        Feature-level permutation importance results.

    Notes
    -----
    Permutation importance explains predictive dependence.

    It does NOT establish:
    - causation,
    - employment-policy relevance,
    - or whether changing a feature would change attrition.
    """

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    if len(X) == 0:
        raise ValueError(
            "X and y cannot be empty."
        )

    if n_repeats < 1:
        raise ValueError(
            "n_repeats must be at least 1."
        )

    feature_names = list(
        feature_names
    )

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "feature_names must match the number "
            "of columns in X."
        )

    if hasattr(
        X,
        "columns",
    ):
        actual_columns = list(
            X.columns
        )

        if actual_columns != feature_names:
            raise ValueError(
                "feature_names must match X columns "
                "in the same order."
            )

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_seed,
        n_jobs=1,
    )

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": (
                result.importances_mean
            ),
            "importance_std": (
                result.importances_std
            ),
        }
    )

    table = (
        table
        .sort_values(
            by=[
                "importance_mean",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    table.insert(
        0,
        "rank",
        range(
            1,
            len(table) + 1,
        ),
    )

    return table