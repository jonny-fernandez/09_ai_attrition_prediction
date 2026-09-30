import pandas as pd

from src.model_report import (
    generate_model_report,
    save_model_report,
)


def make_report_inputs():
    metadata = {
        "model_label": "Gradient Boosting",
        "model_name": "GradientBoostingClassifier",
        "prediction_horizon_days": 90,
        "target": "attrition_within_90_days",
        "feature_count": 2,
        "operating_threshold": 0.125,
        "random_seed": 42,
        "test_used_for_model_selection": False,
        "splits": {
            "train": {
                "start": "2025-01-01",
                "end": "2025-12-31",
                "rows": 100,
            },
            "validation": {
                "start": "2026-01-01",
                "end": "2026-03-31",
                "rows": 20,
            },
            "test": {
                "start": "2026-04-01",
                "end": "2026-06-30",
                "rows": 20,
            },
        },
        "model_file": "model.joblib",
        "model_sha256": "abc123",
        "dataset_file": "data.csv",
        "library_versions": {
            "python": "3.10",
            "scikit_learn": "1.5",
            "pandas": "2.2",
            "numpy": "2.0",
        },
    }

    model_lock = {
        "threshold_selected_on": "validation",
        "calibration_method": None,
        "hyperparameter_tuning_performed": False,
    }

    final_metrics = {
        "target_prevalence": 0.10,
        "roc_auc": 0.65,
        "pr_auc": 0.20,
        "brier_score": 0.09,
        "precision": 0.20,
        "recall": 0.40,
        "f1": 0.27,
        "predicted_positive_rate": 0.20,
        "true_negative": 15,
        "false_positive": 3,
        "false_negative": 1,
        "true_positive": 1,
    }

    top_risk = pd.DataFrame(
        [
            {
                "requested_top_percentage": 0.10,
                "actual_population_rate": 0.10,
                "attrition_capture_rate": 0.25,
                "observed_attrition_rate": 0.25,
                "lift": 2.5,
            }
        ]
    )

    permutation = pd.DataFrame(
        [
            {
                "rank": 1,
                "feature": "feature_a",
                "importance_mean": 0.05,
            }
        ]
    )

    shap_summary = pd.DataFrame(
        [
            {
                "rank": 1,
                "feature": "feature_a",
                "mean_abs_shap": 0.5,
                "importance_share": 0.7,
            }
        ]
    )

    subgroup = pd.DataFrame(
        [
            {
                "group_column": "department",
                "group_value": "A",
                "record_count": 50,
                "attrition_prevalence": 0.10,
                "predicted_positive_rate": 0.20,
                "roc_auc": 0.65,
                "recall": 0.40,
                "false_positive_rate": 0.15,
                "false_negative_rate": 0.60,
            }
        ]
    )

    errors = pd.DataFrame(
        {
            "error_category": [
                "TRUE_POSITIVE",
                "TRUE_NEGATIVE",
                "FALSE_POSITIVE",
                "FALSE_NEGATIVE",
            ]
        }
    )

    return (
        metadata,
        model_lock,
        final_metrics,
        top_risk,
        permutation,
        shap_summary,
        subgroup,
        errors,
    )


def test_model_report_contains_core_sections():
    inputs = make_report_inputs()

    report = generate_model_report(
        metadata=inputs[0],
        model_lock=inputs[1],
        final_metrics=inputs[2],
        top_risk=inputs[3],
        permutation_importance=inputs[4],
        shap_summary=inputs[5],
        subgroup_metrics=inputs[6],
        error_analysis=inputs[7],
    )

    assert (
        "Final Untouched Test Performance"
        in report
    )

    assert (
        "SHAP Explainability"
        in report
    )

    assert (
        "Synthetic Subgroup Diagnostics"
        in report
    )

    assert (
        "AI Boundary"
        in report
    )


def test_model_report_identifies_synthetic_data():
    inputs = make_report_inputs()

    report = generate_model_report(
        metadata=inputs[0],
        model_lock=inputs[1],
        final_metrics=inputs[2],
        top_risk=inputs[3],
        permutation_importance=inputs[4],
        shap_summary=inputs[5],
        subgroup_metrics=inputs[6],
        error_analysis=inputs[7],
    )

    assert (
        "synthetic workforce data only"
        in report
    )


def test_model_report_can_be_saved(
    tmp_path,
):
    output = (
        tmp_path
        / "model_report.md"
    )

    save_model_report(
        report="# Test Report",
        output_path=output,
    )

    assert output.exists()

    assert (
        output.read_text(
            encoding="utf-8"
        )
        == "# Test Report"
    )