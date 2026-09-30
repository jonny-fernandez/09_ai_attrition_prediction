import json

import pandas as pd

from src.model_report import (
    generate_model_report,
    save_model_report,
)


def main():
    # ---------------------------------------------------------
    # Load only existing verified artifacts.
    #
    # This script does not retrain or rescore the model.
    # ---------------------------------------------------------
    with open(
        "artifacts/model_metadata.json",
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    with open(
        "artifacts/development_model_lock.json",
        "r",
        encoding="utf-8",
    ) as f:
        model_lock = json.load(f)

    with open(
        "outputs/final_test_metrics.json",
        "r",
        encoding="utf-8",
    ) as f:
        final_metrics = json.load(f)

    top_risk = pd.read_csv(
        "outputs/final_test_top_risk.csv"
    )

    permutation_importance = pd.read_csv(
        "outputs/"
        "validation_permutation_importance.csv"
    )

    shap_summary = pd.read_csv(
        "outputs/"
        "validation_shap_summary.csv"
    )

    subgroup_metrics = pd.read_csv(
        "outputs/"
        "validation_subgroup_metrics.csv"
    )

    error_analysis = pd.read_csv(
        "outputs/"
        "validation_error_analysis.csv"
    )

    # ---------------------------------------------------------
    # Build deterministic Markdown report
    # ---------------------------------------------------------
    report = generate_model_report(
        metadata=metadata,
        model_lock=model_lock,
        final_metrics=final_metrics,
        top_risk=top_risk,
        permutation_importance=permutation_importance,
        shap_summary=shap_summary,
        subgroup_metrics=subgroup_metrics,
        error_analysis=error_analysis,
    )

    output_path = (
        "outputs/model_report.md"
    )

    save_model_report(
        report=report,
        output_path=output_path,
    )

    # ---------------------------------------------------------
    # Report generation summary
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Deterministic Model Report"
    )
    print(
        "=" * 38
    )

    print(
        "No model training was performed."
    )

    print(
        "No final-test rescoring was performed."
    )

    print(
        "No AI was used."
    )

    print()

    print(
        f"Model:      "
        f"{metadata['model_label']}"
    )

    print(
        f"Threshold:  "
        f"{metadata['operating_threshold']:.3f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{final_metrics['roc_auc']:.3f}"
    )

    print(
        f"Test PR-AUC:  "
        f"{final_metrics['pr_auc']:.3f}"
    )

    print(
        f"Test F1:      "
        f"{final_metrics['f1']:.3f}"
    )

    print()

    print(
        f"Saved report: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()