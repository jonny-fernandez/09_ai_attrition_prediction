# AI Interpretation of Verified Model Results

> AI-generated explanation based only on verified deterministic analytical outputs. The AI did not generate predictions, metrics, or thresholds.

API model: `gpt-6-astra`

## Executive Summary

The Gradient Boosting model uses 19 features for a 90-day attrition modeling task on synthetic workforce data only. The final test results show modest discrimination and enrichment of observed attrition cases among higher-ranked records, alongside substantial classification errors at the operating threshold of 0.125. The test set was not used for model selection. These results describe synthetic-data model behavior, not real-world workforce outcomes.

## Performance Interpretation

On 507 test records, ROC-AUC was 0.6156236323851203 and PR-AUC was 0.19942038019456926, with target prevalence of 0.09861932938856016. At the supplied threshold, precision was 0.16101694915254236 and recall was 0.38. The confusion matrix contained 19 true positives, 358 true negatives, 99 false positives, and 31 false negatives. Thus, false positives outnumbered true positives, and many observed attrition cases were missed. The Brier score was 0.08718630417497854; the supplied evidence is insufficient to establish calibration quality.

## Ranking Interpretation

For the requested top-population fractions of 0.05, 0.1, and 0.2, the supplied lifts were 1.95, 1.9882352941176469, and 1.69, respectively. Their attrition capture rates were 0.1, 0.2, and 0.34. These results indicate enrichment relative to overall observed attrition prevalence in the synthetic test set, but the ranked subsets did not capture all observed cases. Actual selected population fractions differ from the requested fractions, as documented in the supplied outputs.

## Model Driver Notes

- manager_changes_12m ranked first in both permutation importance and mean absolute SHAP importance. tenure_months ranked second in permutation importance and third in the SHAP summary.
- engagement_score ranked second in the SHAP summary but fifth in permutation importance. These summaries describe different aspects of model behavior: attribution magnitude and sensitivity of measured performance to feature permutation.
- For several listed permutation importances, including engagement_score and commute_minutes, the reported standard deviation exceeded the mean. This cautions against treating their precise importance ordering as firmly established.
- The supplied aggregate SHAP summaries do not establish how increasing a feature changes individual model scores. Feature importance and attribution describe model associations, not causes of employee attrition.

## Subgroup Diagnostic Notes

- The supplied highest-recall list includes Development, Programs, and L2. Development had recall of 0.8 across 61 records. The lowest-recall list includes People, Program Delivery, and Part-Time; People had recall of 0.0 and false-negative rate of 1.0 across 68 records.
- The Data job family appears first in the supplied highest-false-positive-rate list, with a false-positive rate of 0.3188405797101449 across 79 records. L2 and People also appear in that list.
- These differences are model-performance diagnostics across synthetic subgroups only. The lists span different grouping dimensions and do not establish independent populations, real-world group characteristics, or fairness or unfairness.

## Limitations

- Synthetic data alone are insufficient evidence of real-world validity or deployment readiness.
- No confidence intervals or repeated-evaluation results are supplied, so the stability of overall and subgroup performance is not established.
- Only selected feature summaries and subgroup extremes are provided; the evidence is insufficient for a complete feature-level or subgroup assessment.
- No alternative-threshold comparison is supplied, so the evidence is insufficient to conclude that the operating threshold is optimal.

## Governance Note

This explanation is limited to the supplied verified analytical outputs. It does not generate predictions, change metrics or thresholds, or recommend employment actions. Model associations must remain explicitly separate from real-world causation, and subgroup diagnostics must not be treated as proof of fairness or unfairness.
