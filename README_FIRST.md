# Project 9 - Start Here

## AI Attrition Prediction & Explainability

This V2 starter is intentionally ready for **inspection first**. You should not need to redesign the supplied dataset or repair starter files before beginning.

### First session

1. Open this project folder in VS Code.
2. Read `config/project_config.json`.
3. Inspect `data/reference/data_dictionary.csv` and the policy/reference files.
4. Run the dataset inspection command below.
5. Run the starter tests.
6. Stop and review the results before advancing to modeling.

```powershell
python -m pytest tests -q
python -m src.inspect_data
```

### Supplied dataset

3,000 synthetic employee snapshots with a future 90-day attrition target, multiple categorical and numeric predictors, and deliberately included post-outcome leakage fields documented in `feature_policy.csv`.

### Important

- Synthetic data only.
- Do not use the OpenAI API during the first checkpoint.
- Do not replace the supplied files simply to reorganize them.
- Do not treat the starter baseline as the final model.
- The optional AI layer comes only after the deterministic/data-science pipeline is validated.
