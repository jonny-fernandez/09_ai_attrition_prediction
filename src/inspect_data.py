import json
import pandas as pd
from src.data_contract import load_data

def main():
    df = load_data()
    cfg = json.load(open("config/project_config.json", encoding="utf-8"))
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Snapshot range:", df.snapshot_date.min().date(), "to", df.snapshot_date.max().date())
    print("Target rate:", f"{df[cfg['target']].mean():.1%}")
    print("Target counts:")
    print(df[cfg["target"]].value_counts().sort_index().to_string())
    print("\nDepartment counts:")
    print(df.department.value_counts().to_string())
    policy = pd.read_csv("data/reference/feature_policy.csv")
    print("\nFeature policy:")
    print(policy.policy.value_counts().to_string())

if __name__ == "__main__":
    main()
