from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"employee_id","snapshot_date","attrition_within_90_days","department","job_family","job_level","location","engagement_score","tenure_months"}
PROHIBITED_FEATURES = {"employee_id","termination_date","termination_reason","exit_interview_complete","final_pay_indicator","attrition_within_90_days","snapshot_date"}

def load_data(path="data/raw/employee_attrition_snapshots_synthetic.csv"):
    df=pd.read_csv(Path(path), parse_dates=["snapshot_date"])
    validate_data(df)
    return df

def validate_data(df):
    missing=REQUIRED_COLUMNS-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty: raise ValueError("Dataset is empty")
    if df["employee_id"].duplicated().any(): raise ValueError("employee_id must be unique in this starter dataset")
    if not set(df["attrition_within_90_days"].dropna().unique()).issubset({0,1}): raise ValueError("Target must be binary 0/1")
    if not df["engagement_score"].between(0,100).all(): raise ValueError("engagement_score must be 0-100")
    return True
