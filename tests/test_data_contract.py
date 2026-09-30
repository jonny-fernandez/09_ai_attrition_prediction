import pandas as pd
from src.data_contract import load_data, PROHIBITED_FEATURES

def test_dataset_contract():
    df=load_data(); assert len(df)>=2500; assert df.employee_id.is_unique; assert set(df.attrition_within_90_days.unique()) <= {0,1}

def test_target_has_both_classes():
    df=load_data(); assert df.attrition_within_90_days.nunique()==2; assert df.attrition_within_90_days.mean() > 0.05

def test_leakage_policy_present():
    policy=pd.read_csv("data/reference/feature_policy.csv").set_index("column")
    for c in ["termination_date","termination_reason","exit_interview_complete","final_pay_indicator"]: assert policy.loc[c,"policy"]=="EXCLUDED_LEAKAGE"

def test_prohibited_features_are_known():
    assert "termination_date" in PROHIBITED_FEATURES; assert "attrition_within_90_days" in PROHIBITED_FEATURES

def test_time_windows_do_not_overlap():
    df=load_data(); train=df[df.snapshot_date <= "2025-12-31"]; test=df[df.snapshot_date >= "2026-04-01"]; assert train.snapshot_date.max() < test.snapshot_date.min()
