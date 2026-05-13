# src/rules_engine.py
import pandas as pd
from typing import Tuple

RULES_CONFIG = {
    "high_value_amount": 50000.0,   # tune as needed
    "daily_tx_count_threshold": 10,
    "weight_ml": 0.7,
    "weight_rules": 0.3
}

def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # high value rule
    df["rule_high_value"] = (df.get("amount", 0.0) >= RULES_CONFIG["high_value_amount"]).astype(int)

    # daily counts (if date present)
    if "date" in df.columns:
        tmp = df.groupby(["user_id", "date"]).size().rename("daily_tx_count").reset_index()
        df = df.merge(tmp, on=["user_id", "date"], how="left")
    else:
        df["daily_tx_count"] = 0

    df["rule_daily_count"] = (df["daily_tx_count"] >= RULES_CONFIG["daily_tx_count_threshold"]).astype(int)

    df["rule_score"] = df["rule_high_value"] + df["rule_daily_count"]
    return df

def combine_risk(df: pd.DataFrame, ml_col: str = "fraud_proba") -> pd.DataFrame:
    df = df.copy()
    df["rule_score_norm"] = 0.0
    if df["rule_score"].max() > 0:
        df["rule_score_norm"] = df["rule_score"] / df["rule_score"].max()
    w_ml = RULES_CONFIG["weight_ml"]
    w_rules = RULES_CONFIG["weight_rules"]
    df["final_risk"] = w_ml * df[ml_col].fillna(0) + w_rules * df["rule_score_norm"]
    return df
