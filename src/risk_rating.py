# src/risk_rating.py

import pandas as pd

def compute_customer_risk(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer risk scores using amount velocity and behavior profiles.
    """
    df = tx.copy()

    # Aggregate per customer
    agg = df.groupby("customer_id").agg(
        total_volume=("amount", "sum"),
        txn_count=("amount", "count"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
    ).reset_index()

    # Compute risk scoring
    agg["volume_risk"] = agg["total_volume"] / (agg["total_volume"].max() + 1)
    agg["txn_freq_risk"] = agg["txn_count"] / (agg["txn_count"].max() + 1)
    agg["high_value_risk"] = agg["max_amount"] / (agg["max_amount"].max() + 1)

    agg["composite_risk"] = (
        0.4 * agg["volume_risk"] +
        0.3 * agg["txn_freq_risk"] +
        0.3 * agg["high_value_risk"]
    )

    agg["risk_band"] = agg["composite_risk"].apply(
        lambda x: "High" if x >= 0.7 else ("Medium" if x >= 0.4 else "Low")
    )

    return agg
