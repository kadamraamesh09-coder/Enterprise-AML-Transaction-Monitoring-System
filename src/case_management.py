# src/case_management.py

import pandas as pd

def create_cases(alerts_df: pd.DataFrame, risk_df: pd.DataFrame):
    """
    Combine alerts and customer risk to create AML cases.
    """

    df = alerts_df.copy()

    # Ensure required columns exist
    required = ["customer_id", "txn_id"]
    for col in required:
        if col not in df.columns:
            df[col] = None   # fallback if missing

    # Merge risk band
    df = df.merge(
        risk_df[["customer_id", "risk_band"]],
        on="customer_id",
        how="left"
    )

    # Create unique case ID
    df["case_id"] = "CASE-" + df["txn_id"].astype(str)

    # Select final case structure
    return df[["case_id", "customer_id", "txn_id", "typology", "risk_band"]]
