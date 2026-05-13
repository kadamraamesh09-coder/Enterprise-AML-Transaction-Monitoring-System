# src/data_loader.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

def load_transactions() -> pd.DataFrame:
    """
    Load the Kaggle-style transaction dataset and prepare basic columns.
    """
    df = pd.read_csv(DATA_DIR / "transactions.csv")

    # Rename columns to AML-friendly names
    df = df.rename(columns={
        "nameOrig": "customer_id",
        "nameDest": "counterparty_id"
    })

    # Convert step to datetime (step = hours)
    df["txn_datetime"] = pd.to_datetime(df["step"], unit="h", origin="2020-01-01")

    df.sort_values(["customer_id", "txn_datetime"], inplace=True)

    return df
