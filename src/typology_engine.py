# typology_engine.py
import pandas as pd
from config import ALERT_THRESHOLDS


# -------------------------------
# HELPER: Safe column accessor
# -------------------------------
def col(df, name):
    """Safe access to column; prevents KeyErrors."""
    return df[name] if name in df.columns else pd.Series([0] * len(df))


# -------------------------------
# 1. STRUCTURING / SMURFING
# -------------------------------
def detect_structuring(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()

    # Trigger: Repeated transactions below reporting threshold within short time
    threshold = ALERT_THRESHOLDS["structuring_cash_threshold"]
    window_hours = ALERT_THRESHOLDS["structuring_window_hours"]

    df = df.sort_values(["nameOrig", "txn_datetime"])

    df["is_structuring_amount"] = (df["amount"] < threshold).astype(int)

    df["time_diff"] = (
        df.groupby("nameOrig")["txn_datetime"]
        .diff()
        .dt.total_seconds()
        .fillna(99999) / 3600
    )

    structuring_alerts = df[
        (df["is_structuring_amount"] == 1) &
        (df["time_diff"] <= window_hours)
    ].copy()

    structuring_alerts["typology"] = "Structuring / Smurfing"
    return structuring_alerts


# -------------------------------
# 2. DORMANT → ACTIVE BEHAVIOR
# -------------------------------
def detect_dormant_then_active(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()

    old_org = col(df, "oldbalanceOrg")
    new_org = col(df, "newbalanceOrig")

    df["balance_change"] = (new_org - old_org).abs()

    # Dormant = no activity
    dormant_accounts = df[df["balance_change"] == 0]["nameOrig"].unique()

    # Active = sudden activity
    active_tx = df[
        (df["nameOrig"].isin(dormant_accounts)) &
        (df["balance_change"] > 0)
    ].copy()

    active_tx["typology"] = "Dormant-to-Active Spike"
    return active_tx


# -------------------------------
# 3. MULE DESTINATION DETECTION
# -------------------------------
def detect_mule_accounts(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()

    mule_threshold = ALERT_THRESHOLDS["mule_threshold"]

    # Count how many senders send to a single destination
    dest_counts = df["nameDest"].value_counts()

    mule_candidates = dest_counts[dest_counts >= mule_threshold].index.tolist()

    mule_alerts = df[df["nameDest"].isin(mule_candidates)].copy()
    mule_alerts["typology"] = "Possible Money Mule Destination"

    return mule_alerts


# -------------------------------
# 4. RAPID TRANSACTIONS (Velocity)
# -------------------------------
def detect_rapid_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()

    df = df.sort_values(["nameOrig", "txn_datetime"])

    df["time_diff"] = (
        df.groupby("nameOrig")["txn_datetime"]
        .diff()
        .dt.total_seconds()
        .fillna(99999)
    )

    # Threshold: Rapid transfers within 60 seconds
    rapid_alerts = df[df["time_diff"] < 60].copy()
    rapid_alerts["typology"] = "Rapid Transaction Velocity"

    return rapid_alerts


# -------------------------------
# FINAL TYPOLOGY AGGREGATOR
# -------------------------------
def generate_typology_alerts(tx: pd.DataFrame) -> pd.DataFrame:
    alerts = []

    a1 = detect_structuring(tx)
    if not a1.empty:
        alerts.append(a1)

    a2 = detect_dormant_then_active(tx)
    if not a2.empty:
        alerts.append(a2)

    a3 = detect_mule_accounts(tx)
    if not a3.empty:
        alerts.append(a3)

    a4 = detect_rapid_transactions(tx)
    if not a4.empty:
        alerts.append(a4)

    if not alerts:
        return pd.DataFrame(columns=tx.columns.tolist() + ["typology"])

    return pd.concat(alerts, ignore_index=True)
