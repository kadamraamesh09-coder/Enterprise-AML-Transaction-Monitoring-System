# src/preprocess.py
import pandas as pd
import numpy as np
import json
from pathlib import Path

DEFAULT_FEATURES = [
    # these are example features used previously. When you retrain, feature_names.json will be authoritative.
    "type_CASH-IN", "type_CASH-OUT", "type_TRANSFER", "type_PAYMENT",
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
]

def load_feature_list(out_dir: str = "artifacts"):
    p = Path(out_dir) / "feature_names.json"
    if p.exists():
        return json.loads(p.read_text())
    return DEFAULT_FEATURES

def preprocess_df(df: pd.DataFrame, feature_list: list = None):
    """
    Read transactions df (raw), drop noisy cols, one-hot encode 'type', align to feature_list.
    Returns DataFrame ready for model.transform/predict.
    """
    df = df.copy()
    # drop common unused columns if present
    df = df.drop([c for c in ["Unnamed: 0", "nameOrig", "nameDest", "y_prednew"] if c in df.columns], axis=1, errors="ignore")

    # Ensure columns we expect exist
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], prefix="type")
    else:
        # create placeholder columns if original data had no 'type'
        pass

    # Fill NaNs in numeric
    for c in df.select_dtypes(include=["number"]).columns:
        df[c] = df[c].fillna(0)

    # Align features
    if feature_list is None:
        feature_list = load_feature_list()

    # Add missing cols
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # Trim extras and keep order
    X = df[feature_list].astype(float)
    return X
