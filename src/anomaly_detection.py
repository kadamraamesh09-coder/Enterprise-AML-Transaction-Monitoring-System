# src/anomaly_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest

def run_anomaly_detection(tx: pd.DataFrame):
    df = tx.copy()

    df["hour"] = df["txn_datetime"].dt.hour

    features = df[["amount", "hour"]]

    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(features)

    df["anomaly_score"] = -model.decision_function(features)
    df["is_anomaly"] = model.predict(features).astype(int)

    anomalies = df[df["is_anomaly"] == -1].copy()
    anomalies["typology"] = "Anomalous Transaction Pattern"

    return anomalies, model
