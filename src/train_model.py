# src/train_model.py
import argparse
from pathlib import Path
import joblib
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def featurize(df):
    # simple featurization similar to pipeline above
    df = df.copy()
    df = df.drop([c for c in ["Unnamed: 0", "nameOrig", "nameDest", "y_prednew"] if c in df.columns], axis=1, errors="ignore")
    df = pd.get_dummies(df, columns=["type"], prefix="type")
    # choose columns for training (adjust if dataset differs)
    candidate = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFlaggedFraud"]
    # include type columns if present
    type_cols = [c for c in df.columns if c.startswith("type_")]
    features = type_cols + candidate
    X = df[features].fillna(0)
    # default label column names may be 'isFraud' or 'label'
    if "isFraud" in df.columns:
        y = df["isFraud"].astype(int)
    elif "label" in df.columns:
        y = df["label"].astype(int)
    else:
        raise RuntimeError("No label column found (isFraud/label)")
    return X, y, features

def main(csv_path):
    df = load_data(csv_path)
    X, y, features = featurize(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_s, y_train)
    # save artifacts
    joblib.dump(clf, ARTIFACT_DIR / "model.pkl")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
    (ARTIFACT_DIR / "feature_names.json").write_text(json.dumps(features))
    print("Saved model, scaler, and feature_names.json to artifacts/")
    # quick eval
    print("Train score:", clf.score(X_train_s, y_train))
    print("Test score:", clf.score(X_test_s, y_test))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_model.py data/transactions.csv")
    else:
        main(sys.argv[1])
