# src/case_manager.py
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime

CASES_PATH = Path("artifacts") / "cases.csv"
CASES_PATH.parent.mkdir(parents=True, exist_ok=True)

def create_cases(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    df = df.copy()
    if "final_risk" not in df.columns:
        return pd.DataFrame(columns=["case_id", "tx_id", "user_id", "final_risk", "created_at"])
    flagged = df[df["final_risk"] >= threshold].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["case_id", "tx_id", "user_id", "final_risk", "created_at"])
    flagged["case_id"] = ["CASE-" + datetime.utcnow().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8] for _ in range(len(flagged))]
    cases = flagged[["case_id", "tx_id", "user_id", "final_risk"]].copy()
    cases["created_at"] = datetime.utcnow().isoformat()
    # Append to artifacts/cases.csv (dedupe)
    if CASES_PATH.exists():
        existing = pd.read_csv(CASES_PATH)
        combined = pd.concat([existing, cases], ignore_index=True)
        combined.drop_duplicates(subset=["case_id"], inplace=True)
        combined.to_csv(CASES_PATH, index=False)
    else:
        cases.to_csv(CASES_PATH, index=False)
    return cases
