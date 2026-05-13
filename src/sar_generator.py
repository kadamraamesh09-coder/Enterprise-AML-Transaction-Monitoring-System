# src/sar_generator.py

def generate_sar(case_row):
    return f"""
SAR REPORT
===========
Case ID: {case_row['case_id']}
Customer ID: {case_row['customer_id']}
Transaction ID: {case_row['txn_id']}
Typology Triggered: {case_row['typology']}
Risk Band: {case_row['risk_band']}

Narrative:
The system detected suspicious behavior consistent with AML typology: {case_row['typology']}.
Further review recommended.
"""
