import sys, os

# Add src folder to python path
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_transactions
from rules_engine import apply_rules
from typology_engine import generate_typology_alerts
from risk_rating import compute_customer_risk
from anomaly_detection import run_anomaly_detection
from network_graph import build_transaction_graph, detect_multi_hop_layering
from case_management import create_cases
from sar_generator import generate_sar

# ---------------------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="AML Transaction Monitoring System", layout="wide")
st.title("🛡️ Enterprise AML Transaction Monitoring Dashboard ")


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
page = st.sidebar.selectbox(
    "Navigation",
    [
        "📂 Data Overview",
        "⚠️ Rule Engine Alerts",
        "🔍 AML Typologies",
        "📊 Customer Risk Rating",
        "🚨 Anomaly Detection",
        "🕸️ Network Analysis",
        "📁 Case Management & SAR",
    ]
)

uploaded = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])

if not uploaded:
    st.info("Upload your AML transaction dataset to continue.")
    st.stop()

# ---------------------------------------------------------------------
# LOAD & PREPARE DATA
# ---------------------------------------------------------------------
df = pd.read_csv(uploaded)

# Ensure required core columns exist
required_cols = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in uploaded file: {missing}")
    st.stop()

# Synthetic transaction ID
if "txn_id" not in df.columns:
    df["txn_id"] = df.index.astype(str)

# Customer / counterparty IDs for other modules
df["customer_id"] = df["nameOrig"]
df["counterparty_id"] = df["nameDest"]

# Datetime for time-based analysis
df["txn_datetime"] = pd.to_datetime(
    df["step"], unit="h", origin="2020-01-01", errors="coerce"
)

# Helper for safe table rendering
def show_table(data: pd.DataFrame, limit: int = 500):
    if len(data) > limit:
        st.warning(f"Large dataset: showing first {limit} rows out of {len(data)}.")
        st.dataframe(data.head(limit))
    else:
        st.dataframe(data)


# ---------------------------------------------------------------------
# PAGE 1 — DATA OVERVIEW
# ---------------------------------------------------------------------
if page == "📂 Data Overview":
    st.header("📂 Dataset Overview")

    # KPI Metrics
    total_tx = len(df)
    unique_customers = df["nameOrig"].nunique()
    fraud_rate = df["isFraud"].mean() * 100 if "isFraud" in df.columns else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Transactions", f"{total_tx:,}")
    kpi2.metric("Unique Customers", f"{unique_customers:,}")
    kpi3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

    st.subheader("Sample Records")
    show_table(df)

    # Transaction Amount Distribution
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["amount"], bins=50, ax=ax)
    ax.set_title("Histogram of Transaction Amounts")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

    # Transaction Type Distribution
    st.subheader("Transaction Type Breakdown")
    if "type" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="type", data=df, ax=ax)
        ax.set_title("Count by Transaction Type")
        ax.set_xlabel("Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Column Info")
    st.write(df.dtypes)


# ---------------------------------------------------------------------
# PAGE 2 — RULE ENGINE ALERTS
# ---------------------------------------------------------------------
elif page == "⚠️ Rule Engine Alerts":
    st.header("⚠️ Rule-Based AML Alerts")

    alerts = apply_rules(df)

    # KPIs
    total_alerts = len(alerts)
    high_value_count = alerts["rule_high_value"].sum() if "rule_high_value" in alerts.columns else 0
    daily_count_violations = alerts["rule_daily_count"].sum() if "rule_daily_count" in alerts.columns else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Rule Alerts", f"{total_alerts:,}")
    k2.metric("High-Value Alerts", f"{high_value_count:,}")
    k3.metric("High Daily Count Alerts", f"{daily_count_violations:,}")

    st.subheader("Rule Alerts Table")
    show_table(alerts)

    # Rule score distribution
    if "rule_score" in alerts.columns:
        st.subheader("Rule Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(alerts["rule_score"], bins=10, discrete=True, ax=ax)
        ax.set_title("Distribution of Rule Scores")
        ax.set_xlabel("Rule Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------
# PAGE 3 — AML TYPOLOGIES
# ---------------------------------------------------------------------
elif page == "🔍 AML Typologies":
    st.header("🔍 AML Typology Detection")

    typ_alerts = generate_typology_alerts(df)

    total_typ_alerts = len(typ_alerts)
    unique_typologies = typ_alerts["typology"].nunique() if "typology" in typ_alerts.columns else 0
    impacted_customers = typ_alerts["nameOrig"].nunique() if "nameOrig" in typ_alerts.columns else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Typology Alerts", f"{total_typ_alerts:,}")
    k2.metric("Typology Types", f"{unique_typologies:,}")
    k3.metric("Impacted Customers", f"{impacted_customers:,}")

    st.subheader("Typology Alerts Table")
    if typ_alerts.empty:
        st.info("No typology-based alerts detected.")
    else:
        show_table(typ_alerts)

        # Typology breakdown
        if "typology" in typ_alerts.columns:
            st.subheader("Typology Breakdown")
            fig, ax = plt.subplots(figsize=(6, 4))
            typ_counts = typ_alerts["typology"].value_counts()
            sns.barplot(x=typ_counts.index, y=typ_counts.values, ax=ax)
            ax.set_title("Count by Typology")
            ax.set_xlabel("Typology")
            ax.set_ylabel("Count")
            plt.xticks(rotation=30)
            st.pyplot(fig)
            plt.close(fig)


# ---------------------------------------------------------------------
# PAGE 4 — CUSTOMER RISK RATING
# ---------------------------------------------------------------------
elif page == "📊 Customer Risk Rating":
    st.header("📊 Customer Risk Rating Engine")

    risk_df = compute_customer_risk(df)

    st.subheader("Customer Risk Table")
    show_table(risk_df)

    if "risk_band" in risk_df.columns:
        high_risk_count = (risk_df["risk_band"] == "High").sum()
        med_risk_count = (risk_df["risk_band"] == "Medium").sum()
        low_risk_count = (risk_df["risk_band"] == "Low").sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("High-Risk Customers", f"{high_risk_count:,}")
        k2.metric("Medium-Risk Customers", f"{med_risk_count:,}")
        k3.metric("Low-Risk Customers", f"{low_risk_count:,}")

        st.subheader("Risk Band Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        band_counts = risk_df["risk_band"].value_counts()
        sns.barplot(x=band_counts.index, y=band_counts.values, ax=ax)
        ax.set_title("Customer Count by Risk Band")
        ax.set_xlabel("Risk Band")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    if "risk_score" in risk_df.columns:
        st.subheader("Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(risk_df["risk_score"], bins=30, ax=ax)
        ax.set_title("Distribution of Risk Scores")
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------
# PAGE 5 — ANOMALY DETECTION
# ---------------------------------------------------------------------
elif page == "🚨 Anomaly Detection":
    st.header("🚨 Machine Learning Anomaly Detection")

    try:
        anomalies, model = run_anomaly_detection(df)
    except Exception as e:
        st.error(f"Error in anomaly detection: {e}")
        st.stop()

    total_anomalies = len(anomalies)
    anomaly_pct = (total_anomalies / len(df) * 100) if len(df) > 0 else 0.0

    k1, k2 = st.columns(2)
    k1.metric("Total Anomalies Detected", f"{total_anomalies:,}")
    k2.metric("Anomaly Percentage", f"{anomaly_pct:.2f}%")

    st.subheader("Anomalous Transactions (Sample)")
    show_table(anomalies)

    if "anomaly_score" in anomalies.columns:
        st.subheader("Anomaly Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(anomalies["anomaly_score"], bins=30, ax=ax)
        ax.set_title("Distribution of Anomaly Scores")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------
# PAGE 6 — NETWORK ANALYSIS
# ---------------------------------------------------------------------
elif page == "🕸️ Network Analysis":
    st.header("🕸️ Transaction Network Analysis")

    # Limit rows for performance
    max_rows = 5000
    if len(df) > max_rows:
        st.warning(f"Dataset too large for full graph analysis ({len(df)} rows). Using first {max_rows} rows.")
        df_small = df.head(max_rows)
    else:
        df_small = df

    try:
        G = build_transaction_graph(df_small)
        layering = detect_multi_hop_layering(G)
    except Exception as e:
        st.error(f"Error during network analysis: {e}")
        st.stop()

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    layering_count = len(layering)

    k1, k2, k3 = st.columns(3)
    k1.metric("Graph Nodes (Accounts)", f"{total_nodes:,}")
    k2.metric("Graph Edges (Transactions)", f"{total_edges:,}")
    k3.metric("Layering Paths Detected", f"{layering_count:,}")

    st.subheader("Layering Paths (Sample)")
    if layering.empty:
        st.info("No layering paths detected.")
    else:
        show_table(layering)

        if "hops" in layering.columns:
            st.subheader("Distribution of Path Lengths (Hops)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="hops", data=layering, ax=ax)
            ax.set_title("Count of Layering Paths by Hop Length")
            ax.set_xlabel("Number of Hops")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)


# ---------------------------------------------------------------------
# PAGE 7 — CASE MANAGEMENT & SAR
# ---------------------------------------------------------------------
elif page == "📁 Case Management & SAR":
    st.header("📁 AML Case Management & SAR Generator")

    # Generate alerts from rules + typologies
    rule_alerts = apply_rules(df)
    typ_alerts = generate_typology_alerts(df)
    alerts = pd.concat([rule_alerts, typ_alerts], ignore_index=True)

    # Compute customer risk
    risk_df = compute_customer_risk(df)

    # Create cases
    cases = create_cases(alerts, risk_df)

    total_cases = len(cases)
    high_risk_cases = 0
    if "risk_band" in cases.columns:
        high_risk_cases = (cases["risk_band"] == "High").sum()

    k1, k2 = st.columns(2)
    k1.metric("Total AML Cases", f"{total_cases:,}")
    k2.metric("High-Risk Cases", f"{high_risk_cases:,}")

    st.subheader("Generated AML Cases")
    show_table(cases)

    if "risk_band" in cases.columns:
        st.subheader("Cases by Risk Band")
        fig, ax = plt.subplots(figsize=(6, 4))
        band_counts = cases["risk_band"].value_counts()
        sns.barplot(x=band_counts.index, y=band_counts.values, ax=ax)
        ax.set_title("Case Count by Risk Band")
        ax.set_xlabel("Risk Band")
        ax.set_ylabel("Case Count")
        st.pyplot(fig)
        plt.close(fig)

    if "typology" in cases.columns:
        st.subheader("Cases by Typology")
        fig, ax = plt.subplots(figsize=(6, 4))
        typ_counts = cases["typology"].value_counts()
        sns.barplot(x=typ_counts.index, y=typ_counts.values, ax=ax)
        ax.set_title("Case Count by Typology")
        ax.set_xlabel("Typology")
        ax.set_ylabel("Case Count")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    # SAR Generation
    st.subheader("Generate SAR for Selected Case")
    selected = st.selectbox("Choose Case ID", cases["case_id"] if not cases.empty else [])

    if selected:
        sar_text = generate_sar(cases[cases["case_id"] == selected].iloc[0])
        st.text_area("Suspicious Activity Report (SAR)", sar_text, height=300)
