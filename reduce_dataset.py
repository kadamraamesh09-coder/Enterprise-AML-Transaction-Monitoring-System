import pandas as pd

# Load full dataset
df = pd.read_csv("Synthetic_Financial_datasets_log.csv")

# Keep all fraud cases
fraud = df[df["isFraud"] == 1]

# Sample non-fraud cases (change size if needed)
non_fraud = df[df["isFraud"] == 0].sample(n=150000, random_state=42)

# Combine
df_small = pd.concat([fraud, non_fraud])

# Save reduced dataset
df_small.to_csv("fraud_dataset_small.csv", index=False)

print("Reduced dataset saved as fraud_dataset_small.csv")
print("Original size:", len(df))
print("New size:", len(df_small))
