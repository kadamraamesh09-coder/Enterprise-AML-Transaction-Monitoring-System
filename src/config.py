# config.py

ALERT_THRESHOLDS = {
    "structuring_cash_threshold": 5000,       # Amount used to detect structuring/smurfing
    "structuring_window_hours": 4,            # Transactions within 4 hours
    "mule_threshold": 5,                      # Destination receiving money from many senders
    "high_value_amount": 50000,               # Large high-risk transactions
    "daily_tx_count_threshold": 10            # High velocity threshold
}
