"""
04_split_and_scale_fraud.py

Splits the Credit Card Fraud data into train/test sets (stratified,
80/20, so both keep the same tiny fraud rate), and scales the Time and
Amount columns. V1-V28 are already PCA-transformed so they don't need
additional scaling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fraud = pd.read_csv("data/creditcard.csv")

X = fraud.drop(columns=["Class"])
y = fraud["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train fraud rate:", round(y_train.mean() * 100, 3), "%")
print("Test fraud rate:", round(y_test.mean() * 100, 3), "%")

# Scale only Time and Amount (V1-V28 are already PCA-scaled)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

print("Scaling done.")
