"""
02_split_and_scale.py

Splits the encoded German Credit data into train/test sets (stratified,
80/20) and applies StandardScaler to the features. Scaling is required
for SVM and does no harm to Decision Tree.

Run 01_data_loading_and_preprocessing.py first to produce
data/german_credit_encoded.csv.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/german_credit_encoded.csv")

X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train target balance:\n", y_train.value_counts(normalize=True).round(3))
print("Test target balance:\n", y_test.value_counts(normalize=True).round(3))

# Scale numeric features (needed for SVM; harmless for Decision Tree)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling done. X_train_scaled shape:", X_train_scaled.shape)
