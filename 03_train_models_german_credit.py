"""
03_train_models_german_credit.py

Trains Decision Tree and SVM on the German Credit data and prints
classification reports. Decision Tree uses unscaled features (doesn't
need scaling); SVM uses scaled features (distance-based, requires it).

Run 01_data_loading_and_preprocessing.py first to produce
data/german_credit_encoded.csv.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

df = pd.read_csv("data/german_credit_encoded.csv")
X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

print("=== Decision Tree ===")
print(classification_report(y_test, dt_preds, target_names=["Bad (0)", "Good (1)"]))

# SVM
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)

print("=== SVM ===")
print(classification_report(y_test, svm_preds, target_names=["Bad (0)", "Good (1)"]))
