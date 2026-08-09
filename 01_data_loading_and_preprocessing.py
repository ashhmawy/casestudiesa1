"""
01_data_loading_and_preprocessing.py

Individual Task 1 - Case Studies in Data Science, RMIT University
Loads and checks both datasets, then preprocesses the German Credit data
(one-hot encoding of categorical columns) ready for modelling.

Datasets (download separately, place in a local data/ folder - see README):
  - german_credit.csv : Statlog German Credit Data (UCI), cleaned/labelled mirror
  - creditcard.csv    : Credit Card Fraud Detection (Kaggle, mlg-ulb)
"""

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load and sanity-check both datasets
# ---------------------------------------------------------------------------

german = pd.read_csv("data/german_credit.csv")
print("German Credit shape:", german.shape)
print("German Credit missing values:", german.isnull().sum().sum())
print("German Credit target balance:\n", german["credit_risk"].value_counts())
print()

fraud = pd.read_csv("data/creditcard.csv")
print("Fraud shape:", fraud.shape)
print("Fraud missing values:", fraud.isnull().sum().sum())
print("Fraud target balance:\n", fraud["Class"].value_counts())
print()

# ---------------------------------------------------------------------------
# 2. Preprocess German Credit: one-hot encode categorical columns
# ---------------------------------------------------------------------------

cat_cols = german.select_dtypes(include="object").columns.tolist()
print("Categorical columns to encode:", cat_cols)

german_encoded = pd.get_dummies(german, columns=cat_cols, drop_first=True)
print("Shape before encoding:", german.shape)
print("Shape after encoding:", german_encoded.shape)

german_encoded.to_csv("data/german_credit_encoded.csv", index=False)
print("Saved encoded data to data/german_credit_encoded.csv")
