# Data Science Case Study: Kroo Bank — Credit Risk & Fraud Detection

**Individual Task 1 — Case Studies in Data Science, RMIT University**
Author: Ziad Ashmawy (s4165181)

## Overview

Code supporting Individual Task 1, exploring two datasets relevant to a **Data Scientist** role at Kroo Bank, a UK digital-first challenger bank. The role covers credit risk modelling and fraud detection — this repo applies and evaluates two machine learning models against datasets from each of those problem areas.

Job listing: https://uk.indeed.com/viewjob?jk=7124376041e0f05d

## Datasets

Raw data is **not included** in this repo (large files, and both are third-party licensed). Download separately and place in a local `data/` folder:

1. **Credit Card Fraud Detection (ULB)** — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   284,807 anonymised European card transactions, 492 frauds (0.172%). Features: `Time`, `Amount`, PCA components `V1`–`V28`, target `Class`.

2. **Statlog German Credit Data (UCI)** — https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
   1,000 loan applicants, 20 attributes (financial history, employment, demographics), target: good/bad credit risk, with an accompanying misclassification cost matrix.

## Repo Structure

```
├── data/              # raw datasets (gitignored — see download links above)
├── notebooks/         # analysis notebooks / scripts
├── requirements.txt   # Python dependencies
└── README.md
```

## Methods

Two ML models applied to each dataset:
- **Decision Tree**
- **Support Vector Machine (SVM)**

Both datasets are class-imbalanced (German Credit ~70/30, Fraud ~99.8/0.2), so evaluation uses precision, recall, F1, and AUC rather than plain accuracy, plus cost-weighted evaluation for German Credit using its provided cost matrix.

## Setup

```bash
pip install -r requirements.txt
```

## Author

Ziad Ashmawy (s4165181), RMIT University
s4165181@student.rmit.edu.au
