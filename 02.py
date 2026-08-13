"""
Combined German Credit Pipeline (Splitting, Scaling, and Model Training)

Splits the encoded German Credit data into train/test sets (stratified, 80/20),
applies StandardScaler to the features, and trains Decision Tree and SVM models.

Run 01_data_loading_and_preprocessing.py first to produce data/german_credit_encoded.csv.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # standardises numerical features so theyre on a similar scale.
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

df = pd.read_csv("data/german_credit_encoded.csv")

X = df.drop(columns=["credit_risk"]) # data without the credit risk column as that is what we are trying to precict
y = df["credit_risk"] # only credit risk column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # 20% for testing & keeps roughly the same proportion of good/bad credit cases in both training and testing
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train target balance:\n", y_train.value_counts(normalize=True).round(3)) # Shows the proportion of each target class in the training set rounded to 3dp
print("Test target balance:\n", y_test.value_counts(normalize=True).round(3)) # same for the test set to confirm that stratify=y worked

# Scale numeric features (needed for SVM but harmless for Decision Tree) ----------> 10, 50, 90 -------> -1.22, 0, +1.22
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit = learn mean and SD for each feature | transform = use those values to standardise

X_test_scaled = scaler.transform(X_test) # not fitted as we only want it to learn from the training data, not testing data

print("Scaling done. X_train_scaled shape:", X_train_scaled.shape)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=5) # creates and limits tree to maximum depth of 5
# without a limit it can become extremely complicated and basically memorise the training data.

dt.fit(X_train, y_train) # tries to learn patterns that allow it to predict credit risk
dt_preds = dt.predict(X_test) # Now we give the trained Decision Tree the test features

print("=== Decision Tree ===")
print(classification_report(y_test, dt_preds, target_names=["Bad (0)", "Good (1)"])) # compares y_test to the dt_preds, 
# target_names=["Bad (0)", "Good (1)"] tells the report what the class numbers mean

# SVM
svm = SVC(kernel="rbf", probability=True, random_state=42) # tells the SVM to use an RBF (Radial Basis Function) kernel and tells SVM to also calculate probability estimates
svm.fit(X_train_scaled, y_train) # changed from x_train to x_train_scaled as SVM is sensitive to scale
svm_preds = svm.predict(X_test_scaled)

print("=== SVM ===")
print(classification_report(y_test, svm_preds, target_names=["Bad (0)", "Good (1)"])) # same as above