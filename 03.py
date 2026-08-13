"""
Combined Credit Card Fraud Pipeline (Splitting, Scaling, and Model Training)

Splits the Credit Card Fraud data into train/test sets (stratified, 80/20),
scales Time and Amount columns, and trains Decision Tree and Linear SVM models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC # we wwill use LinearSVC as this is a huge dataset and RBF version of SVM is extremely computationally expensive.
from sklearn.metrics import classification_report, roc_auc_score

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

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=8, class_weight="balanced") # limits tree to max depth of 8
# IMPORTANT: ADDED CLASS WEIGHT = BALANCED
# Without class_weight="balanced", the model could become biased toward the majority class
# balanced basically tells the model to give fraud cases more importance when learning as they are rare
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1] # we can get its probability of fraud instead of a yes or no

print("=== Decision Tree ===")
print(classification_report(y_test, dt_preds, target_names=["Not Fraud (0)", "Fraud (1)"]))
print("AUC:", round(roc_auc_score(y_test, dt_proba), 4)) 


"""
A completely useless model could say everything is legitimate and it would get:
998 / 1000 = 99.8% accuracy even though it detected: 0 / 2 fraud transactions
Which is why we use AUC
"""




# Linear SVM
svm = LinearSVC(random_state=42, class_weight="balanced", max_iter=5000) # max number of iterations the algorithm is allowed to use while training
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
svm_scores = svm.decision_function(X_test_scaled) # decision_function() gives a score showing how strongly the SVM thinks each transaction belongs to one class or the other
#the further from 0, the stronger the models decision

print("=== Linear SVM ===")
print(classification_report(y_test, svm_preds, target_names=["Not Fraud (0)", "Fraud (1)"]))
print("AUC:", round(roc_auc_score(y_test, svm_scores), 4))