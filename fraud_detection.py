import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

import pandas as pd
df = pd.read_csv("data/creditcard.csv")

# #How's the data
# print(df.head())

# print("\nShape: ")
# print(df.shape)

# print("\nInfo: ")
# print(df.info())

# print("\nDescription: ")
# print(df.describe())

# print("\nMissing Values: ")
# print(df.isnull().sum())


#  Duplicates and dealing with them
print("\nDuplicate Rows: ")
print(df.duplicated().sum())

# duplicates = df[df.duplicated()]
# print(f"Duplicates:\n{duplicates.head()}")

print(f"Rows Before: {df.shape[0]}")

df = df.drop_duplicates()
print("\nDuplicate Rows Now: ")
print(df.duplicated().sum())

print(f"Rows After: {df.shape[0]}")

# Segregating the transactions in classes (Genuine or Fraud), 0-genuine, 1-fraud

print("\nClass Distribution: ")
print(df["Class"].value_counts())

# What percentage of transactions are fraud?
fraud_percentage = (
    df["Class"].value_counts(normalize = True) * 100
)
print(f"Fraud Percentage: \n{fraud_percentage}")

# Transaction Amount Distribution
plt.figure(figsize = (10, 5))
sns.histplot(df["Amount"], bins = 50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction amount")
plt.ylabel("Frequency")
plt.show()


# Visualisation

# sns.countplot(x = "Class", data = df)
# plt.title("Fraud VS Genuine Transactions")
# plt.show() 
# class imbalance because genuine transactions' bar is too big as compared to fraud transactions' bar

print("\nTransaction Amount Statistics: ")
print(df["Amount"].describe())

#Preparing Data For ML

X = df.drop("Class", axis = 1) #features contain everything except "Class"
y = df["Class"] # contains 0-genuine, 1-fraud

# Check shapes
print(f"X Shape: {X.shape}")
print(f"Y Shape: {y.shape}")

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)
# Verify split
print(f"Training Data: {X_train.shape}")
print(f"Testing Data: {X_test.shape}")

model = LogisticRegression(max_iter = 1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# metrics
print("\nLogistic Regression Results: ")

Accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {Accuracy}")

Precision = precision_score(y_test, y_pred)
print(f"\nPrecision: {Precision}")

Recall = recall_score(y_test, y_pred)
print(f"\nRecall: {Recall}")

F1_Score = f1_score(y_test, y_pred)
print(f"\nF1 Score: {F1_Score}")

rf_model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)
print("Random Forest Trained...")

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results: ")

Accuracy = accuracy_score(y_test, rf_pred)
print(f"\nAccuracy: {Accuracy}")

Precision_rf = precision_score(y_test, rf_pred)
print(f"\nPrecision: {Precision_rf}")

Recall_rf = recall_score(y_test, rf_pred)
print(f"\nRecall: {Recall_rf}")

F1_Score_rf = f1_score(y_test, rf_pred)
print(f"\nF1 Score: {F1_Score_rf}")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by = "Importance",
    ascending = False
)

print(feature_importance.head(10))

xgb_model = XGBClassifier(random_state = 42)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)
print("XGBoost Training Complete.")

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Results: ")

Accuracy = accuracy_score(y_test, xgb_pred)
print(f"\nAccuracy: {Accuracy}")

Precision_rf = precision_score(y_test, xgb_pred)
print(f"\nPrecision: {Precision_rf}")

Recall_rf = recall_score(y_test, xgb_pred)
print(f"\nRecall: {Recall_rf}")

F1_Score_rf = f1_score(y_test, xgb_pred)
print(f"\nF1 Score: {F1_Score_rf}")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by = "Importance",
    ascending = False
)

print(feature_importance.head(10))

import pickle
pickle.dump(
    rf_model,
    open("models/fraud_model.pkl", "wb")
)

print("Model Saved Successfully!")