# Credit Card Fraud Detection System

A machine learning-based fraud detection system that identifies potentially fraudulent credit card transactions using Random Forest.

## Project Overview

This project analyzes credit card transaction data and predicts whether a transaction is fraudulent or genuine.

The system was built using:

- Python
- Pandas
- Scikit-Learn
- Random Forest
- Streamlit

## Dataset

Dataset: Credit Card Fraud Detection Dataset from Kaggle

Target Variable:

- 0 = Genuine Transaction
- 1 = Fraudulent Transaction

## Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Duplicate Removal
4. Class Imbalance Analysis
5. Model Training
6. Model Evaluation
7. Streamlit Deployment

## Models Tested

### Logistic Regression

Accuracy: 99.92%

Precision: 84.85%

Recall: 62.22%

F1 Score: 71.79%

### Random Forest (Best Model)

Accuracy: 99.95%

Precision: 97.06%

Recall: 73.33%

F1 Score: 83.54%

### XGBoost

Accuracy: 99.69%

Precision: 9.00%

Recall: 10.00%

F1 Score: 9.47%

## Final Model

Random Forest was selected as the final model due to its superior precision, recall and F1 score.

## Features

- Upload CSV transactions
- Detect fraudulent transactions
- Fraud summary dashboard
- Download prediction results

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit

## Author

Smmayan Gupta