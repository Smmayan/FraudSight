# 👁️ FraudSight

### Spot Fraud Before It Costs You

FraudSight is a Machine Learning-powered fraud detection dashboard that analyzes credit card transaction data and identifies potentially fraudulent transactions using a trained Random Forest model.

🌐 **Live Demo:**
https://fraud-detection-system-smmayan.streamlit.app

🔗 **GitHub Repository:**
https://github.com/Smmayan/FraudSight

---

## 📖 Project Overview

FraudSight is an end-to-end Machine Learning project designed to detect fraudulent credit card transactions.

The project covers the complete ML lifecycle:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Model Training
- Model Evaluation
- Model Selection
- Model Deployment
- Cloud Hosting

Users can upload transaction datasets, analyze fraud predictions, view visual insights, and download prediction results through an interactive web interface.

---

## ✨ Features

✅ Upload transaction datasets

✅ Detect fraudulent transactions instantly

✅ Fraud vs Genuine transaction analysis

✅ Interactive dashboard

✅ Donut chart visualization

✅ Download prediction results

✅ Responsive and modern UI

✅ Cloud deployment with Streamlit

---

## 🧠 Machine Learning Models Tested

### Logistic Regression

- Accuracy: 99.92%
- Precision: 84.85%
- Recall: 62.22%
- F1 Score: 71.79%

### Random Forest ✅ Selected Model

- Accuracy: 99.95%
- Precision: 97.06%
- Recall: 73.33%
- F1 Score: 83.54%

### XGBoost

- Accuracy: 99.69%
- Precision: 9.00%
- Recall: 10.00%
- F1 Score: 9.47%

### Final Model

Random Forest was selected as the final model due to its superior performance and reliability in detecting fraudulent transactions.

---

## 📊 Dataset Information

Dataset Used:

**Credit Card Fraud Detection Dataset**

Source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Statistics

- Total Transactions: 284,807
- Fraudulent Transactions: 492
- Genuine Transactions: 284,315

### Features

- Time
- Amount
- V1 to V28 (PCA-transformed features)
- Class (Target Variable)

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Plotly
- Streamlit
- Pickle

### Tools

- VS Code
- Git
- GitHub
- Streamlit Community Cloud

---

## 📂 Project Structure

```text
FraudSight
│
├── assets/
│ └── logo.png
│
├── models/
│ └── fraud_model.pkl
│
├── .streamlit/
│ └── config.toml
│
├── app.py
├── fraud_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How To Run Locally

### Clone Repository

```bash
git clone https://github.com/Smmayan/FraudSight.git
```

### Move To Project Folder

```bash
cd FraudSight
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🔄 Project Workflow

```text
Transaction Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Selection
↓
Model Training
↓
Model Evaluation
↓
Random Forest Selection
↓
Model Serialization (Pickle)
↓
Streamlit Dashboard
↓
Cloud Deployment
```

---

## 📈 Model Performance Summary

- Accuracy: 99.95%
- Precision: 97.06%
- Recall: 73.33%
- F1 Score: 83.54%

---

## 🌐 Live Application

Access FraudSight here:

https://fraud-detection-system-smmayan.streamlit.app

---

## 🚧 Future Improvements

- Real-time transaction monitoring
- REST API integration
- Advanced fraud analytics dashboard
- User authentication
- Transaction risk scoring
- Real-world banking data integration

---

## 👨‍💻 Author

**Smmayan Gupta**

B.Tech Computer Science & Engineering

Aspiring Applied AI & Machine Learning Engineer

GitHub:
https://github.com/Smmayan

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational, learning, and portfolio purposes.