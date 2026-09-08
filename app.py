import streamlit as st
import pickle
import pandas as pd

model = pickle.load(
    open("models/fraud_model.pkl", "rb")
)

st.title("Credit Card Fraud Detection System")
st.markdown(
    """
    Upload a transaction dataset and indentify potentially fraudulent transactions using a trained Random Forest Model.
    """
)

st.subheader("Model Performance")
st.write("Algorithm: Random Forest")
st.write("Accuracy: 99.95%")
st.write("Precision: 97.06%")
st.write("Recall: 73.33%")
st.write("F1 Score: 83.54%")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV File",
    type = ['csv']
)

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(uploaded_data.head())

if st.button("Predict Fraud"):
    predictions = model.predict(uploaded_data)

    uploaded_data["Fraud_Prediction"] = predictions

    fraud_count = (
        uploaded_data["Fraud_Prediction"].sum()
    )
    genuine_count = (
        len(uploaded_data) - fraud_count
    )

    st.subheader("Prediction Results")

    st.dataframe(uploaded_data.head())

    st.subheader("Prediction Summary")
    st.write(f"Fraud Transactions: {fraud_count}")
    st.write(f"Genuine Transactions: {genuine_count}")

    fraud_transactions = uploaded_data[
        uploaded_data["Fraud_Prediction"] == 1
    ]

    st.subheader("Detected Fraud Transactions")
    st.dataframe(fraud_transactions)

    csv = uploaded_data.to_csv(index = False)

    st.download_button(
        label = "Download Results",
        data = csv,
        file_name = "fraud_predictions.csv",
        mime = "text/csv"
    )