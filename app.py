import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="FraudSight",
    page_icon="👁️",
    layout="wide"
)

model = pickle.load(
    open("models/fraud_model.pkl", "rb")
)

st.image(
    "assets/logo.png",
    width = 200
    )

st.title("👁️ FraudSight")
st.markdown(
    """
    ### Spot Fraud Before It Costs You
    Real-Time Credit Card Transaction Risk Analysis using a trained Random Forest model.
    \nUpload transaction data and instantly identify suspicious activities.
    """
)

with st.expander("ℹ️ About FraudSight"):
    st.write(
        """
        FraudSight is a machine learning-powered fraud
        detection dashboard that identifies suspicious
        credit card transactions.

        Models Tested:
        • Logistic Regression
        • Random Forest
        • XGBoost

        Final Model:
        • Random Forest

        Accuracy:
        • 99.95%
        """
)

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.95%")
col2.metric("Precision", "97.06%")
col3.metric("Recall", "73.33%")
col4.metric("F1 Score", "83.54%")

st.sidebar.title("👁️ FraudSight") 
page = st.sidebar.radio(
    "Go To",
    ["Home", "Fraud Detection"]
)

uploaded_file = st.file_uploader(
    "Upload Transaction CSV File",
    type = ['csv']
)

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(uploaded_data.head())

if st.button("Predict Fraud"):

    if "Class" in uploaded_data.columns:
        uploaded_data = uploaded_data.drop(columns = ["Class"])

        expected_columns = list(model.feature_names_in_)
        if list(uploaded_data.columns) != expected_columns:
            st.error("Please upload a file with the correct columns")
            st.stop()

    predictions = model.predict(uploaded_data)

    st.success("✅ Prediction Completed Successfully!")

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
    st.error(f"🚨 Fraud Transactions: {fraud_count}")
    st.success(f"✅ Genuine Transactions: {genuine_count}")

    # PIE CHART
    chart_data = {
    "Type": ["Fraud", "Genuine"],
    "Count": [fraud_count, genuine_count]
    }

    fig = px.pie(
    values=chart_data["Count"],
    names=chart_data["Type"],
    title="Fraud vs Genuine Transactions",
    hole = 0.5
    )

    st.plotly_chart(fig)

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


# Footer
st.markdown("---")

st.markdown("Developed by **Smmayan Gupta** | Machine Learning Project")
