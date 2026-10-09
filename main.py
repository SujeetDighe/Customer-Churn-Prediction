import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------
# Page configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# Load model files
# ------------------------------------------------
model = joblib.load("churn_model.pkl")
feature_names = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------------------------
# Title
# ------------------------------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to leave the company."
)

st.divider()

# ------------------------------------------------
# Customer Information
# ------------------------------------------------
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])

with col3:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

col1, col2 = st.columns(2)

with col1:
    partner = st.selectbox("Partner", ["Yes", "No"])

with col2:
    dependents = st.selectbox("Dependents", ["Yes", "No"])

# ------------------------------------------------
# Phone & Internet Information
# ------------------------------------------------
st.subheader("📱 Phone & Internet Services")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

# ------------------------------------------------
# Contract & Billing Information
# ------------------------------------------------
st.subheader("💳 Contract & Billing Information")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=monthly_charges * tenure,
        step=10.0
    )

# ------------------------------------------------
# Prediction
# ------------------------------------------------
st.divider()

if st.button("🔮 Predict Customer Churn", use_container_width=True):

    # --------------------------------------------
    # Create input dataframe
    # --------------------------------------------
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # --------------------------------------------
    # One-hot encoding
    # --------------------------------------------
    input_data = pd.get_dummies(
        input_data,
        drop_first=True,
        dtype=int
    )

    # --------------------------------------------
    # Match training columns
    # --------------------------------------------
    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )

    # --------------------------------------------
    # Scaling
    # --------------------------------------------
    input_scaled = scaler.transform(input_data)

    # --------------------------------------------
    # Prediction
    # --------------------------------------------
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # --------------------------------------------
    # Result
    # --------------------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if probability >= 0.50:
        st.write(
            "The model estimates a higher probability that this "
            "customer will leave the company."
        )
    else:
        st.write(
            "The model estimates a higher probability that this "
            "customer will continue with the company."
        )

# ------------------------------------------------
# Project information
# ------------------------------------------------
with st.expander("📘 About this Project"):
    st.write(
        "This project uses Machine Learning to predict customer churn. "
        "The model is trained using the Telco Customer Churn dataset. "
        "Categorical data is converted using one-hot encoding and "
        "numeric features are scaled before Logistic Regression prediction."
    )
