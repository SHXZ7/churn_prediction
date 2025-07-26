import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load('churn_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# Load the feature names in the correct order
try:
    feature_names = joblib.load('feature_names.pkl')
    print(f"Loaded {len(feature_names)} features")
except:
    feature_names = None
    print("Could not load feature names")

st.title("Customer Churn Predictor")

st.header("Customer Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
    partner = st.radio("Partner", ["No", "Yes"])
    dependents = st.radio("Dependents", ["No", "Yes"])

with col2:
    tenure = st.slider("Tenure (months)", 0, 72)
    phone_service = st.radio("Phone Service", ["No", "Yes"])
    paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])

st.header("Services Information")
col1, col2 = st.columns(2)

with col1:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.header("Contract & Payment Information")
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0)

# Create input dataframe with all features
data = {
    # Binary features
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
    
    # Numerical features (will be scaled)
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
}

# Initialize all one-hot encoded columns to 0
# MultipleLines
data['MultipleLines_No phone service'] = 1 if multiple_lines == 'No phone service' else 0
data['MultipleLines_Yes'] = 1 if multiple_lines == 'Yes' else 0

# InternetService 
data['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0
data['InternetService_No'] = 1 if internet_service == 'No' else 0

# OnlineSecurity
data['OnlineSecurity_No internet service'] = 1 if online_security == 'No internet service' else 0
data['OnlineSecurity_Yes'] = 1 if online_security == 'Yes' else 0

# OnlineBackup
data['OnlineBackup_No internet service'] = 1 if online_backup == 'No internet service' else 0
data['OnlineBackup_Yes'] = 1 if online_backup == 'Yes' else 0

# DeviceProtection
data['DeviceProtection_No internet service'] = 1 if device_protection == 'No internet service' else 0
data['DeviceProtection_Yes'] = 1 if device_protection == 'Yes' else 0

# TechSupport
data['TechSupport_No internet service'] = 1 if tech_support == 'No internet service' else 0
data['TechSupport_Yes'] = 1 if tech_support == 'Yes' else 0

# StreamingTV
data['StreamingTV_No internet service'] = 1 if streaming_tv == 'No internet service' else 0
data['StreamingTV_Yes'] = 1 if streaming_tv == 'Yes' else 0

# StreamingMovies
data['StreamingMovies_No internet service'] = 1 if streaming_movies == 'No internet service' else 0
data['StreamingMovies_Yes'] = 1 if streaming_movies == 'Yes' else 0

# Contract
data['Contract_One year'] = 1 if contract == 'One year' else 0
data['Contract_Two year'] = 1 if contract == 'Two year' else 0

# PaymentMethod
data['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == 'Credit card (automatic)' else 0
data['PaymentMethod_Electronic check'] = 1 if payment_method == 'Electronic check' else 0
data['PaymentMethod_Mailed check'] = 1 if payment_method == 'Mailed check' else 0

# Create DataFrame and scale numeric features
user_input = pd.DataFrame([data])

# Scale numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
try:
    user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])
except:
    st.warning("Using standard scaling. For better results, save the scaler from your model training.")
    # Apply standard scaling manually if scaler.pkl is not available
    for col in numeric_cols:
        if user_input[col].std() != 0:
            user_input[col] = (user_input[col] - user_input[col].mean()) / user_input[col].std()

# Ensure feature order matches the training data
if feature_names is not None:
    # Create a DataFrame with all features in the correct order
    missing_cols = set(feature_names) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0  # Add missing columns with default value
    
    # Reorder columns to match training data
    user_input = user_input[feature_names]
    
    st.info(f"Using {len(feature_names)} features in the exact order they were during training")
    
# Predict
if st.button("Predict Churn"):
    try:
        prediction = model.predict(user_input)
        # Show result
        st.write("Prediction:", "⚠️ Likely to Churn" if prediction[0] == 1 else "✅ Not Likely to Churn")
        
        # Get probability scores
        try:
            prob = model.predict_proba(user_input)[0][1]
            st.write(f"Churn Probability: {prob:.2f}")
            
            # Visual indicator
            st.progress(prob)
            if prob > 0.7:
                st.error("High risk of churn! Immediate action recommended.")
            elif prob > 0.4:
                st.warning("Medium risk of churn. Consider customer retention strategies.")
            else:
                st.success("Low risk of churn. Customer appears satisfied.")
        except:
            pass
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please ensure the model was trained with the exact same features as provided in the app.")
