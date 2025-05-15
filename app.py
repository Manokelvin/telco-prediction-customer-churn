import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and the scaler
try:
    model = joblib.load('churn_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Error: Make sure 'churn_model.joblib' and 'scaler.joblib' are in the same directory as this app.")
    st.stop()

st.title("Telco Customer Churn Prediction")
st.write("Enter customer features to predict churn.")

# Get the feature names that the model was trained on (excluding 'Churn')
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                 'InternetService_Fiber optic', 'InternetService_No',
                 'Contract_One year', 'Contract_Two year',
                 'PaymentMethod_Credit card (automatic)',
                 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# Create input fields based on the feature names
user_inputs = {}
for feature in feature_names:
    if feature == 'gender':
        user_inputs[feature] = st.selectbox(f"Select {feature}", ["Female", "Male"])
        user_inputs[feature] = 1 if user_inputs[feature] == "Female" else 0 # Assuming Female is 1, Male is 0
    elif feature in ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                   'InternetService_Fiber optic', 'InternetService_No',
                   'Contract_One year', 'Contract_Two year',
                   'PaymentMethod_Credit card (automatic)',
                   'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']:
        user_inputs[feature] = st.selectbox(f"Select {feature}", ["No", "Yes"])
        user_inputs[feature] = 1 if user_inputs[feature] == "Yes" else 0
    elif feature in ['SeniorCitizen']:
        user_inputs[feature] = st.selectbox(
            f"Is the customer a Senior Citizen?", ["No", "Yes"]
        )
        user_inputs[feature] = 1 if user_inputs[feature] == "Yes" else 0
    elif feature in ['tenure']:
        user_inputs[feature] = st.number_input(f"Enter customer tenure (months)", min_value=0)
    elif feature in ['MonthlyCharges']:
        user_inputs[feature] = st.number_input(f"Enter monthly charges", min_value=0.0)
    elif feature in ['TotalCharges']:
        user_inputs[feature] = st.number_input(f"Enter total charges", min_value=0.0)
def predict_churn():
    input_df = pd.DataFrame([user_inputs])

    # Ensure the order of columns matches the training data
    input_df = input_df[feature_names]

    # Scale the numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1] # Probability of churning

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.warning(f"This customer is likely to churn (Probability: {probability[0]:.2f})")
    else:
        st.success(f"This customer is not likely to churn (Probability: {1 - probability[0]:.2f})")

if st.button("Predict"):
    predict_churn()
