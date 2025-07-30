import streamlit as st
import joblib
import numpy as np
import pandas as pd

import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'premium_predictor_xgb.pkl')
model_path = os.path.abspath(model_path)
model = joblib.load(model_path)


st.set_page_config(page_title="Health Premium Predictor", layout="centered")
st.title("💰 Smart Health Insurance Premium Predictor")
st.write("Fill out the details below to get your estimated annual premium:")

# Basic fields
age = st.slider("Age", 18, 100, 30)
gender = st.radio("Gender", ["Male", "Female"])
marital_status = st.radio("Marital Status", ["Married", "Unmarried"])
dependents = st.slider("Number of Dependents", 0, 5, 0)
income = st.number_input("Income (in Lakhs)", min_value=0.0, step=0.5)
smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker", "Occasionally", "Heavy"])
insurance_plan = st.selectbox("Insurance Plan", [0, 1, 2])  # assuming 0 = basic, 1 = silver, 2 = gold

# Region (includes Central now)
region = st.selectbox("Region", ["Northwest", "Southeast", "Southwest", "Northeast", "Central"])

# BMI Category
bmi_category = st.selectbox("BMI Category", ["Normal", "Underweight", "Overweight", "Obesity"])

# Employment Status
employment = st.selectbox("Employment Status", ["Unemployed", "Salaried", "Self-Employed"])

# Income Level
income_level = st.selectbox("Income Level", ["<10L", "10L-25L", "25L - 40L", "> 40L"])

# Medical History
medical_history = st.selectbox("Medical History", [
    "No Disease",
    "Diabetes & Heart disease",
    "Diabetes & High blood pressure",
    "Diabetes & Thyroid",
    "Heart disease",
    "High blood pressure",
    "High blood pressure & Heart disease",
    "Thyroid"
])

# --- Create Input Feature Vector ---

# Severity score mapping
severity_scores = {
    "No Disease": 0,
    "Thyroid": 1,
    "Diabetes & Thyroid": 2,
    "High blood pressure": 2,
    "Diabetes & High blood pressure": 3,
    "Heart disease": 4,
    "High blood pressure & Heart disease": 5,
    "Diabetes & Heart disease": 6
}
severity_score = severity_scores.get(medical_history, 0)

# Base features
input_data = {
    "Age": age,
    "Gender": 1 if gender == "Male" else 0,
    "Marital_status": 1 if marital_status == "Married" else 0,
    "Number Of Dependants": dependents,
    "Smoking_Status": {"Non-Smoker": 0, "Smoker": 1, "Occasionally": 2, "Heavy": 3}[smoking_status],
    "Income_Lakhs": income,
    "Insurance_Plan": insurance_plan,
    "Medical_Severity": severity_score,

    # Regions (Central added)
    "Region_Northwest": region == "Northwest",
    "Region_Southeast": region == "Southeast",
    "Region_Southwest": region == "Southwest",
    "Region_Central": region == "Central",  # <-- added
    "Region_Northeast": region == "Northeast",

    # BMI
    "BMI_Category_Obesity": bmi_category == "Obesity",
    "BMI_Category_Overweight": bmi_category == "Overweight",
    "BMI_Category_Underweight": bmi_category == "Underweight",

    # Employment
    "Employment_Status_Salaried": employment == "Salaried",
    "Employment_Status_Self-Employed": employment == "Self-Employed",

    # Income Level
    "Income_Level_25L - 40L": income_level == "25L - 40L",
    "Income_Level_<10L": income_level == "<10L",
    "Income_Level_> 40L": income_level == "> 40L",

    # Medical History (keep them too)
    "Medical History_Diabetes & Heart disease": medical_history == "Diabetes & Heart disease",
    "Medical History_Diabetes & High blood pressure": medical_history == "Diabetes & High blood pressure",
    "Medical History_Diabetes & Thyroid": medical_history == "Diabetes & Thyroid",
    "Medical History_Heart disease": medical_history == "Heart disease",
    "Medical History_High blood pressure": medical_history == "High blood pressure",
    "Medical History_High blood pressure & Heart disease": medical_history == "High blood pressure & Heart disease",
    "Medical History_No Disease": medical_history == "No Disease",
    "Medical History_Thyroid": medical_history == "Thyroid"
}

# Create DataFrame with correct feature order
features = list(model.get_booster().feature_names)
input_df = pd.DataFrame([[input_data.get(col, 0) for col in features]], columns=features)

# --- Predict ---
if st.button("Predict Premium"):
    prediction = model.predict(input_df)[0]
    st.success(f"💸 Estimated Annual Premium: ₹{prediction:,.2f}")
