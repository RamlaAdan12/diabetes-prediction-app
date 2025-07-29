import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('random_forest_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')  

# App Title
st.title(" Diabetes Prediction Model")
st.subheader("Includes SQL-Based Feature Engineering (Risk Level & BMI Category)")

# Input fields
st.markdown("###  Enter Patient Information:")

Pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0)
BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0)
SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0)
Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
Age = st.number_input("Age", min_value=1, max_value=120)

# RiskLevel based on Glucose
st.markdown("###  Select Glucose-Based Risk Level")
risk_label = st.selectbox("Choose Risk Level", ["Low", "Medium", "High"])
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
RiskLevel = risk_mapping[risk_label]

# BMICategory based on BMI
st.markdown("### ⚖ Select BMI Category")
bmi_label = st.selectbox("Choose BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
BMICategory = bmi_mapping[bmi_label]

# Predict Button
if st.button(" Predict Diabetes"):
    # Define feature columns
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
               'RiskLevel', 'BMICategory']
    
    # Create DataFrame
    input_df = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DiabetesPedigreeFunction, Age,
                              RiskLevel, BMICategory]], columns=columns)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)

    # Display result
    st.markdown("### 🧾 Prediction Result:")
    if prediction[0] == 1:
        st.error(" The patient is likely Diabetic (Predicted: 1)")
    else:
        st.success(" The patient is likely Not Diabetic (Predicted: 0)")

# Footer
st.markdown("---")
st.caption("Created by Ramla Adan Yare · BSc Data Science · Supervised by Clive Onsomu")