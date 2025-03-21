import streamlit as st
import numpy as np
from model import train_model
from utils import load_model, preprocess_input, predict

# Train the model if it's not already trained
try:
    load_model()
except:
    print("Model not found, training...")
    train_model()

# Streamlit user interface
st.title("AI-Powered Diabetes Prediction")

# User input fields with number validation
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input("Age", min_value=18, max_value=120, value=18)

# Ensure all inputs are numeric and properly passed as a list of numbers
features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

# Check that the inputs are being captured correctly
st.write(f"Features for prediction: {features}")

# Preprocess the input (standardize the features)
features_scaled = preprocess_input(features)

# Load the trained model
model = load_model()

# Predict button
if st.button("Predict"):
    result = predict(model, features_scaled)
    st.write(f"Prediction: {result}")
