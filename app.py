import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'ML', 'Logistic_Regression.pkl')
scaler_path = os.path.join(base_dir, 'ML', 'scaler.pkl')
features_path = os.path.join(base_dir, 'ML', 'feature_names.pkl')

try:
    models = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    expected_features = joblib.load(features_path)
except FileNotFoundError as e:
    st.error(f"Missing model artifact: {e.filename}")
    st.stop()

st.title("Diabetes Prediction App")
st.write("Enter the following details to predict diabetes:")
input_data = {}
for feature in expected_features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = models.predict(input_scaled)
    st.write(f"Predicted Class: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
    
