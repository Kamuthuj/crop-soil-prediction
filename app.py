import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and encoder
model = joblib.load("crop_recommendation_model.pkl")
scaler = joblib.load("StandardScaler.pkl")
encoder = joblib.load("LabelEncoder.pkl")

# Define feature names
feature_names = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "PH", "rainfall"]

# Streamlit app layout
st.title("🌾Crop Recommendation System")

st.markdown("Enter the soil and environmental conditions below to get a crop recommendation:")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", format="%.2f")
humidity = st.number_input("Humidity", format="%.2f")
PH = st.number_input("Soil pH", format="%.2f")
rainfall = st.number_input("Rainfall (mm)", format="%.2f")

# Predict button
if st.button("Predict Crop"):
    # Prepare input DataFrame with correct feature names
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, PH, rainfall]], columns=feature_names)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_scaled)
    predicted_crop = encoder.inverse_transform(prediction)[0]

    # Display the result
    st.success(f"The recommended crop is: **{predicted_crop}**")