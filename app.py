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
st.markdown("*Nitrogen Min value 0 and max value 180*")
N = st.number_input("Nitrogen (N)", min_value=0.0,max_value=180.0)
st.markdown("*Phosphorus (P) Min value 5 and max value 150*")
P = st.number_input("Phosphorus (P)", min_value=5.0,max_value=150.0)
st.markdown("*Potassium (K) Min value 5 and max value 220*")
K = st.number_input("Potassium (K)", min_value=5.0,max_value=220.0)
st.markdown("*Temperature (°C) Min value 7 and max value 45*")
temperature = st.number_input("Temperature (°C)", min_value=7.0,max_value=45.0)
st.markdown("*Humidity Min value 10 and max value 120*")
humidity = st.number_input("Humidity", min_value=10.0,max_value=120.0)
st.markdown("*Soil pH Min value 3.5 and max value 10*")
PH = st.number_input("Soil pH", min_value=3.5,max_value=10.0)
st.markdown("*Rainfall Min value 20 and max value 300*")
rainfall = st.number_input("Rainfall (mm)", min_value=20.0,max_value=300.0)

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
    st.balloons()
    st.write("### Thank you for using the Crop Recommendation System!")