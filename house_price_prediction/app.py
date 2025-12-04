# -----------------------------
# HOUSE PRICE PREDICTION APP
# -----------------------------

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction")
st.write("Predict California house prices based on property features.")

# Input widgets
median_income = st.number_input("Median Income", 0.0, 20.0, 5.0)
house_age = st.number_input("House Age", 0.0, 100.0, 20.0)
avg_rooms = st.number_input("Average Rooms", 0.0, 50.0, 5.0)
avg_bedrooms = st.number_input("Average Bedrooms", 0.0, 50.0, 1.0)
population = st.number_input("Population", 0.0, 50000.0, 1000.0)
avg_occupancy = st.number_input("Average Occupancy", 0.0, 10.0, 3.0)
latitude = st.number_input("Latitude", -90.0, 90.0, 34.0)
longitude = st.number_input("Longitude", -180.0, 180.0, -118.0)

if st.button("Predict Price"):
    input_data = np.array([[median_income, house_age, avg_rooms, avg_bedrooms,
                            population, avg_occupancy, latitude, longitude]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted House Price: **${prediction*100000:.2f}**")
