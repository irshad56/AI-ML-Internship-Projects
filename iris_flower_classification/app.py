# -----------------------------
# IRIS FLOWER CLASSIFICATION APP
# -----------------------------

import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("iris_knn_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

st.title("🌸 Iris Flower Classification")
st.write("Predict the Iris flower species based on input features.")

# Input widgets
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Predict button
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"The predicted species is: **{species_map[prediction]}**")
