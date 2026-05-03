import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="LifeLens", layout="centered")

st.title("🧠 Clinipredict-Cancer-Predictor")
st.markdown("### Predict cancer risk using medical features")

# Load model
model = joblib.load("cancer_model.pkl")

st.markdown("### Enter Patient Data")

# Inputs (simplified key features)
mean_radius = st.number_input("Mean Radius", 0.0, 50.0, 10.0)
mean_texture = st.number_input("Mean Texture", 0.0, 50.0, 15.0)
texture_error = st.number_input("Texture Error", 0.0, 10.0, 1.0)
worst_radius = st.number_input("Worst Radius", 0.0, 50.0, 20.0)
compactness_error = st.number_input("Compactness Error", 0.0, 1.0, 0.05)

if st.button("Predict Risk"):
    input_data = np.array([[mean_radius, mean_texture, texture_error, worst_radius, compactness_error]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 0:
        st.error(f"⚠️ High Risk (Malignant) — Confidence: {1 - probability:.2f}")
    else:
        st.success(f"✅ Low Risk (Benign) — Confidence: {probability:.2f}")
