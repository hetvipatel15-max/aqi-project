import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('aqi_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="AQI Prediction", page_icon="🌍")

# Title
st.title("🌍 AQI Prediction System")
st.markdown("**Explainable ML for Air Quality Index Prediction — Delhi**")
st.markdown("---")

# Input sliders
st.subheader("Enter Pollutant Values")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.slider("PM2.5", 22.0, 263.0, 120.0)
    pm10 = st.slider("PM10", 44.0, 529.0, 200.0)
    no2  = st.slider("NO₂",  34.0, 231.0, 150.0)

with col2:
    so2  = st.slider("SO₂",  46.0, 293.0, 100.0)
    co   = st.slider("CO",   0.43, 4.41,  2.0)
    o3   = st.slider("O₃",   38.0, 243.0, 180.0)

st.markdown("---")

# Predict button
if st.button("🔍 Predict AQI"):
    features = np.array([[pm25, pm10, no2, so2, co, o3]])
    features_scaled = scaler.transform(features)
    prediction = round(float(model.predict(features_scaled)[0]), 1)

    # Show result
    st.markdown("### Predicted AQI")
    st.metric(label="AQI Value", value=prediction)

    # Health category
    if prediction <= 50:
        st.success("✅ Good — Air quality is satisfactory. Safe for outdoor activities.")
    elif prediction <= 100:
        st.success("🟡 Satisfactory — Air quality is acceptable.")
    elif prediction <= 200:
        st.warning("⚠️ Moderate — Sensitive groups should take care.")
    elif prediction <= 300:
        st.error("😷 Poor — Everyone may experience health effects. Wear a mask.")
    elif prediction <= 400:
        st.error("🚨 Very Poor — Avoid outdoor activities.")
    else:
        st.error("☠️ Severe — Stay indoors. Seek medical help if needed.")