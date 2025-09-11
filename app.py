#Week 3
# Creating an Application

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("air_quality_model.pkl")

st.title("🌍 Air Quality Prediction App")
st.write("Predict **PM2.5 concentration** based on pollutant levels.")

# Input fields for features
CO = st.number_input("CO (GT)", min_value=0.0, step=0.1)
NOx = st.number_input("NOx (GT)", min_value=0.0, step=1.0)
NO2 = st.number_input("NO2 (GT)", min_value=0.0, step=1.0)
O3 = st.number_input("O3 (GT)", min_value=0.0, step=1.0)
SO2 = st.number_input("SO2 (GT)", min_value=0.0, step=1.0)

# Button to predict
if st.button("Predict PM2.5"):
    # Put inputs into dataframe
    input_df = pd.DataFrame([[CO, NOx, NO2, O3, SO2]],
                            columns=["CO(GT)", "NOx(GT)", "NO2(GT)", "O3(GT)", "SO2(GT)"])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted PM2.5 concentration: {prediction:.2f} µg/m³")
