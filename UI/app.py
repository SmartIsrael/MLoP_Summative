import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the model and scaler
model = load_model('./models/heart_failure_model.h5')
scaler = joblib.load('./models/scaler.pkl')

# Define the Streamlit app
st.title("Heart Failure Prediction")
st.write("Enter the patient's clinical data to predict the risk of heart failure.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120)
anaemia = st.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0)
diabetes = st.selectbox("Diabetes", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0)
sex = st.selectbox("Sex", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])
time = st.number_input("Follow-up Period (days)", min_value=0)

# Prediction button
if st.button("Predict"):
    # Create input data array
    input_data = np.array([[
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
    ]])
    
    # Preprocess input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Interpret the prediction
    death_probability = float(prediction[0][0])
    death_risk = "High" if death_probability > 0.5 else "Low"
    
    # Display the result
    st.write(f"Death Probability: {death_probability:.4f}")
    st.write(f"Death Risk: {death_risk}")

if __name__ == "__main__":
    st.run()
