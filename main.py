from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Heart Failure Prediction API", description="API for predicting heart failure based on clinical features")

# Load the saved model and scaler
model = load_model('models/heart_failure_model.h5')
scaler = joblib.load('models/scaler.pkl')

class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the patient")
    anaemia: int = Field(..., ge=0, le=1, description="If the patient had anaemia (0: No, 1: Yes)")
    creatinine_phosphokinase: int = Field(..., ge=0, description="Level of creatine phosphokinase in blood (mcg/L)")
    diabetes: int = Field(..., ge=0, le=1, description="If the patient was diabetic (0: No, 1: Yes)")
    ejection_fraction: int = Field(..., ge=0, le=100, description="Percentage of blood leaving the heart at each contraction")
    high_blood_pressure: int = Field(..., ge=0, le=1, description="If the patient had hypertension (0: No, 1: Yes)")
    platelets: float = Field(..., ge=0, description="Platelet count of blood (kiloplatelets/mL)")
    serum_creatinine: float = Field(..., ge=0, description="Level of serum creatinine in blood (mg/dL)")
    serum_sodium: int = Field(..., ge=0, description="Level of serum sodium in blood (mEq/L)")
    sex: int = Field(..., ge=0, le=1, description="Sex of the patient (0: Female, 1: Male)")
    smoking: int = Field(..., ge=0, le=1, description="If the patient smokes (0: No, 1: Yes)")
    time: int = Field(..., ge=0, description="Follow-up period (days)")

class TrainingData(BaseModel):
    patients: List[PatientData]
    labels: List[int] = Field(..., description="Death event (0: No, 1: Yes)")

@app.post("/predict")
async def predict(patient: PatientData):
    # Convert input data to numpy array
    input_data = np.array([[
        patient.age, patient.anaemia, patient.creatinine_phosphokinase, patient.diabetes,
        patient.ejection_fraction, patient.high_blood_pressure, patient.platelets,
        patient.serum_creatinine, patient.serum_sodium, patient.sex, patient.smoking, patient.time
    ]])
    
    # Preprocess the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Interpret the prediction
    death_probability = float(prediction[0][0])
    death_risk = "High" if death_probability > 0.5 else "Low"
    
    return {
        "death_probability": death_probability,
        "death_risk": death_risk
    }

@app.post("/retrain")
async def retrain(data: TrainingData):
    # Convert input data to numpy arrays
    X = np.array([[
        p.age, p.anaemia, p.creatinine_phosphokinase, p.diabetes,
        p.ejection_fraction, p.high_blood_pressure, p.platelets,
        p.serum_creatinine, p.serum_sodium, p.sex, p.smoking, p.time
    ] for p in data.patients])
    y = np.array(data.labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and compile the model
    global model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(12,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Save the updated model and scaler
    model.save('models/heart_failure_model.h5')
    joblib.dump(scaler, 'models/scaler.pkl')

    return {
        "message": "Model retrained successfully",
        "accuracy": accuracy,
        "loss": loss
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Heart Failure Prediction API"}