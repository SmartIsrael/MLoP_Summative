# src/prediction.py

import pandas as pd
from keras.models import load_model
from preprocessing import load_scaler

def load_trained_model(model_path='../models/heart_failure_model.h5'):
    """
    Load the trained model.
    
    Args:
    model_path (str): Path to the saved model file.
    
    Returns:
    keras.models.Model: Loaded model.
    """
    return load_model(model_path)

def predict(model, new_data):
    """
    Make predictions using the trained model.
    
    Args:
    model (keras.models.Model): Trained model.
    new_data (pd.DataFrame): New data for prediction.
    
    Returns:
    np.array: Predicted probabilities.
    """
    scaler = load_scaler()
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)

if __name__ == "__main__":
    # Example usage
    model = load_trained_model()
    
    # Create some example data (replace this with your actual new data)
    example_data = pd.DataFrame({
        'age': [60],
        'anaemia': [0],
        'creatinine_phosphokinase': [500],
        'diabetes': [1],
        'ejection_fraction': [38],
        'high_blood_pressure': [0],
        'platelets': [250000],
        'serum_creatinine': [1.9],
        'serum_sodium': [130],
        'sex': [1],
        'smoking': [0],
        'time': [6]
    })
    
    predictions = predict(model, example_data)
    print("Prediction (probability of death event):", predictions[0][0])