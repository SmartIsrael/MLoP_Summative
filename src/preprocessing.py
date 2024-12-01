# src/preprocessing.py

import os
import pandas as pd
from sklearn import preprocessing
import joblib

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the heart failure dataset.
    
    Args:
    file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    tuple: Preprocessed features (X) and target variable (y).
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Split features and target
    X = data.drop(["DEATH_EVENT"], axis=1)
    y = data["DEATH_EVENT"]
    
    # Scale features
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the scaler
    if not os.path.exists('../models'):
        os.makedirs('../models')
    joblib.dump(scaler, '../models/scaler.pkl')
    
    return X_df, y

def load_scaler():
    """
    Load the saved StandardScaler object.
    
    Returns:
    sklearn.preprocessing.StandardScaler: Loaded scaler object.
    """
    return joblib.load('../models/scaler.pkl')

if __name__ == "__main__":
    # Example usage
    data_path = '../data/train/heart_failure_clinical_records_dataset.csv'
    X, y = load_and_preprocess_data(data_path)
    print("Data preprocessed and scaler saved.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)