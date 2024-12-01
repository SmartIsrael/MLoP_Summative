import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(X_df):
    # One-hot encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X_df.select_dtypes(include=['object']))
    X_encoded_df = pd.DataFrame.sparse.from_spmatrix(X_encoded, columns=encoder.get_feature_names_out(X_df.select_dtypes(include=['object']).columns))

    # Standardize numerical features
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(X_df.select_dtypes(include=['number']))
    X_numerical_df = pd.DataFrame(X_numerical, columns=X_df.select_dtypes(include=['number']).columns)

    # Combine processed data
    X_processed = pd.concat([X_numerical_df, X_encoded_df], axis=1)

    # Impute missing values (if any)
    X_processed = X_processed.fillna(X_processed.mean())

    return X_processed, encoder, scaler

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
