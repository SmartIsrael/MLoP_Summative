# src/model.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import callbacks
from preprocessing import load_and_preprocess_data

def create_model():
    """
    Create and compile the neural network model.
    
    Returns:
    keras.models.Sequential: Compiled neural network model.
    """
    model = Sequential([
        Dense(16, activation='relu', input_dim=12),
        Dense(8, activation='relu'),
        Dropout(0.25),
        Dense(4, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(X_train, y_train, epochs=500, batch_size=32):
    """
    Train the neural network model.
    
    Args:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    
    Returns:
    tuple: Trained model and training history.
    """
    model = create_model()
    
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001,
        patience=20,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Save the model
    if not os.path.exists('../models'):
        os.makedirs('../models')
    model.save('../models/heart_failure_model.h5')
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
    model (keras.models.Sequential): Trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.
    
    Returns:
    dict: Evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "classification_report": report,
        "confusion_matrix": cm
    }

def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
    history (keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../notebook/training_history.png')
    plt.close()

if __name__ == "__main__":
    # Load and preprocess data
    data_path = '../data/train/heart_failure_clinical_records_dataset.csv'
    X, y = load_and_preprocess_data(data_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train the model
    model, history = train_model(X_train, y_train)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    evaluation = evaluate_model(model, X_test, y_test)
    
    print("Classification Report:")
    print(pd.DataFrame(evaluation['classification_report']).transpose())
    
    print("\nConfusion Matrix:")
    print(evaluation['confusion_matrix'])