# MLOps Summative Project: Diabetes Classification Model

## Project Overview

This project involves building a machine learning model to classify whether a patient has diabetes or not. The project follows a comprehensive MLOps workflow, including model training, deployment, API creation for prediction and retraining, and integration with a Streamlit application.

## Features

- **Model Training:** Trained a machine learning model to classify diabetes based on health indicators.
- **Model Deployment:** Saved and imported the trained model into a new environment.
- **API Creation:** Created an API to facilitate predictions and retraining of the model.
- **Streamlit Integration:** Connected the API to a Streamlit application for an interactive user interface.

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual Environment (venv)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/MLOps-Engineering.git
    cd mlops-summative-diabetes
    ```

2. **Set Up Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Model Training

1. **Train the Model:**

    ```bash
    python train_model.py
    ```

2. **Save the Model:**

    The trained model will be saved as `diabetes_model.pkl` in the `models` directory.

### Model Deployment

1. **Import the Model into a New Environment:**

    ```bash
    python import_model.py
    ```

2. **Create the API:**

    The API is created using FastAPI. Run the API server:

    ```bash
    uvicorn api:app --reload
    ```

    The API will be available at `http://127.0.0.1:8000`.

### Streamlit Application

1. **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

    The Streamlit application will be available at `http://localhost:8501`.

## API Endpoints

- **Predict:** `/predict`
    - Method: POST
    - Request Body: JSON containing the patient's health indicators
    - Response: Prediction result (Diabetes: Yes/No)

- **Retrain:** `/retrain`
    - Method: POST
    - Request Body: JSON containing the new training data
    - Response: Retraining status



## Usage

1. **Predict Diabetes:**

    Use the Streamlit app or send a POST request to the `/predict` endpoint with the patient's health indicators to get a diabetes prediction.

2. **Retrain the Model:**

    Send a POST request to the `/retrain` endpoint with new training data to retrain the model.
