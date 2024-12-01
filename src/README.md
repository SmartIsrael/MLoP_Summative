# Source Code Directory

This directory contains the Python scripts that make up the core functionality of the project.

## Contents

- `preprocessing.py`: Contains functions for data cleaning, feature engineering, and data preprocessing.
- `model.py`: Defines the model architecture, training process, and evaluation metrics.
- `prediction.py`: Includes functions for making predictions using the trained model.

## Usage

These scripts are typically imported and used in the main Jupyter notebook or in a production environment. To use them:

1. Ensure you're in the project's root directory.
2. Import the required functions in your Python script or notebook:

```python
from src.preprocessing import preprocess_data
from src.model import train_model
from src.prediction import make_prediction
```

3. Call the imported functions as needed in your workflow.

Note: Make sure all required dependencies are installed before using these scripts.
