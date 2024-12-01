
# Models Directory

This directory contains saved machine learning models and associated files.

## Contents

- `heart_failure_model.pkl`: Serialized scikit-learn model.
- `modelh5`: Saved TensorFlow model.

## Usage

To load and use a saved model:

1. For pickle (.pkl) files:

```python
import pickle

with open('models/heart_failure_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Use the model to make predictions
predictions = model.predict(X_test)
```

2. For TensorFlow (.tf) files:

```python
import tensorflow as tf

model = tf.keras.models.load_model('models/model.h5')

# Use the model to make predictions
predictions = model.predict(X_test)
```

Note: nnsure you're using the same version of the libraries that were used to save the model to avoid compatibility issues.
