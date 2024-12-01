import tensorflow as tf
import os


def load_model():
    model_path = os.path.join(r'C:\Users\daluc\MyProjects\MLOPS', 'models', 'model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model
