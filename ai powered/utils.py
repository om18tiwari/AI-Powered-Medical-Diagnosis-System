import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model():
    """Load the pre-trained model"""
    return tf.keras.models.load_model('diabetes_model.h5')

def preprocess_input(features):
    """Preprocess input data (standard scaling)"""
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def predict(model, features):
    """Use the trained model to make predictions"""
    prediction = model.predict(features)
    return "Diabetes" if prediction > 0.5 else "No Diabetes"
