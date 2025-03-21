import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model():
    """
    Trains and saves the model. This is to be called once for model training.
    """
    # Load the Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)

    # Preprocessing the Data
    X = data.drop('Outcome', axis=1)  # Features
    y = data['Outcome']  # Target (Outcome)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build the Neural Network Model
    model = Sequential()
    model.add(Dense(16, input_dim=8, activation='relu'))  # Input layer and first hidden layer
    model.add(Dense(8, activation='relu'))  # Second hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

    # Save the trained model for later use
    model.save('diabetes_model.h5')
    print("Model trained and saved as 'diabetes_model.h5'")


