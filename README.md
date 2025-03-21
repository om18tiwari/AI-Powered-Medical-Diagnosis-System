AI-Powered Diabetes Diagnosis System
This is a machine learning-powered web application that predicts whether a person is diabetic based on various medical features such as glucose level, blood pressure, BMI, and more. The model is built using a neural network and trained on the Pima Indians Diabetes Dataset. The app is deployed with Streamlit for a user-friendly interface.

Features
Predict if a person is diabetic or non-diabetic based on medical data.
Inputs include features like glucose level, blood pressure, BMI, age, and more.
The model uses a trained neural network for binary classification (diabetic or non-diabetic).
Technologies Used
TensorFlow for building and training the machine learning model.
Streamlit for creating the web application interface.
Scikit-learn for data preprocessing (scaling and splitting).
Pandas for handling and processing data.
Setup Instructions
1. Clone the Repository

git clone https://github.com/your-username/ai-medical-diagnosis-system.git
cd ai-medical-diagnosis-system
2. Create a Virtual Environment (optional but recommended)

python -m venv venv
3. Install the Dependencies
Make sure you have pip installed, then run the following command to install the required libraries:
pip install -r requirements.txt
4. Train the Model (Once)
Run the training script to train and save the model. You only need to do this once.


python model.py
This will train the model using the Pima Indians Diabetes Dataset and save the model as diabetes_model.h5.

5. Run the Streamlit Web Application
After training the model, you can run the app locally using the following command:


streamlit run app.py
This will start a local web server and open the app in your browser at http://localhost:8501.

