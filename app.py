# app.py - Python Backend with Flask and SVM
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from flask_cors import CORS
import os

# Initialize Flask app, setting the template folder explicitly
app = Flask(__name__, template_folder='templates')
CORS(app) # Enable CORS for frontend communication

# Global variables for model and scaler
svm_model = None
scaler = None

def train_model():
    """Initializes and trains the SVM model using the diabetes dataset."""
    global svm_model, scaler
    try:
        # Load the dataset
        df = pd.read_csv('diabetes.csv')

        # Define features (X) and target (y)
        # Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # Standardize the features (crucial for SVM performance)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the dataset (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize and train the Support Vector Classifier (SVC)
        # 'kernel="linear"' is a simple, effective choice for many classification problems
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)

        print("SVM model trained successfully and ready for predictions.")
        return True
    except FileNotFoundError:
        print("CRITICAL ERROR: 'diabetes.csv' not found. Ensure it is in the project root directory.")
        return False
    except Exception as e:
        print(f"CRITICAL ERROR during model training: {e}")
        return False

# Train the model when the application starts
if not train_model():
    print("Application started but model is not functional.")


@app.route('/')
def home():
    """Renders the main HTML page (index.html)."""
    # Flask looks for this file inside the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request.
    Takes JSON input, standardizes it, and returns the prediction (0 or 1).
    """
    if svm_model is None or scaler is None:
        return jsonify({'error': 'Prediction service is unavailable. Model or scaler is missing.'}), 503

    try:
        # Get JSON data from the request
        data = request.json

        # Ensure the feature order matches the training data exactly
        features = [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]

        # Extract values in the correct order and convert to float
        input_data = [float(data.get(feat, 0)) for feat in features]

        # Standardize the input data using the fitted scaler
        input_data_scaled = scaler.transform([input_data])

        # Make the prediction
        prediction = svm_model.predict(input_data_scaled)

        # Convert prediction (numpy array of 0 or 1) to a standard integer
        result = int(prediction[0])
        return jsonify({'prediction': result})

    except Exception as e:
        print(f"Prediction failed with error: {e}")
        return jsonify({'error': f'Invalid input data or server processing error: {e}'}), 400

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
