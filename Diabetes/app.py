import os
import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Define the path to the current directory
basedir = os.path.abspath(os.path.dirname(__file__))

# Load the scaler and model using relative paths
sc_path = os.path.join(basedir, 'sc.pkl')
model_path = os.path.join(basedir, 'classifier.pkl')

# Ensure the files exist
if not os.path.exists(sc_path) or not os.path.exists(model_path):
    raise FileNotFoundError("One or more required files are missing.")

sc = pickle.load(open(sc_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collecting all the necessary features in the expected order
    float_features = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodPressure']),
        float(request.form['skinThickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]
    final_features = [np.array(float_features)]
    pred = model.predict(sc.transform(final_features))
    output = 'Diabetes' if pred[0] == 1 else 'No Diabetes'
    return render_template('result.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
