from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model, scaler, and scale_factors
model = joblib.load(r'C:\Users\vyoms\Desktop\Sem 6\AI\Power_consumption\models\model.pkl')
scaler = joblib.load(r'C:\Users\vyoms\Desktop\Sem 6\AI\Power_consumption\models\model.pkl')
scale_factors = joblib.load(r'C:\Users\vyoms\Desktop\Sem 6\AI\Power_consumption\models\model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return send_file(r"C:\Users\vyoms\Desktop\Sem 6\AI\Power_consumption\templates\index.html")

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_values = [float(request.form['input1']), float(request.form['input2']), float(request.form['input3']), float(request.form['input4']), float(request.form['input5'])]
    print(input_values)
    # Scale the input using the scaler used for training
    scaled_input_values = scaler.transform(np.array([input_values]))
    
    # Make predictions
    predictions = model.predict(scaled_input_values.reshape((scaled_input_values.shape[0], 1, scaled_input_values.shape[1])))
    
    # Inverse transform the predictions to get actual values
    actual_predictions = predictions * scale_factors.reshape((-1, 1))
    
    # Prepare the predictions to be sent back to the frontend
    prediction_values = {
        'zone1': actual_predictions[0][0],
        'zone2': actual_predictions[0][1],
        'zone3': actual_predictions[0][2]
    }
    
    # Return predictions as JSON
    return jsonify(prediction_values)

if __name__ == '__main__':
    app.run(debug=True)
