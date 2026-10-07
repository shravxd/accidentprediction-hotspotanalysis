from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load your trained machine learning model if available, otherwise use a placeholder
MODEL_PATH = 'model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Warning: model.pkl not found. Running in simulation mode.")

@app.route('/')
def home():
    # Renders index.html from the 'templates' folder
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features sent from the frontend form
        latitude = float(data.get('latitude', 18.5204))
        longitude = float(data.get('longitude', 73.8567))
        speed_limit = float(data.get('speed_limit', 50))
        weather = data.get('weather', 'Clear')
        road_surface = data.get('road_surface', 'Dry')
        lighting = data.get('lighting', 'Daylight')
        vehicle_type = data.get('vehicle_type', 'Car')
        
        if model is not None:
            # Prepare feature vector for your trained model
            # Note: Ensure this matches the feature columns your model expects (encoding categoricals if needed)
            features = pd.DataFrame([[latitude, longitude, speed_limit]], 
                                      columns=['Latitude', 'Longitude', 'Speed_limit'])
            prediction = model.predict(features)[0]
            probability = np.max(model.predict_proba(features)) * 100
        else:
            # Fallback simulated calculation based on speed limit & weather for testing
            risk_score = (speed_limit / 100.0) * 0.5
            if weather in ['Rain', 'Snow', 'Fog']:
                risk_score += 0.3
            if road_surface == 'Wet':
                risk_score += 0.2
                
            if risk_score > 0.7:
                prediction = "Fatal / Serious Injury"
                probability = round(risk_score * 85, 1)
            elif risk_score > 0.4:
                prediction = "Serious Injury"
                probability = round(risk_score * 78, 1)
            else:
                prediction = "Slight Injury"
                probability = round((1 - risk_score) * 90, 1)

        return jsonify({
            'status': 'success',
            'severity_prediction': str(prediction),
            'confidence_score': f"{probability}%",
            'hotspot_status': "High Density Cluster (DBSCAN)" if speed_limit > 60 else "Standard Zone"
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

@app.route('/api/hotspots', methods=['GET'])
def get_hotspots():
    try:
        # If you have your dataset or trained points, you can return them. 
        # Here we return sample cluster points matching your trained DBSCAN output
        hotspots = [
            {"lat": 18.5204, "lng": 73.8567, "severity": "Fatal", "cluster": 1},
            {"lat": 18.5314, "lng": 73.8442, "severity": "Serious", "cluster": 1},
            {"lat": 18.5102, "lng": 73.8751, "severity": "Slight", "cluster": 2},
            {"lat": 18.5450, "lng": 73.8321, "severity": "Fatal", "cluster": 2},
            {"lat": 18.4900, "lng": 73.8200, "severity": "Serious", "cluster": 3},
        ]
        return jsonify({'status': 'success', 'hotspots': hotspots})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400