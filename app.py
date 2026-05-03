"""
==================================================================
app.py - Weather Prediction + AI Daily Advisor API
Software Engineering in AI - Project
==================================================================
Flask REST API with:
  - POST /predict  : ML prediction (RainTomorrow)
  - POST /advice   : LLM-generated weather advice (Groq API)
  
SDG 13: Climate Action
==================================================================
"""
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import requests
import os

app = Flask(__name__)

# Load model artifacts
print("Loading model...")
artifacts = joblib.load('model.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
imputer = artifacts['imputer']
label_encoders = artifacts['label_encoders']
numeric_cols = artifacts['numeric_cols']
categorical_cols = artifacts['categorical_cols']
feature_cols = artifacts['feature_cols']
print("Model loaded successfully!")

# Groq API Config
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'gsk_your_api_key_here')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

EXPECTED_FEATURES = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
    'Temp9am', 'Temp3pm', 'RainToday'
]

@app.route('/')
def home():
    return '''
    <h1>Weather Prediction + AI Advisor API</h1>
    <p><b>POST /predict</b> - Predict rain tomorrow</p>
    <p><b>POST /advice</b> - Get AI weather advice</p>
    <p>SDG 13: Climate Action</p>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data'}), 400
        
        df = pd.DataFrame([data])
        
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df[numeric_cols] = imputer.transform(df[numeric_cols])
        
        for col in categorical_cols:
            if col in df.columns:
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except:
                    df[col] = 0
        
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['HumidityChange'] = df['Humidity3pm'] - df['Humidity9am']
        df['PressureChange'] = df['Pressure3pm'] - df['Pressure9am']
        
        df = df[feature_cols]
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0]
        confidence = float(max(prob))
        
        return jsonify({
            'prediction': 'Yes' if pred == 1 else 'No',
            'rain_prediction': bool(pred),
            'confidence': confidence,
            'probability_rain': float(prob[1]),
            'probability_no_rain': float(prob[0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/advice', methods=['POST'])
def advice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data'}), 400
        
        location = data.get('location', 'your area')
        weather = data.get('weather_data', {})
        prediction = data.get('prediction', {})
        
        rain_status = "rain" if prediction.get('rain_prediction') else "no rain"
        confidence = prediction.get('confidence', 0)
        
        prompt = f"Weather for {location}: Temp {weather.get('MinTemp', 'N/A')} to {weather.get('MaxTemp', 'N/A')}C, Rainfall {weather.get('Rainfall', 'N/A')}mm, Humidity9am {weather.get('Humidity9am', 'N/A')}%, Humidity3pm {weather.get('Humidity3pm', 'N/A')}%, Wind {weather.get('WindSpeed3pm', 'N/A')}km/h, Cloud {weather.get('Cloud3pm', 'N/A')} oktas. Rain prediction: {rain_status} (Confidence: {confidence:.0%}). Give 5-6 sentences of practical weather advice."

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return jsonify({'error': f'Groq API error: {response.text}'}), 500
        
        result = response.json()
        advice_text = result['choices'][0]['message']['content']
        
        return jsonify({
            'location': location,
            'rain_prediction': prediction.get('rain_prediction'),
            'confidence': confidence,
            'advice': advice_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)