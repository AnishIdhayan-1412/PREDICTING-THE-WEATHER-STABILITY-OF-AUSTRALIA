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
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
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
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather AI Advisor</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #1a73e8, #0d47a1); min-height: 100vh; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: white; text-align: center; margin-bottom: 10px; font-size: 2em; }
            .subtitle { color: #e0e0e0; text-align: center; margin-bottom: 30px; }
            .card { background: white; border-radius: 15px; padding: 30px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
            .card h2 { color: #1a73e8; margin-bottom: 20px; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; font-weight: 600; color: #333; margin-bottom: 5px; }
            input, select { width: 100%; padding: 10px; border: 2px solid #ddd; border-radius: 8px; font-size: 14px; }
            input:focus { border-color: #1a73e8; outline: none; }
            .row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            button { background: #1a73e8; color: white; padding: 12px 30px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; width: 100%; margin-top: 10px; }
            button:hover { background: #0d5bbd; }
            .result { margin-top: 20px; padding: 20px; background: #f5f5f5; border-radius: 10px; display: none; }
            .result.show { display: block; }
            .prediction-yes { color: #d32f2f; font-weight: bold; }
            .prediction-no { color: #2e7d32; font-weight: bold; }
            .advice-box { background: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #2e7d32; }
            .loading { text-align: center; color: #666; display: none; }
            .loading.show { display: block; }
            .confidence-bar { background: #e0e0e0; border-radius: 10px; height: 20px; margin-top: 10px; overflow: hidden; }
            .confidence-fill { background: linear-gradient(90deg, #2e7d32, #4caf50); height: 100%; border-radius: 10px; transition: width 0.5s; }
            .tab { display: flex; margin-bottom: 20px; }
            .tab button { background: #e0e0e0; color: #333; border: none; padding: 10px 20px; cursor: pointer; border-radius: 8px 8px 0 0; margin-right: 5px; width: auto; }
            .tab button.active { background: #1a73e8; color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌤️ Weather AI Advisor</h1>
            <p class="subtitle">ML-Powered Prediction + AI Advice | SDG 13: Climate Action</p>
            
            <div class="card">
                <div class="tab">
                    <button class="active" onclick="switchTab('predict')">🔮 Predict Rain</button>
                    <button onclick="switchTab('advice')">💬 Get Advice</button>
                </div>
                
                <div id="predict-tab" class="tab-content active">
                    <h2>Enter Weather Details</h2>
                    <div class="row">
                        <div class="form-group"><label>Min Temp (°C)</label><input type="number" id="MinTemp" value="13.4" step="0.1"></div>
                        <div class="form-group"><label>Max Temp (°C)</label><input type="number" id="MaxTemp" value="22.9" step="0.1"></div>
                        <div class="form-group"><label>Rainfall (mm)</label><input type="number" id="Rainfall" value="0.6" step="0.1"></div>
                        <div class="form-group"><label>Evaporation (mm)</label><input type="number" id="Evaporation" value="4.8" step="0.1"></div>
                        <div class="form-group"><label>Sunshine (hours)</label><input type="number" id="Sunshine" value="1.2" step="0.1"></div>
                        <div class="form-group"><label>Wind Gust Dir</label><input type="text" id="WindGustDir" value="W"></div>
                        <div class="form-group"><label>Wind Gust Speed (km/h)</label><input type="number" id="WindGustSpeed" value="44"></div>
                        <div class="form-group"><label>Wind Dir 9am</label><input type="text" id="WindDir9am" value="W"></div>
                        <div class="form-group"><label>Wind Dir 3pm</label><input type="text" id="WindDir3pm" value="WNW"></div>
                        <div class="form-group"><label>Wind Speed 9am</label><input type="number" id="WindSpeed9am" value="20"></div>
                        <div class="form-group"><label>Wind Speed 3pm</label><input type="number" id="WindSpeed3pm" value="24"></div>
                        <div class="form-group"><label>Humidity 9am (%)</label><input type="number" id="Humidity9am" value="71"></div>
                        <div class="form-group"><label>Humidity 3pm (%)</label><input type="number" id="Humidity3pm" value="22"></div>
                        <div class="form-group"><label>Pressure 9am (hPa)</label><input type="number" id="Pressure9am" value="1007.7" step="0.1"></div>
                        <div class="form-group"><label>Pressure 3pm (hPa)</label><input type="number" id="Pressure3pm" value="1007.1" step="0.1"></div>
                        <div class="form-group"><label>Cloud 9am (oktas)</label><input type="number" id="Cloud9am" value="8"></div>
                        <div class="form-group"><label>Cloud 3pm (oktas)</label><input type="number" id="Cloud3pm" value="5"></div>
                        <div class="form-group"><label>Temp 9am (°C)</label><input type="number" id="Temp9am" value="16.9" step="0.1"></div>
                        <div class="form-group"><label>Temp 3pm (°C)</label><input type="number" id="Temp3pm" value="21.8" step="0.1"></div>
                        <div class="form-group"><label>Rain Today</label><select id="RainToday"><option value="No">No</option><option value="Yes">Yes</option></select></div>
                    </div>
                    <button onclick="predictRain()">🔮 Predict Rain Tomorrow</button>
                    <div id="predict-loading" class="loading">⏳ Analyzing weather data...</div>
                    <div id="predict-result" class="result"></div>
                </div>
                
                <div id="advice-tab" class="tab-content">
                    <h2>Get AI Weather Advice</h2>
                    <div class="form-group"><label>📍 Location</label><input type="text" id="location" value="Sydney" placeholder="Enter city name"></div>
                    <button onclick="getAdvice()">💬 Get AI Advice</button>
                    <div id="advice-loading" class="loading">🤖 AI is generating advice...</div>
                    <div id="advice-result" class="result"></div>
                </div>
            </div>
        </div>
        
        <script>
            function switchTab(tab) {
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab button').forEach(b => b.classList.remove('active'));
                document.getElementById(tab + '-tab').classList.add('active');
                event.target.classList.add('active');
            }
            
            async function predictRain() {
                document.getElementById('predict-loading').classList.add('show');
                document.getElementById('predict-result').classList.remove('show');
                
                const data = {};
                ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday'].forEach(id => {
                    data[id] = document.getElementById(id).value;
                });
                
                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await res.json();
                    
                    const predClass = result.rain_prediction ? 'prediction-yes' : 'prediction-no';
                    const emoji = result.rain_prediction ? '🌧️' : '☀️';
                    
                    document.getElementById('predict-result').innerHTML = 
                        `<h3>${emoji} Prediction: <span class="${predClass}">${result.prediction}</span></h3>
                         <p>Confidence: <b>${(result.confidence * 100).toFixed(1)}%</b></p>
                         <div class="confidence-bar"><div class="confidence-fill" style="width:${result.confidence * 100}%"></div></div>
                         <p style="margin-top:10px;">Rain Probability: ${(result.probability_rain * 100).toFixed(1)}%</p>`;
                    document.getElementById('predict-result').classList.add('show');
                } catch(e) {
                    document.getElementById('predict-result').innerHTML = '<p style="color:red;">Error: ' + e.message + '</p>';
                    document.getElementById('predict-result').classList.add('show');
                }
                document.getElementById('predict-loading').classList.remove('show');
            }
            
            async function getAdvice() {
                document.getElementById('advice-loading').classList.add('show');
                document.getElementById('advice-result').classList.remove('show');
                
                const weatherData = {};
                ['MinTemp','MaxTemp','Rainfall','Humidity9am','Humidity3pm','WindSpeed3pm','Cloud3pm'].forEach(id => {
                    weatherData[id] = document.getElementById(id).value;
                });
                
                // First get prediction
                const allData = {};
                ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday'].forEach(id => {
                    allData[id] = document.getElementById(id).value;
                });
                
                try {
                    const predRes = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(allData)
                    });
                    const predResult = await predRes.json();
                    
                    const advRes = await fetch('/advice', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            location: document.getElementById('location').value,
                            weather_data: weatherData,
                            prediction: {rain_prediction: predResult.rain_prediction, confidence: predResult.confidence}
                        })
                    });
                    const advResult = await advRes.json();
                    
                    const predClass = predResult.rain_prediction ? 'prediction-yes' : 'prediction-no';
                    const emoji = predResult.rain_prediction ? '🌧️' : '☀️';
                    
                    document.getElementById('advice-result').innerHTML = 
                        `<h3>${emoji} Rain Prediction: <span class="${predClass}">${predResult.prediction}</span> (${(predResult.confidence*100).toFixed(1)}% confidence)</h3>
                         <div class="advice-box"><strong>🤖 AI Advice:</strong><br>${advResult.advice || advResult.error}</div>`;
                    document.getElementById('advice-result').classList.add('show');
                } catch(e) {
                    document.getElementById('advice-result').innerHTML = '<p style="color:red;">Error: ' + e.message + '</p>';
                    document.getElementById('advice-result').classList.add('show');
                }
                document.getElementById('advice-loading').classList.remove('show');
            }
        </script>
    </body>
    </html>
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