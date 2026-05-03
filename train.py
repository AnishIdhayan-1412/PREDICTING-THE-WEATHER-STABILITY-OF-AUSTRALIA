"""
==================================================================
train.py - Weather Prediction Model Training
Software Engineering in AI - Project
==================================================================

Base Repository: PREDICTING-THE-WEATHER-STABILITY-OF-AUSTRALIA
  GitHub: https://github.com/jamjewel/PREDICTING-THE-WEATHER-STABILITY-OF-AUSTRALIA
  
Original Work: Statistical analysis proving weather instability 
  in Australia using paired t-tests (JamJewel, 2019)

Dataset: Rain in Australia (Kaggle)
  URL: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

Our Contributions:
  - Upgraded from statistical testing to ML prediction (Random Forest)
  - Added feature engineering for improved accuracy
  - Integrated LLM-based weather advisory using Groq API
  - Containerized Flask REST API deployment
  
SDG 13: Climate Action - Improving extreme weather preparedness
==================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("WEATHER PREDICTION MODEL TRAINING PIPELINE")
print("Based on: PREDICTING-THE-WEATHER-STABILITY-OF-AUSTRALIA")
print("=" * 60)

# Load dataset
print("\n[1/5] Loading dataset...")
df = pd.read_csv('dataset/weatherAUS.csv')
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Clean data
print("\n[2/5] Cleaning data...")
columns_to_drop = ['RISK_MM', 'Date', 'Location']
df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['RainTomorrow'])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'RainTomorrow' in numeric_cols:
    numeric_cols.remove('RainTomorrow')

imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna('Unknown')
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df = df.dropna()
print(f"Cleaned: {df.shape[0]} rows")

# Feature Engineering
print("\n[3/5] Engineering features...")
df['TempRange'] = df['MaxTemp'] - df['MinTemp']
df['HumidityChange'] = df['Humidity3pm'] - df['Humidity9am']
df['PressureChange'] = df['Pressure3pm'] - df['Pressure9am']

feature_cols = [col for col in df.columns if col != 'RainTomorrow']
X = df[feature_cols]
y = df['RainTomorrow']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print(f"Features: {len(feature_cols)}")

# Train model
print("\n[4/5] Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 60)

# Save model
print("\n[5/5] Saving model...")
artifacts = {
    'model': model,
    'scaler': scaler,
    'imputer': imputer,
    'label_encoders': label_encoders,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'feature_cols': feature_cols
}
joblib.dump(artifacts, 'model.pkl')
print("Model saved as model.pkl")
print("\nTraining complete!")