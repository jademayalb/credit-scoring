from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

print("API Credit Scoring v1.0.0")

# Charger le modèle au démarrage
try:
    model_data = joblib.load("models/model_complet.pkl")
    print("Modèle chargé")
except Exception as e:
    print(f"Erreur: {e}")
    model_data = None

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        "message": "API Credit Scoring v1.0.0",
        "status": "active",
        "model_loaded": model_data is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction basique"""
    if model_data is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Pas de données"}), 400
            
        # Conversion simple
        df = pd.DataFrame([data])
        X = df[model_data['features']]
        
        # Preprocessing
        X_imputed = model_data['imputer'].transform(X)
        X_scaled = model_data['scaler'].transform(X_imputed)
        
        # Prédiction avec seuil 0.5 (basique)
        proba = model_data['model'].predict_proba(X_scaled)[0, 1]
        decision = "REFUSE" if proba > 0.5 else "ACCEPTE"
        
        return jsonify({
            "probabilite_defaut": float(proba),
            "decision": decision
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 