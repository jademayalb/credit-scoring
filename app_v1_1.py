from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

print("API Credit Scoring v1.1.0")

# Charger le modèle au démarrage
try:
    model_data = joblib.load("models/model_complet.pkl")
    print("Modèle chargé")
    
    # Afficher des infos sur le modèle
    expected_features = model_data['features']
    print(f"Features attendues: {len(expected_features)}")
    print(f"Premières features: {expected_features[:5]}")
    
except Exception as e:
    print(f"Erreur: {e}")
    model_data = None
    expected_features = []

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        "message": "API Credit Scoring v1.1.0",
        "status": "active",
        "model_loaded": model_data is not None,
        "features_count": len(expected_features) if expected_features else 0,
        "version": "1.1.0",
        "improvements": [
            "Validation des données d'entrée",
            "Gestion intelligente des features manquantes",
            "Messages d'erreur détaillés",
            "Endpoint /features pour voir les features requises"
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check amélioré"""
    return jsonify({
        "status": "healthy",
        "model_status": "loaded" if model_data else "error",
        "version": "1.1.0"
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Retourne la liste des features attendues par le modèle"""
    if model_data is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    return jsonify({
        "features_count": len(expected_features),
        "features": expected_features,
        "sample_data": {
            "description": "Voici quelques features importantes",
            "important_features": expected_features[:10] if expected_features else []
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction avec gestion intelligente des features manquantes"""
    if model_data is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Pas de données fournies",
                "help": "Envoyez un JSON avec les features du client"
            }), 400
        
        # Créer un DataFrame avec toutes les features requises, remplies à 0
        df_full = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Remplir avec les données fournies
        for key, value in data.items():
            if key in expected_features:
                df_full[key] = value
        
        # Compter les features fournies vs manquantes
        provided_features = [k for k in data.keys() if k in expected_features]
        missing_features = [f for f in expected_features if f not in data.keys()]
        
        # Preprocessing
        X_imputed = model_data['imputer'].transform(df_full)
        X_scaled = model_data['scaler'].transform(X_imputed)
        
        # Prédiction
        proba = model_data['model'].predict_proba(X_scaled)[0, 1]
        
        # Déterminer la décision (seuil basique à 0.5)
        decision = "REFUSE" if proba > 0.5 else "ACCEPTE"
        
        # Déterminer le niveau de risque
        if proba < 0.3:
            risk_level = "FAIBLE"
        elif proba < 0.7:
            risk_level = "MOYEN"
        else:
            risk_level = "ÉLEVÉ"
        
        return jsonify({
            "probabilite_defaut": float(proba),
            "decision": decision,
            "niveau_risque": risk_level,
            "details": {
                "features_fournies": len(provided_features),
                "features_manquantes": len(missing_features),
                "total_features": len(expected_features),
                "completude": f"{len(provided_features)/len(expected_features)*100:.1f}%"
            },
            "warning": "Prédiction basée sur des features partielles - résultat indicatif" if len(missing_features) > len(expected_features) * 0.5 else None
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "help": "Vérifiez le format des données et consultez /features pour voir les features attendues"
        }), 500

@app.route('/predict/simple', methods=['POST'])
def predict_simple():
    """Prédiction simplifiée avec les features les plus importantes"""
    if model_data is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Pas de données"}), 400
        
        # Features importantes minimales
        important_features = [
            'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_ANNUITY',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'
        ]
        
        # Vérifier si on a au moins quelques features importantes
        provided_important = [f for f in important_features if f in data]
        
        if len(provided_important) < 3:
            return jsonify({
                "error": "Features importantes manquantes",
                "required_minimum": important_features,
                "help": "Fournissez au moins 3 des features importantes listées"
            }), 400
        
        # Même logique que predict
        df_full = pd.DataFrame(0, index=[0], columns=expected_features)
        for key, value in data.items():
            if key in expected_features:
                df_full[key] = value
        
        X_imputed = model_data['imputer'].transform(df_full)
        X_scaled = model_data['scaler'].transform(X_imputed)
        proba = model_data['model'].predict_proba(X_scaled)[0, 1]
        
        decision = "REFUSE" if proba > 0.5 else "ACCEPTE"
        
        return jsonify({
            "probabilite_defaut": float(proba),
            "decision": decision,
            "qualite_prediction": "simplifiee",
            "features_importantes_fournies": provided_important
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)