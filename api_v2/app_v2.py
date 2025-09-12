from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Chargement du modèle complet
try:
    model_data = joblib.load('model_complet.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    imputer = model_data['imputer']
    features = model_data['features']
    optimal_threshold = model_data['optimal_threshold']
    model_name = model_data['model_name']
    
    logger.info(f"Modèle {model_name} chargé avec succès")
    logger.info(f"Seuil optimal: {optimal_threshold}")
    logger.info(f"Nombre de features: {len(features)}")
    
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil avec informations sur l'API V2"""
    if model is None:
        return jsonify({
            "error": "Modèle non chargé",
            "status": "unhealthy"
        }), 500
    
    return jsonify({
        "message": "API de Prédiction de Défaut de Crédit - Version 2.0",
        "model": model_name,
        "version": "2.0",
        "optimal_threshold": optimal_threshold,
        "features_count": len(features),
        "endpoints": {
            "/predict": "POST - Prédiction avec données JSON",
            "/health": "GET - Statut de l'API",
            "/model-info": "GET - Informations détaillées du modèle"
        },
        "improvements": [
            "Seuil optimal calibré (0.52)",
            "Modèle LightGBM optimisé",
            "Meilleur équilibre Précision/Rappel",
            "Preprocessing intégré"
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Informations détaillées sur le modèle"""
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    return jsonify({
        "model_name": model_name,
        "model_type": "LightGBM",
        "version": "2.0",
        "optimal_threshold": optimal_threshold,
        "features": features,
        "features_count": len(features),
        "preprocessing": {
            "scaler": "StandardScaler",
            "imputer": "SimpleImputer",
            "missing_value_strategy": "median"
        },
        "performance_notes": "Optimisé pour équilibrer précision et rappel"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction avec seuil optimal"""
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    try:
        # Récupération des données
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Aucune donnée fournie"}), 400
        
        # Conversion en DataFrame
        df = pd.DataFrame([data])
        logger.info(f"Données reçues: {df.shape}")
        
        # Vérification des features manquantes
        missing_features = set(features) - set(df.columns)
        if missing_features:
            return jsonify({
                "error": "Features manquantes",
                "missing_features": list(missing_features),
                "required_features": features
            }), 400
        
        # Sélection et ordre des features
        df_processed = df[features].copy()
        
        # Preprocessing : Imputation puis Scaling
        df_imputed = pd.DataFrame(
            imputer.transform(df_processed), 
            columns=features
        )
        
        df_scaled = pd.DataFrame(
            scaler.transform(df_imputed), 
            columns=features
        )
        
        # Prédiction
        probabilities = model.predict_proba(df_scaled)
        probability_default = probabilities[0][1]  # Probabilité de défaut
        
        # Application du seuil optimal
        prediction = 1 if probability_default >= optimal_threshold else 0
        
        # Niveau de confiance
        confidence = abs(probability_default - 0.5) * 2
        
        # Classification du risque
        if probability_default < 0.3:
            risk_level = "Faible"
        elif probability_default < optimal_threshold:
            risk_level = "Modéré"
        elif probability_default < 0.7:
            risk_level = "Élevé"
        else:
            risk_level = "Très Élevé"
        
        logger.info(f"Prédiction: {prediction}, Proba: {probability_default:.3f}")
        
        return jsonify({
            "prediction": int(prediction),
            "probability_default": round(float(probability_default), 4),
            "probability_no_default": round(float(1 - probability_default), 4),
            "risk_level": risk_level,
            "confidence": round(float(confidence), 4),
            "model_info": {
                "name": model_name,
                "version": "2.0",
                "threshold_used": optimal_threshold
            },
            "interpretation": {
                "result": "Défaut prédit" if prediction == 1 else "Pas de défaut prédit",
                "recommendation": "Attention particulière requise" if risk_level in ["Élevé", "Très Élevé"] else "Risque acceptable"
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({
            "error": "Erreur lors de la prédiction",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)