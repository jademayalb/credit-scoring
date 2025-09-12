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

# Chargement du modèle et des données
try:
    model_data = joblib.load('model_complet.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    imputer = model_data['imputer']
    features = model_data['features']
    optimal_threshold = model_data.get('optimal_threshold', 0.52)  
    model_name = model_data.get('model_name', 'LightGBM_Balanced')
    
    # Chargement des données clients
    clients_data = pd.read_csv('application_test.csv')
    logger.info(f"Données clients chargées: {len(clients_data)} clients")
    
    logger.info(f"Modèle {model_name} chargé avec succès")
    logger.info(f"Seuil optimal: {optimal_threshold}")  
    logger.info(f"Nombre de features: {len(features)}")
    
except Exception as e:
    logger.error(f"Erreur lors du chargement: {e}")
    model = None
    clients_data = None

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil avec informations sur l'API V2"""
    if model is None or clients_data is None:
        return jsonify({
            "error": "Modèle ou données non chargés",
            "status": "unhealthy"
        }), 500
    
    return jsonify({
        "message": "API Scoring Crédit v2.0 - VRAIES DONNÉES KAGGLE",
        "version": "2.0",
        "status": "OK",
        "modele_charge": True,
        "donnees_chargees": True,
        "nb_clients_total": len(clients_data),
        "seuil_optimal": float(optimal_threshold),  
        "source_donnees": "application_test.csv (Kaggle Home Credit)",
        "description": "API avec prédictions sur vrais clients du challenge Kaggle",
        "endpoints": {
            "GET /": "Informations API",
            "GET /clients?page=1&per_page=20": "Liste clients avec pagination",
            "GET /client/<client_id>/info": "Profil détaillé du client",
            "GET /predict/<client_id>": "Prédiction crédit pour client réel",
            "GET /model_info": "Informations techniques du modèle"
        },
        "exemples": {
            "info_client": "/client/100001/info",
            "test_prediction": "/predict/100001"
        }
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Informations détaillées sur le modèle"""
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    return jsonify({
        "model_name": model_name,
        "model_type": str(type(model)),
        "version": "2.0",
        "seuil_optimal": float(optimal_threshold), 
        "nb_features": len(features),
        "nb_clients_disponibles": len(clients_data) if clients_data is not None else 0,
        "source_donnees": "application_test.csv",
        "preprocessing": {
            "scaler": str(type(scaler)),
            "imputer": str(type(imputer))
        },
        "exemples_clients": list(clients_data['SK_ID_CURR'].head(5)) if clients_data is not None else []
    })

@app.route('/clients', methods=['GET'])
def list_clients():
    """Liste des clients avec pagination"""
    if clients_data is None:
        return jsonify({"error": "Données clients non chargées"}), 500
    
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 20)), 100)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    clients_page = clients_data.iloc[start_idx:end_idx]
    
    return jsonify({
        "clients": [
            {
                "client_id": int(row['SK_ID_CURR']),
                "index": idx
            } for idx, row in clients_page.iterrows()
        ],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": len(clients_data),
            "total_pages": (len(clients_data) + per_page - 1) // per_page
        }
    })

@app.route('/client/<int:client_id>/info', methods=['GET'])
def client_info(client_id):
    """Informations détaillées d'un client"""
    if clients_data is None:
        return jsonify({"error": "Données clients non chargées"}), 500
    
    client_row = clients_data[clients_data['SK_ID_CURR'] == client_id]
    
    if client_row.empty:
        return jsonify({"error": f"Client {client_id} non trouvé"}), 404
    
    client_data = client_row.iloc[0]
    
    # Features importantes à afficher
    important_features = {
        'CODE_GENDER': 'Genre',
        'DAYS_BIRTH': 'Âge (en jours)',
        'DAYS_EMPLOYED': 'Ancienneté emploi (en jours)',
        'AMT_INCOME_TOTAL': 'Revenus totaux',
        'AMT_CREDIT': 'Montant crédit',
        'AMT_ANNUITY': 'Annuités'
    }
    
    client_info_dict = {
        "client_id": int(client_id),
        "features_disponibles": len([col for col in client_data.index if pd.notna(client_data[col])]),
        "features_manquantes": len([col for col in client_data.index if pd.isna(client_data[col])]),
        "caracteristiques": {}
    }
    
    for feature, description in important_features.items():
        if feature in client_data.index:
            value = client_data[feature]
            if pd.notna(value):
                client_info_dict["caracteristiques"][description] = value
            else:
                client_info_dict["caracteristiques"][description] = "Non renseigné"
    
    return jsonify(client_info_dict)

@app.route('/predict/<int:client_id>', methods=['GET'])
def predict_client(client_id):
    """Prédiction pour un client spécifique"""
    if model is None or clients_data is None:
        return jsonify({"error": "Modèle ou données non chargés"}), 500
    
    client_row = clients_data[clients_data['SK_ID_CURR'] == client_id]
    
    if client_row.empty:
        return jsonify({"error": f"Client {client_id} non trouvé"}), 404
    
    try:
        # Préparer les données client
        client_data = client_row.iloc[0]
        
        # Vérifier si toutes les features sont disponibles
        missing_features = [f for f in features if f not in clients_data.columns]
        if missing_features:
            logger.warning(f"Features manquantes dans CSV: {len(missing_features)}")
        
        # Sélectionner les features disponibles
        available_features = [f for f in features if f in clients_data.columns]
        client_features = client_data[available_features].values.reshape(1, -1)
        
        # Créer un DataFrame avec toutes les features (NaN pour manquantes)
        full_features_data = pd.DataFrame(columns=features)
        for i, feature in enumerate(available_features):
            full_features_data.loc[0, feature] = client_features[0][i]
        
        # Preprocessing
        client_imputed = imputer.transform(full_features_data)
        client_scaled = scaler.transform(client_imputed)
        
        # Prédiction
        probabilities = model.predict_proba(client_scaled)
        probability_default = probabilities[0][1]
        
        # Application du seuil 0.52 ✅
        prediction = 1 if probability_default >= optimal_threshold else 0
        
        # Classification du risque avec seuil 0.52
        if probability_default < 0.3:
            risk_level = "Faible"
            color = "green"
        elif probability_default < optimal_threshold:  # < 0.52
            risk_level = "Modéré"
            color = "orange"
        elif probability_default < 0.7:
            risk_level = "Élevé"
            color = "red"
        else:
            risk_level = "Très Élevé"
            color = "darkred"
        
        logger.info(f"Prédiction client {client_id}: {prediction}, Proba: {probability_default:.3f}")
        
        return jsonify({
            "client_id": int(client_id),
            "prediction": int(prediction),
            "probability_default": round(float(probability_default), 4),
            "probability_no_default": round(float(1 - probability_default), 4),
            "risk_level": risk_level,
            "risk_color": color,
            "decision": "CRÉDIT REFUSÉ" if prediction == 1 else "CRÉDIT ACCORDÉ",
            "confidence": round(abs(probability_default - 0.5) * 2, 4),
            "model_info": {
                "name": model_name,
                "threshold_used": float(optimal_threshold),  
                "version": "2.0",
                "features_used": len(available_features),
                "features_missing": len(missing_features)
            },
            "interpretation": {
                "message": f"Probabilité de défaut: {probability_default:.1%}",
                "recommendation": "Attention particulière requise" if risk_level in ["Élevé", "Très Élevé"] else "Dossier acceptable"
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur prédiction client {client_id}: {e}")
        return jsonify({
            "error": "Erreur lors de la prédiction",
            "details": str(e),
            "client_id": int(client_id)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
