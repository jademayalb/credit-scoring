from flask import Flask, jsonify, request
import joblib
import numpy as np
import os
from datetime import datetime

# Créer l'application Flask
app = Flask(__name__)

print("DÉMARRAGE API CREDIT SCORING V1.0")
print(f"Heure de démarrage: {datetime.now()}")

# Charger le modèle au démarrage
MODEL_PATH = "model_complet.pkl"
model_data = None

try:
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        print("Modèle chargé avec succès")
        print(f"Algorithme: {type(model_data['model'])}")
        print(f"Nombre de features: {len(model_data['features'])}")
        print(f"Seuil optimal: {model_data['optimal_threshold']:.3f}")
    else:
        print("Fichier modèle introuvable")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")

# Route de base pour tester que l'API fonctionne
@app.route('/')
def accueil():
    """Endpoint de base pour vérifier le statut de l'API"""
    return jsonify({
        "message": "🚀 API Scoring Crédit V1.0",
        "status": "OK",
        "modele_charge": model_data is not None,
        "timestamp": datetime.now().isoformat(),
        "endpoints_disponibles": [
            "/",
            "/health",
            "/test_prediction",
            "/model_info"
        ]
    })

@app.route('/health')
def health_check():
    """Endpoint de santé pour le monitoring"""
    return jsonify({
        "status": "healthy" if model_data is not None else "unhealthy",
        "model_loaded": model_data is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/model_info')
def model_info():
    """Informations sur le modèle chargé"""
    if model_data is None:
        return jsonify({"erreur": "Modèle non chargé"}), 500
    
    return jsonify({
        "algorithme": str(type(model_data['model']).__name__),
        "nombre_features": len(model_data['features']),
        "seuil_optimal": float(model_data['optimal_threshold']),
        "features_sample": model_data['features'][:10],
        "version": "1.0"
    })

@app.route('/test_prediction', methods=['GET'])
def test_prediction():
    """Test de prédiction avec données aléatoires"""
    if model_data is None:
        return jsonify({"erreur": "Modèle non chargé"}), 500
    
    try:
        # Créer des données bidon pour tester
        np.random.seed(42)  # Pour reproductibilité
        donnees_bidon = np.random.rand(1, len(model_data['features']))
        
        # Faire le preprocessing
        donnees_preprocessed = model_data['scaler'].transform(
            model_data['imputer'].transform(donnees_bidon)
        )
        
        # Prédiction
        probabilite = model_data['model'].predict_proba(donnees_preprocessed)[0, 1]
        decision = "REFUSÉ" if probabilite >= model_data['optimal_threshold'] else "ACCEPTÉ"
        
        return jsonify({
            "message": "Test de prédiction réussi",
            "probabilite_defaut": float(probabilite),
            "probabilite_pourcentage": f"{probabilite*100:.2f}%",
            "seuil_optimal": float(model_data['optimal_threshold']),
            "decision": decision,
            "confiance": "HAUTE" if abs(probabilite - 0.5) > 0.3 else "MOYENNE",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "erreur": f"Problème lors de la prédiction: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# Gestionnaire d'erreur global
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "erreur": "Endpoint non trouvé",
        "endpoints_disponibles": ["/", "/health", "/test_prediction", "/model_info"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "erreur": "Erreur interne du serveur",
        "message": "Contactez l'administrateur"
    }), 500

# Lancer l'API
if __name__ == '__main__':
    print("API disponible sur: http://localhost:5001")
    print("Endpoints disponibles:")
    print("   - GET / : Informations générales")
    print("   - GET /health : Statut de santé")
    print("   - GET /test_prediction : Test de prédiction")
    print("   - GET /model_info : Informations du modèle")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)