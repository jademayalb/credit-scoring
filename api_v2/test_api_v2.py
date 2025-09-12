import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:5000"
TIMEOUT = 5

def print_test_header(test_name):
    """Affiche un en-tête de test"""
    print(f"\n{'='*50}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*50}")

def test_home():
    """Test de l'endpoint d'accueil"""
    print_test_header("Home Page")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print("Home Page: SUCCESS")
            print(f"   Version: {data.get('version')}")
            print(f"   Modèle: {data.get('model')}")
            print(f"   Seuil optimal: {data.get('optimal_threshold')}")
            print(f"   Nombre de features: {data.get('features_count')}")
            return True
        else:
            print(f"Home Page: FAILED (Code: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Home Page: ERROR - {e}")
        return False

def test_health():
    """Test de l'endpoint de santé"""
    print_test_header("Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print("Health Check: SUCCESS")
            print(f"   Status: {data.get('status')}")
            print(f"   Modèle chargé: {data.get('model_loaded')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"Health Check: FAILED (Code: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Health Check: ERROR - {e}")
        return False

def test_model_info():
    """Test de l'endpoint d'informations détaillées"""
    print_test_header("Model Info Detailed")
    
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print("Model Info: SUCCESS")
            print(f"   Nom: {data.get('model_name')}")
            print(f"   Type: {data.get('model_type')}")
            print(f"   Features: {data.get('features_count')}")
            print(f"   Preprocessing: {data.get('preprocessing', {}).get('scaler')}")
            return True
        else:
            print(f"Model Info: FAILED (Code: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Model Info: ERROR - {e}")
        return False

def test_prediction_with_data():
    """Test de prédiction avec données d'exemple"""
    print_test_header("Prediction with Sample Data")
    
    # Données d'exemple (à adapter selon vos features réelles)
    sample_data = {
        "feature1": 0.5,
        "feature2": -0.2,
        "feature3": 1.0,
        "feature4": 0.0
        # Ajoutez d'autres features selon votre modèle
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_data,
            headers={'Content-Type': 'application/json'},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print("Prediction: SUCCESS")
            print(f"   Prédiction: {data.get('prediction')}")
            print(f"   Probabilité défaut: {data.get('probability_default')}")
            print(f"   Niveau de risque: {data.get('risk_level')}")
            print(f"   Confiance: {data.get('confidence')}")
            print(f"   Résultat: {data.get('interpretation', {}).get('result')}")
            return True
        else:
            print(f"Prediction: FAILED (Code: {response.status_code})")
            print(f"   Réponse: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Prediction: ERROR - {e}")
        return False

def test_prediction_error_handling():
    """Test de gestion d'erreurs pour prédiction"""
    print_test_header("Prediction Error Handling")
    
    # Test avec données manquantes
    incomplete_data = {"feature1": 0.5}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=incomplete_data,
            headers={'Content-Type': 'application/json'},
            timeout=TIMEOUT
        )
        
        if response.status_code == 400:
            data = response.json()
            print("Error Handling: SUCCESS")
            print(f"   Erreur détectée: {data.get('error')}")
            print(f"   Features manquantes: {len(data.get('missing_features', []))}")
            return True
        else:
            print(f"Error Handling: FAILED - Devrait retourner 400")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error Handling: ERROR - {e}")
        return False

def check_api_availability():
    """Vérifie si l'API est disponible"""
    print_test_header("Vérification disponibilité API")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("API V2 disponible")
            return True
        else:
            print("API V2 non disponible")
            return False
    except:
        print("API V2 non accessible")
        print(f"Assurez-vous que l'API tourne sur {BASE_URL}")
        return False

def run_all_tests():
    """Exécute tous les tests"""
    print("🚀 DÉMARRAGE DES TESTS API V2")
    print(f"URL de base: {BASE_URL}")
    
    # Vérification de disponibilité
    if not check_api_availability():
        print("\nTests arrêtés - API non disponible")
        sys.exit(1)
    
    # Tests des endpoints
    tests = [
        ("Home Page", test_home),
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Prediction", test_prediction_with_data),
        ("Error Handling", test_prediction_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        time.sleep(1)  # Pause entre les tests
    
    # Résumé
    print_test_header("RÉSUMÉ DES TESTS")
    success_count = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\nRésultat: {success_count}/{len(results)} tests réussis")
    
    if success_count == len(results):
        print("TOUS LES TESTS SONT PASSÉS!")
        return True
    else:
        print("⚠CERTAINS TESTS ONT ÉCHOUÉ")
        return False

if __name__ == "__main__":
    run_all_tests()