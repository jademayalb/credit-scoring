import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:5001"
TIMEOUT = 5

def print_test_header(test_name):
    """Affiche un en-tête de test"""
    print(f"\n{'='*50}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*50}")

def test_api_status():
    """Test de l'endpoint de status"""
    print_test_header("API Status")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print("Status API: SUCCESS")
            print(f"   Version: {data.get('version')}")
            print(f"   Modèle chargé: {data.get('modele_charge')}")
            print(f"   Seuil optimal: {data.get('seuil_optimal')}")
            return True
        else:
            print(f"Status API: FAILED (Code: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Status API: ERROR - {e}")
        return False

def test_prediction():
    """Test de l'endpoint de prédiction"""
    print_test_header("Test Prediction")
    
    try:
        response = requests.get(f"{BASE_URL}/test_prediction", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print("Test Prediction: SUCCESS")
            print(f"   Probabilité défaut: {data.get('probabilite_defaut')}")
            print(f"   Décision: {data.get('decision')}")
            print(f"   Type: {data.get('type')}")
            return True
        else:
            print(f"Test Prediction: FAILED (Code: {response.status_code})")
            print(f"   Réponse: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Test Prediction: ERROR - {e}")
        return False

def test_model_info():
    """Test de l'endpoint d'informations modèle"""
    print_test_header("Model Info")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print("Model Info: SUCCESS")
            print(f"   Type de modèle: {data.get('model_type')}")
            print(f"   Nombre de features: {data.get('nb_features')}")
            print(f"   Nom du modèle: {data.get('model_name')}")
            return True
        else:
            print(f"Model Info: FAILED (Code: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Model Info: ERROR - {e}")
        return False

def check_api_availability():
    """Vérifie si l'API est disponible"""
    print_test_header("Vérification disponibilité API")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        if response.status_code == 200:
            print("API V1 disponible")
            return True
        else:
            print("API V1 non disponible")
            return False
    except:
        print("API V1 non accessible")
        print(f"Assurez-vous que l'API tourne sur {BASE_URL}")
        return False

def run_all_tests():
    """Exécute tous les tests"""
    print("DÉMARRAGE DES TESTS API V1")
    print(f"URL de base: {BASE_URL}")
    
    # Vérification de disponibilité
    if not check_api_availability():
        print("\nTests arrêtés - API non disponible")
        sys.exit(1)
    
    # Tests des endpoints
    tests = [
        ("Status API", test_api_status),
        ("Test Prediction", test_prediction),
        ("Model Info", test_model_info)
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