import pytest
import requests
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Ajouter le chemin vers api_v1
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api_v1'))

class TestAPIV1:
    """Tests unitaires pour l'API V1"""
    
    BASE_URL = "http://localhost:5001"
    
    def test_api_status(self):
        """Test endpoint status API V1"""
        try:
            response = requests.get(f"{self.BASE_URL}/")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "OK"
            assert data["version"] == "1.0"
            assert "seuil_optimal" in data
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API V1 non démarrée")
    
    def test_model_info(self):
        """Test endpoint model_info"""
        try:
            response = requests.get(f"{self.BASE_URL}/model_info")
            assert response.status_code == 200
            
            data = response.json()
            assert "model_type" in data
            assert "nb_features" in data
            assert data["version"] == "1.0"
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API V1 non démarrée")
    
    def test_prediction(self):
        """Test endpoint test_prediction"""
        try:
            response = requests.get(f"{self.BASE_URL}/test_prediction")
            assert response.status_code == 200
            
            data = response.json()
            assert "probabilite_defaut" in data
            assert "decision" in data
            assert data["decision"] in ["ACCEPTÉ", "REFUSÉ"]
            assert 0 <= data["probabilite_defaut"] <= 1
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API V1 non démarrée")

# Tests sans serveur (mocking)
class TestAPIV1Unit:
    """Tests unitaires avec mocking"""
    
    @patch('joblib.load')
    def test_model_loading(self, mock_joblib):
        """Test chargement du modèle"""
        # Mock du modèle
        mock_model_data = {
            'model': MagicMock(),
            'scaler': MagicMock(), 
            'imputer': MagicMock(),
            'features': ['feature1', 'feature2'],
            'optimal_threshold': 0.52,
            'model_name': 'LightGBM_Test'
        }
        mock_joblib.return_value = mock_model_data
        
        # Import après le mock
        from app import model_data
        assert model_data is not None
    
    def test_threshold_logic(self):
        """Test logique du seuil"""
        threshold = 0.52
        
        # Test cas accepté
        prob_low = 0.3
        decision_low = "REFUSÉ" if prob_low >= threshold else "ACCEPTÉ"
        assert decision_low == "ACCEPTÉ"
        
        # Test cas refusé  
        prob_high = 0.7
        decision_high = "REFUSÉ" if prob_high >= threshold else "ACCEPTÉ"
        assert decision_high == "REFUSÉ"
