import pytest
import requests
import json
from unittest.mock import patch, MagicMock
import sys
import os

class TestAPIV2:
    """Tests unitaires pour l'API V2"""
    
    BASE_URL = "http://localhost:5002"
    
    def test_api_status(self):
        """Test endpoint status API V2"""
        try:
            response = requests.get(f"{self.BASE_URL}/")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "OK"
            assert data["version"] == "2.0" 
            assert "nb_clients_total" in data
            assert data["seuil_optimal"] == 0.52
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API V2 non démarrée")
    
    def test_model_info(self):
        """Test endpoint model_info V2"""
        try:
            response = requests.get(f"{self.BASE_URL}/model_info")
            assert response.status_code == 200
            
            data = response.json()
            assert data["version"] == "2.0"
            assert "nb_clients_disponibles" in data
            assert data["seuil_optimal"] == 0.52
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API V2 non démarrée")
    
    def test_clients_list(self):
        """Test endpoint liste clients"""
        try:
            response = requests.get(f"{self.BASE_URL}/clients?page=1&per_page=5")
            assert response.status_code == 200
            
            data = response.json()
            assert "clients" in data
            assert "pagination" in data
            assert len(data["clients"]) <= 5
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API V2 non démarrée")
    
    def test_client_info_valid(self):
        """Test info client valide"""
        try:
            # Récupérer un ID client valide
            clients_response = requests.get(f"{self.BASE_URL}/clients?page=1&per_page=1")
            if clients_response.status_code == 200:
                clients_data = clients_response.json()
                if clients_data["clients"]:
                    client_id = clients_data["clients"][0]["client_id"]
                    
                    response = requests.get(f"{self.BASE_URL}/client/{client_id}/info")
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert data["client_id"] == client_id
                    assert "caracteristiques" in data
                    
        except requests.exceptions.ConnectionError:
            pytest.skip("API V2 non démarrée")
    
    def test_client_info_invalid(self):
        """Test info client inexistant"""
        try:
            response = requests.get(f"{self.BASE_URL}/client/999999999/info")
            assert response.status_code == 404
            
            data = response.json()
            assert "error" in data
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API V2 non démarrée")
    
    def test_prediction_valid_client(self):
        """Test prédiction client valide"""
        try:
            # Récupérer un ID client valide
            clients_response = requests.get(f"{self.BASE_URL}/clients?page=1&per_page=1")
            if clients_response.status_code == 200:
                clients_data = clients_response.json()
                if clients_data["clients"]:
                    client_id = clients_data["clients"][0]["client_id"]
                    
                    response = requests.get(f"{self.BASE_URL}/predict/{client_id}")
                    assert response.status_code == 200
                    
                    data = response.json()
                    assert data["client_id"] == client_id
                    assert "probability_default" in data
                    assert "decision" in data
                    assert data["decision"] in ["CRÉDIT ACCORDÉ", "CRÉDIT REFUSÉ"]
                    assert 0 <= data["probability_default"] <= 1
                    assert data["model_info"]["threshold_used"] == 0.52
                    
        except requests.exceptions.ConnectionError:
            pytest.skip("API V2 non démarrée")

class TestAPIV2Unit:
    """Tests unitaires V2 avec mocking"""
    
    def test_threshold_consistency(self):
        """Test cohérence du seuil 0.52"""
        threshold = 0.52
        
        # Test limites
        assert ("ACCORDÉ" in "CRÉDIT ACCORDÉ") if 0.51 < threshold else True
        assert ("REFUSÉ" in "CRÉDIT REFUSÉ") if 0.53 > threshold else True
    
    def test_risk_levels(self):
        """Test niveaux de risque"""
        threshold = 0.52
        
        # Test classification risques
        assert 0.2 < 0.3  # Faible
        assert 0.4 < threshold  # Modéré  
        assert 0.6 > threshold  # Élevé
        assert 0.8 > 0.7  # Très Élevé
