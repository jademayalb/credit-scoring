import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_client_not_found(client):
    response = client.get('/predict/999999999')  # ID inexistant
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "NOT_FOUND"

def test_api_client_found(client):
    response = client.get('/predict/100001')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "OK"
    assert "probabilite_defaut" in data
    assert "decision" in data