# API Credit Scoring - Version 1.0

## Description

API simple pour tester le modèle de scoring crédit avec données aléatoires. Parfaite pour les démonstrations et tests rapides.

## Installation

```bash
cd api_v1
pip install -r requirements.txt
python app.py
```

## Endpoints

### `GET /` - Status API
Vérification du statut de l'API et informations de base.

**Réponse :**
```json
{
    "message": "API Scoring Crédit v1.0 - Simple",
    "status": "OK",
    "version": "1.0",
    "modele_charge": true,
    "seuil_optimal": 0.52,
    "description": "API simple avec données aléatoires pour test"
}
```

### `GET /test_prediction` - Test avec données aléatoires
Génère des données aléatoires et effectue une prédiction de test.

**Réponse :**
```json
{
    "message": "Test de prédiction réussi !",
    "probabilite_defaut": 0.3456,
    "seuil_optimal": 0.52,
    "decision": "ACCEPTÉ",
    "version": "1.0",
    "type": "test_aleatoire"
}
```

### `GET /model_info` - Informations modèle
Détails techniques sur le modèle chargé.

**Réponse :**
```json
{
    "model_type": "LightGBM",
    "nb_features": 120,
    "seuil_optimal": 0.52,
    "model_name": "LightGBM_v1",
    "version": "1.0"
}
```

## 🧪 Tests rapides

```bash
# Test status
curl http://localhost:5001/

# Test prédiction
curl http://localhost:5001/test_prediction

# Info modèle
curl http://localhost:5001/model_info
```

## Configuration

- **Port :** 5001
- **Modèle :** model_complet.pkl
- **Seuil :** 0.52 (optimal)

## Usage recommandé

- Démonstrations rapides
- Tests de fonctionnement
- Validation du modèle
- Développement initial

## Évolution

Pour des fonctionnalités avancées, voir [API V2](../api_v2/README.md)
