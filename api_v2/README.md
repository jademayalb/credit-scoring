# API Credit Scoring - Version 2.0

## Nouveautés V2

Cette version apporte des améliorations significatives par rapport à la V1 :
- **Modèle optimisé** : LightGBM avec hyperparamètres tunés
- **Seuil calibré** : 0.52 pour équilibre optimal Précision/Rappel
- **Endpoints enrichis** : Prédictions avec données réelles + monitoring
- **Preprocessing intégré** : Imputation et scaling automatiques
- **Logging avancé** : Traçabilité complète des opérations
- **Gestion d'erreurs** : Validation robuste des données d'entrée

## Installation

```bash
cd api_v2
pip install -r requirements.txt
python app_v2.py
```

## Endpoints

### `GET /` - Page d'accueil
Informations complètes sur l'API V2 et ses améliorations.

**Réponse :**
```json
{
    "message": "API de Prédiction de Défaut de Crédit - Version 2.0",
    "model": "LightGBM_v1",
    "version": "2.0",
    "optimal_threshold": 0.52,
    "features_count": 120,
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
}
```

### `POST /predict` - Prédiction complète
Effectue une prédiction avec des données client réelles.

**Requête :**
```json
{
    "feature1": 0.5,
    "feature2": 0.8,
    "feature3": -0.2,
    "...": "autres features"
}
```

**Réponse :**
```json
{
    "prediction": 0,
    "probability_default": 0.3456,
    "probability_no_default": 0.6544,
    "risk_level": "Modéré",
    "confidence": 0.6912,
    "model_info": {
        "name": "LightGBM_v1",
        "version": "2.0",
        "threshold_used": 0.52
    },
    "interpretation": {
        "result": "Pas de défaut prédit",
        "recommendation": "Risque acceptable"
    }
}
```

### `GET /health` - Monitoring
Statut de santé de l'API pour monitoring en production.

**Réponse :**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-09-12T09:30:15.123456",
    "version": "2.0"
}
```

### `GET /model-info` - Informations détaillées
Métadonnées complètes du modèle et preprocessing.

**Réponse :**
```json
{
    "model_name": "LightGBM_v1",
    "model_type": "LightGBM",
    "version": "2.0",
    "optimal_threshold": 0.52,
    "features": ["feature1", "feature2", "..."],
    "features_count": 120,
    "preprocessing": {
        "scaler": "StandardScaler",
        "imputer": "SimpleImputer",
        "missing_value_strategy": "median"
    },
    "performance_notes": "Optimisé pour équilibrer précision et rappel"
}
```

## Tests avec curl

```bash
# Status de l'API
curl http://localhost:5000/

# Santé de l'API
curl http://localhost:5000/health

# Informations modèle
curl http://localhost:5000/model-info

# Prédiction (exemple)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 0.5, "feature2": 0.8}'
```

## Classification des risques

| Probabilité | Niveau de risque | Action recommandée |
|-------------|------------------|-------------------|
| < 0.3 | Faible | Acceptation standard |
| 0.3 - 0.52 | Modéré | Évaluation supplémentaire |
| 0.52 - 0.7 | Élevé | Attention particulière |
| > 0.7 | Très Élevé | Refus recommandé |

## Seuil optimal : 0.52

Le seuil de 0.52 a été déterminé par optimisation pour :
- Maximiser le F1-Score
- Équilibrer Précision et Rappel
- Minimiser les faux négatifs critiques
- Optimiser la valeur business

## Configuration

- **Port :** 5000
- **Modèle :** model_complet.pkl (LightGBM optimisé)
- **Seuil :** 0.52
- **Preprocessing :** SimpleImputer + StandardScaler
- **Logging :** INFO level

## Usage recommandé

- **Production** : Intégration dans systèmes métier
- **API REST** : Appels depuis applications web/mobile
- **Batch processing** : Scoring de portefeuilles
- **Monitoring** : Endpoint /health pour supervision

## Différences avec V1

| Aspect | V1 | V2 |
|--------|----|----|
| Données | Aléatoires | Réelles |
| Endpoints | 3 basiques | 4 enrichis |
| Validation | Minimale | Complète |
| Monitoring | Aucun | /health |
| Logging | Basic | Avancé |
| Erreurs | Simple | Détaillées |
| Production | Non | Oui |

## Documentation API

Pour une documentation interactive, utilisez des outils comme Postman ou Swagger en important les exemples ci-dessus.
