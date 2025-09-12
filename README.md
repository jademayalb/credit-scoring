Projet MLOps - Scoring Crédit avec API et déploiement
# Credit Scoring MLOps Project

## Objectif du projet
Développement d'un système de scoring crédit avec pipeline MLOps complet, de l'entraînement du modèle au déploiement en production.

## Description
Projet consistant à :
- Créer un modèle de classification pour prédire le risque de défaut de paiement
- Mettre en place une approche MLOps complète
- Déployer une API de prédiction sur le cloud

## Structure du projet

```
credit-scoring-mlops/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── notebooks/               # Notebooks Jupyter
│   ├── modélisation/        # Notebook de modélisation ML
│   └── data_drift/          # Analyse de data drift
├── api/                     # Code de l'API Flask
│   ├── app.py              # API principale
│   ├── verif_modele.py   # Script de vérification
│   └── model_complet.pkl   # Modèle sauvegardé
├── data/                    # Données (non versionnées)
└── docs/                    # Documentation
```

## Version actuelle : V1.0 - MVP

### Fonctionnalités disponibles :
- Notebook de modélisation complet avec MLflow
- Analyse de data drift avec Evidently
- API Flask basique avec endpoints de test
- Modèle LightGBM optimisé avec seuil métier

### Endpoints API :
- `GET /` : Informations générales de l'API
- `GET /health` : Statut de santé
- `GET /test_prediction` : Test avec données aléatoires
- `GET /model_info` : Informations sur le modèle

## 🛠️ Installation et utilisation

### Prérequis
- Python 3.8+
- pip

### Installation
```bash
# Cloner le repository
git clone https://github.com/votre-username/credit-scoring-mlops.git
cd credit-scoring-mlops

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'API
```bash
cd api
python app.py
```

L'API sera disponible sur : http://localhost:5001

### Tests
```bash
# Test de base
curl http://localhost:5001/

# Test de prédiction
curl http://localhost:5001/test_prediction
```

## Modèle

- **Algorithme** : LightGBM avec gestion du déséquilibre
- **Métrique principale** : AUC + coût métier (FN:FP = 10:1)
- **Features** : 243 variables après feature engineering
- **Performance** : AUC ~0.75, seuil optimal ~0.42

## Roadmap

### V2.0 - API avec données réelles (à venir)
- Intégration avec fichiers CSV
- Prédiction par ID client
- Gestion des erreurs avancée

### V3.0 - Pipeline MLOps complet (à venir)
- GitHub Actions CI/CD
- Tests unitaires automatisés
- Déploiement Heroku
- Interface Streamlit
