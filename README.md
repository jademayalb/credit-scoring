# Credit Scoring API

Ce projet propose une API de scoring crédit, un notebook de modélisation, une interface de test Streamlit, et un suivi MLOps.

## Structure du projet

- `app/` : code et artefacts de l'API Flask
- `notebooks/` : notebooks de modélisation et data drift
- `tableau_html/` : rapport HTML Evidently
- `tests/` : tests unitaires
- `app_streamlit.py` : interface Streamlit de test de l'API
- `requirements.txt` : dépendances minimales pour l'API (déploiement)
- `requirements_dev.txt` : dépendances complètes pour le développement et la modélisation

## Installation

### Pour l'API uniquement (déploiement)

```bash
pip install -r requirements.txt
```

### Pour tout le projet (modélisation, notebooks, tests, Streamlit)

```bash
pip install -r requirements_dev.txt
```

## Lancer l'API Flask

```bash
cd app
python api.py
```

## Lancer l'app Streamlit

```bash
streamlit run app_streamlit.py
```

## Lancer les tests unitaires

```bash
pytest tests/
```

## Démo API en production

L'API est déployée sur Heroku.

## Auteur

jademayalb
