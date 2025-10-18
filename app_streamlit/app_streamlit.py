import streamlit as st
import pandas as pd
import requests

st.title("Scoring Crédit – Test de l'API")

API_URL = "https://credit-scoring-jademayalb-db8bcc609fed.herokuapp.com/predict/"

# Charge le CSV et récupère la liste des IDs
@st.cache_data
def load_client_ids():
    df = pd.read_csv("app/application_test.csv")
    return df["SK_ID_CURR"].sort_values().tolist()

client_ids = load_client_ids()

selected_id = st.selectbox("Choisissez l'identifiant client (SK_ID_CURR)", client_ids)

if st.button("Obtenir la prédiction"):
    with st.spinner("Appel à l'API..."):
        try:
            response = requests.get(f"{API_URL}{selected_id}")
            if response.status_code == 200:
                data = response.json()
                # Affichage conditionnel selon la décision
                if data['decision'].upper() == "ACCEPTÉ":
                    st.success(f"Décision : {data['decision']}")
                else:
                    st.error(f"Décision : {data['decision']}")
                st.write(f"Probabilité de défaut : {data['probabilite_defaut']:.3f}")
                st.write(f"Seuil optimal : {data['seuil_optimal']:.3f}")
            elif response.status_code == 404:
                st.error("Client non trouvé.")
            else:
                st.error(f"Erreur API : {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")