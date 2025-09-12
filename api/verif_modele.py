# scripts/verif_model.py
import joblib
import os
import sys

def verifier_modele(chemin_modele="model_complet.pkl"):
    """Vérifier l'intégrité du modèle sauvegardé"""
    
    print("=" * 60)
    print("VÉRIFICATION DU MODÈLE")
    print("=" * 60)
    
    # Vérifier l'existence du fichier
    if not os.path.exists(chemin_modele):
        print(f"Fichier {chemin_modele} non trouvé")
        print("Assurez-vous d'avoir exécuté le notebook de modélisation")
        return False
    
    # Informations sur le fichier
    taille_mo = os.path.getsize(chemin_modele) / 1024 / 1024
    print(f"Fichier trouvé: {chemin_modele}")
    print(f"Taille: {taille_mo:.2f} MB")
    
    try:
        # Charger le modèle
        print("Chargement du modèle...")
        model_data = joblib.load(chemin_modele)
        print("Modèle chargé avec succès")
        
        # Analyser le contenu
        print(f"\nANALYSE DU CONTENU:")
        print(f"Type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"Clés disponibles: {list(model_data.keys())}")
            
            # Vérifier les composants essentiels
            composants_requis = ['model', 'features', 'scaler', 'imputer', 'optimal_threshold']
            for composant in composants_requis:
                if composant in model_data:
                    print(f"{composant}: {type(model_data[composant])}")
                else:
                    print(f"{composant}: MANQUANT")
            
            # Informations détaillées
            if 'model' in model_data:
                print(f"Algorithme: {type(model_data['model']).__name__}")
            
            if 'features' in model_data:
                print(f"Nombre de features: {len(model_data['features'])}")
            
            if 'optimal_threshold' in model_data:
                print(f"Seuil optimal: {model_data['optimal_threshold']:.4f}")
        
        print(f"\nVÉRIFICATION TERMINÉE - Modèle valide")
        return True
        
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return False

if __name__ == "__main__":
    # Permettre de spécifier le chemin en argument
    chemin = sys.argv[1] if len(sys.argv) > 1 else "model_complet.pkl"
    
    # Vérifier dans le dossier api/ aussi
    if not os.path.exists(chemin) and os.path.exists(f"api/{chemin}"):
        chemin = f"api/{chemin}"
    
    succes = verifier_modele(chemin)
    sys.exit(0 if succes else 1)