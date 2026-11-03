"""Module de prediction sur de nouvelles donnees."""
import pickle
from automl._config import RESULTS_DIR
from automl.utils.io import load_test_data
from automl.utils.logging import log


def predict(data_path: str):
    """
    Charge le meilleur modele entraine et predit sur un nouveau dataset.
    
    Args:
        data_path: Chemin vers le fichier .data (sans extension ou avec)
        
    Returns:
        list: Les predictions
    """
    model_path = RESULTS_DIR / "best_model.pkl"
    
    if not model_path.exists():
        print("Aucun modele entraine trouve. Veuillez lancer fit() d'abord.")
        return []

    log("predict", f"Demarrage prediction sur : {data_path}")

    # Pipeline complet serialise (preprocessor + modele + metadata)
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
        
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    task_type = pipeline["task_type"]
    
    log("predict", f"   Modele charge : {type(model).__name__} ({task_type})")

    # Chargement features uniquement, sans labels
    try:
        X_raw = load_test_data(data_path)
    except Exception as e:
        print(f"Erreur chargement donnees : {e}")
        return []

    # Transform seul pour garantir coherence avec l'entrainement
    try:
        X_clean = preprocessor.transform(X_raw)
        log("predict", f"   Donnees nettoyees. Shape: {X_clean.shape}")
    except Exception as e:
        print(f"Erreur preprocessing : {e}")
        return []

    try:
        predictions = model.predict(X_clean)
        
        # Conversion ndarray vers list pour serialisation JSON
        preds_list = predictions.tolist()
        
        log("predict", f"{len(preds_list)} predictions generees.")
        return preds_list
        
    except Exception as e:
        print(f"Erreur prediction : {e}")
        return []