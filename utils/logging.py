"""Logging et checkpoints."""
import pickle
from datetime import datetime
from automl._config import LOG_DIR, CHECKPOINT_DIR


def log(name: str, message: str) -> None:
    """Ecrit un message horodate dans un fichier log."""
    # Un fichier par composant pour faciliter le debug
    log_file = LOG_DIR / f"{name}_log.txt"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def save_checkpoint(data_name: str, mode: str, model_results: dict) -> None:
    """Sauvegarde le meilleur modele pour reprise ou reutilisation ulterieure."""
    try:
        path = CHECKPOINT_DIR / f"best_model_{data_name}_{mode}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model_results, f)
        log("checkpoint", f"Checkpoint sauvegarde: {path}")
    except Exception as e:
        # Echec silencieux: ne doit pas bloquer le pipeline
        log("checkpoint", f"Erreur sauvegarde checkpoint: {e}")