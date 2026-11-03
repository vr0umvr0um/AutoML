"""Orchestration des modeles."""
import gc
import traceback
import multiprocessing as mp
from automl.utils.logging import log, save_checkpoint

# Separation stable/explo pour isoler les dependances experimentales
from automl.classification import stable as classif_stable
from automl.classification import explo as classif_explo
from automl.regression import stable as reg_stable
from automl.regression import explo as reg_explo


def run_all_models(X, y, task_type, mode="stable", iterations=50, n_jobs=2):
    """
    Lance les modeles adaptes au type de tache.
    """
    log_name = f"runner_{mode}"
    log(log_name, f"Tache: {task_type} | Mode: {mode}")

    try:
        if task_type in ["binary-classification", "multi-classification", 
                         "multi-label-classification"]:
            # Dispatch dynamique selon le mode
            module = classif_explo if mode == "explo" else classif_stable
            func = (module.run_classification_models_explo if mode == "explo" 
                    else module.run_classification_models)
            results = func(X, y, n_jobs=n_jobs, model_type=task_type, 
                          iterations=iterations)

        elif task_type == "regression":
            module = reg_explo if mode == "explo" else reg_stable
            func = (module.run_regression_models_explo if mode == "explo" 
                    else module.run_regression_models)
            results = func(X, y, n_jobs=n_jobs, iterations=iterations)

        else:
            log(log_name, f"Type inconnu: {task_type}")
            return []

        log(log_name, f"{len(results)} modeles testes")

        # Sauvegarde du meilleur modele pour reprise ulterieure
        if results:
            best = max(results, key=lambda x: x["best_score"])
            save_checkpoint(task_type, mode, best)
            log(log_name, f"Meilleur: {best['model']} ({best['best_score']:.4f})")

        # Liberation memoire entre chaque run
        gc.collect()
        return results

    except Exception as e:
        log(log_name, f"Erreur: {e}")
        log(log_name, traceback.format_exc())
        gc.collect()
        return []