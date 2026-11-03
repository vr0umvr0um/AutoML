"""
Classification - Mode Stable
Ensemble reduit de modeles fiables et rapides.
"""
import gc
import warnings
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from automl.utils.logging import log
from automl.classification._base import (
    MULTILABEL_INCOMPATIBLE_STABLE,
    get_models_stable, 
    DEFAULT_PARAMS_STABLE,
    evaluate_model,
    MULTILABEL_INCOMPATIBLE
)

LOG_NAME = "classification_stable"


def run_classification_models(
    X, y, 
    n_jobs=2, 
    model_type="binary-classification", 
    iterations=50, 
    sample_ratio=0.3
):
    """
    Teste les modeles de classification en mode stable.
    Utilise Optuna pour l'optimisation des hyperparametres.
    """
    warnings.filterwarnings("ignore")
    
    models = get_models_stable()
    
    multilabel = model_type in ["multi-label-classification", "multi-output"]

    log(LOG_NAME, f"Classification stable: {len(models)} modeles, type={model_type}")

    # Sous-echantillonnage pour accelerer l'optimisation Optuna
    if 0 < sample_ratio < 1.0:
        try:
            # Stratify preserve la distribution des classes
            stratify = y if model_type != "regression" else None
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=sample_ratio, random_state=42, stratify=stratify
            )
            log(LOG_NAME, f"Sous-echantillonnage: {sample_ratio*100:.0f}% ({X_sub.shape[0]} samples)")
        except:
            # Classe trop rare pour stratifier
            X_sub, _, y_sub, _ = train_test_split(X, y, train_size=sample_ratio, random_state=42)
            log(LOG_NAME, f"Sous-echantillonnage (sans stratify): {sample_ratio*100:.0f}%")
    else:
        X_sub, y_sub = X, y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )
    except Exception as e:
        log(LOG_NAME, f"Erreur split train/test: {e}")
        return []

    model_names = list(models.keys())

    # Certains boosters ne gerent pas les sorties multi-label
    if multilabel:
        model_names = [m for m in model_names if m not in MULTILABEL_INCOMPATIBLE_STABLE]
        log(LOG_NAME, f"Multi-label: exclusion de {MULTILABEL_INCOMPATIBLE_STABLE}")

    tasks = [
        (name, models, DEFAULT_PARAMS_STABLE, X_train, X_test, 
         y_train, y_test, multilabel, iterations, LOG_NAME)
        for name in model_names
    ]

    results = []

    try:
        # spawn requis pour XGBoost/LightGBM (incompatibilite fork + threads)
        effective_n_jobs = 1 if len(X_sub) < 1000 else n_jobs
        
        ctx = mp.get_context("spawn")
        with ctx.Pool(effective_n_jobs) as pool:
            for res in pool.imap_unordered(evaluate_model, tasks):
                if res:
                    results.append(res)
            pool.close()
            pool.join()
    except Exception as e:
        log(LOG_NAME, f"Erreur multiprocessing ({e}). Mode sequentiel.")
        for args in tasks:
            res = evaluate_model(args)
            if res:
                results.append(res)
    finally:
        gc.collect()

    # Filtrage des echecs (score sentinelle)
    results.sort(key=lambda x: x["best_score"], reverse=True)
    valid_results = [r for r in results if r["best_score"] > -1e10]

    if not valid_results:
        log(LOG_NAME, "Aucun modele valide trouve")
    else:
        log(LOG_NAME, f"{len(valid_results)} modeles evalues avec succes")

    return valid_results