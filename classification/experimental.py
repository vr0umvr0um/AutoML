"""
Classification - Mode Exploratoire
Ensemble complet de modeles sklearn + boosting.
"""
import gc
import warnings
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from automl.utils.logging import log
from automl.classification._base import (
    MULTILABEL_INCOMPATIBLE_EXPLO,
    get_models_explo,
    DEFAULT_PARAMS_EXPLO,
    evaluate_model
)

LOG_NAME = "classification_explo"


def run_classification_models_explo(
    X, y,
    n_jobs=2,
    model_type="binary-classification",
    iterations=10,
    sample_ratio=1.0
):
    """
    Exploration complete des modeles de classification.
    Plus de modeles que le mode stable, mais plus lent.
    """
    warnings.filterwarnings("ignore")
    
    models = get_models_explo()
    multilabel = model_type in ["multi-label-classification", "multi-output"]

    log(LOG_NAME, f"Classification explo: {len(models)} modeles, type={model_type}")

    # Exclusion des modeles a complexite O(n2) ou plus sur gros volumes
    dataset_size = X.shape[0] * X.shape[1]
    if dataset_size > 500_000:
        heavy = ["GaussianProcessClassifier", "RadiusNeighborsClassifier", "NuSVC", "SVC", "KNeighborsClassifier"]
        for m in heavy:
            if m in models:
                del models[m]
                log(LOG_NAME, f"Dataset volumineux: {m} exclu")

    if 0 < sample_ratio < 1.0:
        try:
            # Stratify conserve la distribution des classes
            stratify = y if model_type != "regression" else None
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=sample_ratio, random_state=42, stratify=stratify
            )
            log(LOG_NAME, f"Sous-echantillonnage: {sample_ratio*100:.0f}%")
        except:
            # Classe trop rare pour stratifier
            X_sub, _, y_sub, _ = train_test_split(X, y, train_size=sample_ratio, random_state=42)
            log(LOG_NAME, f"Sous-echantillonnage (no stratify): {sample_ratio*100:.0f}%")
    else:
        X_sub, y_sub = X, y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )
    except Exception as e:
        log(LOG_NAME, f"Erreur split: {e}")
        return []

    model_names = list(models.keys())

    if multilabel:
        model_names = [m for m in model_names if m not in MULTILABEL_INCOMPATIBLE_EXPLO]
        log(LOG_NAME, f"Multi-label: exclusion de {MULTILABEL_INCOMPATIBLE_EXPLO}")

    # Structure de tuple alignee avec evaluate_model()
    tasks = [
        (name, models, DEFAULT_PARAMS_EXPLO, X_train, X_test,
         y_train, y_test, multilabel, iterations, LOG_NAME)
        for name in model_names
    ]

    results = []

    # Ajustement du parallelisme selon la taille du dataset
    effective_n_jobs = 1 if len(X_sub) < 1000 else n_jobs
    if X.shape[0] > 50_000 and effective_n_jobs > 2:
        effective_n_jobs = 2
        log(LOG_NAME, f"Reduction n_jobs a {effective_n_jobs}")

    try:
        # spawn requis pour XGBoost/LightGBM (incompatibilite fork + threads)
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

    log(LOG_NAME, f"{len(valid_results)}/{len(model_names)} modeles valides")

    return valid_results