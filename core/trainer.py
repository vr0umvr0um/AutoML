"""Entrainement AutoML avec garantie d'entrainement final sur le dataset complet."""
import os
import pickle
import time
import signal
import functools
from pathlib import Path
from typing import List, Dict, Optional

from automl._config import (
    RESULTS_DIR, CHECKPOINT_DIR, 
    DEFAULT_MODE, AUTOSAVE_INTERVAL
)
from automl.utils.logging import log
from automl.utils.io import load_data
from automl.utils.detection import detect_type_model
from automl.utils.cleaning import AutoMLPreprocessor
from automl.core.runner import run_all_models

# Modeles disponibles selon le mode (stable/explo)
from automl.classification._base import get_models_stable as classif_stable, get_models_explo as classif_explo
from automl.regression._base import get_models_stable as reg_stable, get_models_explo as reg_explo


def _get_model_class(name, task_type, mode):
    """Resout le nom du modele vers sa classe selon le contexte."""
    if "classification" in task_type:
        models = classif_explo() if mode == "explo" else classif_stable()
    else:
        models = reg_explo() if mode == "explo" else reg_stable()
    return models.get(name)


def _handle_exit(sig, frame, checkpoint_path, all_results):
    """Sauvegarde l'etat courant avant arret brutal (Ctrl+C, kill)."""
    log("automl", f"Interruption ({sig}). Sauvegarde...")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(all_results, f)
    log("automl", f"Checkpoint: {checkpoint_path}")
    raise SystemExit(0)


def fit(
    data_path: str,
    mode: str = "stable",
    iterations: int = 50,
    n_jobs: int = 2,
    autosave_interval: int = AUTOSAVE_INTERVAL
) -> Dict:
    """
    Pipeline complet : charge, pretraite, optimise les hyperparametres,
    puis re-entraine le meilleur modele sur 100% des donnees.
    """
    
    log_name = f"automl_{mode}"
    log(log_name, f"AutoML fit: {data_path} (mode={mode})")

    save_path = RESULTS_DIR / f"results_{mode}.pkl"
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{mode}.pkl"

    log(log_name, "Chargement des donnees completes...")
    df = load_data(data_path)
    
    # Convention : colonnes prefixees "target" = cibles (supporte multi-output)
    target_cols = [c for c in df.columns if str(c).startswith("target")]
    
    # Fallback sur derniere colonne si aucun prefixe "target"
    if not target_cols:
        target_cols = [df.columns[-1]]

    y = df[target_cols]
    X = df.drop(columns=target_cols)
    
    # Sklearn attend une Series pour single-output
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
        
    log(log_name, f"Dataset: {X.shape[0]} lignes. Features: {X.shape[1]}, Targets: {len(target_cols)}")

    task_type = detect_type_model(df)
    log(log_name, f"Type detecte: {task_type}")

    log(log_name, "Nettoyage complet...")
    preprocessor = AutoMLPreprocessor()
    X_clean = preprocessor.fit_transform(X)
    
    log(log_name, f"Shape apres clean: {X_clean.shape}")

    # Optimisation hyperparametres via cross-validation
    log(log_name, f"Optimisation modele ({task_type})...")
    
    results_list = run_all_models(
        X_clean, y, task_type, 
        mode=mode, 
        iterations=iterations,
        n_jobs=n_jobs
    )

    if not results_list:
        log(log_name, "Aucun modele valide trouve.")
        return {}

    best_result = max(results_list, key=lambda x: x["best_score"])
    model_name = best_result["model"]
    best_params = best_result["best_params"]
    
    # Refit sur 100% des donnees (pas de split)
    log(log_name, f"Gagnant: {model_name} ({best_result['best_score']:.4f})")
    log(log_name, "Re-entrainement final...")

    try:
        ModelClass = _get_model_class(model_name, task_type, mode)
        final_model = ModelClass(**best_params)
        
        # Wrapper OvR pour modeles sans support natif multi-label
        is_multilabel = task_type == "multi-label-classification"
        native_multilabel = ["RandomForestClassifier", "ExtraTreesClassifier", "KNeighborsClassifier", "MLPClassifier", "RadiusNeighborsClassifier"]
        
        if is_multilabel and model_name not in native_multilabel:
             from sklearn.multiclass import OneVsRestClassifier
             final_model = OneVsRestClassifier(final_model, n_jobs=1)

        final_model.fit(X_clean, y)
        
        # Pipeline complet pour predict()
        final_pipeline = {
            "preprocessor": preprocessor,
            "model": final_model,
            "task_type": task_type,
            "info": best_result
        }
        
        pipeline_path = RESULTS_DIR / "best_model.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(final_pipeline, f)
            
        log(log_name, f"Pipeline sauvegarde : {pipeline_path}")
        
    except Exception as e:
        log(log_name, f"Erreur Refit : {e}")
        import traceback
        log(log_name, traceback.format_exc())
        raise e

    results = {
        "data_path": data_path,
        "task_type": task_type,
        "best_models": results_list[:5], 
        "n_models_tested": len(results_list)
    }
    
    with open(save_path, "wb") as f:
        pickle.dump({data_path: results}, f)
    log(log_name, f"Resultats: {save_path}")

    if results_list:
        best = max(results_list, key=lambda x: x["best_score"])
        log(log_name, f"Meilleur: {best['model']} (score={best['best_score']:.4f})")

    return results


def fit_multiple(
    datasets: List[Dict],
    mode: str = "stable",
    **kwargs
) -> Dict:
    """
    Entraine sur plusieurs datasets avec reprise automatique sur interruption.
    
    Args:
        datasets: Liste de {"file": path, "expected": type}
        mode: "stable" ou "explo"
        
    Example:
        >>> automl.fit_multiple([
        ...     {"file": "data/A", "expected": "classification"},
        ...     {"file": "data/B", "expected": "regression"}
        ... ])
    """
    all_results = {}
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_multi_{mode}.pkl"

    # Reprise : recharge les resultats deja calcules
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            all_results = pickle.load(f)
        done = set(all_results.keys())
        datasets = [d for d in datasets if d["file"] not in done]
        log("automl", f"Reprise: {len(done)} faits, {len(datasets)} restants")

    # Interception signaux pour sauvegarde propre
    handler = functools.partial(
        _handle_exit, 
        checkpoint_path=checkpoint_path, 
        all_results=all_results
    )
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    last_save = time.time()

    for data_info in datasets:
        try:
            result = fit(data_info["file"], mode=mode, **kwargs)
            all_results[data_info["file"]] = result

            # Autosave periodique pour eviter perte sur crash
            if time.time() - last_save > kwargs.get("autosave_interval", 600):
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(all_results, f)
                last_save = time.time()

        except Exception as e:
            # Continue les autres datasets meme si un echoue
            log("automl", f"Erreur {data_info['file']}: {e}")
            all_results[data_info["file"]] = {"error": str(e)}

    save_path = RESULTS_DIR / f"results_{mode}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)

    # Checkpoint inutile si tout est termine
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return all_results