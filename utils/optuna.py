"""Optimisation Optuna pour la recherche d'hyperparametres."""
import numpy as np
import optuna
from sklearn.metrics import r2_score, f1_score
from optuna import TrialPruned

from automl.utils.logging import log

# Evite la pollution des logs pendant l'optimisation
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_search(
    model_class,
    param_grid,
    X_train,
    y_train,
    X_val,
    y_val,
    task_type="regression",
    base_params=None,
    n_trials=50,
    log_name="optuna_search"
):
    """
    Recherche d'hyperparametres avec Optuna (classification ou regression).

    Args:
        model_class: classe du modele (ex: XGBRegressor, LGBMClassifier)
        param_grid: dict des plages de recherche, ex: {"n_estimators": (10, 200)}
        X_train, y_train: donnees d'entrainement
        X_val, y_val: donnees de validation
        task_type: "regression" ou "classification"
        base_params: parametres fixes
        n_trials: nombre d'essais Optuna
        log_name: nom du logger

    Returns:
        best_params: dict des meilleurs parametres
        best_value: meilleur score obtenu
    """

    if base_params is None:
        base_params = {}

    def objective(trial):
        """Fonction objectif evaluee a chaque trial Optuna."""
        params = base_params.copy()

        # Infere le type (int/float) selon le nom du parametre
        for k, v_range in param_grid.items():
            if not isinstance(v_range, (list, tuple)) or len(v_range) != 2:
                continue
            vmin, vmax = v_range

            # Parametres intrinsequement entiers dans sklearn/boosting
            if any(kw in k for kw in ["depth", "n_estimators", "num_leaves", 
                                       "n_neighbors", "max_depth", "hidden_layer_sizes"]):
                params[k] = int(trial.suggest_int(k, int(vmin), int(vmax)))
            else:
                params[k] = float(trial.suggest_float(k, float(vmin), float(vmax), log=False))

        try:
            model = model_class(**params)

            # Early stopping: API specifique a chaque framework
            fit_args = {}
            name = model_class.__name__
            if "XGB" in name:
                fit_args = {
                    "eval_set": [(X_val, y_val)],
                    "verbose": False
                }
            elif "LGBM" in name:
                fit_args = {
                    "eval_set": [(X_val, y_val)],
                    "eval_metric": "r2" if task_type == "regression" else "logloss"
                }
            elif "CatBoost" in name:
                fit_args = {
                    "eval_set": [(X_val, y_val)],
                    "silent": True
                }

            model.fit(X_train, y_train, **fit_args)

            preds = model.predict(X_val)
            if task_type == "regression":
                score = r2_score(y_val, preds)
            else:
                score = f1_score(y_val, preds, average='macro')

            # Ecarte les resultats invalides de l'etude
            if np.isnan(score) or np.isinf(score):
                raise TrialPruned()

            return score

        except Exception as e:
            # Params incompatibles ou erreur de convergence
            log(log_name, f"Trial echoue pour {model_class.__name__}: {str(e)}")
            raise TrialPruned()

    # TPESampler: echantillonnage bayesien, plus efficace que grid search
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log(log_name, f"{model_class.__name__} | meilleur score={study.best_value:.4f} | params={study.best_params}")

    return study.best_params, study.best_value