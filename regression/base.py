"""Base commune pour les modeles de regression."""

import gc

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
    BayesianRidge, HuberRegressor, Lars, LassoLars
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.dummy import DummyRegressor

# Librairies de boosting optionnelles
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    XGBRegressor = None
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    LGBMRegressor = None
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    CatBoostRegressor = None
    HAS_CATBOOST = False

from automl.utils.logging import log
from automl.utils.optuna import optuna_search


# =============================================================================
# PARAMETRES PAR DEFAUT
# =============================================================================

# Hyperparametres pour entrainement sans optimisation
DEFAULT_PARAMS_STABLE = {
    "LinearRegression": {},
    "Ridge": {"alpha": 1.0},
    "Lasso": {"alpha": 0.1},
    "ElasticNet": {"alpha": 0.1, "l1_ratio": 0.5},
    "MLPRegressor": {"hidden_layer_sizes": (100,), "alpha": 0.0001, "max_iter": 1000},
    "RandomForestRegressor": {"n_estimators": 100, "n_jobs": 1},
    "GradientBoostingRegressor": {"n_estimators": 100, "learning_rate": 0.1},
    "BaggingRegressor": {"n_estimators": 10, "n_jobs": 1},
    "ExtraTreesRegressor": {"n_estimators": 100, "n_jobs": 1},
    "DecisionTreeRegressor": {"max_depth": 10},
    "SVR": {"C": 1.0, "kernel": "rbf"},
    "KNeighborsRegressor": {"n_neighbors": 5, "n_jobs": 1},
    "SGDRegressor": {"alpha": 0.0001},
    "XGBRegressor": {
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5,
        "n_jobs": 1, "verbosity": 0
    },
    "LGBMRegressor": {
        "n_estimators": 100, "learning_rate": 0.1,
        "n_jobs": 1, "verbosity": -1
    },
    "CatBoostRegressor": {
        "iterations": 100, "learning_rate": 0.1, "depth": 6,
        "thread_count": 1, "verbose": 0
    }
}

# Etend les params stables avec les modeles experimentaux
DEFAULT_PARAMS_EXPLO = {
    **DEFAULT_PARAMS_STABLE,
    "BayesianRidge": {},
    "HuberRegressor": {"epsilon": 1.35},
    "Lars": {},
    "LassoLars": {"alpha": 0.1},
    "AdaBoostRegressor": {"n_estimators": 50, "learning_rate": 1.0},
    "HistGradientBoostingRegressor": {"max_iter": 100},
    "LinearSVR": {"C": 1.0},
    "NuSVR": {"nu": 0.5},
    "GaussianProcessRegressor": {},
    "DummyRegressor": {"strategy": "mean"},
}

# Bornes (min, max) pour optimisation Optuna
PARAM_GRIDS_STABLE = {
    "Ridge": {"alpha": (0.001, 50)},
    "Lasso": {"alpha": (1e-5, 10)},
    "ElasticNet": {"alpha": (1e-5, 10), "l1_ratio": (0.05, 0.95)},
    "MLPRegressor": {"alpha": (1e-6, 1e-1), "hidden_layer_sizes": (10, 300)},
    "RandomForestRegressor": {"n_estimators": (10, 800), "max_depth": (3, 50)},
    "GradientBoostingRegressor": {"learning_rate": (0.001, 0.5), "n_estimators": (10, 500)},
    "BaggingRegressor": {"n_estimators": (10, 800)},
    "ExtraTreesRegressor": {"n_estimators": (10, 800), "max_depth": (3, 50)},
    "DecisionTreeRegressor": {"max_depth": (2, 50)},
    "SVR": {"C": (0.001, 100)},
    "KNeighborsRegressor": {"n_neighbors": (2, 50)},
    "SGDRegressor": {"alpha": (1e-6, 1e-1)},
    "XGBRegressor": {
        "learning_rate": (0.001, 0.5), "n_estimators": (50, 800),
        "max_depth": (3, 20), "subsample": (0.5, 1.0)
    },
    "LGBMRegressor": {
        "learning_rate": (0.001, 0.5), "num_leaves": (20, 300),
        "max_depth": (3, 20), "feature_fraction": (0.5, 1.0)
    },
    "CatBoostRegressor": {
        "learning_rate": (0.001, 0.5), "depth": (4, 12),
        "l2_leaf_reg": (1, 10)
    },
}


# =============================================================================
# MODELES DISPONIBLES
# =============================================================================

def get_models_stable():
    """Retourne les modeles pour le mode stable."""
    models = {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "MLPRegressor": MLPRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "BaggingRegressor": BaggingRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "SVR": SVR,
        "KNeighborsRegressor": KNeighborsRegressor,
        "SGDRegressor": SGDRegressor,
    }
    # Ajout conditionnel des librairies de boosting
    if HAS_XGB:
        models["XGBRegressor"] = XGBRegressor
    if HAS_LGBM:
        models["LGBMRegressor"] = LGBMRegressor
    if HAS_CATBOOST:
        models["CatBoostRegressor"] = CatBoostRegressor
    return models


def get_models_explo():
    """Retourne les modeles pour le mode explo."""
    models = get_models_stable()
    # Modeles additionnels moins courants ou plus couteux
    models.update({
        "BayesianRidge": BayesianRidge,
        "HuberRegressor": HuberRegressor,
        "Lars": Lars,
        "LassoLars": LassoLars,
        "AdaBoostRegressor": AdaBoostRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "LinearSVR": LinearSVR,
        "NuSVR": NuSVR,
        "GaussianProcessRegressor": GaussianProcessRegressor,
        "DummyRegressor": DummyRegressor,
    })
    return models


# =============================================================================
# FONCTION D EVALUATION
# =============================================================================

def evaluate_model(args):
    """
    Evalue un modele de regression avec Optuna.
    
    Args (tuple):
        model_class, param_grid, X_train, y_train, X_test, y_test, iterations, log_name
    
    Returns:
        dict ou None
    """
    model_class, param_grid, X_train, y_train, X_test, y_test, iterations, log_name = args
    
    # Liberation memoire avant chaque modele
    gc.collect()
    
    try:
        best_params, best_score = optuna_search(
            model_class=model_class,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            task_type="regression",
            n_trials=iterations,
            log_name=log_name
        )
        
        result = {
            "model": model_class.__name__,
            "best_params": best_params,
            "best_score": best_score
        }
        
        log(log_name, f"[OK] {model_class.__name__} | R2={best_score:.4f}")
        
        gc.collect()
        return result
        
    except Exception as e:
        # Capture silencieuse pour continuer avec les autres modeles
        log(log_name, f"[ERREUR] {model_class.__name__}: {str(e)}")
        return None