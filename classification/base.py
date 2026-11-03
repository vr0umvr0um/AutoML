"""
Base commune pour les modeles de classification.
Contient les parametres par defaut et la fonction d'evaluation partagee.
"""
from automl.utils.logging import log
from automl.utils.optuna import optuna_search

import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    Perceptron, PassiveAggressiveClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier

# Boosters externes (dependances optionnelles)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    CatBoostClassifier = None
    HAS_CATBOOST = False

# =============================================================================
# PARAMETRES PAR DEFAUT
# =============================================================================

# Mode stable : hyperparametres conservateurs pour limiter l'overfitting
DEFAULT_PARAMS_STABLE = {
    "LogisticRegression": {"C": 1.0},
    "RidgeClassifier": {"alpha": 1.0},
    "SGDClassifier": {"alpha": 0.01},
    "MLPClassifier": {"hidden_layer_sizes": (100,), "alpha": 0.0005, "max_iter": 200},
    "RandomForestClassifier": {"n_estimators": 100, "max_depth": None, "n_jobs": 1},
    "GradientBoostingClassifier": {"n_estimators": 100, "learning_rate": 0.1},
    "ExtraTreesClassifier": {"n_estimators": 100, "n_jobs": 1},
    "BaggingClassifier": {"n_estimators": 10, "n_jobs": 1},
    "SVC": {"C": 1.0, "kernel": "rbf", "probability": True},
    "GaussianNB": {},
    "XGBClassifier": {
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5,
        "use_label_encoder": False, "eval_metric": "logloss",
        "n_jobs": 1, "verbosity": 0
    },
    "LGBMClassifier": {
        "n_estimators": 100, "learning_rate": 0.1,
        "max_depth": -1, "n_jobs": 1, "verbosity": -1
    },
    "CatBoostClassifier": {
        "iterations": 100, "learning_rate": 0.1, "depth": 6,
        "thread_count": 1, "verbose": 0
    }
}

# Mode exploration : parametres de base, Optuna optimise ensuite
DEFAULT_PARAMS_EXPLO = {
    "LogisticRegression": {"C": 1.0, "solver": "lbfgs", "n_jobs": 1},
    "RidgeClassifier": {"alpha": 1.0},
    "SGDClassifier": {"alpha": 0.0001, "n_jobs": 1},
    "Perceptron": {"alpha": 0.0001, "n_jobs": 1},
    "PassiveAggressiveClassifier": {"C": 1.0, "n_jobs": 1},
    "MLPClassifier": {"hidden_layer_sizes": (100,), "alpha": 0.0001, "max_iter": 1000},
    "RandomForestClassifier": {"n_estimators": 100, "n_jobs": 1},
    "GradientBoostingClassifier": {"n_estimators": 100, "learning_rate": 0.1},
    "AdaBoostClassifier": {"n_estimators": 50, "learning_rate": 1.0},
    "ExtraTreesClassifier": {"n_estimators": 100, "n_jobs": 1},
    "BaggingClassifier": {"n_estimators": 10, "n_jobs": 1},
    "HistGradientBoostingClassifier": {"max_iter": 100},
    "SVC": {"C": 1.0, "kernel": "rbf", "probability": True},
    "NuSVC": {"nu": 0.5, "kernel": "rbf", "probability": True},
    "LinearSVC": {"C": 1.0, "dual": "auto"},
    "KNeighborsClassifier": {"n_neighbors": 5, "n_jobs": 1},
    "RadiusNeighborsClassifier": {"radius": 1.0, "n_jobs": 1},
    "NearestCentroid": {},
    "GaussianNB": {},
    "MultinomialNB": {"alpha": 1.0},
    "ComplementNB": {"alpha": 1.0},
    "BernoulliNB": {"alpha": 1.0},
    "LinearDiscriminantAnalysis": {},
    "QuadraticDiscriminantAnalysis": {},
    "GaussianProcessClassifier": {"n_jobs": 1},
    "DummyClassifier": {"strategy": "prior"},
    "XGBClassifier": {"n_estimators": 100, "learning_rate": 0.1, "n_jobs": 1, "verbosity": 0},
    "LGBMClassifier": {"n_estimators": 100, "learning_rate": 0.1, "n_jobs": 1, "verbosity": -1},
    "CatBoostClassifier": {"iterations": 100, "learning_rate": 0.1, "thread_count": 1, "verbose": 0}
}


# =============================================================================
# MODELES DISPONIBLES
# =============================================================================

def get_models_stable():
    """
    Retourne les modeles pour le mode stable.
    Fonction pour gerer les imports conditionnels des boosters.
    """
    models = {
        "LogisticRegression": LogisticRegression,
        "RidgeClassifier": RidgeClassifier,
        "SGDClassifier": SGDClassifier,
        "MLPClassifier": MLPClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "BaggingClassifier": BaggingClassifier,
        "SVC": SVC,
        "GaussianNB": GaussianNB,
    }
    if HAS_XGB:
        models["XGBClassifier"] = XGBClassifier
    if HAS_LGBM:
        models["LGBMClassifier"] = LGBMClassifier
    if HAS_CATBOOST:
        models["CatBoostClassifier"] = CatBoostClassifier
    return models


def get_models_explo():
    """
    Retourne les modeles pour le mode explo.
    Inclut des modeles plus lents ou specialises.
    """
    models = get_models_stable()
    models.update({
        "Perceptron": Perceptron,
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "NuSVC": NuSVC,
        "LinearSVC": LinearSVC,
        "KNeighborsClassifier": KNeighborsClassifier,
        "RadiusNeighborsClassifier": RadiusNeighborsClassifier,
        "NearestCentroid": NearestCentroid,
        "MultinomialNB": MultinomialNB,
        "ComplementNB": ComplementNB,
        "BernoulliNB": BernoulliNB,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
        "GaussianProcessClassifier": GaussianProcessClassifier,
        "DummyClassifier": DummyClassifier,
    }
    )
    if HAS_XGB:
        models["XGBClassifier"] = XGBClassifier
    if HAS_LGBM:
        models["LGBMClassifier"] = LGBMClassifier
    if HAS_CATBOOST:
        models["CatBoostClassifier"] = CatBoostClassifier
    return models


# =============================================================================
# CONFIGURATION MULTI-LABEL
# =============================================================================

# Modeles gerant nativement les sorties multi-colonnes
NATIVE_MULTILABEL = [
    "RandomForestClassifier", "ExtraTreesClassifier",
    "KNeighborsClassifier", "MLPClassifier", "RadiusNeighborsClassifier"
]

# Modeles incompatibles avec le multi-label meme via OneVsRest
MULTILABEL_INCOMPATIBLE_EXPLO = ["QuadraticDiscriminantAnalysis", "LinearDiscriminantAnalysis", "ComplementNB"]

MULTILABEL_INCOMPATIBLE_STABLE = ["CatBoostClassifier", "XGBClassifier", "LGBMClassifier", "QDA", "LinearDiscriminantAnalysis"]

# =============================================================================
# FONCTION D'EVALUATION COMMUNE
# =============================================================================

def evaluate_model(args):
    """
    Evalue un modele de classification avec optimisation Optuna.
    Concue pour execution parallele via multiprocessing.Pool.

    Returns:
        dict avec model, best_params, best_score (ou -999 si echec)
    """
    (name, models, default_params, X_train, X_test,
     y_train, y_test, multilabel, iterations, log_name) = args

    # Liberation memoire en multiprocessing pour eviter l'accumulation
    gc.collect()

    base_params = default_params.get(name, {}).copy()
    model_class = models[name]

    try:
        best_params, best_score = optuna_search(
            model_class=model_class,
            param_grid={},
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            task_type="classification",
            base_params=base_params,
            n_trials=iterations,
            log_name=log_name
        )

        model = model_class(**best_params)

        # Wrapper pour classifieurs binaires en contexte multi-label
        if multilabel and name not in NATIVE_MULTILABEL:
            model = OneVsRestClassifier(model, n_jobs=1)

        # Selection du scaler selon les contraintes du modele
        if name in ["MultinomialNB", "ComplementNB"]:
            # Requiert des features non negatives
            estimator = Pipeline([
                ("scaler", MinMaxScaler()),
                ("model", model)
            ])
        elif any(kw in name for kw in ["Logistic", "Ridge", "SVC", "SGD", "MLP",
                                        "Perceptron", "PassiveAggressive",
                                        "NearestCentroid", "LinearDiscriminant",
                                        "KNeighbors", "RadiusNeighbors"]):
            # Modeles sensibles a l'echelle des features
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
        else:
            # Arbres : invariants aux transformations monotones
            estimator = model

        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)

        result = {
            "model": name,
            "best_params": best_params,
            "best_score": float(score)
        }

        log(log_name, f" {name} | Score: {score:.4f}")

        del model, estimator
        gc.collect()

        return result

    except Exception as e:
        log(log_name, f" {name}: {str(e)}")
        # Score sentinelle pour exclusion lors du tri final
        return {
            "model": name,
            "best_params": {},
            "best_score": -999.0,
            "error": str(e)
        }