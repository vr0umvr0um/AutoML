"""Module de regression."""

# stable: modeles optimises, rapide | explo: plus de modeles, plus lent
from automl.regression.stable import run_regression_models
from automl.regression.explo import run_regression_models_explo

# API publique du module
__all__ = ["run_regression_models", "run_regression_models_explo"]