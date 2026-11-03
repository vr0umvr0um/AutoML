"""Module de classification."""
from automl.classification.stable import run_classification_models
from automl.classification.explo import run_classification_models_explo

# stable: modeles valides, peu nombreux, rapide | explo: large selection, plus lent
__all__ = ["run_classification_models", "run_classification_models_explo"]