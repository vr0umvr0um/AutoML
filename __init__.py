"""
AutoML - Pipeline automatique de Machine Learning
"""

# API publique du package
from automl.core.trainer import fit, fit_multiple
from automl.core.evaluator import eval, show_results
from automl.core.predictor import predict

__version__ = "0.2.0"

# Restreint les exports pour "from automl import *"
__all__ = [
    "fit",
    "fit_multiple",
    "predict",
    "eval",
    "show_results"
]