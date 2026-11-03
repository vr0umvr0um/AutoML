"""Detection automatique du type de probleme ML."""
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from automl.utils.logging import log


def detect_type_model(data) -> str:
    """
    Detecte le type de modele a appliquer.
    
    Returns:
        str: "binary-classification", "multi-classification", 
             "multi-label-classification", ou "regression"
    """
    # Extraction de y selon le format d'entree (Bunch, DataFrame, tuple, dict)
    if hasattr(data, "data") and hasattr(data, "target"):
        y = data.target
    elif isinstance(data, pd.DataFrame):
        if "target" in data.columns:
            y = data["target"]
        else:
            y = data.iloc[:, -1]
    elif isinstance(data, tuple) and len(data) == 2:
        _, y = data
    elif isinstance(data, dict) and "target" in data:
        y = data["target"]
    else:
        raise ValueError(f"Format non supporte: {type(data)}")

    # Uniformisation vers array 1D pour traitement homogene
    if hasattr(y, "values"):
        y = y.values
    y = np.array(y)
    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y.ravel()

    # Detection initiale via sklearn
    y_type = type_of_target(y)
    log("detection", f"Type sklearn detecte: {y_type}")

    # Cas multi-label: plusieurs labels simultanement par echantillon
    if y_type in ["multilabel-indicator", "multiclass-multioutput"]:
        return "multi-label-classification"

    # Heuristiques pour cibles numeriques: distinguer classes encodees vs valeurs continues
    if np.issubdtype(y.dtype, np.number):
        n_unique = len(np.unique(y))
        ratio = n_unique / len(y)

        if n_unique == 2:
            return "binary-classification"
        # Seuils empiriques: peu de valeurs distinctes suggere un encodage de classes
        if n_unique < 20 or ratio < 0.01:
            return "multi-classification"
        return "regression"

    # Cibles non numeriques: forcement categoriel
    n_unique = len(np.unique(y))
    return "binary-classification" if n_unique == 2 else "multi-classification"