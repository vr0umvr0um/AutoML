"""Chargement des donnees."""
import os
import numpy as np
import pandas as pd
from scipy import sparse


def load_data(path: str) -> pd.DataFrame:
    """Charge un dataset d'entrainement (.data + .solution) en un seul DataFrame."""
    solution_path = f"{path}.solution"
    data_path = f"{path}.data"
    
    # Separateur flexible pour gerer formats heterogenes (espaces, tabs, mix)
    y_df = pd.read_csv(
        solution_path, sep=r"\s+", header=None, 
        engine="python", on_bad_lines="skip"
    )
    
    # Adaptation au format mono ou multi-target
    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0].rename("target")
    else:
        y = y_df.copy()
        y.columns = [f"target_{i}" for i in range(y.shape[1])]

    X = _read_features(data_path)
    
    df = pd.concat([X, y], axis=1)
    return df


def load_test_data(path: str) -> pd.DataFrame:
    """Charge un dataset de test (features uniquement, sans .solution)."""
    data_path = f"{path}" if path.endswith(".data") else f"{path}.data"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
        
    return _read_features(data_path)


def _read_features(data_path: str) -> pd.DataFrame:
    """
    Lecture interne des features.
    Detecte automatiquement le format sparse (SVMLight) vs dense.
    """
    with open(data_path, "r") as f:
        lines = f.readlines()
    
    # Format SVMLight "idx:val" typique pour donnees textuelles ou genomiques
    if any(":" in line for line in lines):
        max_col = 0
        data_rows = []
        for line in lines:
            row = {}
            for entry in line.strip().split():
                if ":" in entry:
                    idx, val = entry.split(":")
                    row[int(idx)] = float(val)
                    max_col = max(max_col, int(idx))
            data_rows.append(row)
        
        # Conversion sparse vers dense avec zeros explicites
        X = pd.DataFrame(0, index=np.arange(len(data_rows)), 
                         columns=np.arange(max_col + 1))
        for i, row in enumerate(data_rows):
            for col_idx, val in row.items():
                X.at[i, col_idx] = val
    else:
        X = pd.read_csv(
            data_path, sep=r"\s+", header=None, 
            engine="python", on_bad_lines="skip"
        )

    X.columns = X.columns.astype(str)
    # Convention sparse: valeurs absentes interpretees comme zero
    X = X.fillna(0).reset_index(drop=True)
    return X


def importer(datafile: str, max_dense_cols: int = 5000):
    """
    Import alternatif retournant (X, y) separement.
    Optimise memoire: conserve le format sparse si > max_dense_cols colonnes.
    """
    data_path = f"{datafile}.data"
    solution_path = f"{datafile}.solution"

    if not os.path.exists(data_path) or not os.path.exists(solution_path):
        raise FileNotFoundError(f"Fichier manquant: {data_path} ou {solution_path}")

    y_df = pd.read_csv(solution_path, sep=r"\s+", header=None, engine="python")
    y = y_df.iloc[:, 0] if y_df.shape[1] == 1 else y_df

    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    is_svmlight = any(":" in line for line in lines)

    if is_svmlight:
        # Construction par triplets COO pour efficacite memoire
        entries = []
        max_col = 0
        for row_idx, line in enumerate(lines):
            for entry in line.split():
                if ":" in entry:
                    col, val = entry.split(":")
                    col, val = int(col), float(val)
                    entries.append((row_idx, col, val))
                    max_col = max(max_col, col)

        n_rows, n_cols = len(lines), max_col + 1
        rows, cols, vals = zip(*entries)
        # CSR optimal pour slicing par lignes (operations sklearn)
        X_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))

        # Conservation du format sparse en haute dimension pour economie memoire
        if n_cols <= max_dense_cols:
            X = pd.DataFrame(X_sparse.toarray())
        else:
            X = pd.DataFrame.sparse.from_spmatrix(X_sparse)
        return X, y

    X_arr = np.loadtxt(data_path)
    # Cas particulier: fichier mono-ligne
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, len(X_arr))
    return pd.DataFrame(X_arr), y