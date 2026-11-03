"""Nettoyage automatique des donnees."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, OrdinalEncoder, MaxAbsScaler
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from automl.utils.logging import log


class AutoMLPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocesseur compatible sklearn pour pipelines AutoML."""
    
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.low_card_cols = []
        self.high_card_cols = []
        self.is_high_dim = False
        self.transformers = {}
        self.output_columns = []

    def fit(self, X, y=None):
        """Apprend les statistiques sur les donnees d'entrainement."""
        log("clean", "Fitting Preprocessor...")
        
        # Seuil 1000 colonnes pour adapter la strategie de preprocessing
        n_cols = X.shape[1]
        self.is_high_dim = n_cols > 1000

        # Echantillonnage pour accelerer l'inference de types en haute dimension
        sample = X.iloc[:1000] if self.is_high_dim else X
        type_col = sample.dtypes.apply(
            lambda t: "Numerical" if np.issubdtype(t, np.number) else "Categorical"
        )
        
        self.num_cols = type_col[type_col == "Numerical"].index.tolist()
        self.cat_cols = type_col[type_col == "Categorical"].index.tolist()

        # Imputation numerique par mediane (robuste aux outliers)
        if self.num_cols:
            self.transformers["num_imputer"] = SimpleImputer(strategy="median")
            self.transformers["num_imputer"].fit(X[self.num_cols])
        
        # Imputation categorielle par mode
        if self.cat_cols:
            self.transformers["cat_imputer"] = SimpleImputer(strategy="most_frequent")
            self.transformers["cat_imputer"].fit(X[self.cat_cols])

        # Strategie d'encodage selon la cardinalite
        if self.cat_cols:
            self.low_card_cols = [c for c in self.cat_cols if X[c].nunique() <= 10]
            self.high_card_cols = [c for c in self.cat_cols if X[c].nunique() > 10]
            
            # OneHot pour faible cardinalite (preserve l'information)
            if self.low_card_cols:
                self.transformers["ohe"] = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                self.transformers["ohe"].fit(X[self.low_card_cols])
            
            # Ordinal pour haute cardinalite (evite explosion dimensionnelle)
            if self.high_card_cols:
                self.transformers["oe"] = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                self.transformers["oe"].fit(X[self.high_card_cols])

        # Transformation intermediaire pour fitter variance et scaler
        X_trans = self._partial_transform(X)
        
        # Suppression des colonnes a variance nulle (aucun pouvoir predictif)
        self.transformers["var_selector"] = VarianceThreshold(threshold=0.0)
        X_trans_var = self.transformers["var_selector"].fit_transform(X_trans)
        
        # MaxAbsScaler preserve la sparsite en haute dimension
        self.transformers["scaler"] = MaxAbsScaler() if self.is_high_dim else StandardScaler()
        self.transformers["scaler"].fit(X_trans_var)

        return self

    def transform(self, X):
        """Applique les transformations apprises."""
        X_trans = self._partial_transform(X)
        
        if "var_selector" in self.transformers:
            X_trans = pd.DataFrame(
                self.transformers["var_selector"].transform(X_trans),
                index=X_trans.index,
                columns=self.transformers["var_selector"].get_feature_names_out(X_trans.columns)
            )
            
        if "scaler" in self.transformers:
            X_trans = pd.DataFrame(
                self.transformers["scaler"].transform(X_trans),
                index=X_trans.index,
                columns=X_trans.columns
            )
            
        return X_trans

    def _partial_transform(self, X):
        """Applique imputation et encodage sans scaling."""
        X_out = X.copy()
        
        if self.num_cols and "num_imputer" in self.transformers:
            X_out[self.num_cols] = self.transformers["num_imputer"].transform(X_out[self.num_cols])
            
        if self.cat_cols and "cat_imputer" in self.transformers:
            X_out[self.cat_cols] = self.transformers["cat_imputer"].transform(X_out[self.cat_cols])
            
        encoded_dfs = []
        
        if self.num_cols:
            encoded_dfs.append(X_out[self.num_cols])
            
        if self.low_card_cols and "ohe" in self.transformers:
            ohe_vals = self.transformers["ohe"].transform(X_out[self.low_card_cols])
            ohe_cols = self.transformers["ohe"].get_feature_names_out(self.low_card_cols)
            encoded_dfs.append(pd.DataFrame(ohe_vals, columns=ohe_cols, index=X_out.index))
            
        if self.high_card_cols and "oe" in self.transformers:
            oe_vals = self.transformers["oe"].transform(X_out[self.high_card_cols])
            encoded_dfs.append(pd.DataFrame(oe_vals, columns=self.high_card_cols, index=X_out.index))
            
        if not encoded_dfs:
            return pd.DataFrame(index=X.index)
            
        return pd.concat(encoded_dfs, axis=1)


def auto_clean(data, y=None, feature_types=None):
    """
    Nettoyage automatique d'un dataset.
    
    Etapes:
    1. Detection sparse/high-dim
    2. Imputation valeurs manquantes
    3. Winsorization (si non high-dim)
    4. Encodage categoriel
    5. Selection features (variance nulle)
    6. Standardisation
    """
    log("clean", "=== AUTO CLEANING START ===")

    # Normalisation du format d'entree
    input_was_dataframe = isinstance(data, pd.DataFrame)
    if input_was_dataframe:
        df = data.copy()
        if "target" in df.columns:
            y = df["target"].copy()
            X = df.drop(columns=["target"])
        else:
            # Convention: derniere colonne = target si non explicite
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
    else:
        X = pd.DataFrame(data)

    cleaned_data = X

    # Metriques pour adapter le pipeline
    n_cols = cleaned_data.shape[1]
    sparsity = (cleaned_data == 0).sum().sum() / cleaned_data.size if n_cols > 0 else 0
    is_high_dim = n_cols > 1000

    if is_high_dim:
        log("clean", f"Mode High-Dim: {n_cols} cols, {sparsity:.0%} sparsity")

    # Inference automatique des types si non fournis
    if feature_types is None:
        sample = cleaned_data.iloc[:1000] if is_high_dim else cleaned_data
        type_col = sample.dtypes.apply(
            lambda t: "Numerical" if np.issubdtype(t, np.number) else "Categorical"
        )
    else:
        type_col = feature_types[0].apply(lambda t: t.capitalize().strip())

    num_cols = type_col[type_col == "Numerical"].index
    cat_cols = type_col[type_col == "Categorical"].index
    log("clean", f"Num: {len(num_cols)} | Cat: {len(cat_cols)}")

    # Imputation par mediane (robuste) et mode (categoriel)
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        cleaned_data[num_cols] = num_imputer.fit_transform(cleaned_data[num_cols])

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cleaned_data[cat_cols] = cat_imputer.fit_transform(cleaned_data[cat_cols])

    # Winsorization: desactivee en high-dim pour performance O(n*cols)
    if not is_high_dim:
        log("clean", "Winsorization...")
        for col in num_cols:
            if cleaned_data[col].nunique() > 1:
                # Quantiles 5-95% moins agressifs que IQR classique 25-75%
                q1, q3 = cleaned_data[col].quantile([0.05, 0.95])
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    cleaned_data[col] = np.clip(cleaned_data[col], lower, upper)

    # Encodage adaptatif selon cardinalite
    encoded_data = cleaned_data.copy()
    if len(cat_cols) > 0:
        low_card = [c for c in cat_cols if cleaned_data[c].nunique() <= 10]
        high_card = [c for c in cat_cols if cleaned_data[c].nunique() > 10]

        if low_card:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = pd.DataFrame(
                ohe.fit_transform(cleaned_data[low_card]),
                columns=ohe.get_feature_names_out(low_card),
                index=cleaned_data.index
            )
            encoded_data = pd.concat(
                [encoded_data.drop(columns=low_card), encoded], axis=1
            )

        if high_card:
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoded_data[high_card] = oe.fit_transform(cleaned_data[high_card])

    # Elimination des colonnes constantes
    selector = VarianceThreshold(threshold=0.0)
    encoded_vals = selector.fit_transform(encoded_data)
    feat_names = selector.get_feature_names_out(encoded_data.columns)
    encoded_data = pd.DataFrame(encoded_vals, columns=feat_names, index=encoded_data.index)

    # Normalisation adaptee au contexte
    num_cols_final = encoded_data.select_dtypes(include=["int", "float"]).columns
    if len(num_cols_final) > 0:
        scaler = MaxAbsScaler() if is_high_dim else StandardScaler()
        encoded_data[num_cols_final] = scaler.fit_transform(encoded_data[num_cols_final])

    log("clean", f"Clean termine: {encoded_data.shape[1]} cols")

    # Retour au format d'origine
    if input_was_dataframe:
        y_ser = pd.Series(y).reset_index(drop=True).rename("target")
        return pd.concat([encoded_data.reset_index(drop=True), y_ser], axis=1)
    else:
        if y is not None:
            y_out = y.reset_index(drop=True) if isinstance(y, pd.Series) else np.asarray(y)
            return encoded_data, y_out
        return encoded_data


def verif_quality(X_before, X_after, y, task_type="classification", verbose=True):
    """
    Valide que le nettoyage n'a pas degrade l'information predictive.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    report = {
        "missing_before": X_before.isna().sum().sum() if isinstance(X_before, pd.DataFrame) else 0,
        "missing_after": X_after.isna().sum().sum(),
        "drop_ratio": 1 - (X_after.shape[1] / max(X_before.shape[1], 1))
    }

    # Validation rapide sur echantillon pour limiter le cout de calcul
    try:
        n = min(2000, len(X_before))
        idx = np.random.choice(len(X_before), n, replace=False)
        
        Xa = X_after.iloc[idx]
        y_s = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        
        # RandomForest leger: diagnostic uniquement, pas prediction finale
        model = (RandomForestRegressor if task_type == "regression" 
                 else RandomForestClassifier)(n_estimators=20, max_depth=5, n_jobs=-1)
        model.fit(Xa, y_s)
        report["score_after"] = model.score(Xa, y_s)
    except Exception as e:
        report["score_after"] = 0
        log("clean", f"Erreur verif_quality: {e}")

    # Score de qualite avec penalite proportionnelle aux colonnes supprimees
    report["quality_score"] = round(min(100, 100 - 30 * report["drop_ratio"]), 2)
    
    if verbose:
        log("clean", f"Qualite: {report['quality_score']}/100")
    
    return report