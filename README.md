# Master 1 AutoML - Pipeline de Machine Learning Automatisé

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.2+-orange.svg)](https://scikit-learn.org/)

AutoML est une pipeline automatisée de Machine Learning qui détecte automatiquement le type de problème (classification binaire, multi-classe, multi-label, régression), nettoie les données, optimise les hyperparamètres via Optuna, et sélectionne le meilleur modèle.

---

## Table des matières

1. [Fonctionnalités](#fonctionnalités)
2. [Installation](#installation)
3. [Démarrage rapide](#démarrage-rapide)
4. [Format des données](#format-des-données)
5. [Pipeline de traitement](#pipeline-de-traitement)
6. [API Reference](#api-reference)
7. [Modèles disponibles](#modèles-disponibles)
8. [Configuration avancée](#configuration-avancée)
9. [Interface CLI](#interface-cli)
10. [Fichiers générés](#fichiers-générés)
11. [Limitations connues](#limitations-connues)
12. [Contribution](#contribution)
13. [Auteurs](#auteurs)

---

## Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| Détection automatique | Classification binaire, multi-classe, multi-label, régression |
| Preprocessing adaptatif | Imputation, encodage, scaling avec gestion haute dimension et sparse |
| Optimisation Optuna | Recherche automatique des hyperparamètres par modèle |
| Parallélisation | Multiprocessing avec contexte spawn (compatible XGBoost/LightGBM) |
| Deux modes | `stable` (rapide, modèles éprouvés) et `explo` (exhaustif) |
| Reprise sur interruption | Sauvegarde automatique du meilleur modèle |
| Prédiction | Pipeline complet sauvegardé pour inference sur nouvelles données |

---

## Installation

### Depuis les sources

```bash
git clone [lien du dépôt]
cd master-1-automl
pip install -e .
```

### Dependencies

Les dépendances sont installées automatiquement. Pour une installation manuelle :

```bash
pip install -r requirements.txt
```

**Dépendances principales :**

| Package | Version minimale |
|---------|------------------|
| numpy | 1.23 |
| pandas | 1.5 |
| scikit-learn | 1.2 |
| optuna | 3.0 |
| scipy | 1.7 |
| xgboost | 1.5 |
| lightgbm | 3.3 |
| catboost | 1.0 |

---

## Démarrage rapide

### Entraînement

```python
import automl

# Entraînement sur un dataset (chemin sans extension)
results = automl.fit("chemin/vers/mon_dataset/data_X", mode="stable")

# Afficher les résultats
automl.eval()
```

### Prédiction sur de nouvelles données

```python
from automl.core.predictor import predict

# Charge le pipeline entraîné et prédit
predictions = predict("chemin/vers/mon_dataset/data_X")
```

### Exemple complet

```python
import automl

# Mode stable : 13-16 modèles selon les boosters installés
results = automl.fit(
    data_path="chemin/vers/mon_dataset/data_X",
    mode="stable",
    iterations=50,      # Itérations Optuna par modèle
    n_jobs=2            # Processus parallèles
)

# Structure du résultat
print(f"Type detecte : {results['task_type']}")
print(f"Modeles testes : {results['n_models_tested']}")
print(f"Meilleur : {results['best_models'][0]['model']}")
print(f"Score : {results['best_models'][0]['best_score']:.4f}")

# Mode exploratoire : plus de modèles, plus lent
results_explo = automl.fit("chemin/vers/mon_dataset/data_X", mode="explo", iterations=10)
```

---

## Format des données

Le package attend deux fichiers par dataset :

```
mon_dataset/
├── data_X.data      # Features (obligatoire)
└── data_X.solution  # Labels (obligatoire pour fit, absent pour predict)
```

### Format dense (.data)

Valeurs séparées par espaces ou tabulations, sans en-tête :

```
1.5 2.3 0.8 4.1
2.1 1.9 0.5 3.8
0.9 3.1 1.2 2.7
```

### Format sparse SVMLight (.data)

Pour données haute dimension :

```
0:1.5 2:0.8 10:4.1
1:2.1 2:0.5 8:3.8
0:0.9 5:1.2
```

Les indices absents sont interprétés comme zéro. Au-delà de 5000 colonnes, le format sparse est conservé en mémoire.

### Labels (.solution)

**Mono-label :**
```
0
1
0
2
```

**Multi-label :**
```
1 0 1
0 1 0
1 1 0
```

---

## Pipeline de traitement

```
Chargement
    load_data()
    Lecture .data + .solution
    Detection auto sparse/dense
           |
           v
Detection du type
    detect_type_model()
    binary-classification
    multi-classification
    multi-label-classification
    regression
           |
           v
Preprocessing
    AutoMLPreprocessor.fit_transform()
    Imputation (mediane/mode)
    Encodage (OneHot <= 10 cat, Ordinal > 10)
    VarianceThreshold (supprime variance=0)
    Scaling (StandardScaler ou MaxAbsScaler si >1000 cols)
           |
           v
Optimisation
    run_all_models()
    Sous-echantillonnage 30% pour Optuna
    n_trials iterations par modele
    Parallelisation spawn
           |
           v
Refit final
    Meilleur modele re-entraine sur 100% des donnees
    Pipeline complet sauvegarde (preprocessor + model)
           |
           v
Resultats
    results_{mode}.pkl : scores de tous les modeles
    best_model.pkl : pipeline pret pour predict()
```

Pour plus de détails sur le pipeline, consultez [README_PACKAGE.md](README_PACKAGE.md).

---

## API Reference

### automl.fit()

```python
def fit(
    data_path: str,
    mode: str = "stable",
    iterations: int = 50,
    n_jobs: int = 2,
    autosave_interval: int = 600
) -> dict
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `data_path` | str | requis | Chemin vers le dataset (sans extension) |
| `mode` | str | `"stable"` | `"stable"` ou `"explo"` |
| `iterations` | int | `50` | Nombre d'essais Optuna par modèle |
| `n_jobs` | int | `2` | Processus parallèles (réduit auto si < 1000 samples) |
| `autosave_interval` | int | `600` | Intervalle sauvegarde checkpoint (secondes) |

**Retour :**

```python
{
    "data_path": str,
    "task_type": str,  # "binary-classification", "multi-classification", etc.
    "best_models": [   # Top 5 modèles triés par score
        {
            "model": str,
            "best_params": dict,
            "best_score": float
        },
        ...
    ],
    "n_models_tested": int
}
```

### automl.eval() / automl.show_results()

```python
def eval(mode: str = "stable") -> dict
def show_results(mode: str = "stable", top_n: int = 5) -> dict
```

Affiche et retourne les résultats d'un run précédent.

### predict() (non exporté par défaut)

```python
from automl.core.predictor import predict

def predict(data_path: str) -> list
```

Charge le pipeline `best_model.pkl` et retourne les prédictions sur un nouveau dataset (fichier `.data` uniquement).

---

## Modèles disponibles

### Classification

**Mode stable (10-13 modèles) :** LogisticRegression, RidgeClassifier, SGDClassifier, MLPClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, SVC, GaussianNB, + XGB/LGBM/CatBoost (optionnels)

**Mode explo (ajouts) :** Perceptron, PassiveAggressiveClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, NuSVC, LinearSVC, KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid, MultinomialNB, ComplementNB, BernoulliNB, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, GaussianProcessClassifier, DummyClassifier

### Régression

**Mode stable (13-16 modèles) :** LinearRegression, Ridge, Lasso, ElasticNet, MLPRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor, DecisionTreeRegressor, SVR, KNeighborsRegressor, SGDRegressor, + XGB/LGBM/CatBoost (optionnels)

**Mode explo (ajouts) :** BayesianRidge, HuberRegressor, Lars, LassoLars, AdaBoostRegressor, HistGradientBoostingRegressor, LinearSVR, NuSVR, GaussianProcessRegressor, DummyRegressor

### Métriques

| Type | Métrique |
|------|----------|
| Classification | Accuracy Micro, Accuracy Macro |
| Régression | R2, RMSE |

Pour la liste complète des modèles et leurs configurations, consultez [README_PACKAGE.md](README_PACKAGE.md).

---

## Configuration avancée

### Modifier les répertoires par défaut

```python
from automl._config import LOG_DIR, RESULTS_DIR, CHECKPOINT_DIR
from pathlib import Path

# Les répertoires sont créés automatiquement à l'import
# Pour personnaliser avant utilisation :
import automl._config as config
config.LOG_DIR = Path("/custom/logs")
config.RESULTS_DIR = Path("/custom/results")
config.CHECKPOINT_DIR = Path("/custom/checkpoints")
```

### Paramètres par défaut

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `DEFAULT_MODE` | `"stable"` | Mode par défaut |
| `DEFAULT_ITERATIONS` | `50` | Essais Optuna |
| `DEFAULT_SAMPLE_RATIO` | `0.3` | Fraction pour optimisation |
| `DEFAULT_N_JOBS` | `2` | Parallélisme |
| `AUTOSAVE_INTERVAL` | `600` | Secondes entre sauvegardes |

---

## Interface CLI

### Afficher les résultats

```bash
# Via le point d'entrée installé
automl-results --mode stable --top 10

# Options
automl-results --mode explo --top 5 --dir /chemin/vers/projet
```

| Option | Description | Défaut |
|--------|-------------|--------|
| `--mode` | `stable` ou `explo` | `stable` |
| `--top` | Nombre de modèles à afficher | `5` |
| `--dir` | Répertoire contenant `results/` | `.` |

---

## Fichiers générés

```
projet/
├── logs/
│   ├── automl_{mode}_log.txt         # Log principal
│   ├── classification_{mode}_log.txt # Logs par type
│   ├── regression_{mode}_log.txt
│   ├── runner_{mode}_log.txt
│   ├── clean_log.txt                 # Preprocessing
│   └── detection_log.txt             # Detection type
│
├── checkpoints/
│   └── best_model_{type}_{mode}.pkl  # Meilleur modèle par run
│
└── results/
    ├── results_{mode}.pkl            # Tous les scores
    └── best_model.pkl                # Pipeline complet pour predict()
```

---

## Limitations connues

| Limitation | Détail |
|------------|--------|
| Format de données | Uniquement `.data` + `.solution` (pas de CSV/Parquet natif) |
| Métriques | Accuracy Macro et Micro ainsi que R2 et RMSE |
| Cross-validation | Non disponible (split train/test simple 80/20) |
| Feature importance | Non exportée automatiquement |
| Export modèle | Format pickle uniquement (pas ONNX/PMML) |
| GPU | Non supporté (même pour XGBoost/LightGBM/CatBoost) |
| Modèles lourds | GaussianProcess, SVC, KNN exclus automatiquement sur gros datasets |

---

## Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit (`git commit -m 'Ajout fonctionnalite X'`)
4. Push (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

### Développement local

```bash
pip install -e ".[dev]"  # Installe pytest, black, flake8
```

---

## Auteurs

- ANTIVACKIS VANESSA
- MABIRE AYMERIC
- REMARS Eliott
- STIEVENARD Emma

---

## Remerciements

- [scikit-learn](https://scikit-learn.org/)
- [Optuna](https://optuna.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [CatBoost](https://catboost.ai/)

---

## Documentation Complète

Pour une documentation détaillée incluant l'API complète, la configuration avancée et les exemples d'utilisation, consultez [README_PACKAGE.md](README_PACKAGE.md).
