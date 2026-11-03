# AutoML

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.0+-orange.svg)](https://scikit-learn.org/)

AutoML est une pipeline automatisee de Machine Learning qui detecte le type de probleme, nettoie les donnees, optimise les hyperparametres via Optuna, et selectionne le meilleur modele.

---

## Table des matieres

1. [Fonctionnalites](#fonctionnalites)
2. [Installation](#installation)
3. [Demarrage rapide](#demarrage-rapide)
4. [Format des donnees](#format-des-donnees)
5. [Pipeline de traitement](#pipeline-de-traitement)
6. [API Reference](#api-reference)
7. [Modeles disponibles](#modeles-disponibles)
8. [Configuration avancee](#configuration-avancee)
9. [Interface CLI](#interface-cli)
10. [Fichiers generes](#fichiers-generes)
11. [Limitations connues](#limitations-connues)
12. [Contribution](#contribution)

---

## Fonctionnalites

| Fonctionnalite | Description |
|----------------|-------------|
| Detection automatique | Classification binaire, multi-classe, multi-label, regression |
| Preprocessing adaptatif | Imputation, encodage, scaling avec gestion haute dimension et sparse |
| Optimisation Optuna | Recherche automatique des hyperparametres par modele |
| Parallelisation | Multiprocessing avec contexte spawn (compatible XGBoost/LightGBM) |
| Deux modes | `stable` (rapide, modeles eprouves) et `explo` (exhaustif) |
| Reprise sur interruption | Sauvegarde automatique du meilleur modele |
| Prediction | Pipeline complet sauvegarde pour inference sur nouvelles donnees |

---

## Installation

### Depuis les sources

```bash
git clone [lien du depot]
cd master-1-automl
pip install -e .
```

### Dependances

Les dependances sont installees automatiquement. Pour une installation manuelle :

```bash
pip install -r requirements.txt
```

**Dependances principales :**

| Package | Version minimale |
|---------|------------------|
| numpy | 1.21 |
| pandas | 1.3 |
| scikit-learn | 1.0 |
| optuna | 3.0 |
| scipy | 1.7 |
| xgboost | 1.5 |
| lightgbm | 3.3 |
| catboost | 1.0 |

---

## Demarrage rapide

### Entrainement

```python
import automl

# Entrainement sur un dataset (chemin sans extension)
results = automl.fit("chemin/vers/mon_dataset/data_X", mode="stable")

# Afficher les resultats
automl.eval()
```

### Prediction sur de nouvelles donnees

```python
from automl.core.predictor import predict

# Charge le pipeline entraine et predit
predictions = predict("chemin/vers/mon_dataset/data_X")
```

### Exemple complet

```python
import automl

# Mode stable : 13-16 modeles selon les boosters installes
results = automl.fit(
    data_path="chemin/vers/mon_dataset/data_X",
    mode="stable",
    iterations=50,      # Iterations Optuna par modele
    n_jobs=2            # Processus paralleles
)

# Structure du resultat
print(f"Type detecte : {results['task_type']}")
print(f"Modeles testes : {results['n_models_tested']}")
print(f"Meilleur : {results['best_models'][0]['model']}")
print(f"Score : {results['best_models'][0]['best_score']:.4f}")

# Mode exploratoire : plus de modeles, plus lent
results_explo = automl.fit("chemin/vers/mon_dataset/data_X", mode="explo", iterations=10)
```

---

## Format des donnees

Le package attend deux fichiers par dataset :

```
mon_dataset/
├── data_X.data      # Features (obligatoire)
└── data_X.solution  # Labels (obligatoire pour fit, absent pour predict)
```

### Format dense (.data)

Valeurs separees par espaces ou tabulations, sans en-tete :

```
1.5 2.3 0.8 4.1
2.1 1.9 0.5 3.8
0.9 3.1 1.2 2.7
```

### Format sparse SVMLight (.data)

Pour donnees haute dimension :

```
0:1.5 2:0.8 10:4.1
1:2.1 2:0.5 8:3.8
0:0.9 5:1.2
```

Les indices absents sont interpretes comme zero. Au-dela de 5000 colonnes, le format sparse est conserve en memoire.

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

### Detection du type de probleme

| Condition | Type detecte |
|-----------|--------------|
| 2 valeurs uniques | `binary-classification` |
| < 20 valeurs uniques OU ratio < 1% | `multi-classification` |
| sklearn detecte `multilabel-indicator` | `multi-label-classification` |
| Valeurs continues (> 20 uniques ET ratio > 1%) | `regression` |

### Preprocessing adaptatif

| Contexte | Strategie |
|----------|-----------|
| Colonnes numeriques | Imputation mediane |
| Colonnes categorielles | Imputation mode |
| Cardinalite <= 10 | OneHotEncoder |
| Cardinalite > 10 | OrdinalEncoder |
| Haute dimension (> 1000 cols) | MaxAbsScaler (preserve sparsity) |
| Dimension standard | StandardScaler |
| Winsorization | Quantiles 5-95%, desactivee si > 1000 cols |

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

**Parametres :**

| Parametre | Type | Defaut | Description |
|-----------|------|--------|-------------|
| `data_path` | str | requis | Chemin vers le dataset (sans extension) |
| `mode` | str | `"stable"` | `"stable"` ou `"explo"` |
| `iterations` | int | `50` | Nombre d'essais Optuna par modele |
| `n_jobs` | int | `2` | Processus paralleles (reduit auto si < 1000 samples) |
| `autosave_interval` | int | `600` | Intervalle sauvegarde checkpoint (secondes) |

**Retour :**

```python
{
    "data_path": str,
    "task_type": str,  # "binary-classification", "multi-classification", etc.
    "best_models": [   # Top 5 modeles tries par score
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

Affiche et retourne les resultats d'un run precedent.

### predict() (non exporte par defaut)

```python
from automl.core.predictor import predict

def predict(data_path: str) -> list
```

Charge le pipeline `best_model.pkl` et retourne les predictions sur un nouveau dataset (fichier `.data` uniquement).

---

## Modeles disponibles

### Classification

**Mode stable (10-13 modeles) :**

| Modele | Source |
|--------|--------|
| LogisticRegression | sklearn |
| RidgeClassifier | sklearn |
| SGDClassifier | sklearn |
| MLPClassifier | sklearn |
| RandomForestClassifier | sklearn |
| GradientBoostingClassifier | sklearn |
| ExtraTreesClassifier | sklearn |
| BaggingClassifier | sklearn |
| SVC | sklearn |
| GaussianNB | sklearn |
| XGBClassifier | xgboost (optionnel) |
| LGBMClassifier | lightgbm (optionnel) |
| CatBoostClassifier | catboost (optionnel) |

**Mode explo (ajouts) :**

| Modele | Note |
|--------|------|
| Perceptron, PassiveAggressiveClassifier | Lineaires |
| AdaBoostClassifier, HistGradientBoostingClassifier | Boosting |
| NuSVC, LinearSVC | SVM variants |
| KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid | Voisins |
| MultinomialNB, ComplementNB, BernoulliNB | Naive Bayes |
| LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis | Discriminant |
| GaussianProcessClassifier | Exclu si dataset > 500k elements |
| DummyClassifier | Baseline |

**Gestion multi-label :**

Les modeles suivants supportent nativement le multi-label :
- RandomForestClassifier
- ExtraTreesClassifier
- KNeighborsClassifier
- MLPClassifier
- RadiusNeighborsClassifier

Les autres sont wrapes automatiquement avec `OneVsRestClassifier`.

Modeles exclus en multi-label : CatBoost, XGBoost, LightGBM, QDA, LDA, ComplementNB.

### Regression

**Mode stable (13-16 modeles) :**

| Modele | Source |
|--------|--------|
| LinearRegression | sklearn |
| Ridge, Lasso, ElasticNet | sklearn |
| MLPRegressor | sklearn |
| RandomForestRegressor | sklearn |
| GradientBoostingRegressor | sklearn |
| BaggingRegressor, ExtraTreesRegressor | sklearn |
| DecisionTreeRegressor | sklearn |
| SVR | sklearn |
| KNeighborsRegressor | sklearn |
| SGDRegressor | sklearn |
| XGBRegressor | xgboost (optionnel) |
| LGBMRegressor | lightgbm (optionnel) |
| CatBoostRegressor | catboost (optionnel) |

**Mode explo (ajouts) :**

| Modele | Note |
|--------|------|
| BayesianRidge, HuberRegressor | Robustes |
| Lars, LassoLars | Selection variables |
| AdaBoostRegressor, HistGradientBoostingRegressor | Boosting |
| LinearSVR, NuSVR | SVM variants |
| GaussianProcessRegressor | Exclu si > 10k samples |
| DummyRegressor | Baseline |

### Metriques

| Type | Metrique |
|------|----------|
| Classification | Accuracy Micro, Accuracy Macro |
| Regression | R2, RMSE  |

---

## Configuration avancee

### Modifier les repertoires par defaut

```python
from automl._config import LOG_DIR, RESULTS_DIR, CHECKPOINT_DIR
from pathlib import Path

# Les repertoires sont crees automatiquement a l'import
# Pour personnaliser avant utilisation :
import automl._config as config
config.LOG_DIR = Path("/custom/logs")
config.RESULTS_DIR = Path("/custom/results")
config.CHECKPOINT_DIR = Path("/custom/checkpoints")
```

### Parametres par defaut

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| `DEFAULT_MODE` | `"stable"` | Mode par defaut |
| `DEFAULT_ITERATIONS` | `50` | Essais Optuna |
| `DEFAULT_SAMPLE_RATIO` | `0.3` | Fraction pour optimisation |
| `DEFAULT_N_JOBS` | `2` | Parallelisme |
| `AUTOSAVE_INTERVAL` | `600` | Secondes entre sauvegardes |

### Modifier les hyperparametres par defaut

```python
from automl.classification._base import DEFAULT_PARAMS_STABLE

# Exemple : augmenter le nombre d'arbres
DEFAULT_PARAMS_STABLE["RandomForestClassifier"]["n_estimators"] = 200
```

---

## Interface CLI

### Afficher les resultats

```bash
# Via le point d'entree installe
automl-results --mode stable --top 10

# Options
automl-results --mode explo --top 5 --dir /chemin/vers/projet
```

| Option | Description | Defaut |
|--------|-------------|--------|
| `--mode` | `stable` ou `explo` | `stable` |
| `--top` | Nombre de modeles a afficher | `5` |
| `--dir` | Repertoire contenant `results/` | `.` |

### Exemple de sortie

```
==================================================
Dataset: chemin/vers/mon_dataset/data_X
Type: multi-classification

Top 5 modeles:
1. RandomForestClassifier          | Score: 0.9667
2. GradientBoostingClassifier      | Score: 0.9667
3. XGBClassifier                   | Score: 0.9333
4. LGBMClassifier                  | Score: 0.9333
5. SVC                             | Score: 0.9333
```

---

## Fichiers generes

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
│   └── best_model_{type}_{mode}.pkl  # Meilleur modele par run
│
└── results/
    ├── results_{mode}.pkl            # Tous les scores
    └── best_model.pkl                # Pipeline complet pour predict()
```

### Structure de best_model.pkl

```python
{
    "preprocessor": AutoMLPreprocessor,  # Fitted
    "model": sklearn_estimator,          # Fitted sur 100% des donnees
    "task_type": str,
    "info": {
        "model": str,
        "best_params": dict,
        "best_score": float
    }
}
```

---

## Limitations connues

| Limitation | Detail |
|------------|--------|
| Format de donnees | Uniquement `.data` + `.solution` (pas de CSV/Parquet natif) |
| Metriques | Accuracy Macro et Micro ainsi que R2 et RMSE  |
| Cross-validation | Non disponible (split train/test simple 80/20) |
| Feature importance | Non exportee automatiquement |
| Export modele | Format pickle uniquement (pas ONNX/PMML) |
| GPU | Non supporte (meme pour XGBoost/LightGBM/CatBoost) |
| Modeles lourds | GaussianProcess, SVC, KNN exclus automatiquement sur gros datasets |

### Gestion des erreurs

| Situation | Comportement |
|-----------|--------------|
| Interruption (Ctrl+C) | Sauvegarde checkpoint, arret propre |
| Erreur sur un modele | Log erreur, continue avec les autres |
| Dataset < 1000 samples | Parallelisme desactive (n_jobs=1) |
| Dataset > 50k samples | n_jobs reduit a 2 maximum |
| Classe trop rare pour stratify | Fallback sans stratification |

---

## Contribution

1. Fork le projet
2. Creer une branche (`git checkout -b feature/amelioration`)
3. Commit (`git commit -m 'Ajout fonctionnalite X'`)
4. Push (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

### Developpement local

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