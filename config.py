"""Configuration globale du package."""
from pathlib import Path

# Repertoires par defaut
BASE_DIR = Path.cwd()
LOG_DIR = BASE_DIR / "logs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Creation a l'import pour garantir leur existence avant toute operation I/O
for d in [LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Parametres par defaut
DEFAULT_MODE = "stable"           # "stable" = modeles eprouves, "experimental" = tous
DEFAULT_ITERATIONS = 50
DEFAULT_SAMPLE_RATIO = 0.3        # Fraction pour la recherche d'hyperparametres
DEFAULT_N_JOBS = 2                # Limite pour eviter la surcharge memoire
AUTOSAVE_INTERVAL = 600           # Secondes entre sauvegardes automatiques