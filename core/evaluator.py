"""Evaluation et affichage des resultats."""
import pickle
import argparse
from pathlib import Path
from automl._config import RESULTS_DIR


def show_results(mode: str = "stable", top_n: int = 5) -> dict:
    """
    Affiche les resultats d'un run AutoML.
    
    Args:
        mode: "stable" ou "explo"
        top_n: Nombre de modeles a afficher
        
    Returns:
        Dict des resultats
    """
    path = RESULTS_DIR / f"results_{mode}.pkl"
    
    if not path.exists():
        print(f"Fichier non trouve: {path}")
        return {}

    with open(path, "rb") as f:
        results = pickle.load(f)

    for dataset, info in results.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        # Retrocompatibilite avec anciens formats utilisant 'type'
        print(f"Type: {info.get('task_type', info.get('type', 'N/A'))}")
        print(f"\nTop {top_n} modeles:")
        
        models = info.get("best_models", [])
        for i, m in enumerate(models[:top_n], 1):
            score = m.get("best_score", 0)
            print(f"{i}. {m['model']:30s} | Score: {score:.4f}")

    return results


def eval(mode: str = "stable"):
    """Alias pour show_results."""
    return show_results(mode)


def main():
    """Point d'entree CLI: automl-results"""
    parser = argparse.ArgumentParser(description="Afficher les resultats AutoML")
    parser.add_argument("--mode", choices=["stable", "explo"], default="stable")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--dir", default=".", help="Repertoire des resultats")
    args = parser.parse_args()

    # Override du repertoire pour execution hors contexte projet
    global RESULTS_DIR
    RESULTS_DIR = Path(args.dir) / "results"
    
    show_results(args.mode, args.top)


if __name__ == "__main__":
    main()