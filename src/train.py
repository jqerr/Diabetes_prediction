#!/usr/bin/env python3
"""
Experiment runner.
Usage: python src/train.py experiments/<config>.yaml

Each YAML file defines one experiment (one idea).
Metrics are appended to results/experiment_log.csv.
Test set is never loaded here — sealed until end of project.
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate

MODEL_REGISTRY = {
    "LogisticRegression":         LogisticRegression,
    "RandomForestClassifier":     RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "KNeighborsClassifier":       KNeighborsClassifier,
    "SVC":                        SVC,
}

SCORING = {
    "accuracy":  "accuracy",
    "f1":        "f1",
    "roc_auc":   "roc_auc",
    "precision": "precision",
    "recall":    "recall",
}

DATA_DIR  = Path("data/preprocessed_data")
LOG_FILE  = Path("results/experiment_log.csv")


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "N/A"


def load_train_data(use_scaled: bool):
    suffix = "_scaled" if use_scaled else ""
    X = pd.read_csv(DATA_DIR / f"X_train{suffix}.csv", index_col=0)
    y = pd.read_csv(DATA_DIR / "y_train.csv",          index_col=0).squeeze()
    return X, y


def append_row(row: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if LOG_FILE.exists():
        existing = pd.read_csv(LOG_FILE)
        # union columns so experiments with different params stay aligned
        all_cols = list(existing.columns) + [c for c in row if c not in existing.columns]
        new_df   = pd.DataFrame([row]).reindex(columns=all_cols)
        updated  = pd.concat([existing.reindex(columns=all_cols), new_df], ignore_index=True)
        updated.to_csv(LOG_FILE, index=False)
    else:
        pd.DataFrame([row]).to_csv(LOG_FILE, index=False)


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id      = cfg["experiment_id"]
    description = cfg.get("description", "")
    hypothesis  = cfg.get("hypothesis",  "")
    use_scaled  = cfg["data"].get("use_scaled", False)
    model_type  = cfg["model"]["type"]
    model_params = cfg["model"].get("params", {})

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_id}")
    print(f"Description: {description}")
    print(f"Model      : {model_type}({model_params})")
    print(f"Data       : {'scaled' if use_scaled else 'unscaled'} train set only")
    print(f"{'='*60}")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    X_train, y_train = load_train_data(use_scaled)
    model = MODEL_REGISTRY[model_type](**model_params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running 5-fold stratified CV...")
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)

    metrics = {}
    for metric in SCORING:
        scores = cv_results[f"test_{metric}"]
        metrics[f"cv_{metric}_mean"] = round(float(scores.mean()), 4)
        metrics[f"cv_{metric}_std"]  = round(float(scores.std()),  4)
        print(f"  {metric:<10}: {scores.mean():.4f} ± {scores.std():.4f}")

    row = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": exp_id,
        "description":   description,
        "hypothesis":    hypothesis,
        "model_type":    model_type,
        **{f"param_{k}": v for k, v in model_params.items()},
        **metrics,
        "git_hash":      get_git_hash(),
    }

    append_row(row)
    print(f"\nLogged → {LOG_FILE.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train.py <experiments/config.yaml>")
        sys.exit(1)
    run(sys.argv[1])
