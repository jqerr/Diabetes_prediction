#!/usr/bin/env python3
"""
Final test set evaluation.
Usage: python src/evaluate.py experiments/<config>.yaml

Run this ONCE at the end of the project when a final model is chosen.
The test set is sealed — do not use this script during experimentation.
"""
import sys
import importlib
from pathlib import Path

import yaml
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data/preprocessed_data")


def load_model_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id       = cfg["experiment_id"]
    use_scaled   = cfg["data"].get("use_scaled", False)
    model_type   = cfg["model"]["type"]
    model_params = cfg["model"].get("params", {})

    print(f"\n{'='*60}")
    print(f"FINAL TEST SET EVALUATION — {exp_id}")
    print(f"{'='*60}")
    print("WARNING: only run this once. Test set is now unsealed.\n")

    suffix = "_scaled" if use_scaled else ""
    X_train = pd.read_csv(DATA_DIR / f"X_train{suffix}.csv", index_col=0)
    y_train = pd.read_csv(DATA_DIR / "y_train.csv",          index_col=0).squeeze()
    X_test  = pd.read_csv(DATA_DIR / f"X_test{suffix}.csv",  index_col=0)
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv",           index_col=0).squeeze()

    model = load_model_class(model_type)(**model_params)

    if "features" in cfg:
        feature_module = importlib.import_module(cfg["features"]["module"])
        for fn_name in cfg["features"]["transforms"]:
            X_train = getattr(feature_module, fn_name)(X_train)
            X_test  = getattr(feature_module, fn_name)(X_test)

    if "sampling" in cfg:
        sampler = load_model_class(cfg["sampling"]["method"])(
            **cfg["sampling"].get("params", {})
        )
        model = ImbPipeline([("sampler", sampler), ("model", model)])

    # fit on full train set, evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    print("Summary:")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print()

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/evaluate.py <experiments/config.yaml>")
        sys.exit(1)
    run(sys.argv[1])
