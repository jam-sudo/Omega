#!/usr/bin/env python
"""Train the XGBoost direct Cmax predictor.

Reads cmax_training_set.csv, extracts Morgan FP + RDKit descriptors,
trains XGBoost with strong regularization, prints 5-fold CV metrics,
and saves the final model.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict

# Add src to path so we can import the feature extractor
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omega_pbpk.ml.models.direct_pk.xgboost_cmax import smiles_to_features  # noqa: E402

CSV_PATH = REPO_ROOT / "data" / "ml" / "clinical" / "cmax_training_set.csv"
MODEL_DIR = REPO_ROOT / "models" / "direct_pk"
MODEL_PATH = MODEL_DIR / "xgboost_cmax.json"
META_PATH = MODEL_DIR / "meta.json"


def main() -> None:
    # 1. Read training CSV
    drugs = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugs.append(row)
    print(f"Loaded {len(drugs)} drugs from {CSV_PATH.name}")

    # 2. Extract features
    X_list = []
    y_list = []
    names = []
    skipped = []
    for d in drugs:
        feat = smiles_to_features(d["smiles"])
        if feat is None:
            skipped.append(d["drug"])
            continue
        X_list.append(feat)
        y_list.append(float(d["log_obs_cmax_per_mg"]))
        names.append(d["drug"])

    if skipped:
        print(f"Skipped {len(skipped)} drugs (feature extraction failed): {skipped}")

    X = np.array(X_list)
    y = np.array(y_list, dtype=np.float64)
    print(f"Feature matrix: {X.shape[0]} drugs x {X.shape[1]} features")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

    # 3. 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.3,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        verbosity=0,
    )

    cv_preds = cross_val_predict(model, X, y, cv=cv)

    # 4. CV Metrics
    # Fold errors in linear space: exp(pred) vs exp(true)
    pred_linear = np.exp(cv_preds)
    true_linear = np.exp(y)
    fold_errors = np.maximum(pred_linear / true_linear, true_linear / pred_linear)

    aafe = float(np.mean(fold_errors))
    rmse_log = float(np.sqrt(np.mean((cv_preds - y) ** 2)))
    pct_2fold = float(np.mean(fold_errors <= 2.0) * 100)
    pct_3fold = float(np.mean(fold_errors <= 3.0) * 100)

    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation Results")
    print("=" * 60)
    print(f"  AAFE:        {aafe:.3f}")
    print(f"  RMSE(log):   {rmse_log:.3f}")
    print(f"  %2-fold:     {pct_2fold:.1f}%")
    print(f"  %3-fold:     {pct_3fold:.1f}%")
    print("=" * 60)

    # 5. Per-drug errors (worst 10)
    print("\nWorst 10 predictions (by fold error):")
    for idx in np.argsort(fold_errors)[::-1][:10]:
        obs = math.exp(y[idx])
        pred = math.exp(cv_preds[idx])
        print(
            f"  {names[idx]:25s} FE={fold_errors[idx]:.2f}  "
            f"obs_norm={obs:.4f}  pred_norm={pred:.4f}"
        )

    # 6. Train final model on ALL data
    final_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.3,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    # 7. Top 15 feature importances
    importances = final_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]

    feature_names = [f"FP_{i}" for i in range(2048)] + [
        "MolLogP",
        "TPSA/200",
        "MolWt/600",
        "NumHAcceptors/10",
        "NumHDonors/5",
        "NumRotatableBonds/15",
        "RingCount/5",
        "FractionCSP3",
        "MolMR/150",
    ]

    print("\nTop 15 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[idx]:25s}  {importances[idx]:.4f}")

    # 8. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(MODEL_PATH))
    print(f"\nModel saved to {MODEL_PATH}")

    # 9. Save metadata
    meta = {
        "n_drugs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_aafe": round(aafe, 4),
        "cv_rmse_log": round(rmse_log, 4),
        "cv_pct_2fold": round(pct_2fold, 1),
        "cv_pct_3fold": round(pct_3fold, 1),
        "target": "log(obs_cmax / dose_mg)",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.3,
            "min_child_weight": 5,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        },
        "skipped_drugs": skipped,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
