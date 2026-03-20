#!/usr/bin/env python
"""Train XGBoost direct Cmax predictor V2 on MMPK clean dataset (1,128 drugs).

Reads mmpk_clean.csv, extracts Morgan FP + RDKit descriptors via smiles_to_features(),
trains XGBoost with hyperparameters tuned for 15x more data than V1,
prints 5-fold CV metrics, and saves the final model.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

# Add src to path so we can import the feature extractor
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omega_pbpk.ml.models.direct_pk.xgboost_cmax import smiles_to_features  # noqa: E402

CSV_PATH = REPO_ROOT / "data" / "ml" / "clinical" / "mmpk_clean.csv"
MODEL_DIR = REPO_ROOT / "models" / "direct_pk"
MODEL_PATH = MODEL_DIR / "xgboost_cmax_v2.json"
META_PATH = MODEL_DIR / "meta_v2.json"

# V2 hyperparameters — adjusted for 15x more training data
HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 3,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "random_state": 42,
    "verbosity": 0,
}


def main() -> None:
    # 1. Read MMPK clean CSV
    import csv

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
            skipped.append(d["name"])
            continue
        X_list.append(feat)
        y_list.append(float(d["log_cmax_per_dose"]))
        names.append(d["name"])

    if skipped:
        print(f"Skipped {len(skipped)} drugs (feature extraction failed): {skipped}")

    X = np.array(X_list)
    y = np.array(y_list, dtype=np.float64)
    names_arr = np.array(names)
    print(f"Feature matrix: {X.shape[0]} drugs x {X.shape[1]} features")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

    # 3. 5-fold CV with per-fold metrics
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.zeros_like(y)

    print("\nPer-fold results:")
    print("-" * 60)
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        model = xgb.XGBRegressor(**HYPERPARAMS)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        cv_preds[val_idx] = preds

        # Fold metrics
        pred_lin = np.exp(preds)
        true_lin = np.exp(y[val_idx])
        fe = np.maximum(pred_lin / true_lin, true_lin / pred_lin)
        fold_aafe = float(np.exp(np.mean(np.log(fe))))
        fold_2f = float(np.mean(fe <= 2.0) * 100)
        print(f"  Fold {fold_i}: n={len(val_idx)}, AAFE={fold_aafe:.3f}, %2-fold={fold_2f:.1f}%")

    # 4. Overall CV Metrics
    pred_linear = np.exp(cv_preds)
    true_linear = np.exp(y)
    fold_errors = np.maximum(pred_linear / true_linear, true_linear / pred_linear)

    # AAFE = geometric mean of fold errors
    aafe = float(np.exp(np.mean(np.log(fold_errors))))
    rmse_log = float(np.sqrt(np.mean((cv_preds - y) ** 2)))
    pct_2fold = float(np.mean(fold_errors <= 2.0) * 100)
    pct_3fold = float(np.mean(fold_errors <= 3.0) * 100)

    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation Results (V2, MMPK 1128 drugs)")
    print("=" * 60)
    print(f"  AAFE (geomean):  {aafe:.3f}")
    print(f"  RMSE(log):       {rmse_log:.3f}")
    print(f"  %2-fold:         {pct_2fold:.1f}%")
    print(f"  %3-fold:         {pct_3fold:.1f}%")
    print(f"  N drugs:         {X.shape[0]}")
    print(f"  N features:      {X.shape[1]}")
    print("=" * 60)

    # 5. Per-drug errors (worst 10)
    print("\nWorst 10 predictions (by fold error):")
    for idx in np.argsort(fold_errors)[::-1][:10]:
        obs = math.exp(y[idx])
        pred = math.exp(cv_preds[idx])
        print(
            f"  {names_arr[idx]:30s} FE={fold_errors[idx]:.2f}  "
            f"obs_norm={obs:.4f}  pred_norm={pred:.4f}"
        )

    # 6. Train final model on ALL data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(**HYPERPARAMS)
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
        "version": "v2",
        "trained_on": "mmpk_clean.csv",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_drugs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_aafe": round(aafe, 4),
        "cv_rmse_log": round(rmse_log, 4),
        "cv_pct_2fold": round(pct_2fold, 1),
        "cv_pct_3fold": round(pct_3fold, 1),
        "target": "log(cmax_mg_L / dose_mg)",
        "hyperparameters": HYPERPARAMS,
        "skipped_drugs": skipped,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
