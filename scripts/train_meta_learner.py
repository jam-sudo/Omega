#!/usr/bin/env python3
"""Train CmaxMetaLearner: adaptive PBPK/ML blend for Cmax prediction.

Reads MMPK PBPK features, gets DirectCmaxV2 predictions for each drug,
constructs 12-feature meta-feature vectors, and trains an XGBoost model
to predict log10(cmax_obs).

Usage:
    python scripts/train_meta_learner.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_PATH = REPO_ROOT / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"
V2_MODEL_PATH = REPO_ROOT / "models" / "direct_pk" / "xgboost_cmax_v2.json"
OUTPUT_DIR = REPO_ROOT / "models" / "meta_learner"


def main() -> None:
    import xgboost as xgb
    from sklearn.model_selection import KFold

    from omega_pbpk.ml.models.direct_pk.meta_learner import (
        FEATURE_NAMES,
        MetaFeatures,
    )
    from omega_pbpk.ml.models.direct_pk.xgboost_cmax import smiles_to_features

    # --- Load data ---
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} drugs loaded")

    # --- Get DirectCmaxV2 predictions ---
    # NOTE: DirectCmaxPredictor.predict() uses math.exp() but V2 was trained
    # with log10 target. We use the raw XGBoost model + 10** conversion to
    # get correct predictions.
    print(f"Loading DirectCmaxV2 (raw) from {V2_MODEL_PATH}")
    v2_model = xgb.XGBRegressor()
    v2_model.load_model(str(V2_MODEL_PATH))

    # Load V2 metadata
    v2_meta_path = V2_MODEL_PATH.parent / "meta_v2.json"
    if v2_meta_path.exists():
        with open(v2_meta_path) as f:
            v2_meta = json.load(f)
        print(f"  V2 model: {v2_meta.get('version', '?')}, CV AAFE {v2_meta.get('cv_aafe', '?')}")
    else:
        print("  WARNING: V2 model metadata not found")

    print("Generating ML predictions...")
    t0 = time.time()
    cmax_ml_list = []
    skipped = []
    for idx, row in df.iterrows():
        try:
            feat = smiles_to_features(row["smiles"])
            if feat is None:
                skipped.append((idx, row["name"], "feature extraction failed"))
                continue
            # V2 target = log10(cmax / dose_mg), so: cmax = 10^pred * dose_mg
            log_cmax_per_dose = float(v2_model.predict(feat.reshape(1, -1))[0])
            cmax_ml = (10.0**log_cmax_per_dose) * row["dose_mg"]
            if cmax_ml <= 0 or not np.isfinite(cmax_ml):
                skipped.append((idx, row["name"], "non-positive ML prediction"))
                continue
            cmax_ml_list.append((idx, cmax_ml))
        except Exception as e:
            skipped.append((idx, row["name"], str(e)))

    print(f"  {len(cmax_ml_list)} predictions in {time.time() - t0:.1f}s")
    if skipped:
        print(f"  {len(skipped)} skipped: {skipped[:5]}")

    # Filter to valid predictions
    valid_indices = [i for i, _ in cmax_ml_list]
    cmax_ml_arr = np.array([v for _, v in cmax_ml_list])
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    # --- Construct meta-features ---
    print("Constructing 12-feature meta-vectors...")
    X_list = []
    y_list = []
    names = []
    valid_mask = []

    for i, (_, row) in enumerate(df_valid.iterrows()):
        cmax_pbpk = row["cmax_pbpk"]
        cmax_ml = cmax_ml_arr[i]
        cmax_obs = row["cmax_obs"]

        # Skip invalid entries
        if (
            cmax_pbpk <= 0
            or cmax_obs <= 0
            or not np.isfinite(cmax_pbpk)
            or not np.isfinite(cmax_obs)
        ):
            continue

        mf = MetaFeatures(
            cmax_pbpk=cmax_pbpk,
            cmax_ml=cmax_ml,
            dose_mg=row["dose_mg"],
            logP=row["logP"],
            TPSA_norm=row["TPSA_norm"],
            MW_norm=row["MW_norm"],
            fup=row["fup"],
            clint=row["clint"],
            is_acid=float(row["is_acid"]),
            is_base=float(row["is_base"]),
            pgp_flag=float(row["pgp_flag"]),
        )
        arr = mf.to_array()
        if not np.isfinite(arr).all():
            continue

        X_list.append(arr)
        y_list.append(np.log10(cmax_obs))
        names.append(row["name"])
        valid_mask.append(i)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  {len(X)} valid training samples, {len(FEATURE_NAMES)} features")

    # Also store PBPK and ML predictions for comparison
    pbpk_preds = np.array([df_valid.iloc[i]["cmax_pbpk"] for i in valid_mask], dtype=np.float64)
    ml_preds = np.array([cmax_ml_arr[i] for i in valid_mask], dtype=np.float64)

    # --- 5-fold CV ---
    print("\n5-fold Cross-Validation:")
    print("=" * 60)

    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_meta_fe = []
    all_pbpk_fe = []
    all_ml_fe = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, verbose=False)

        # Meta-learner predictions (linear space)
        y_pred_log = model.predict(X_test)
        meta_pred = 10.0**y_pred_log
        obs_test = 10.0**y_test

        # Fold errors: FE = max(pred/obs, obs/pred)
        meta_fe = np.maximum(meta_pred / obs_test, obs_test / meta_pred)
        pbpk_fe = np.maximum(pbpk_preds[test_idx] / obs_test, obs_test / pbpk_preds[test_idx])
        ml_fe = np.maximum(ml_preds[test_idx] / obs_test, obs_test / ml_preds[test_idx])

        all_meta_fe.extend(meta_fe.tolist())
        all_pbpk_fe.extend(pbpk_fe.tolist())
        all_ml_fe.extend(ml_fe.tolist())

        # Per-fold stats
        meta_aafe = np.exp(np.mean(np.log(meta_fe)))
        pbpk_aafe = np.exp(np.mean(np.log(pbpk_fe)))
        ml_aafe = np.exp(np.mean(np.log(ml_fe)))
        print(
            f"  Fold {fold + 1}: MetaLearner AAFE={meta_aafe:.3f}, "
            f"PBPK={pbpk_aafe:.3f}, ML={ml_aafe:.3f} "
            f"(n={len(test_idx)})"
        )

    # Overall CV metrics
    all_meta_fe = np.array(all_meta_fe)
    all_pbpk_fe = np.array(all_pbpk_fe)
    all_ml_fe = np.array(all_ml_fe)

    meta_aafe = np.exp(np.mean(np.log(all_meta_fe)))
    pbpk_aafe = np.exp(np.mean(np.log(all_pbpk_fe)))
    ml_aafe = np.exp(np.mean(np.log(all_ml_fe)))

    meta_2fold = 100.0 * np.mean(all_meta_fe <= 2.0)
    meta_3fold = 100.0 * np.mean(all_meta_fe <= 3.0)
    pbpk_2fold = 100.0 * np.mean(all_pbpk_fe <= 2.0)
    ml_2fold = 100.0 * np.mean(all_ml_fe <= 2.0)

    print("\n" + "=" * 60)
    print("OVERALL CV RESULTS:")
    print("=" * 60)
    print(
        f"  MetaLearner AAFE: {meta_aafe:.4f}  "
        f"(%2-fold: {meta_2fold:.1f}%, %3-fold: {meta_3fold:.1f}%)"
    )
    print(f"  PBPK-only  AAFE:  {pbpk_aafe:.4f}  (%2-fold: {pbpk_2fold:.1f}%)")
    print(f"  ML-only    AAFE:  {ml_aafe:.4f}  (%2-fold: {ml_2fold:.1f}%)")
    print()

    improvement = (pbpk_aafe - meta_aafe) / pbpk_aafe * 100
    print(f"  MetaLearner vs PBPK: {improvement:+.1f}% improvement")
    improvement_ml = (ml_aafe - meta_aafe) / ml_aafe * 100
    print(f"  MetaLearner vs ML:   {improvement_ml:+.1f}% improvement")

    # --- Feature importances ---
    print("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X, y, verbose=False)

    importances = final_model.feature_importances_
    print("\nFeature Importances:")
    print("-" * 40)
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        bar = "#" * int(importances[i] * 50)
        print(f"  {FEATURE_NAMES[i]:20s} {importances[i]:.4f}  {bar}")

    # --- Save model ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "xgboost_meta.json"
    final_model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    meta_info = {
        "version": "v1",
        "trained_on": "mmpk_pbpk_features.csv",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_drugs": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "cv_aafe": round(float(meta_aafe), 4),
        "cv_rmse_log": round(float(np.sqrt(np.mean((all_meta_fe - 1) ** 2))), 4),
        "cv_pct_2fold": round(float(meta_2fold), 1),
        "cv_pct_3fold": round(float(meta_3fold), 1),
        "pbpk_only_aafe": round(float(pbpk_aafe), 4),
        "ml_only_aafe": round(float(ml_aafe), 4),
        "target": "log10(cmax_obs)",
        "hyperparameters": xgb_params,
        "feature_importances": {
            FEATURE_NAMES[i]: round(float(importances[i]), 4) for i in sorted_idx
        },
    }
    meta_path = OUTPUT_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # --- Acceptance check ---
    print("\n" + "=" * 60)
    if meta_aafe < pbpk_aafe:
        print("PASS: MetaLearner CV AAFE < PBPK-only AAFE")
    else:
        print("FAIL: MetaLearner CV AAFE >= PBPK-only AAFE")
    print("=" * 60)


if __name__ == "__main__":
    main()
