#!/usr/bin/env python3
"""Train Ridge correction model on expanded benchmark residuals.

Reads the latest expanded benchmark JSON, extracts molecular features,
computes log-residuals, and trains Ridge regression models for Cmax and AUC.

Usage:
    python scripts/train_correction_model.py
    python scripts/train_correction_model.py --benchmark outputs/expanded_benchmark_2026-03-16.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.ml.models.correction.residual_model import (  # noqa: E402
    FEATURE_NAMES,
    ResidualCorrectionModel,
)


def extract_features(drug_entry: dict, pipeline, smiles: str) -> np.ndarray | None:
    """Extract 6 features for a drug."""
    dose_mg = drug_entry.get("dose_mg", 100.0)

    try:
        adme = pipeline._predict_adme(smiles, [])
        logP = adme.get("logP", 2.0)
        mw = adme.get("mw", 300.0)
        fup = adme.get("fup", 0.1)
        peff = adme.get("peff", 1.0)
    except Exception:
        return None

    pgp = 0
    try:
        from omega_pbpk.ml.models.adme.transporter_lookup import is_pgp_substrate

        pgp = 1 if is_pgp_substrate(smiles=smiles) else 0
    except Exception:
        pass

    if any(v is None or v <= 0 for v in [mw, peff]):
        return None

    return np.array(
        [
            logP,
            np.log10(max(mw, 1.0)),
            fup,
            np.log10(max(dose_mg, 0.1)),
            pgp,
            np.log10(max(peff, 1e-6)),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    # Find latest expanded benchmark
    if args.benchmark:
        bm_path = Path(args.benchmark)
    else:
        bm_files = sorted((repo_root / "outputs").glob("expanded_benchmark_*.json"))
        if not bm_files:
            print("ERROR: No expanded benchmark found. Run run_expanded_benchmark.py first.")
            sys.exit(1)
        bm_path = bm_files[-1]

    print(f"Loading benchmark: {bm_path}")
    with open(bm_path) as f:
        benchmark = json.load(f)

    # Initialize pipeline for feature extraction
    from omega_pbpk.pipeline import OmegaPipeline

    pipeline = OmegaPipeline()
    pipeline._ensure_initialized()

    # Extract features and residuals
    features_list = []
    cmax_residuals = []
    auc_residuals_raw = []
    drug_names = []

    for entry in benchmark["per_drug"]:
        if "error" in entry:
            continue

        smiles = entry.get("smiles", "")
        if not smiles:
            continue

        pred_cmax = entry.get("pred_cmax", 0)
        obs_cmax = entry.get("obs_cmax")

        if obs_cmax is None or obs_cmax <= 0 or pred_cmax <= 0:
            continue

        feats = extract_features(entry, pipeline, smiles)
        if feats is None:
            continue

        log_res = np.log(pred_cmax / obs_cmax)
        features_list.append(feats)
        cmax_residuals.append(log_res)
        drug_names.append(entry["drug"])

        # Track AUC residuals where available
        pred_auc = entry.get("pred_auc", 0)
        obs_auc = entry.get("obs_auc")
        if obs_auc and obs_auc > 0 and pred_auc > 0:
            auc_residuals_raw.append(np.log(pred_auc / obs_auc))
        else:
            auc_residuals_raw.append(None)

    if len(features_list) < 10:
        print(f"ERROR: Only {len(features_list)} drugs with Cmax data. Need ≥10.")
        sys.exit(1)

    X = np.array(features_list)
    y_cmax = np.array(cmax_residuals)

    print(f"\nTraining data: {len(drug_names)} drugs with Cmax")
    print(f"Feature matrix: {X.shape}")
    print(f"Log-residual Cmax: mean={y_cmax.mean():.3f}, std={y_cmax.std():.3f}")

    # Train Cmax correction
    cmax_model = ResidualCorrectionModel(alpha=1.0)
    cmax_model.fit(X, y_cmax)
    loo_cmax = cmax_model.leave_one_out_cv(X, y_cmax)

    uncorrected_aafe = float(np.exp(np.mean(np.abs(y_cmax))))
    loo_aafe = float(np.exp(np.mean(np.abs(loo_cmax))))
    print(f"\nCmax AAFE: uncorrected={uncorrected_aafe:.2f}, LOO-corrected={loo_aafe:.2f}")
    if loo_aafe < uncorrected_aafe:
        print(f"  → Correction IMPROVES AAFE by {(1 - loo_aafe / uncorrected_aafe) * 100:.1f}%")
    else:
        print("  → Correction does NOT improve AAFE (overfitting risk)")

    # Feature importance
    print("\nFeature importance (|coefficient| on standardized features):")
    for name, coef in sorted(zip(FEATURE_NAMES, cmax_model.coef_), key=lambda x: -abs(x[1])):
        print(f"  {name:20s}: {coef:+.4f}")

    # Save
    out_dir = repo_root / "models" / "correction"
    cmax_model.save(out_dir / "ridge_cmax.json")
    print(f"\nSaved: {out_dir / 'ridge_cmax.json'}")

    # Train AUC correction if enough data
    auc_valid = [
        (X[i], auc_residuals_raw[i])
        for i in range(len(auc_residuals_raw))
        if auc_residuals_raw[i] is not None
    ]
    if len(auc_valid) >= 10:
        X_auc = np.array([x[0] for x in auc_valid])
        y_auc = np.array([x[1] for x in auc_valid])
        auc_model = ResidualCorrectionModel(alpha=1.0)
        auc_model.fit(X_auc, y_auc)
        loo_auc = auc_model.leave_one_out_cv(X_auc, y_auc)
        uncorr_auc = float(np.exp(np.mean(np.abs(y_auc))))
        loo_auc_aafe = float(np.exp(np.mean(np.abs(loo_auc))))
        print(f"\nAUC AAFE: uncorrected={uncorr_auc:.2f}, LOO-corrected={loo_auc_aafe:.2f}")
        auc_model.save(out_dir / "ridge_auc.json")
        print(f"Saved: {out_dir / 'ridge_auc.json'}")
    else:
        print(f"\nSkipping AUC model: only {len(auc_valid)} drugs (need ≥10)")


if __name__ == "__main__":
    main()
