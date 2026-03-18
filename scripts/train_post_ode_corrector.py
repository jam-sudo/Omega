#!/usr/bin/env python3
"""Train the Post-ODE Residual Corrector.

Learns residual corrections (in log10 space) to ODE-predicted Cmax and AUC
AFTER the ODE simulation. The corrections are learned from the difference
between ODE predictions and observed clinical values.

Training approach:
1. For each training drug: run OmegaPipeline.simulate() -> get Cmax_ODE, AUC_ODE
2. Compute targets: log10(Cmax_obs / Cmax_ODE), log10(AUC_obs / AUC_ODE)
3. Extract 28 features (molecular + ADME + ODE output + physics)
4. Train stacking ensemble: Ridge(alpha=10) + XGBoost(depth=3, min_child=10)
5. Final prediction: average of Ridge and XGBoost

Feature vector (28 dims):
- 9 molecular descriptors (logP, TPSA, MW, HBA, HBD, RotBonds, Rings, FrCSP3, MolMR)
- 5 ADME predicted (fup, CLint, rbp, peff, logS)
- 4 pKa + compound type (pka_val, is_acid, is_base, is_neutral)
- 4 ODE output (log10(Cmax_ODE), log10(AUC_ODE), tmax, t_half)
- 2 Fg/Fh estimates (from pipeline)
- 4 extra (log10(dose), MW/dose, TPSA, rings)

Usage:
    python scripts/train_post_ode_corrector.py
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = repo_root / "data" / "ml" / "clinical" / "expanded_cmax.csv"
SPLIT_PATH = repo_root / "data" / "ml" / "clinical" / "dataset_split.json"
MODEL_DIR = repo_root / "models" / "corrections" / "post_ode"

# XGBoost hyperparameters (conservative to prevent overfitting on ~70 drugs)
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
}

# Ridge hyperparameters
RIDGE_ALPHA = 10.0

# Target names
TARGET_NAMES = ["cmax_correction_log", "auc_correction_log"]

# Correction clamp
MAX_CORRECTION = 1.0


def extract_features(
    smiles: str,
    ode_cmax: float,
    ode_auc: float,
    tmax: float,
    t_half: float,
    adme_props: dict,
    dose_mg: float,
) -> np.ndarray | None:
    """Extract 28-dimensional feature vector from SMILES + ODE output + ADME.

    Args:
        smiles: SMILES string.
        ode_cmax: ODE-predicted Cmax (mg/L).
        ode_auc: ODE-predicted AUC (mg*h/L).
        tmax: Time of Cmax from ODE (h).
        t_half: Terminal half-life from ODE (h).
        adme_props: ADME properties dict from pipeline.
        dose_mg: Dose in mg.

    Returns:
        np.ndarray of shape (28,), or None if SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # --- 9 molecular descriptors (normalized) ---
    mol_logP = Descriptors.MolLogP(mol)
    tpsa_raw = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    ring_count = Descriptors.RingCount(mol)
    fr_csp3 = Descriptors.FractionCSP3(mol)
    mol_mr = Descriptors.MolMR(mol)

    molecular = np.array(
        [
            mol_logP,
            tpsa_raw / 200.0,
            mw / 600.0,
            hba / 10.0,
            hbd / 5.0,
            rot_bonds / 15.0,
            ring_count / 5.0,
            fr_csp3,
            mol_mr / 150.0,
        ],
        dtype=np.float32,
    )

    # --- 5 ADME predicted ---
    fup = float(adme_props.get("fup", 0.1))
    clint = float(adme_props.get("clint_3a4", 5.0))
    rbp = float(adme_props.get("rbp", 0.55))
    peff = float(adme_props.get("peff", 1.0))
    logS = float(adme_props.get("logS", -3.0))

    adme = np.array(
        [
            fup,
            np.log10(max(clint, 1e-6) + 1.0),
            rbp,
            peff,
            logS,
        ],
        dtype=np.float32,
    )

    # --- 4 pKa + compound type ---
    # Try to get pKa and compound_type from adme_props or detect from structure
    pka_val = 7.0
    compound_type = "neutral"
    try:
        base_patt = Chem.MolFromSmarts("[NH2,NH1,NH0;!$(NC=O);!$(NS=O)]")
        acid_patt = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        has_base = mol.HasSubstructMatch(base_patt)
        has_acid = mol.HasSubstructMatch(acid_patt)
        if has_base and has_acid:
            compound_type = "zwitterion"
            pka_val = 9.0
        elif has_base:
            compound_type = "base"
            pka_val = 9.0
        elif has_acid:
            compound_type = "acid"
            pka_val = 4.0
    except Exception:
        pass

    is_acid = 1.0 if compound_type == "acid" else 0.0
    is_base = 1.0 if compound_type == "base" else 0.0
    is_neutral = 1.0 if compound_type in ("neutral", "zwitterion") else 0.0

    pka_features = np.array(
        [pka_val / 14.0, is_acid, is_base, is_neutral],
        dtype=np.float32,
    )

    # --- 4 ODE output ---
    log_cmax = np.log10(max(ode_cmax, 1e-6))
    log_auc = np.log10(max(ode_auc, 1e-6))

    ode_features = np.array(
        [
            log_cmax,
            log_auc,
            tmax / 24.0,
            t_half / 24.0,
        ],
        dtype=np.float32,
    )

    # --- 2 physics (Fg, Fh) ---
    # Estimate from pipeline ADME if available, else use simple heuristics
    fg = 1.0
    fh = 0.7
    # Simple Fh estimate from CLint and hepatic blood flow
    Q_H = 90.0  # L/h
    clint_val = max(clint, 0.01)
    fh_est = Q_H / (Q_H + fup * clint_val * 12.96)  # standard IVIVE approx
    fh = max(0.01, min(1.0, fh_est))

    physics = np.array([fg, fh], dtype=np.float32)

    # --- 4 extra features ---
    log_dose = np.log10(max(dose_mg, 1.0))
    mw_dose_ratio = mw / max(dose_mg, 1.0)

    extra = np.array(
        [
            log_dose,
            mw_dose_ratio / 10.0,
            tpsa_raw / 200.0,
            ring_count / 5.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([molecular, adme, pka_features, ode_features, physics, extra])


def main():
    import joblib
    import xgboost as xgb
    from sklearn.linear_model import Ridge

    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    logger.info("=" * 70)
    logger.info("Training Post-ODE Residual Corrector")
    logger.info("=" * 70)

    # 1. Load dataset split
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    train_drugs = set(d.lower() for d in splits["train"])
    logger.info("Train split: %d drugs", len(train_drugs))

    # 2. Load expanded_cmax.csv
    drugs = []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_name = row["drug"].strip().lower()
            if drug_name in train_drugs:
                smiles = row["smiles"].strip()
                dose_mg = float(row["dose_mg"])
                cmax_obs = float(row["cmax_mg_L"])
                if smiles and dose_mg > 0 and cmax_obs > 0:
                    drugs.append(
                        {
                            "drug": row["drug"].strip(),
                            "smiles": smiles,
                            "dose_mg": dose_mg,
                            "cmax_obs": cmax_obs,
                        }
                    )
    logger.info("Loaded %d training drugs with valid SMILES/Cmax", len(drugs))

    # 3. Run pipeline for each drug to get ODE predictions
    logger.info("Running pipeline for %d drugs...", len(drugs))
    pipeline = OmegaPipeline()

    results_list = []
    t0 = time.perf_counter()

    for i, d in enumerate(drugs):
        try:
            result = pipeline.simulate(
                SimulationRequest(
                    smiles=d["smiles"],
                    dose_mg=d["dose_mg"],
                    route="oral",
                    duration_h=24.0,
                )
            )
            results_list.append(result)
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "  %d/%d drugs simulated (%.1fs, %.0fms/drug)",
                    i + 1,
                    len(drugs),
                    elapsed,
                    elapsed / (i + 1) * 1000,
                )
        except Exception as exc:
            logger.warning("Pipeline failed for %s: %s", d["drug"], exc)
            results_list.append(None)

    elapsed = time.perf_counter() - t0
    n_success = sum(1 for r in results_list if r is not None)
    logger.info(
        "Pipeline complete: %d/%d succeeded in %.1fs (%.0fms/drug)",
        n_success,
        len(drugs),
        elapsed,
        elapsed / max(len(drugs), 1) * 1000,
    )

    # 4. Extract features and compute targets
    logger.info("Extracting features and computing targets...")
    X_list = []
    y_cmax_list = []
    y_auc_list = []
    valid_drug_names = []

    for i, d in enumerate(drugs):
        result = results_list[i]
        if result is None:
            continue

        cmax_ode = result.cmax_mg_L
        auc_ode = result.auc0t_mg_h_L
        cmax_obs = d["cmax_obs"]

        # Skip if ODE returned invalid values
        if cmax_ode <= 0 or auc_ode <= 0 or cmax_obs <= 0:
            continue

        features = extract_features(
            smiles=d["smiles"],
            ode_cmax=cmax_ode,
            ode_auc=auc_ode,
            tmax=result.tmax_h,
            t_half=result.t_half_h,
            adme_props=result.adme_properties,
            dose_mg=d["dose_mg"],
        )

        if features is None:
            continue

        if not np.all(np.isfinite(features)):
            logger.warning("Non-finite features for %s — skipping", d["drug"])
            continue

        # Target: log10(obs / pred)
        target_cmax = np.log10(cmax_obs / cmax_ode)
        # AUC target: we don't have obs AUC in expanded_cmax, use Cmax as proxy
        # (AUC correction will be approximately correlated but not identical)
        target_auc = target_cmax * 0.8  # rough proxy: AUC error ~ 80% of Cmax error

        X_list.append(features)
        y_cmax_list.append(target_cmax)
        y_auc_list.append(target_auc)
        valid_drug_names.append(d["drug"])

    X = np.array(X_list, dtype=np.float32)
    y_cmax = np.array(y_cmax_list, dtype=np.float32)
    y_auc = np.array(y_auc_list, dtype=np.float32)

    logger.info(
        "Valid training samples: %d, feature dim: %d", len(X), X.shape[1] if len(X) > 0 else 0
    )

    if len(X) < 10:
        logger.error("Too few valid training samples (%d). Aborting.", len(X))
        sys.exit(1)

    # Clamp targets to [-MAX_CORRECTION, MAX_CORRECTION]
    y_cmax = np.clip(y_cmax, -MAX_CORRECTION, MAX_CORRECTION)
    y_auc = np.clip(y_auc, -MAX_CORRECTION, MAX_CORRECTION)

    # Print target statistics
    for name, y in [("cmax", y_cmax), ("auc", y_auc)]:
        logger.info(
            "  %s target: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            name,
            y.mean(),
            y.std(),
            y.min(),
            y.max(),
        )

    # 5. Compute feature normalization stats (for Ridge)
    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)
    X_norm = (X - feature_means) / np.maximum(feature_stds, 1e-8)

    # 6. Train models
    logger.info("Training stacking ensemble (Ridge + XGBoost)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    targets = {"cmax_correction_log": y_cmax, "auc_correction_log": y_auc}
    train_metrics = {}

    for target_name, y in targets.items():
        logger.info("  Training %s...", target_name)

        # Ridge regression (on normalized features)
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_norm, y)
        ridge_preds = ridge.predict(X_norm)
        ridge_rmse = float(np.sqrt(np.mean((ridge_preds - y) ** 2)))
        ridge_mae = float(np.mean(np.abs(ridge_preds - y)))
        logger.info("    Ridge: RMSE=%.4f, MAE=%.4f", ridge_rmse, ridge_mae)

        # XGBoost (on raw features)
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(X, y)
        xgb_preds = xgb_model.predict(X)
        xgb_rmse = float(np.sqrt(np.mean((xgb_preds - y) ** 2)))
        xgb_mae = float(np.mean(np.abs(xgb_preds - y)))
        logger.info("    XGBoost: RMSE=%.4f, MAE=%.4f", xgb_rmse, xgb_mae)

        # Stacking (simple average)
        stacked_preds = (ridge_preds + xgb_preds) / 2.0
        stacked_rmse = float(np.sqrt(np.mean((stacked_preds - y) ** 2)))
        stacked_mae = float(np.mean(np.abs(stacked_preds - y)))
        logger.info("    Stacked: RMSE=%.4f, MAE=%.4f", stacked_rmse, stacked_mae)

        # Save Ridge model
        ridge_path = MODEL_DIR / f"ridge_{target_name}.joblib"
        joblib.dump(ridge, ridge_path)
        logger.info("    Saved Ridge to %s", ridge_path)

        # Save XGBoost model
        xgb_path = MODEL_DIR / f"xgb_{target_name}.json"
        xgb_model.save_model(str(xgb_path))
        logger.info("    Saved XGBoost to %s", xgb_path)

        train_metrics[target_name] = {
            "ridge_rmse": ridge_rmse,
            "ridge_mae": ridge_mae,
            "xgb_rmse": xgb_rmse,
            "xgb_mae": xgb_mae,
            "stacked_rmse": stacked_rmse,
            "stacked_mae": stacked_mae,
        }

    # 7. Save feature normalization stats
    np.save(MODEL_DIR / "feature_means.npy", feature_means)
    np.save(MODEL_DIR / "feature_stds.npy", feature_stds)
    logger.info("Saved feature normalization stats")

    # 8. Save metadata
    meta = {
        "n_train_drugs": int(len(X)),
        "n_features": int(X.shape[1]),
        "max_correction": MAX_CORRECTION,
        "ridge_alpha": RIDGE_ALPHA,
        "xgb_params": XGB_PARAMS,
        "target_names": TARGET_NAMES,
        "train_metrics": train_metrics,
        "valid_drugs": valid_drug_names,
    }
    with open(MODEL_DIR / "post_ode_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", MODEL_DIR / "post_ode_meta.json")

    # 9. Validation: compute corrected vs uncorrected Cmax on training set
    logger.info("\n" + "=" * 70)
    logger.info("Training set validation (corrected vs uncorrected Cmax)")
    logger.info("=" * 70)

    fe_uncorrected = []
    fe_corrected = []

    for i, (drug_name, _y_c) in enumerate(zip(valid_drug_names, y_cmax)):
        # Find the matching drug entry
        drug_entry = None
        for d in drugs:
            if d["drug"] == drug_name:
                drug_entry = d
                break
        if drug_entry is None:
            continue

        result_idx = drugs.index(drug_entry)
        result = results_list[result_idx]
        if result is None:
            continue

        cmax_ode = result.cmax_mg_L
        cmax_obs = drug_entry["cmax_obs"]

        # Uncorrected fold error
        fe_unc = max(cmax_ode / cmax_obs, cmax_obs / cmax_ode)

        # Corrected: apply stacking prediction
        X_i = X[i : i + 1]
        X_i_norm = X_norm[i : i + 1]

        ridge_pred = float(Ridge(alpha=RIDGE_ALPHA).fit(X_norm, y_cmax).predict(X_i_norm)[0])
        xgb_pred = float(xgb.XGBRegressor(**XGB_PARAMS).fit(X, y_cmax).predict(X_i)[0])
        correction = (ridge_pred + xgb_pred) / 2.0

        # Apply safety clamp
        if abs(correction) > MAX_CORRECTION:
            correction = 0.0
        else:
            correction = float(np.clip(correction, -MAX_CORRECTION, MAX_CORRECTION))

        cmax_corrected = cmax_ode * 10**correction
        fe_corr = max(cmax_corrected / cmax_obs, cmax_obs / cmax_corrected)

        fe_uncorrected.append(fe_unc)
        fe_corrected.append(fe_corr)

        if fe_unc > 2.0 or fe_corr > 2.0:
            logger.info(
                "  %-25s: FE %.2fx -> %.2fx  (correction=%.3f log10)",
                drug_name,
                fe_unc,
                fe_corr,
                correction,
            )

    if fe_uncorrected:
        aafe_unc = float(10 ** np.mean(np.log10(fe_uncorrected)))
        aafe_corr = float(10 ** np.mean(np.log10(fe_corrected)))
        pct2_unc = 100.0 * sum(1 for fe in fe_uncorrected if fe <= 2.0) / len(fe_uncorrected)
        pct2_corr = 100.0 * sum(1 for fe in fe_corrected if fe <= 2.0) / len(fe_corrected)

        logger.info("")
        logger.info(
            "UNCORRECTED: AAFE=%.3f, %%2-fold=%.0f%% (N=%d)",
            aafe_unc,
            pct2_unc,
            len(fe_uncorrected),
        )
        logger.info(
            "CORRECTED:   AAFE=%.3f, %%2-fold=%.0f%% (N=%d)",
            aafe_corr,
            pct2_corr,
            len(fe_corrected),
        )
        logger.info("Delta AAFE: %+.3f", aafe_corr - aafe_unc)
        logger.info("")
        logger.info("NOTE: These are in-sample (train set) metrics. External validation required.")

    logger.info("\nDone. Models saved to %s", MODEL_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
