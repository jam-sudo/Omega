#!/usr/bin/env python3
"""Train the Pre-ODE ADME Corrector.

Learns multiplicative corrections (delta_log_fup, delta_log_clint, delta_log_peff)
to predicted ADME parameters BEFORE they enter the ODE solver. The corrections
are optimized to minimize end-to-end Cmax prediction error.

Training approach (Sobol-weighted allocation):
1. For each training drug: run pipeline -> get base Cmax_pred
2. Compute log_error = log10(obs_cmax / pred_cmax)
3. Allocate error correction across ADME params using Sobol sensitivity weights:
   - fup: 40% (Sobol ST ~ 0.418)
   - CLint_gut: 47% (Sobol ST ~ 0.470)
   - peff: 13% (remaining)
4. Clamp targets to [-0.5, 0.5] log units
5. Train 3 separate XGBoost models: features -> delta_i

Feature vector (50 dims):
- 41 Morgan FP bits (top-variance from training set, radius=2, 2048 bits)
- 9 physicochemical descriptors (normalized)

Usage:
    python scripts/train_pre_ode_corrector.py
"""

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
MODEL_DIR = repo_root / "models" / "corrections" / "pre_ode"

# Feature extraction constants
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
N_FP_SELECTED = 41
N_DESCRIPTORS = 9

# Sobol sensitivity weights for error allocation
# From Sobol analysis: gut CLint ST=0.470, fup ST=0.418, peff = remainder
WEIGHT_FUP = 0.40
WEIGHT_CLINT = 0.47
WEIGHT_PEFF = 0.13

# Correction clamp
MAX_CORRECTION = 0.5

# XGBoost hyperparameters (conservative to prevent overfitting)
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

TARGET_NAMES = ["delta_log_fup", "delta_log_clint", "delta_log_peff"]
SOBOL_WEIGHTS = [WEIGHT_FUP, WEIGHT_CLINT, WEIGHT_PEFF]


def extract_features(smiles: str) -> np.ndarray | None:
    """Extract 2048-bit Morgan FP + 9 descriptors from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprint (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
    fp_arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
    for idx in fp.GetOnBits():
        fp_arr[idx] = 1.0

    # 9 physicochemical descriptors (normalized)
    descs = np.array(
        [
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol) / 200.0,
            Descriptors.MolWt(mol) / 600.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumHDonors(mol) / 5.0,
            Descriptors.NumRotatableBonds(mol) / 15.0,
            Descriptors.RingCount(mol) / 5.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol) / 150.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([fp_arr, descs])


def select_top_fp_indices(fp_matrix: np.ndarray, n_select: int = N_FP_SELECTED) -> np.ndarray:
    """Select top-variance fingerprint bit indices from training data.

    Args:
        fp_matrix: (n_drugs, 2048) fingerprint matrix.
        n_select: Number of bits to select.

    Returns:
        Array of selected bit indices, sorted.
    """
    variances = np.var(fp_matrix, axis=0)
    top_indices = np.argsort(variances)[-n_select:]
    return np.sort(top_indices)


def main():
    import csv

    import xgboost as xgb

    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    logger.info("=" * 70)
    logger.info("Training Pre-ODE ADME Corrector")
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

    # 3. Extract features for all drugs
    logger.info("Extracting molecular features...")
    features_full = []  # (n_drugs, 2048 + 9)
    valid_drugs = []
    for d in drugs:
        feat = extract_features(d["smiles"])
        if feat is not None:
            features_full.append(feat)
            valid_drugs.append(d)
        else:
            logger.warning("Invalid SMILES for %s: %s", d["drug"], d["smiles"])

    features_full = np.array(features_full)
    logger.info("Valid features: %d drugs, shape %s", len(valid_drugs), features_full.shape)

    # 4. Select top-variance FP bits
    fp_matrix = features_full[:, :FINGERPRINT_BITS]
    desc_matrix = features_full[:, FINGERPRINT_BITS:]
    top_fp_indices = select_top_fp_indices(fp_matrix, N_FP_SELECTED)
    logger.info("Selected %d FP bits with highest variance", len(top_fp_indices))

    # Build final feature matrix (41 FP + 9 desc = 50)
    X_train = np.concatenate([fp_matrix[:, top_fp_indices], desc_matrix], axis=1)
    logger.info("Final feature matrix: %s", X_train.shape)

    # 5. Run pipeline to get base Cmax predictions
    logger.info("Running pipeline for %d drugs...", len(valid_drugs))
    pipeline = OmegaPipeline()

    cmax_preds = []
    failed_indices = []
    t0 = time.perf_counter()

    for i, d in enumerate(valid_drugs):
        try:
            result = pipeline.simulate(
                SimulationRequest(
                    smiles=d["smiles"],
                    dose_mg=d["dose_mg"],
                    route="oral",
                    duration_h=24.0,
                )
            )
            cmax_preds.append(result.cmax_mg_L)
        except Exception as exc:
            logger.warning("Pipeline failed for %s: %s", d["drug"], exc)
            cmax_preds.append(None)
            failed_indices.append(i)

    elapsed = time.perf_counter() - t0
    n_success = sum(1 for c in cmax_preds if c is not None and c > 0)
    logger.info(
        "Pipeline complete: %d/%d succeeded in %.1fs (%.0fms/drug)",
        n_success,
        len(valid_drugs),
        elapsed,
        elapsed / len(valid_drugs) * 1000,
    )

    # 6. Compute correction targets
    logger.info("Computing correction targets (Sobol-weighted allocation)...")
    y_targets = []  # (n_drugs, 3) — delta_log_fup, delta_log_clint, delta_log_peff
    valid_mask = []

    for i, d in enumerate(valid_drugs):
        cmax_pred = cmax_preds[i]
        cmax_obs = d["cmax_obs"]

        if cmax_pred is None or cmax_pred <= 0 or cmax_obs <= 0:
            y_targets.append([0.0, 0.0, 0.0])
            valid_mask.append(False)
            continue

        # log10(obs / pred): positive means we need to increase Cmax
        log_error = np.log10(cmax_obs / cmax_pred)

        # Allocate across ADME params using Sobol weights
        # fup correction: increasing fup increases Cmax (more free drug in plasma)
        # clint correction: increasing CLint decreases Cmax (more clearance)
        #   => delta_log_clint should be NEGATIVE when log_error is positive
        # peff correction: increasing peff increases Cmax (more absorption)
        delta_fup = np.clip(WEIGHT_FUP * log_error, -MAX_CORRECTION, MAX_CORRECTION)
        delta_clint = np.clip(
            -WEIGHT_CLINT * log_error, -MAX_CORRECTION, MAX_CORRECTION
        )  # negative sign: CLint and Cmax are inversely related
        delta_peff = np.clip(WEIGHT_PEFF * log_error, -MAX_CORRECTION, MAX_CORRECTION)

        y_targets.append([delta_fup, delta_clint, delta_peff])
        valid_mask.append(True)

    y_targets = np.array(y_targets)
    valid_mask = np.array(valid_mask)

    n_valid = valid_mask.sum()
    logger.info("Valid training samples: %d", n_valid)

    if n_valid < 10:
        logger.error("Too few valid training samples (%d). Aborting.", n_valid)
        sys.exit(1)

    # Filter to valid samples only
    X_valid = X_train[valid_mask]
    y_valid = y_targets[valid_mask]

    # Print target statistics
    for i, name in enumerate(TARGET_NAMES):
        vals = y_valid[:, i]
        logger.info(
            "  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            name,
            vals.mean(),
            vals.std(),
            vals.min(),
            vals.max(),
        )

    # 7. Train 3 separate XGBoost models
    logger.info("Training XGBoost models...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = []
    train_metrics = {}

    for i, name in enumerate(TARGET_NAMES):
        logger.info("  Training %s...", name)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_valid, y_valid[:, i])

        # Train RMSE
        preds = model.predict(X_valid)
        rmse = np.sqrt(np.mean((preds - y_valid[:, i]) ** 2))
        mae = np.mean(np.abs(preds - y_valid[:, i]))
        logger.info("    Train RMSE=%.4f, MAE=%.4f", rmse, mae)

        # Save individual model
        model_path = MODEL_DIR / f"pre_ode_xgb_{name}.json"
        model.save_model(str(model_path))
        logger.info("    Saved to %s", model_path)

        models.append(model)
        train_metrics[name] = {"rmse": float(rmse), "mae": float(mae)}

    # 8. Save metadata and FP indices
    np.save(MODEL_DIR / "fp_indices.npy", top_fp_indices)
    logger.info("Saved FP indices to %s", MODEL_DIR / "fp_indices.npy")

    meta = {
        "n_train_drugs": int(n_valid),
        "n_features": int(X_valid.shape[1]),
        "n_fp_bits": int(N_FP_SELECTED),
        "n_descriptors": int(N_DESCRIPTORS),
        "sobol_weights": {
            "fup": WEIGHT_FUP,
            "clint": WEIGHT_CLINT,
            "peff": WEIGHT_PEFF,
        },
        "max_correction": MAX_CORRECTION,
        "xgb_params": XGB_PARAMS,
        "target_names": TARGET_NAMES,
        "train_metrics": train_metrics,
    }
    with open(MODEL_DIR / "pre_ode_xgb.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", MODEL_DIR / "pre_ode_xgb.json")

    # 9. Validation: predict on training set and compute corrected Cmax
    logger.info("\n" + "=" * 70)
    logger.info("Training set validation (corrected vs uncorrected Cmax)")
    logger.info("=" * 70)

    fe_uncorrected = []
    fe_corrected = []
    j = 0
    for i, (d, mask) in enumerate(zip(valid_drugs, valid_mask)):
        if not mask:
            continue

        cmax_pred = cmax_preds[i]
        cmax_obs = d["cmax_obs"]
        fe_unc = max(cmax_pred / cmax_obs, cmax_obs / cmax_pred)

        # Compute corrected Cmax (multiplicative)
        delta_fup = float(models[0].predict(X_valid[j : j + 1])[0])
        delta_clint = float(models[1].predict(X_valid[j : j + 1])[0])
        delta_peff = float(models[2].predict(X_valid[j : j + 1])[0])

        # Approximate Cmax correction factor:
        # Cmax ~ fup^a * CLint^b * peff^c where a,b,c are from Sobol
        # In log space: log(Cmax) += delta_fup - delta_clint + delta_peff
        # (CLint inversely related to Cmax => subtract)
        log_correction = delta_fup - delta_clint + delta_peff
        cmax_corrected = cmax_pred * 10**log_correction

        fe_corr = max(cmax_corrected / cmax_obs, cmax_obs / cmax_corrected)

        fe_uncorrected.append(fe_unc)
        fe_corrected.append(fe_corr)

        if fe_unc > 2.0 or fe_corr > 2.0:
            logger.info(
                "  %-20s: FE %.2fx -> %.2fx  (delta_fup=%.3f, delta_clint=%.3f, delta_peff=%.3f)",
                d["drug"],
                fe_unc,
                fe_corr,
                delta_fup,
                delta_clint,
                delta_peff,
            )
        j += 1

    # Compute AAFEs
    aafe_unc = float(10 ** np.mean(np.log10(fe_uncorrected)))
    aafe_corr = float(10 ** np.mean(np.log10(fe_corrected)))
    pct2_unc = 100.0 * sum(1 for fe in fe_uncorrected if fe <= 2.0) / len(fe_uncorrected)
    pct2_corr = 100.0 * sum(1 for fe in fe_corrected if fe <= 2.0) / len(fe_corrected)

    logger.info("")
    logger.info(
        "UNCORRECTED: AAFE=%.3f, %%2-fold=%.0f%% (N=%d)", aafe_unc, pct2_unc, len(fe_uncorrected)
    )
    logger.info(
        "CORRECTED:   AAFE=%.3f, %%2-fold=%.0f%% (N=%d)", aafe_corr, pct2_corr, len(fe_corrected)
    )
    logger.info("Delta AAFE: %+.3f", aafe_corr - aafe_unc)
    logger.info("")
    logger.info("NOTE: These are in-sample (train set) metrics. External validation required.")

    logger.info("\nDone. Models saved to %s", MODEL_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
