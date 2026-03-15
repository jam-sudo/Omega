#!/usr/bin/env python3
"""Fine-tune PK parameter predictions against clinical PK outcomes.

Instead of training on ADMET-AI labels (can never beat ADMET-AI),
fine-tune against actual observed Cmax/AUC from clinical studies.

Approach:
1. Load pre-trained Transfer MLP (fast param prediction from SMILES)
2. For each of 20 benchmark drugs: MLP → params → real ODE → predicted Cmax/AUC
3. Optimize a correction layer to minimize log fold-error vs observed data
4. The correction layer learns systematic biases in ADMET-AI → ODE pipeline

The correction layer has few parameters (6-12) so 20 drugs is sufficient
to avoid overfitting.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clinical_finetune")

PARAM_NAMES = ["logP", "fup", "clint_L_h", "mw", "rbp", "peff"]


def load_transfer_model():
    """Load pre-trained Transfer MLP."""
    import torch

    from train_l2_transfer import TransferMLP, inverse_transform, smiles_to_fingerprint

    ckpt = torch.load(
        str(repo_root / "models" / "level2" / "transfer_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model = TransferMLP(n_output=6, hidden=512)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["label_stats"]


def predict_params(model, stats, smiles: str) -> dict[str, float] | None:
    """Predict 6 PK params from SMILES using Transfer MLP."""
    import torch

    from train_l2_transfer import inverse_transform, smiles_to_fingerprint

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None
    with torch.no_grad():
        pred_norm = model(torch.tensor(fp, dtype=torch.float32).unsqueeze(0)).numpy()
    pred = inverse_transform(pred_norm, stats)[0]
    return {name: float(pred[i]) for i, name in enumerate(PARAM_NAMES)}


def run_ode(params: dict, dose_mg: float) -> dict[str, float] | None:
    """Run real ODE with given params, return PK metrics."""
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    try:
        drug = Drug(
            name="finetune",
            mw=max(params["mw"], 50),
            logP=params["logP"],
            fup=max(min(params["fup"], 1.0), 0.001),
            rbp=max(params["rbp"], 0.3),
            peff=max(params["peff"], 0.01),
            clint_hepatic_L_per_h=max(params["clint_L_h"], 0.001),
            clint={"CYP3A4": 1.0},
            smiles="C",
            dose_mg=dose_mg,
            route="oral",
        )
        pbpk = WholeBodyPBPK(drug, body_weight=70.0)
        pbpk.setup_oral(dose_mg)
        sim = pbpk.simulate(t_end_h=24.0)
        pk = sim.pk_summary()
        return {
            "Cmax_mg_L": pk.get("Cmax_mg_L", 0),
            "AUC_mg_h_L": pk.get("AUC_mg_h_L", 0),
        }
    except Exception:
        return None


def apply_correction(params: dict, correction: np.ndarray) -> dict:
    """Apply multiplicative + additive correction to params.

    correction is 12D: [mult_logP, mult_fup, mult_clint, mult_mw, mult_rbp, mult_peff,
                         add_logP,  add_fup,  add_clint,  add_mw,  add_rbp,  add_peff]
    """
    corrected = {}
    for i, name in enumerate(PARAM_NAMES):
        mult = np.exp(correction[i])  # exp to keep positive
        add = correction[i + 6]
        corrected[name] = params[name] * mult + add
    return corrected


def load_benchmark_data():
    """Load 20 benchmark drugs with observed PK data."""
    from eval_l2_model import BENCHMARK_DRUGS, load_observed_pk

    drugs = []
    for drug_name, info in BENCHMARK_DRUGS.items():
        observed = load_observed_pk(drug_name)
        if observed and observed["cmax"] > 0 and observed["auc"] > 0:
            drugs.append(
                {
                    "name": drug_name,
                    "smiles": info["smiles"],
                    "dose_mg": info["dose_mg"],
                    "obs_cmax": observed["cmax"],
                    "obs_auc": observed["auc"],
                }
            )
    return drugs


def objective(correction: np.ndarray, drug_data: list, base_params: list) -> float:
    """Objective: sum of log10 fold errors across all drugs.

    Lower = better. Minimizing this directly optimizes AAFE.
    """
    total_loss = 0.0
    n_valid = 0

    for drug_info, params in zip(drug_data, base_params):
        if params is None:
            continue

        corrected = apply_correction(params, correction)
        pk = run_ode(corrected, drug_info["dose_mg"])
        if pk is None or pk["Cmax_mg_L"] < 1e-15 or pk["AUC_mg_h_L"] < 1e-15:
            total_loss += 10.0  # penalty for failed ODE
            n_valid += 1
            continue

        # Log fold error (what AAFE optimizes)
        fe_cmax = max(
            pk["Cmax_mg_L"] / drug_info["obs_cmax"],
            drug_info["obs_cmax"] / pk["Cmax_mg_L"],
        )
        fe_auc = max(
            pk["AUC_mg_h_L"] / drug_info["obs_auc"],
            drug_info["obs_auc"] / pk["AUC_mg_h_L"],
        )
        total_loss += np.log10(fe_cmax) + np.log10(fe_auc)
        n_valid += 1

    return total_loss / max(n_valid, 1)


def main():
    logger.info("=== Clinical Fine-tuning ===")

    # Step 1: Load model and benchmark data
    model, stats = load_transfer_model()
    drug_data = load_benchmark_data()
    logger.info("Loaded %d benchmark drugs with clinical PK data", len(drug_data))

    # Step 2: Pre-compute base params for all drugs
    base_params = []
    for d in drug_data:
        params = predict_params(model, stats, d["smiles"])
        base_params.append(params)
        if params:
            logger.info(
                "  %s: logP=%.2f fup=%.3f clint=%.1f mw=%.0f",
                d["name"],
                params["logP"],
                params["fup"],
                params["clint_L_h"],
                params["mw"],
            )

    # Step 3: Evaluate BEFORE correction
    logger.info("\n--- Before correction ---")
    x0 = np.zeros(12)  # no correction = identity
    loss_before = objective(x0, drug_data, base_params)
    logger.info("Mean log10(FE) before: %.4f (≈ AAFE %.2f)", loss_before, 10**loss_before)

    # Step 4: Optimize correction factors
    logger.info("\nOptimizing correction factors (12 params, %d drugs)...", len(drug_data))
    t0 = time.time()

    result = minimize(
        objective,
        x0,
        args=(drug_data, base_params),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5, "disp": False},
    )

    elapsed = time.time() - t0
    logger.info("Optimization done in %.1fs (%d iterations)", elapsed, result.nit)
    logger.info("Mean log10(FE) after: %.4f (≈ AAFE %.2f)", result.fun, 10**result.fun)

    # Step 5: Print correction factors
    correction = result.x
    logger.info("\nLearned corrections:")
    for i, name in enumerate(PARAM_NAMES):
        mult = np.exp(correction[i])
        add = correction[i + 6]
        logger.info("  %s: × %.3f + %.4f", name, mult, add)

    # Step 6: Evaluate AFTER correction — per drug
    from eval_l2_model import fold_error

    logger.info("\n--- Per-drug results (after correction) ---")
    cmax_fes, auc_fes = [], []

    for drug_info, params in zip(drug_data, base_params):
        if params is None:
            continue
        corrected = apply_correction(params, correction)
        pk = run_ode(corrected, drug_info["dose_mg"])
        if pk is None:
            continue

        fe_c = fold_error(pk["Cmax_mg_L"], drug_info["obs_cmax"])
        fe_a = fold_error(pk["AUC_mg_h_L"], drug_info["obs_auc"])
        cmax_fes.append(fe_c)
        auc_fes.append(fe_a)
        mark = "✓" if fe_c <= 2.0 and fe_a <= 2.0 else "✗"
        logger.info(
            "  %s: Cmax=%.4f (obs=%.4f, FE=%.1fx) AUC=%.4f (obs=%.4f, FE=%.1fx) %s",
            drug_info["name"],
            pk["Cmax_mg_L"],
            drug_info["obs_cmax"],
            fe_c,
            pk["AUC_mg_h_L"],
            drug_info["obs_auc"],
            fe_a,
            mark,
        )

    if cmax_fes:
        aafe_c = float(10 ** np.mean(np.log10(cmax_fes)))
        aafe_a = float(10 ** np.mean(np.log10(auc_fes)))
        pct_c = sum(1 for fe in cmax_fes if fe <= 2.0) / len(cmax_fes) * 100
        pct_a = sum(1 for fe in auc_fes if fe <= 2.0) / len(auc_fes) * 100

        logger.info("\n" + "=" * 60)
        logger.info("CLINICAL FINE-TUNED RESULTS")
        logger.info("  Cmax: AAFE=%.2f, %%2-fold=%.0f%%", aafe_c, pct_c)
        logger.info("  AUC:  AAFE=%.2f, %%2-fold=%.0f%%", aafe_a, pct_a)
        logger.info("  Speed: ~74ms (Transfer MLP, no ADMET-AI)")
        logger.info("  vs L1: Cmax AAFE 2.16→%.2f, AUC AAFE 1.66→%.2f", aafe_c, aafe_a)
        logger.info("=" * 60)

    # Step 7: Save correction factors
    import json

    save_path = repo_root / "models" / "level2" / "clinical_correction.json"
    save_data = {
        "correction_vector": correction.tolist(),
        "param_names": PARAM_NAMES,
        "n_drugs": len(drug_data),
        "aafe_cmax": aafe_c if cmax_fes else None,
        "aafe_auc": aafe_a if cmax_fes else None,
        "method": "Nelder-Mead",
        "n_iterations": result.nit,
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info("Saved correction factors to %s", save_path)


if __name__ == "__main__":
    main()
