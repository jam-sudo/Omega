#!/usr/bin/env python3
"""Validate surrogate model accuracy against real ODE.

For each benchmark drug:
  1. Get ADME params from L1 benchmark results
  2. Run params through surrogate → predicted C(t)
  3. Run params through real ODE (pipeline.simulate) → reference C(t)
  4. Compare: Cmax, AUC, curve RMSE, curve correlation

This reveals whether the surrogate is a bottleneck for L2 training.
If surrogate C(t) diverges significantly from real ODE C(t), then
GNN training through the surrogate will learn wrong gradients.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_surrogate")

SURROGATE_PATH = Path("models/pbpk_surrogate/6param/surrogate_model.pt")
BENCHMARK_RESULTS = Path("outputs/l1_benchmark_results.json")
IVIVE_FACTOR = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0  # ~0.054

# Surrogate input order: [logP, fup, clint_L_h, mw, rbp, peff]


def load_benchmark_drugs() -> list[dict]:
    """Load per-drug results from L1 benchmarks."""
    with open(BENCHMARK_RESULTS) as f:
        data = json.load(f)
    return data["per_drug"]


def drug_to_surrogate_params(drug: dict) -> np.ndarray:
    """Convert benchmark drug ADME dict to 6D surrogate input."""
    adme = drug["adme"]
    clint_L_h = (adme.get("clint_3a4", 0.1) + adme.get("clint_2d6", 0.0)) * IVIVE_FACTOR
    return np.array(
        [
            adme["logP"],
            adme["fup"],
            clint_L_h,
            adme["mw"],
            adme["rbp"],
            adme.get("peff", 1.0),
        ],
        dtype=np.float32,
    )


def run_ode_simulation(smiles: str, dose_mg: float) -> dict | None:
    """Run real ODE via pipeline.simulate()."""
    try:
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        request = SimulationRequest(
            smiles=smiles,
            dose_mg=dose_mg,
            route="oral",
            duration_h=24.0,
        )
        result = pipeline.simulate(request)
        return {
            "times": np.array(result.time_h),
            "cp": np.array(result.cp_mg_L),
            "cmax": float(result.cmax_mg_L),
            "auc": float(result.auc0t_mg_h_L),
        }
    except Exception as exc:
        logger.warning("ODE simulation failed for %s: %s", smiles, exc)
        return None


def compute_metrics(
    surr_curve: np.ndarray,
    ode_times: np.ndarray,
    ode_cp: np.ndarray,
    surr_dt: float = 0.1,
) -> dict:
    """Compare surrogate and ODE C(t) curves.

    Surrogate outputs 241 timepoints at dt=0.1h (0-24h).
    ODE may have different time resolution — interpolate to match.
    """
    # Surrogate timepoints
    surr_times = np.arange(len(surr_curve)) * surr_dt

    # Interpolate ODE to surrogate timepoints
    ode_interp = np.interp(surr_times, ode_times, ode_cp)

    # Cmax comparison
    surr_cmax = float(surr_curve.max())
    ode_cmax = float(ode_interp.max())

    # AUC comparison (trapezoidal)
    surr_auc = float(np.trapz(surr_curve, surr_times))
    ode_auc = float(np.trapz(ode_interp, surr_times))

    # Curve-level metrics
    eps = 1e-12
    rmse = float(np.sqrt(np.mean((surr_curve - ode_interp) ** 2)))
    # Normalize RMSE by ODE Cmax for interpretability
    nrmse = rmse / max(ode_cmax, eps)

    # Pearson correlation
    if np.std(surr_curve) > eps and np.std(ode_interp) > eps:
        corr = float(np.corrcoef(surr_curve, ode_interp)[0, 1])
    else:
        corr = 0.0

    # Fold errors
    cmax_fe = max(surr_cmax, eps) / max(ode_cmax, eps)
    if cmax_fe < 1:
        cmax_fe = 1.0 / cmax_fe
    auc_fe = max(surr_auc, eps) / max(ode_auc, eps)
    if auc_fe < 1:
        auc_fe = 1.0 / auc_fe

    return {
        "surr_cmax": surr_cmax,
        "ode_cmax": ode_cmax,
        "cmax_fold_error": cmax_fe,
        "surr_auc": surr_auc,
        "ode_auc": ode_auc,
        "auc_fold_error": auc_fe,
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": corr,
    }


def main() -> None:
    import torch

    from omega_pbpk.ml.models.surrogate.differentiable_ode import load_surrogate

    if not SURROGATE_PATH.exists():
        logger.error("Surrogate model not found: %s", SURROGATE_PATH)
        sys.exit(1)

    if not BENCHMARK_RESULTS.exists():
        logger.error("Benchmark results not found: %s", BENCHMARK_RESULTS)
        sys.exit(1)

    # Load surrogate
    surrogate = load_surrogate(SURROGATE_PATH)
    ck = torch.load(str(SURROGATE_PATH), map_location="cpu", weights_only=False)
    logger.info(
        "Surrogate loaded: %d inputs, %d outputs, training AAFE=%.3f",
        ck["n_input"],
        ck["n_output"],
        ck.get("aafe", -1),
    )

    # Load benchmark drugs
    drugs = load_benchmark_drugs()
    logger.info("Benchmark drugs: %d", len(drugs))

    # Run comparison
    results = []
    print()
    print(
        f"{'Drug':20s} {'Surr Cmax':>10s} {'ODE Cmax':>10s} {'FE':>6s} "
        f"{'Surr AUC':>10s} {'ODE AUC':>10s} {'FE':>6s} {'NRMSE':>8s} {'Corr':>6s}"
    )
    print("-" * 100)

    cmax_fes = []
    auc_fes = []
    corrs = []

    for drug in drugs:
        name = drug["drug"]
        smiles = drug["smiles"]
        dose_mg = drug["dose_mg"]

        # Get surrogate prediction
        params_6d = drug_to_surrogate_params(drug)
        surr_curve = surrogate.predict(params_6d)

        # Get real ODE prediction
        ode_result = run_ode_simulation(smiles, dose_mg)
        if ode_result is None:
            print(f"{name:20s} {'SKIP — ODE failed':>60s}")
            continue

        # Compare
        metrics = compute_metrics(surr_curve, ode_result["times"], ode_result["cp"])
        results.append({"drug": name, **metrics})

        cmax_fes.append(metrics["cmax_fold_error"])
        auc_fes.append(metrics["auc_fold_error"])
        corrs.append(metrics["correlation"])

        print(
            f"{name:20s} "
            f"{metrics['surr_cmax']:10.4f} {metrics['ode_cmax']:10.4f} {metrics['cmax_fold_error']:6.2f} "
            f"{metrics['surr_auc']:10.3f} {metrics['ode_auc']:10.3f} {metrics['auc_fold_error']:6.2f} "
            f"{metrics['nrmse']:8.4f} {metrics['correlation']:6.3f}"
        )

    if not results:
        logger.error("No drugs compared successfully")
        sys.exit(1)

    # Summary statistics
    print("-" * 100)
    aafe_cmax = float(np.exp(np.mean(np.log(cmax_fes))))
    aafe_auc = float(np.exp(np.mean(np.log(auc_fes))))
    mean_corr = float(np.mean(corrs))
    median_nrmse = float(np.median([r["nrmse"] for r in results]))
    pct_cmax_2fold = sum(1 for fe in cmax_fes if fe <= 2.0) / len(cmax_fes) * 100
    pct_auc_2fold = sum(1 for fe in auc_fes if fe <= 2.0) / len(auc_fes) * 100

    print(f"\n{'=' * 60}")
    print(f"SURROGATE vs REAL ODE — Summary ({len(results)} drugs)")
    print(f"{'=' * 60}")
    print(f"  Cmax AAFE:          {aafe_cmax:.3f}")
    print(f"  Cmax %within 2-fold: {pct_cmax_2fold:.0f}%")
    print(f"  AUC  AAFE:          {aafe_auc:.3f}")
    print(f"  AUC  %within 2-fold: {pct_auc_2fold:.0f}%")
    print(f"  Mean correlation:   {mean_corr:.3f}")
    print(f"  Median NRMSE:       {median_nrmse:.4f}")
    print()

    # Verdict
    if aafe_cmax < 1.5 and mean_corr > 0.9:
        print("  VERDICT: Surrogate is ACCURATE — not a bottleneck for L2 training")
    elif aafe_cmax < 2.0 and mean_corr > 0.7:
        print("  VERDICT: Surrogate is ACCEPTABLE — minor error source")
    else:
        print("  VERDICT: Surrogate is INACCURATE — likely bottleneck for L2 training")
        print("  RECOMMENDATION: Retrain surrogate with more ODE data before L2 training")

    # Save results
    output_path = Path("outputs/surrogate_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "n_drugs": len(results),
                "aafe_cmax": aafe_cmax,
                "aafe_auc": aafe_auc,
                "pct_2fold_cmax": pct_cmax_2fold,
                "pct_2fold_auc": pct_auc_2fold,
                "mean_correlation": mean_corr,
                "median_nrmse": median_nrmse,
                "per_drug": results,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
