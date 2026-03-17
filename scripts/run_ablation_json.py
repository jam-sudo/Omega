#!/usr/bin/env python3
"""Run ablation study and save results to JSON.

Wraps run_ablation.py logic, saving structured output to
outputs/diagnostic_report_ablation.json.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

from run_l1_benchmarks import (  # noqa: E402
    BENCHMARK_DRUGS,
    compute_aafe,
    compute_fold_error,
    load_observed_pk,
)

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402


def run_benchmark_with_config(config_name, pipeline):
    """Run 25-drug benchmark with given pipeline configuration."""
    cmax_fes = []
    auc_fes = []
    per_drug = []

    for drug_name, info in BENCHMARK_DRUGS.items():
        try:
            result = pipeline.simulate(
                SimulationRequest(
                    smiles=info["smiles"],
                    dose_mg=info["dose_mg"],
                    route="oral",
                    duration_h=24.0,
                )
            )
            obs = load_observed_pk(drug_name)
            fe_cmax = compute_fold_error(result.cmax_mg_L, obs.get("cmax", 0))
            fe_auc = compute_fold_error(result.auc0t_mg_h_L, obs.get("auc", 0))
            if not np.isnan(fe_cmax) and fe_cmax > 0:
                cmax_fes.append(fe_cmax)
            if not np.isnan(fe_auc) and fe_auc > 0:
                auc_fes.append(fe_auc)
            per_drug.append(
                {
                    "drug": drug_name,
                    "fe_cmax": round(fe_cmax, 4) if not np.isnan(fe_cmax) else None,
                    "fe_auc": round(fe_auc, 4) if not np.isnan(fe_auc) else None,
                    "pred_cmax": round(result.cmax_mg_L, 6),
                    "pred_auc": round(result.auc0t_mg_h_L, 6),
                }
            )
        except Exception as e:
            print(f"    [{config_name}] {drug_name} FAILED: {e}")
            per_drug.append(
                {
                    "drug": drug_name,
                    "fe_cmax": None,
                    "fe_auc": None,
                    "error": str(e),
                }
            )

    aafe_cmax = compute_aafe(cmax_fes)
    aafe_auc = compute_aafe(auc_fes)
    pct_2fold = 100 * sum(1 for f in cmax_fes if f <= 2.0) / len(cmax_fes) if cmax_fes else 0

    return {
        "config": config_name,
        "aafe_cmax": round(aafe_cmax, 4),
        "aafe_auc": round(aafe_auc, 4),
        "pct_2fold_cmax": round(pct_2fold, 1),
        "n_valid": len(cmax_fes),
        "gt3fold": sum(1 for f in cmax_fes if f > 3.0),
        "per_drug": per_drug,
    }


def make_pipeline_no_hybrid():
    import types

    from omega_pbpk._compat import np_trapz

    p = OmegaPipeline()
    p._ensure_initialized()

    def simulate_no_hybrid(self, request):
        self._ensure_initialized()
        warnings_list = []
        adme_props = self._predict_adme(request.smiles, warnings_list)
        drug = self._build_drug(request.smiles, adme_props, warnings_list)
        time_h, cp = self._run_simulation(drug, request, warnings_list)
        cmax = float(np.max(cp))
        tmax = float(time_h[np.argmax(cp)])
        auc = float(np_trapz(cp, time_h))
        cmax_idx = int(np.argmax(cp))
        threshold = cmax * 0.001
        post_cmax_cp = cp[cmax_idx + 1 :]
        post_cmax_time = time_h[cmax_idx + 1 :]
        mask = post_cmax_cp > threshold
        t_half = float("nan")
        if np.sum(mask) >= 3:
            log_cp = np.log(post_cmax_cp[mask])
            t_fit = post_cmax_time[mask]
            slope, _ = np.polyfit(t_fit, log_cp, 1)
            if slope < -1e-10:
                t_half = float(-np.log(2) / slope)
        confidence = adme_props.get("confidence", "low")
        from omega_pbpk.pipeline import SimulationResult

        return SimulationResult(
            time_h=time_h,
            cp_mg_L=cp,
            cmax_mg_L=cmax,
            tmax_h=tmax,
            auc0t_mg_h_L=auc,
            t_half_h=t_half,
            adme_properties=adme_props,
            confidence=confidence,
            warnings=warnings_list,
        )

    p.simulate = types.MethodType(simulate_no_hybrid, p)
    return p


def make_pipeline_no_pgp():
    import types

    p = OmegaPipeline()
    p._ensure_initialized()
    original_build_drug = p._build_drug.__func__

    def build_drug_no_pgp(self, smiles, adme, warnings_list):
        import omega_pbpk.ml.models.adme.transporter_lookup as tl

        orig_is_pgp = tl.is_pgp_substrate
        tl.is_pgp_substrate = lambda **kwargs: False
        try:
            result = original_build_drug(self, smiles, adme, warnings_list)
        finally:
            tl.is_pgp_substrate = orig_is_pgp
        return result

    p._build_drug = types.MethodType(build_drug_no_pgp, p)
    return p


def make_pipeline_bare():
    import types

    from omega_pbpk._compat import np_trapz

    p = OmegaPipeline()
    p._ensure_initialized()

    def simulate_bare(self, request):
        self._ensure_initialized()
        warnings_list = []
        adme_props = self._predict_adme(request.smiles, warnings_list)
        import omega_pbpk.ml.models.adme.transporter_lookup as tl

        orig_is_pgp = tl.is_pgp_substrate
        tl.is_pgp_substrate = lambda **kwargs: False
        try:
            drug = self._build_drug(request.smiles, adme_props, warnings_list)
        finally:
            tl.is_pgp_substrate = orig_is_pgp
        time_h, cp = self._run_simulation(drug, request, warnings_list)
        cmax = float(np.max(cp))
        tmax = float(time_h[np.argmax(cp)])
        auc = float(np_trapz(cp, time_h))
        cmax_idx = int(np.argmax(cp))
        threshold = cmax * 0.001
        post_cmax_cp = cp[cmax_idx + 1 :]
        post_cmax_time = time_h[cmax_idx + 1 :]
        mask = post_cmax_cp > threshold
        t_half = float("nan")
        if np.sum(mask) >= 3:
            log_cp = np.log(post_cmax_cp[mask])
            t_fit = post_cmax_time[mask]
            slope, _ = np.polyfit(t_fit, log_cp, 1)
            if slope < -1e-10:
                t_half = float(-np.log(2) / slope)
        confidence = adme_props.get("confidence", "low")
        from omega_pbpk.pipeline import SimulationResult

        return SimulationResult(
            time_h=time_h,
            cp_mg_L=cp,
            cmax_mg_L=cmax,
            tmax_h=tmax,
            auc0t_mg_h_L=auc,
            t_half_h=t_half,
            adme_properties=adme_props,
            confidence=confidence,
            warnings=warnings_list,
        )

    p.simulate = types.MethodType(simulate_bare, p)
    return p


def main():
    configs = {}

    print("=" * 80)
    print("ABLATION STUDY (JSON output)")
    print("=" * 80)

    # 1. FULL
    print("\n[1/5] FULL...")
    t0 = time.time()
    configs["FULL"] = run_benchmark_with_config("FULL", OmegaPipeline())
    print(f"  Done in {time.time() - t0:.1f}s")

    # 2. NO_RIDGE (no Ridge correction in pipeline currently, but verify)
    print("\n[2/5] NO_RIDGE...")
    t0 = time.time()
    p2 = OmegaPipeline()
    p2._ensure_initialized()
    p2._correction_model_cmax = None
    p2._correction_model_auc = None
    configs["NO_RIDGE"] = run_benchmark_with_config("NO_RIDGE", p2)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 3. NO_HYBRID
    print("\n[3/5] NO_HYBRID...")
    t0 = time.time()
    configs["NO_HYBRID"] = run_benchmark_with_config("NO_HYBRID", make_pipeline_no_hybrid())
    print(f"  Done in {time.time() - t0:.1f}s")

    # 4. NO_PGP
    print("\n[4/5] NO_PGP...")
    t0 = time.time()
    configs["NO_PGP"] = run_benchmark_with_config("NO_PGP", make_pipeline_no_pgp())
    print(f"  Done in {time.time() - t0:.1f}s")

    # 5. BARE
    print("\n[5/5] BARE...")
    t0 = time.time()
    configs["BARE"] = run_benchmark_with_config("BARE", make_pipeline_bare())
    print(f"  Done in {time.time() - t0:.1f}s")

    # Build delta table
    full = configs["FULL"]
    deltas = {}
    for name, result in configs.items():
        if name == "FULL":
            continue
        deltas[name] = {
            "d_aafe_cmax": round(result["aafe_cmax"] - full["aafe_cmax"], 4),
            "d_pct_2fold": round(result["pct_2fold_cmax"] - full["pct_2fold_cmax"], 1),
            "d_aafe_auc": round(result["aafe_auc"] - full["aafe_auc"], 4),
            "verdict": (
                "HELPS"
                if result["aafe_cmax"] - full["aafe_cmax"] > 0.05
                else ("HURTS" if result["aafe_cmax"] - full["aafe_cmax"] < -0.05 else "NEUTRAL")
            ),
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Ablation study: impact of each pipeline correction on 25-drug benchmark",
        "configs": configs,
        "deltas_vs_full": deltas,
    }

    out_path = repo_root / "outputs" / "diagnostic_report_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 90)
    print(
        f"{'Config':20s} {'Cmax AAFE':>10s} {'%2-fold':>8s} {'AUC AAFE':>10s} {'Delta':>10s} {'Verdict':>10s}"
    )
    print("-" * 90)
    for name, result in configs.items():
        d = deltas.get(name, {})
        print(
            f"{name:20s} {result['aafe_cmax']:10.3f} {result['pct_2fold_cmax']:7.0f}% "
            f"{result['aafe_auc']:10.3f} {d.get('d_aafe_cmax', 0):+10.3f} "
            f"{d.get('verdict', 'BASELINE'):>10s}"
        )


if __name__ == "__main__":
    main()
