#!/usr/bin/env python3
"""Ablation study: measure impact of each pipeline correction.

Tests the OmegaPipeline with corrections progressively stripped to understand
which post-hoc corrections help and which hurt on the 25-drug benchmark.

Configurations:
  FULL          - Current pipeline with ALL corrections (baseline)
  NO_ENSEMBLE   - Disable PBPK+ML ensemble (direct Cmax predictor)
  NO_RIDGE      - Disable Ridge residual correction model
  NO_VDSS       - Disable XGBoost VDss correction
  NO_ENS_RIDGE  - Disable both ensemble and Ridge
  NO_HYBRID     - Disable hybrid Cmax selector (use raw ODE Cmax)
  NO_PGP        - Disable P-gp efflux correction
  BARE          - Disable ALL corrections (raw ODE output)
"""

import sys
import time
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
            per_drug.append((drug_name, fe_cmax, fe_auc, result.cmax_mg_L, result.auc0t_mg_h_L))
        except Exception as e:
            print(f"    [{config_name}] {drug_name} FAILED: {e}")
            per_drug.append((drug_name, float("nan"), float("nan"), 0, 0))

    aafe_cmax = compute_aafe(cmax_fes)
    aafe_auc = compute_aafe(auc_fes)
    pct_2fold = 100 * sum(1 for f in cmax_fes if f <= 2.0) / len(cmax_fes) if cmax_fes else 0
    gt3 = sum(1 for f in cmax_fes if f > 3.0)
    median_cmax = float(np.median(cmax_fes)) if cmax_fes else float("nan")
    median_auc = float(np.median(auc_fes)) if auc_fes else float("nan")

    return {
        "config": config_name,
        "aafe_cmax": aafe_cmax,
        "aafe_auc": aafe_auc,
        "median_cmax": median_cmax,
        "median_auc": median_auc,
        "pct_2fold": pct_2fold,
        "gt3fold": gt3,
        "n_valid": len(cmax_fes),
        "per_drug": per_drug,
    }


def make_pipeline_no_hybrid():
    """Create pipeline that skips hybrid Cmax selector (uses raw ODE Cmax).

    We monkey-patch the simulate method to bypass the hybrid selector,
    ridge correction, and ensemble — returning raw ODE Cmax/AUC.
    """
    import types

    from omega_pbpk._compat import np_trapz

    p = OmegaPipeline()
    p._ensure_initialized()

    def simulate_no_hybrid(self, request):
        """Simulate but skip hybrid selector — use raw ODE Cmax."""
        self._ensure_initialized()
        warnings_list = []
        adme_props = self._predict_adme(request.smiles, warnings_list)
        drug = self._build_drug(request.smiles, adme_props, warnings_list)

        time_h, cp = self._run_simulation(drug, request, warnings_list)
        cmax = float(np.max(cp))
        tmax = float(time_h[np.argmax(cp)])
        auc = float(np_trapz(cp, time_h))

        # Curve-fit t_half only (no analytical)
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
    """Create pipeline that skips P-gp efflux correction."""
    import types

    p = OmegaPipeline()
    p._ensure_initialized()

    original_build_drug = p._build_drug.__func__

    def build_drug_no_pgp(self, smiles, adme, warnings_list):
        """Build drug without P-gp correction by temporarily disabling the import."""
        # Call original but patch is_pgp_substrate to always return False
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
    """Create pipeline with ALL corrections disabled: raw ODE output, no P-gp."""
    import types

    from omega_pbpk._compat import np_trapz

    p = OmegaPipeline()
    p._ensure_initialized()

    def simulate_bare(self, request):
        """Simulate with zero corrections — raw ODE Cmax/AUC, no P-gp."""
        self._ensure_initialized()
        warnings_list = []
        adme_props = self._predict_adme(request.smiles, warnings_list)

        # Build drug without P-gp
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

        # Curve-fit t_half only
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

    # 1. FULL (current pipeline)
    print("=" * 80)
    print("ABLATION STUDY: Impact of each pipeline correction")
    print("=" * 80)

    print("\n[1/8] Running FULL configuration (all corrections)...")
    t0 = time.time()
    p1 = OmegaPipeline()
    configs["FULL"] = run_benchmark_with_config("FULL", p1)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 2. NO_ENSEMBLE — disable direct Cmax ML predictor
    print("\n[2/8] Running NO_ENSEMBLE (disable PBPK+ML ensemble)...")
    t0 = time.time()
    p2 = OmegaPipeline()
    p2._ensure_initialized()
    p2._direct_cmax = None
    configs["NO_ENSEMBLE"] = run_benchmark_with_config("NO_ENSEMBLE", p2)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 3. NO_RIDGE — disable Ridge residual correction
    print("\n[3/8] Running NO_RIDGE (disable Ridge correction)...")
    t0 = time.time()
    p3 = OmegaPipeline()
    p3._ensure_initialized()
    p3._correction_model_cmax = None
    p3._correction_model_auc = None
    configs["NO_RIDGE"] = run_benchmark_with_config("NO_RIDGE", p3)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 4. NO_VDSS — disable XGBoost VDss correction
    print("\n[4/8] Running NO_VDSS (disable XGBoost VDss predictor)...")
    t0 = time.time()
    p4 = OmegaPipeline()
    p4._ensure_initialized()
    p4._vdss_predictor = None
    configs["NO_VDSS"] = run_benchmark_with_config("NO_VDSS", p4)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 5. NO_ENS_RIDGE — disable both ensemble and Ridge
    print("\n[5/8] Running NO_ENS_RIDGE (disable ensemble + Ridge)...")
    t0 = time.time()
    p5 = OmegaPipeline()
    p5._ensure_initialized()
    p5._direct_cmax = None
    p5._correction_model_cmax = None
    p5._correction_model_auc = None
    configs["NO_ENS_RIDGE"] = run_benchmark_with_config("NO_ENS_RIDGE", p5)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 6. NO_HYBRID — disable hybrid Cmax selector entirely (raw ODE Cmax)
    print("\n[6/8] Running NO_HYBRID (raw ODE Cmax, no blending/Ridge/ensemble)...")
    t0 = time.time()
    p6 = make_pipeline_no_hybrid()
    configs["NO_HYBRID"] = run_benchmark_with_config("NO_HYBRID", p6)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 7. NO_PGP — disable P-gp efflux correction
    print("\n[7/8] Running NO_PGP (disable P-gp efflux correction)...")
    t0 = time.time()
    p7 = make_pipeline_no_pgp()
    configs["NO_PGP"] = run_benchmark_with_config("NO_PGP", p7)
    print(f"  Done in {time.time() - t0:.1f}s")

    # 8. BARE — ALL corrections disabled
    print("\n[8/8] Running BARE (ALL corrections disabled — raw ODE)...")
    t0 = time.time()
    p8 = make_pipeline_bare()
    configs["BARE"] = run_benchmark_with_config("BARE", p8)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Print results table ----
    print("\n" + "=" * 100)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':20s} {'Cmax AAFE':>10s} {'Med Cmax':>10s} {'%2-fold':>8s} "
        f"{'>3-fold':>8s} {'AUC AAFE':>10s} {'Med AUC':>10s} {'N':>4s}"
    )
    print("-" * 100)
    for name, result in configs.items():
        print(
            f"{name:20s} {result['aafe_cmax']:10.3f} {result['median_cmax']:10.3f} "
            f"{result['pct_2fold']:7.0f}% {result['gt3fold']:8d} "
            f"{result['aafe_auc']:10.3f} {result['median_auc']:10.3f} {result['n_valid']:4d}"
        )

    # ---- Delta table (vs FULL) ----
    print("\n" + "=" * 100)
    print("DELTA vs FULL (positive = correction HELPS, negative = correction HURTS)")
    print("=" * 100)
    full = configs["FULL"]
    print(
        f"{'Config':20s} {'dCmax AAFE':>12s} {'d%2-fold':>10s} {'dAUC AAFE':>12s} {'Verdict':>10s}"
    )
    print("-" * 100)
    for name, result in configs.items():
        if name == "FULL":
            continue
        d_cmax = result["aafe_cmax"] - full["aafe_cmax"]
        d_pct = result["pct_2fold"] - full["pct_2fold"]
        d_auc = result["aafe_auc"] - full["aafe_auc"]
        # Positive delta means removing the correction INCREASED error (correction helps)
        verdict = "HELPS" if d_cmax > 0.05 else ("HURTS" if d_cmax < -0.05 else "NEUTRAL")
        print(f"{name:20s} {d_cmax:+12.3f} {d_pct:+9.0f}% {d_auc:+12.3f} {verdict:>10s}")

    # ---- Per-drug comparison ----
    print("\n" + "=" * 120)
    print("PER-DRUG Cmax FOLD ERRORS (all drugs, sorted by max FE)")
    print("=" * 120)

    config_names = list(configs.keys())
    header = f"{'Drug':20s}"
    for name in config_names:
        header += f" {name:>12s}"
    print(header)
    print("-" * 120)

    # Build drug FE map
    drug_names = [d[0] for d in configs["FULL"]["per_drug"]]
    drug_fes = {}
    for drug in drug_names:
        drug_fes[drug] = {}
        for name, result in configs.items():
            for d in result["per_drug"]:
                if d[0] == drug:
                    drug_fes[drug][name] = d[1]  # Cmax FE

    sorted_drugs = sorted(
        drug_fes.keys(),
        key=lambda d: max(
            (v for v in drug_fes[d].values() if not np.isnan(v)),
            default=0,
        ),
        reverse=True,
    )

    for drug in sorted_drugs:
        row = f"{drug:20s}"
        for name in config_names:
            fe = drug_fes[drug].get(name, float("nan"))
            if np.isnan(fe):
                row += f" {'FAIL':>12s}"
            else:
                row += f" {fe:11.2f}x"
        print(row)

    # ---- Per-drug AUC comparison ----
    print("\n" + "=" * 120)
    print("PER-DRUG AUC FOLD ERRORS (all drugs, sorted by max FE)")
    print("=" * 120)

    print(header)
    print("-" * 120)

    drug_auc_fes = {}
    for drug in drug_names:
        drug_auc_fes[drug] = {}
        for name, result in configs.items():
            for d in result["per_drug"]:
                if d[0] == drug:
                    drug_auc_fes[drug][name] = d[2]  # AUC FE

    sorted_drugs_auc = sorted(
        drug_auc_fes.keys(),
        key=lambda d: max(
            (v for v in drug_auc_fes[d].values() if not np.isnan(v)),
            default=0,
        ),
        reverse=True,
    )

    for drug in sorted_drugs_auc:
        row = f"{drug:20s}"
        for name in config_names:
            fe = drug_auc_fes[drug].get(name, float("nan"))
            if np.isnan(fe):
                row += f" {'FAIL':>12s}"
            else:
                row += f" {fe:11.2f}x"
        print(row)

    # ---- Winners/Losers analysis ----
    print("\n" + "=" * 100)
    print("DRUG-LEVEL WINNERS/LOSERS: Which corrections help which drugs?")
    print("=" * 100)
    for name in config_names:
        if name == "FULL":
            continue
        improved = []
        worsened = []
        for drug in drug_names:
            fe_full = drug_fes[drug].get("FULL", float("nan"))
            fe_ablated = drug_fes[drug].get(name, float("nan"))
            if np.isnan(fe_full) or np.isnan(fe_ablated):
                continue
            delta = fe_ablated - fe_full
            if delta > 0.3:
                improved.append((drug, fe_full, fe_ablated, delta))
            elif delta < -0.3:
                worsened.append((drug, fe_full, fe_ablated, delta))
        print(f"\n--- Removing {name} ---")
        if improved:
            print(f"  Correction HELPS ({len(improved)} drugs worse without it):")
            for drug, full_fe, abl_fe, delta in sorted(improved, key=lambda x: -x[3]):
                print(f"    {drug:20s}: {full_fe:.2f}x -> {abl_fe:.2f}x (delta={delta:+.2f})")
        if worsened:
            print(f"  Correction HURTS ({len(worsened)} drugs better without it):")
            for drug, full_fe, abl_fe, delta in sorted(worsened, key=lambda x: x[3]):
                print(f"    {drug:20s}: {full_fe:.2f}x -> {abl_fe:.2f}x (delta={delta:+.2f})")
        if not improved and not worsened:
            print("  No significant changes (all deltas < 0.3)")


if __name__ == "__main__":
    main()
