#!/usr/bin/env python3
"""Evaluate L2 GNN model vs L1 pipeline on 20 benchmark drugs.

Compares:
1. Parameter prediction accuracy (GNN vs ensemble ADME)
2. PK metrics (Cmax, AUC) against observed clinical data
3. Inference speed (L2 vs L1)
4. Surrogate vs real ODE agreement

Usage:
    python scripts/eval_l2_model.py
    python scripts/eval_l2_model.py --model models/level2/final.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Same 20 benchmark drugs as L1
BENCHMARK_DRUGS = {
    "caffeine": {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "dose_mg": 100},
    "metoprolol": {"smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1", "dose_mg": 100},
    "midazolam": {"smiles": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2", "dose_mg": 2},
    "propranolol": {"smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "dose_mg": 80},
    "warfarin": {"smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "dose_mg": 5},
    "d_amphetamine": {"smiles": "C[C@@H](N)Cc1ccccc1", "dose_mg": 20},
    "methanol": {"smiles": "CO", "dose_mg": 100},
    "ibuprofen": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "dose_mg": 400},
    "acetaminophen": {"smiles": "CC(=O)Nc1ccc(O)cc1", "dose_mg": 1000},
    "amoxicillin": {
        "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "dose_mg": 500,
    },
    "atorvastatin": {
        "smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "dose_mg": 40,
    },
    "carbamazepine": {"smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21", "dose_mg": 200},
    "diazepam": {"smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "dose_mg": 10},
    "digoxin": {
        "smiles": "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@@H](O[C@@H]4C[C@H](O)[C@@H](OC5CC(CO)=CC(=O)O5)C(C)O4)C(C)O3)C(C)O2)C[C@H](O)[C@H]1O",
        "dose_mg": 0.5,
    },
    "fluoxetine": {
        "smiles": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        "dose_mg": 20,
    },
    "nifedipine": {
        "smiles": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
        "dose_mg": 10,
    },
    "omeprazole": {
        "smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "dose_mg": 20,
    },
    "phenytoin": {"smiles": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1", "dose_mg": 300},
    "theophylline": {"smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "dose_mg": 300},
    "verapamil": {
        "smiles": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
        "dose_mg": 80,
    },
}


def load_observed_pk(drug_name: str) -> dict:
    """Load observed Cmax and AUC from benchmark CSV dataset."""
    datasets_dir = repo_root / "benchmarks" / "datasets"
    for csv_file in datasets_dir.glob("*.csv"):
        if drug_name.replace("_", "") in csv_file.stem.replace("_", "").lower():
            data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            time_h = data[:, 0]
            conc = data[:, 1]
            cmax = float(conc.max())
            auc = float(np.trapz(conc, time_h))
            return {"cmax": cmax, "auc": auc}
    return {}


def fold_error(pred, obs):
    if abs(pred) < 1e-12 or abs(obs) < 1e-12:
        return float("nan")
    ratio = pred / obs
    return max(ratio, 1.0 / ratio)


def compute_aafe(fold_errors):
    valid = [fe for fe in fold_errors if not np.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return float(10.0 ** np.mean(np.log10(valid)))


def eval_l2(model_path: str, device: str = "cpu"):
    """Evaluate L2 GNN model."""
    from omega_pbpk.ml.models.foundation.inference import MLPKPredictor

    print("=" * 80)
    print(f"L2 EVALUATION: GNN model = {model_path}")
    print(f"Device: {device}")
    print("=" * 80)

    predictor = MLPKPredictor(model_path=model_path, device=device)
    if predictor.model is None:
        print("ERROR: Model not loaded. Check path.")
        return None

    results = []
    cmax_fes, auc_fes = [], []
    latencies = []

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]
        observed = load_observed_pk(drug_name)

        print(f"\n--- {drug_name} (dose={dose_mg}mg) ---")

        try:
            t0 = time.perf_counter()
            result = predictor.predict(
                smiles, dose_mg=dose_mg, route="oral", t_end_h=24.0
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            params = result["params"]
            pk = result.get("pk_profile", {})
            surr_curve = result.get("surrogate_curve")

            # Extract PK metrics
            pred_cmax = pk.get("Cmax_mg_L", 0)
            pred_auc = pk.get("AUC_mg_h_L", 0)
            pred_thalf = pk.get("half_life_h", 0)

            # Surrogate Cmax for comparison
            surr_cmax = float(np.max(surr_curve)) if surr_curve is not None else 0

            print(f"  Params: logP={params.get('logP', 0):.2f}, "
                  f"fup={params.get('fup', 0):.4f}, "
                  f"clint={params.get('clint_L_h', params.get('clint_3a4', 0)):.3f}, "
                  f"mw={params.get('mw', 0):.0f}")
            print(f"  PK (ODE): Cmax={pred_cmax:.4f} mg/L, "
                  f"AUC={pred_auc:.4f} mg*h/L, t½={pred_thalf:.2f}h")
            print(f"  Surrogate Cmax: {surr_cmax:.4f} mg/L")
            print(f"  Latency: {latency_ms:.0f}ms")

            entry = {
                "drug": drug_name,
                "params": params,
                "pred_cmax": pred_cmax,
                "pred_auc": pred_auc,
                "pred_thalf": pred_thalf,
                "surr_cmax": surr_cmax,
                "latency_ms": latency_ms,
                "success": True,
            }

            if observed:
                fe_c = fold_error(pred_cmax, observed["cmax"])
                fe_a = fold_error(pred_auc, observed["auc"])
                cmax_fes.append(fe_c)
                auc_fes.append(fe_a)
                print(f"  Observed: Cmax={observed['cmax']:.4f}, AUC={observed['auc']:.4f}")
                print(f"  Fold-error: Cmax={fe_c:.2f}x, AUC={fe_a:.2f}x "
                      f"{'✓' if fe_c <= 2.0 and fe_a <= 2.0 else '✗'}")
                entry["obs_cmax"] = observed["cmax"]
                entry["obs_auc"] = observed["auc"]
                entry["fe_cmax"] = fe_c
                entry["fe_auc"] = fe_a

            results.append(entry)

        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({"drug": drug_name, "success": False, "error": str(e)})

    # --- Summary ---
    print("\n" + "=" * 80)
    print("L2 AGGREGATE RESULTS")
    print("=" * 80)

    n_success = sum(1 for r in results if r.get("success"))
    print(f"Drugs: {n_success}/{len(results)} succeeded")

    aafe_cmax = compute_aafe(cmax_fes)
    aafe_auc = compute_aafe(auc_fes)
    valid_c = [fe for fe in cmax_fes if not np.isnan(fe)]
    valid_a = [fe for fe in auc_fes if not np.isnan(fe)]
    pct_2f_c = sum(1 for fe in valid_c if fe <= 2.0) / max(len(valid_c), 1) * 100
    pct_2f_a = sum(1 for fe in valid_a if fe <= 2.0) / max(len(valid_a), 1) * 100

    avg_latency = np.mean(latencies) if latencies else 0
    p50_latency = np.median(latencies) if latencies else 0

    print(f"\nCmax: AAFE={aafe_cmax:.2f}, %2-fold={pct_2f_c:.0f}%")
    print(f"AUC:  AAFE={aafe_auc:.2f}, %2-fold={pct_2f_a:.0f}%")
    print(f"\nLatency: mean={avg_latency:.0f}ms, p50={p50_latency:.0f}ms")

    print("\n--- L2 EXIT CRITERIA ---")
    print(f"  AAFE < 2.0:   Cmax={'PASS' if aafe_cmax < 2.0 else 'FAIL'} ({aafe_cmax:.2f}), "
          f"AUC={'PASS' if aafe_auc < 2.0 else 'FAIL'} ({aafe_auc:.2f})")
    print(f"  %2-fold ≥70%: Cmax={'PASS' if pct_2f_c >= 70 else 'FAIL'} ({pct_2f_c:.0f}%), "
          f"AUC={'PASS' if pct_2f_a >= 70 else 'FAIL'} ({pct_2f_a:.0f}%)")
    print(f"  Speed <500ms: {'PASS' if avg_latency < 500 else 'FAIL'} ({avg_latency:.0f}ms)")

    # --- Load L1 results for comparison ---
    l1_path = repo_root / "outputs" / "l1_benchmark_results.json"
    if l1_path.exists():
        l1 = json.load(open(l1_path))
        print("\n--- L2 vs L1 COMPARISON ---")
        print(f"  {'Metric':<20s} {'L1':>10s} {'L2':>10s} {'Delta':>10s}")
        print(f"  {'Cmax AAFE':<20s} {l1['aafe_cmax']:>10.2f} {aafe_cmax:>10.2f} "
              f"{'↑' if aafe_cmax > l1['aafe_cmax'] else '↓'}{abs(aafe_cmax - l1['aafe_cmax']):.2f}")
        print(f"  {'AUC AAFE':<20s} {l1['aafe_auc']:>10.2f} {aafe_auc:>10.2f} "
              f"{'↑' if aafe_auc > l1['aafe_auc'] else '↓'}{abs(aafe_auc - l1['aafe_auc']):.2f}")
        print(f"  {'Cmax %2-fold':<20s} {l1['pct_2fold_cmax']:>9.0f}% {pct_2f_c:>9.0f}%")
        print(f"  {'AUC %2-fold':<20s} {l1['pct_2fold_auc']:>9.0f}% {pct_2f_a:>9.0f}%")

    # Save
    summary = {
        "model_path": str(model_path),
        "device": device,
        "n_drugs": len(results),
        "n_success": n_success,
        "aafe_cmax": aafe_cmax,
        "aafe_auc": aafe_auc,
        "pct_2fold_cmax": pct_2f_c,
        "pct_2fold_auc": pct_2f_a,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "exit_criteria": {
            "aafe_lt_2": aafe_cmax < 2.0 and aafe_auc < 2.0,
            "pct_2fold_gte_70": pct_2f_c >= 70 and pct_2f_a >= 70,
            "speed_lt_500ms": avg_latency < 500,
        },
        "per_drug": results,
    }

    out_path = repo_root / "outputs" / "l2_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate L2 GNN model")
    parser.add_argument(
        "--model",
        default="models/level2/final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu/cuda)",
    )
    args = parser.parse_args()
    eval_l2(args.model, args.device)
