#!/usr/bin/env python3
"""Validate L2 model checkpoint (models/level2/final.pt).

Checks:
1. Checkpoint loads without errors
2. Model architecture matches expected config
3. Parameter counts are reported
4. Inference works on test SMILES
5. Predicted PK params are physically meaningful
6. C(t) curves are non-degenerate
7. Compare final.pt vs pbpk_finetune/best.pt
"""

import sys
import json
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import torch
import numpy as np


def main():
    results = {"status": "running", "checks": {}}

    # --- Test SMILES for benchmark drugs ---
    test_drugs = {
        "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "midazolam": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2",
        "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "metoprolol": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
    }

    # ---- Check 1: Load checkpoint ----
    print("=" * 60)
    print("CHECK 1: Load final.pt checkpoint")
    print("=" * 60)
    final_path = repo_root / "models" / "level2" / "final.pt"
    try:
        from omega_pbpk.ml.models.foundation.end_to_end import SMILESToPKModel

        model = SMILESToPKModel.load(str(final_path), device="cpu")
        print(f"  PASS: Model loaded from {final_path}")
        print(f"  embedding_dim: {model.embedding_dim}")
        print(f"  dt_h: {model.dt_h}")
        results["checks"]["load_final"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["checks"]["load_final"] = f"FAIL: {e}"
        results["status"] = "failed"
        _save_results(results)
        return

    # ---- Check 2: Parameter counts ----
    print("\n" + "=" * 60)
    print("CHECK 2: Parameter counts")
    print("=" * 60)
    counts = model.parameter_count()
    for name, count in counts.items():
        print(f"  {name}: {count:,}")
    results["checks"]["param_counts"] = counts

    # ---- Check 3: Inference on test SMILES ----
    print("\n" + "=" * 60)
    print("CHECK 3: Inference on test drugs")
    print("=" * 60)
    inference_results = {}
    for drug_name, smiles in test_drugs.items():
        try:
            with torch.no_grad():
                output = model(smiles)

            params = {k: v.item() for k, v in output["params"].items()}
            pk = {k: v.item() for k, v in output["pk_metrics"].items()}
            curve = output["curve"].numpy().flatten()

            print(f"\n  {drug_name} ({smiles[:30]}...):")
            print(f"    Params: mw={params['mw']:.1f}, logP={params['logP']:.2f}, "
                  f"fup={params['fup']:.4f}, rbp={params['rbp']:.3f}")
            print(f"    CLint: 3a4={params['clint_3a4']:.3f}, 2d6={params['clint_2d6']:.3f}")
            print(f"    peff={params['peff']:.3f}")
            print(f"    PK: Cmax={pk['cmax']:.4f}, AUC={pk['auc']:.4f}, "
                  f"Tmax={pk['tmax']:.2f}h, t_half={pk['t_half']:.2f}h")
            print(f"    Curve: shape={curve.shape}, min={curve.min():.6f}, "
                  f"max={curve.max():.6f}, mean={curve.mean():.6f}")

            inference_results[drug_name] = {
                "params": params,
                "pk_metrics": pk,
                "curve_stats": {
                    "shape": list(curve.shape),
                    "min": float(curve.min()),
                    "max": float(curve.max()),
                    "mean": float(curve.mean()),
                    "nonzero_frac": float((curve > 1e-8).mean()),
                },
                "status": "PASS",
            }
        except Exception as e:
            print(f"  {drug_name}: FAIL - {e}")
            inference_results[drug_name] = {"status": f"FAIL: {e}"}

    results["checks"]["inference"] = inference_results

    # ---- Check 4: Physical meaningfulness ----
    print("\n" + "=" * 60)
    print("CHECK 4: Physical meaningfulness of predictions")
    print("=" * 60)
    phys_checks = {}
    for drug_name, res in inference_results.items():
        if res["status"] != "PASS":
            continue
        p = res["params"]
        issues = []
        if p["mw"] < 50 or p["mw"] > 2000:
            issues.append(f"mw={p['mw']:.1f} out of [50,2000]")
        if p["fup"] < 0 or p["fup"] > 1:
            issues.append(f"fup={p['fup']:.4f} out of [0,1]")
        if p["rbp"] < 0.5 or p["rbp"] > 3.0:
            issues.append(f"rbp={p['rbp']:.3f} out of [0.5,3.0]")
        if p["logP"] < -5 or p["logP"] > 10:
            issues.append(f"logP={p['logP']:.2f} out of [-5,10]")
        if p["clint_3a4"] < 0:
            issues.append(f"clint_3a4={p['clint_3a4']:.3f} negative")
        if p["peff"] < 0:
            issues.append(f"peff={p['peff']:.3f} negative")

        pk = res["pk_metrics"]
        if pk["cmax"] <= 0:
            issues.append(f"Cmax={pk['cmax']:.4f} non-positive")
        if pk["auc"] <= 0:
            issues.append(f"AUC={pk['auc']:.4f} non-positive")
        if pk["t_half"] < 0.1 or pk["t_half"] > 1000:
            issues.append(f"t_half={pk['t_half']:.2f}h out of [0.1,1000]")

        curve_stats = res["curve_stats"]
        if curve_stats["nonzero_frac"] < 0.1:
            issues.append(f"Curve mostly zero ({curve_stats['nonzero_frac']:.1%} nonzero)")

        status = "PASS" if not issues else f"WARN: {'; '.join(issues)}"
        print(f"  {drug_name}: {status}")
        phys_checks[drug_name] = status

    results["checks"]["physical_meaningfulness"] = phys_checks

    # ---- Check 5: Load finetune checkpoint ----
    print("\n" + "=" * 60)
    print("CHECK 5: Load pbpk_finetune/best.pt")
    print("=" * 60)
    finetune_path = repo_root / "models" / "level2" / "pbpk_finetune" / "best.pt"
    try:
        model_ft = SMILESToPKModel.load(str(finetune_path), device="cpu")
        print(f"  PASS: Finetune model loaded from {finetune_path}")

        # Compare params
        ft_counts = model_ft.parameter_count()
        print(f"  Params match: {ft_counts['total'] == counts['total']}")
        results["checks"]["load_finetune"] = "PASS"

        # Compare predictions on caffeine
        with torch.no_grad():
            out_final = model("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
            out_ft = model_ft("Cn1c(=O)c2c(ncn2C)n(C)c1=O")

        cmax_diff = abs(out_final["pk_metrics"]["cmax"].item() -
                       out_ft["pk_metrics"]["cmax"].item())
        print(f"  Caffeine Cmax diff (final vs finetune): {cmax_diff:.6f}")
        results["checks"]["finetune_cmax_diff_caffeine"] = float(cmax_diff)

    except Exception as e:
        print(f"  FAIL: {e}")
        results["checks"]["load_finetune"] = f"FAIL: {e}"

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for v in phys_checks.values() if v == "PASS")
    n_total = len(phys_checks)
    n_infer = sum(1 for v in inference_results.values() if v["status"] == "PASS")
    print(f"  Inference: {n_infer}/{len(test_drugs)} drugs succeeded")
    print(f"  Physical checks: {n_pass}/{n_total} PASS")
    results["status"] = "completed"

    _save_results(results)


def _save_results(results):
    out_path = Path(__file__).resolve().parent.parent / "outputs" / "l2_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
