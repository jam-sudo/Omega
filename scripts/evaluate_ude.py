#!/usr/bin/env python3
"""Evaluate UDE model on permanent holdout set.

Compares to pipeline baseline AAFE 3.520 [2.57, 5.00].

Usage:
    python scripts/evaluate_ude.py
    python scripts/evaluate_ude.py --model models/ude/multitask_pk_phase1.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.ml.models.ude.data import _smiles_to_features  # noqa: E402
from omega_pbpk.ml.models.ude.model import MultiTaskPKModel  # noqa: E402


def bootstrap_aafe_ci(
    fold_errors: list[float], n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    boots = [float(10 ** np.mean(np.abs(log_fe[rng.integers(0, n, n)]))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(model_path: str | None = None):
    # Load model
    if model_path is None:
        model_path = repo_root / "models" / "ude" / "multitask_pk_phase1.pt"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    model = MultiTaskPKModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load holdout
    with open(repo_root / "data" / "clinical" / "holdout_split.json") as f:
        split = json.load(f)
    with open(repo_root / "data" / "clinical" / "platinum_reference.json") as f:
        plat = json.load(f)

    holdout_drugs = split["holdout"]
    results = []
    fold_errors = []

    print(f"\nEvaluating on {len(holdout_drugs)} holdout drugs")
    print("=" * 70)

    for drug_name in sorted(holdout_drugs):
        entry = plat["drugs"].get(drug_name)
        if entry is None:
            continue

        features = _smiles_to_features(entry["smiles"])
        if features is None:
            print(f"  SKIP {drug_name}: features failed")
            results.append({"drug": drug_name, "success": False, "error": "features"})
            continue

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        dose = torch.tensor([entry["dose_mg"]], dtype=torch.float32)

        with torch.no_grad():
            cmax_pred, params = model.predict_cmax(x, dose)

        pred = cmax_pred.item()
        obs = entry["cmax_mg_L"]

        if pred > 0 and obs > 0:
            fe = max(pred / obs, obs / pred)
        else:
            fe = float("nan")

        if not np.isnan(fe):
            fold_errors.append(fe)

        results.append(
            {
                "drug": drug_name,
                "success": True,
                "pred_cmax": round(pred, 6),
                "obs_cmax": round(obs, 6),
                "fold_error": round(fe, 4) if not np.isnan(fe) else None,
                "F": round(params["F"].item(), 4),
                "Vd_L": round(params["Vd"].item(), 2),
                "ka_per_h": round(params["ka"].item(), 4),
                "ke_per_h": round(params["ke"].item(), 4),
            }
        )

        symbol = "ok" if fe <= 2.0 else ("~" if fe <= 3.0 else "X")
        print(
            f"  {symbol:2s} {drug_name:25s} FE={fe:7.2f}x  "
            f"pred={pred:.4f}  obs={obs:.4f}  "
            f"F={params['F'].item():.2f} Vd={params['Vd'].item():.0f}L"
        )

    # Metrics
    valid_fe = [fe for fe in fold_errors if not np.isnan(fe)]
    if not valid_fe:
        print("No valid predictions!")
        return

    log_fe = np.log10(valid_fe)
    aafe = float(10 ** np.mean(np.abs(log_fe)))
    pct_2fold = sum(1 for fe in valid_fe if fe <= 2.0) / len(valid_fe) * 100
    pct_3fold = sum(1 for fe in valid_fe if fe <= 3.0) / len(valid_fe) * 100
    ci_lo, ci_hi = bootstrap_aafe_ci(valid_fe)

    # Pipeline baseline
    baseline_aafe = 3.520
    improvement = (baseline_aafe - aafe) / baseline_aafe * 100

    # Top errors
    sorted_results = sorted(
        [r for r in results if r.get("success") and r.get("fold_error")],
        key=lambda r: -r["fold_error"],
    )

    output = {
        "model": str(model_path.name),
        "phase": 1,
        "n_holdout": len(holdout_drugs),
        "n_evaluated": len(valid_fe),
        "aafe": round(aafe, 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
        "pct_2fold": round(pct_2fold, 1),
        "pct_3fold": round(pct_3fold, 1),
        "baseline_aafe": baseline_aafe,
        "improvement_pct": round(improvement, 1),
        "top_10_errors": [
            {"drug": r["drug"], "fold_error": r["fold_error"]} for r in sorted_results[:10]
        ],
        "per_drug": results,
    }

    print(f"\n{'=' * 70}")
    print(f"UDE Phase 1:  AAFE = {aafe:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"              %2-fold = {pct_2fold:.1f}%  %3-fold = {pct_3fold:.1f}%")
    print(f"Baseline:     AAFE = {baseline_aafe:.3f} [2.567, 4.997]")
    print(f"Improvement:  {improvement:+.1f}%")
    print("\nTop 5 errors:")
    for r in sorted_results[:5]:
        print(f"  {r['drug']:25s} FE={r['fold_error']:.1f}x")

    if aafe < baseline_aafe:
        print(f"\n>>> UDE BEATS BASELINE by {improvement:.1f}%")
    else:
        print(f"\n>>> UDE does NOT beat baseline (delta: {improvement:+.1f}%)")

    out_path = repo_root / "outputs" / "ude_phase1_holdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    main(model_path=args.model)
