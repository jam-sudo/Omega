#!/usr/bin/env python3
"""Demo: Pragmatic L3 — warfarin PK with patient covariates.

Shows how weight and CYP2C9 genotype affect warfarin PK predictions,
and demonstrates few-shot individual fitting.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest  # noqa: E402

WARFARIN_SMILES = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"


def main():
    pipeline = OmegaPipeline()

    scenarios = [
        {"label": "Reference (70kg)", "weight": 70.0},
        {"label": "Light patient (40kg)", "weight": 40.0},
        {"label": "Heavy patient (100kg)", "weight": 100.0},
    ]

    print("=" * 70)
    print("Omega L3 Demo: Warfarin 5mg Oral — Weight-Based Scaling")
    print("=" * 70)

    for scenario in scenarios:
        request = SimulationRequest(
            smiles=WARFARIN_SMILES,
            dose_mg=5.0,
            route="oral",
            subject_weight_kg=scenario["weight"],
        )
        result = pipeline.simulate(request)
        print(f"\n{scenario['label']}:")
        print(f"  Cmax  = {result.cmax_mg_L:.4f} mg/L")
        print(f"  AUC   = {result.auc0t_mg_h_L:.4f} mg*h/L")
        print(f"  t_half = {result.t_half_h:.1f} h")

    # Few-shot demo
    print("\n" + "=" * 70)
    print("Few-shot Individual Fitting (3 observations)")
    print("=" * 70)

    observations = [(1.0, 0.15), (4.0, 0.13), (12.0, 0.05)]
    request = SimulationRequest(smiles=WARFARIN_SMILES, dose_mg=5.0)
    fit = pipeline.fit_individual(request, observations)

    print(f"\nObservations: {observations}")
    print(f"Population CL: {fit['cl_pop']:.2f} L/h, Vd: {fit['vd_pop']:.1f} L")
    print(f"Individual CL: {fit['cl_individual']:.2f} L/h (scale={fit['cl_scale']:.3f})")
    print(f"Individual Vd: {fit['vd_individual']:.1f} L (scale={fit['vd_scale']:.3f})")
    print(f"Residual: {fit['residual']:.6f}")


if __name__ == "__main__":
    main()
