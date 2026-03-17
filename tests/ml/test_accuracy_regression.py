"""Accuracy regression tests using held-out validation drugs.

These drugs are NOT in CLint anchors or dev-set tuning.
Catches catastrophic regressions without being brittle.
"""

import numpy as np
import pytest

# 5 validation drugs (from run_l1_benchmarks.py, never used for tuning)
VALIDATION_DRUGS = {
    "atenolol": {
        "smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "dose_mg": 50,
        "obs_cmax": 0.38,
    },
    "fluconazole": {
        "smiles": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
        "dose_mg": 200,
        "obs_cmax": 7.20,
    },
    "furosemide": {
        "smiles": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
        "dose_mg": 40,
        "obs_cmax": 2.00,
    },
    "gabapentin": {
        "smiles": "OC(=O)CC1(CN)CCCCC1",
        "dose_mg": 300,
        "obs_cmax": 2.90,
    },
    "metformin": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "dose_mg": 500,
        "obs_cmax": 1.35,
    },
}


@pytest.mark.benchmark
def test_validation_cmax_aafe_below_threshold():
    """Validation set AAFE should stay below 3.0 (current ~1.87)."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    fold_errors = []

    for info in VALIDATION_DRUGS.values():
        result = pipeline.simulate(
            SimulationRequest(
                smiles=info["smiles"],
                dose_mg=info["dose_mg"],
                route="oral",
            )
        )
        fe = max(
            result.cmax_mg_L / info["obs_cmax"],
            info["obs_cmax"] / result.cmax_mg_L,
        )
        fold_errors.append(fe)

    aafe = float(np.exp(np.mean(np.log(fold_errors))))
    assert aafe < 3.0, f"Validation AAFE {aafe:.2f} exceeds threshold 3.0"


@pytest.mark.benchmark
def test_no_catastrophic_single_drug_error():
    """No single validation drug should have >10x fold error."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()

    for name, info in VALIDATION_DRUGS.items():
        result = pipeline.simulate(
            SimulationRequest(
                smiles=info["smiles"],
                dose_mg=info["dose_mg"],
                route="oral",
            )
        )
        fe = max(
            result.cmax_mg_L / info["obs_cmax"],
            info["obs_cmax"] / result.cmax_mg_L,
        )
        assert fe < 10.0, f"{name} Cmax fold error {fe:.2f}x exceeds 10x"
