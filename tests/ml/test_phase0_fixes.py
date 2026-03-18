# tests/ml/test_phase0_fixes.py
"""Phase 0 bug fixes: compound_type mapping and adaptive simulation."""

import pytest


@pytest.mark.slow
def test_warfarin_treated_as_acid():
    """Warfarin (enol-lactone, pKa=5.0) must be treated as 'acid' in Kp calculation."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    # Warfarin SMILES
    smiles = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
    result = pipeline.simulate(SimulationRequest(smiles=smiles, dose_mg=10.0, duration_h=24.0))
    # With correct acid Kp, Cmax should be > 0.2 mg/L (was 0.184 as neutral, observed: 1.278)
    # Acid logD correction reduces Kp → lower Vd → higher Cmax
    # Remaining gap (0.24 vs 1.28) is due to fup=0.009 and Vd issues (separate from compound_type)
    assert result.cmax_mg_L > 0.22, (
        f"Warfarin Cmax {result.cmax_mg_L:.4f} too low — compound_type likely still 'neutral'"
    )


def test_enol_lactone_in_drug_type_map():
    """_DRUG_TYPE_MAP must map 'enol_lactone' to 'acid'."""
    from omega_pbpk.core.heuristics import _DRUG_TYPE_MAP

    assert "enol_lactone" in _DRUG_TYPE_MAP, "enol_lactone missing from _DRUG_TYPE_MAP"
    assert _DRUG_TYPE_MAP["enol_lactone"] == "acid"


def test_acid_kp_lower_than_neutral():
    """Acid Kp should be much lower than neutral at same logP (logD correction)."""
    from omega_pbpk.core.heuristics import berezhkovskiy_kp

    # pKa=5.0 acid at pH 7.0: logD = logP - 2.0 → 100x reduction in lipid partition
    kp_neutral = berezhkovskiy_kp(
        logP=2.0, pka=5.0, compound_type="neutral", tissue_name="adipose", fup=0.005
    )
    kp_acid = berezhkovskiy_kp(
        logP=2.0, pka=5.0, compound_type="acid", tissue_name="adipose", fup=0.005
    )
    assert kp_acid < kp_neutral * 0.5, (
        f"Acid Kp ({kp_acid}) not sufficiently lower than neutral ({kp_neutral})"
    )
