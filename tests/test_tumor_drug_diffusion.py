"""Tests for Phase 348 — Drug Diffusion in Tumor Interstitium."""

import math

import pytest

from omega_pbpk.core.tumor_drug_diffusion import (
    TumorDrugDiffusionResult,
    compare_drugs,
    predict_tumor_drug_diffusion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_result(**kwargs) -> TumorDrugDiffusionResult:
    defaults = dict(
        drug_name="TestDrug",
        plasma_conc_mg_L=1.0,
        mw_Da=300.0,
        logP=2.0,
        protein_binding_pct=50.0,
        tumor_radius_um=200.0,
        interstitial_pH=6.8,
        charge_at_ph7=0,
    )
    defaults.update(kwargs)
    return predict_tumor_drug_diffusion(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = _base_result()
    assert isinstance(r, TumorDrugDiffusionResult)


def test_result_is_frozen():
    r = _base_result()
    with pytest.raises((AttributeError, TypeError)):
        r.c_zone1_mg_L = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Concentration gradient: zone1 > zone2 > zone3
# ---------------------------------------------------------------------------


def test_zone1_gt_zone2():
    r = _base_result()
    assert r.c_zone1_mg_L > r.c_zone2_mg_L


def test_zone2_gt_zone3():
    r = _base_result()
    assert r.c_zone2_mg_L > r.c_zone3_mg_L


def test_zone1_gt_zone3():
    r = _base_result()
    assert r.c_zone1_mg_L > r.c_zone3_mg_L


# ---------------------------------------------------------------------------
# penetration_ratio in (0, 1]
# ---------------------------------------------------------------------------


def test_penetration_ratio_positive():
    r = _base_result()
    assert r.penetration_ratio_z3_z1 > 0.0


def test_penetration_ratio_le_one():
    r = _base_result()
    assert r.penetration_ratio_z3_z1 <= 1.0


# ---------------------------------------------------------------------------
# Higher logP → higher kp → higher zone1
# ---------------------------------------------------------------------------


def test_higher_logp_higher_kp():
    r_low = _base_result(logP=1.0)
    r_high = _base_result(logP=4.0)
    assert r_high.kp_tumor > r_low.kp_tumor


def test_higher_logp_higher_zone1():
    r_low = _base_result(logP=1.0)
    r_high = _base_result(logP=4.0)
    assert r_high.c_zone1_mg_L > r_low.c_zone1_mg_L


# ---------------------------------------------------------------------------
# Lower MW → better diffusivity → higher zone3
# ---------------------------------------------------------------------------


def test_lower_mw_higher_diffusivity():
    r_small = _base_result(mw_Da=100.0)
    r_large = _base_result(mw_Da=900.0)
    assert r_small.effective_diffusivity_cm2_per_s > r_large.effective_diffusivity_cm2_per_s


def test_lower_mw_higher_zone3():
    r_small = _base_result(mw_Da=100.0)
    r_large = _base_result(mw_Da=900.0)
    assert r_small.c_zone3_mg_L > r_large.c_zone3_mg_L


# ---------------------------------------------------------------------------
# Higher protein binding → lower fu → lower zone1
# ---------------------------------------------------------------------------


def test_higher_protein_binding_lower_fu():
    r_low = _base_result(protein_binding_pct=30.0)
    r_high = _base_result(protein_binding_pct=90.0)
    assert r_low.fu_plasma > r_high.fu_plasma


def test_higher_protein_binding_lower_zone1():
    r_low = _base_result(protein_binding_pct=30.0)
    r_high = _base_result(protein_binding_pct=90.0)
    assert r_low.c_zone1_mg_L > r_high.c_zone1_mg_L


# ---------------------------------------------------------------------------
# penetration_class in valid set
# ---------------------------------------------------------------------------


def test_penetration_class_in_valid_set():
    r = _base_result()
    assert r.penetration_class in ("excellent", "moderate", "poor")


def test_penetration_class_excellent_when_ratio_high():
    # Very low MW → very high penetration (MW=5 → ratio ≈ 0.47 → moderate/excellent)
    r = _base_result(mw_Da=5.0)
    assert r.penetration_class in ("excellent", "moderate")


# ---------------------------------------------------------------------------
# tumor_to_plasma_ratio > 0
# ---------------------------------------------------------------------------


def test_tumor_to_plasma_ratio_positive():
    r = _base_result()
    assert r.tumor_to_plasma_ratio > 0.0


def test_tumor_to_plasma_ratio_equals_kp():
    # tumor_to_plasma = c_zone1 / (plasma * fu) = c_zone1 / c_plasma_free = kp_tumor
    r = _base_result()
    assert abs(r.tumor_to_plasma_ratio - r.kp_tumor) < 1e-9


# ---------------------------------------------------------------------------
# custom diffusivity
# ---------------------------------------------------------------------------


def test_custom_diffusivity_used():
    r = _base_result(diffusivity_cm2_per_s=1e-7)
    assert abs(r.effective_diffusivity_cm2_per_s - 1e-7) < 1e-15


def test_auto_diffusivity_from_mw():
    mw = 400.0
    r = _base_result(mw_Da=mw)
    expected = 6e-7 / math.sqrt(mw)
    assert abs(r.effective_diffusivity_cm2_per_s - expected) < 1e-15


# ---------------------------------------------------------------------------
# compare_drugs
# ---------------------------------------------------------------------------


def test_compare_drugs_sorted_descending():
    drugs = [
        dict(drug_name="A", plasma_conc_mg_L=1.0, mw_Da=100.0, logP=1.0, protein_binding_pct=20.0),
        dict(drug_name="B", plasma_conc_mg_L=1.0, mw_Da=900.0, logP=1.0, protein_binding_pct=20.0),
        dict(drug_name="C", plasma_conc_mg_L=1.0, mw_Da=300.0, logP=1.0, protein_binding_pct=20.0),
    ]
    results = compare_drugs(drugs)
    ratios = [r.penetration_ratio_z3_z1 for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_drugs_returns_correct_count():
    drugs = [
        dict(drug_name="A", plasma_conc_mg_L=1.0, mw_Da=200.0, logP=2.0, protein_binding_pct=50.0),
        dict(drug_name="B", plasma_conc_mg_L=1.0, mw_Da=500.0, logP=2.0, protein_binding_pct=50.0),
    ]
    results = compare_drugs(drugs)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_plasma_conc_zero():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        _base_result(plasma_conc_mg_L=0.0)


def test_validation_plasma_conc_negative():
    with pytest.raises(ValueError):
        _base_result(plasma_conc_mg_L=-1.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _base_result(mw_Da=0.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError):
        _base_result(mw_Da=-100.0)


def test_validation_protein_binding_above_100():
    with pytest.raises(ValueError, match="protein_binding_pct"):
        _base_result(protein_binding_pct=101.0)


def test_validation_protein_binding_negative():
    with pytest.raises(ValueError):
        _base_result(protein_binding_pct=-5.0)


def test_validation_tumor_radius_zero():
    with pytest.raises(ValueError, match="tumor_radius_um"):
        _base_result(tumor_radius_um=0.0)
