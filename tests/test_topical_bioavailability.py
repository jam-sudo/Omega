"""Tests for Phase 1130 — topical_bioavailability.py"""

import pytest

from omega_pbpk.prediction.topical_bioavailability import (
    _FORMULATION_MODIFIERS,
    TopicalBioavailabilityResult,
    compare_formulations,
    predict_topical_bioavailability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logP=2.5,
        dose_mg=50.0,
        application_area_cm2=100.0,
        formulation_type="cream",
        application_duration_h=8.0,
        vehicle_solubility_mg_mL=5.0,
    )
    defaults.update(kwargs)
    return predict_topical_bioavailability(**defaults)


# ---------------------------------------------------------------------------
# Return type & field preservation
# ---------------------------------------------------------------------------


def test_returns_result_type():
    r = _run_default()
    assert isinstance(r, TopicalBioavailabilityResult)


def test_drug_name_preserved():
    r = _run_default(drug_name="Hydrocortisone")
    assert r.drug_name == "Hydrocortisone"


def test_formulation_type_preserved():
    r = _run_default(formulation_type="gel")
    assert r.formulation_type == "gel"


def test_dose_preserved():
    r = _run_default(dose_mg=20.0)
    assert r.dose_mg == pytest.approx(20.0)


def test_mw_preserved():
    r = _run_default(mw_Da=500.0)
    assert r.mw_Da == pytest.approx(500.0)


def test_logp_preserved():
    r = _run_default(logP=1.0)
    assert r.logP == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Kp calculation
# ---------------------------------------------------------------------------


def test_kp_base_positive():
    r = _run_default()
    assert r.kp_base_cm_per_h > 0.0


def test_kp_effective_equals_kp_base_times_modifier():
    r = _run_default(formulation_type="ointment")
    expected = r.kp_base_cm_per_h * _FORMULATION_MODIFIERS["ointment"]
    assert r.kp_effective_cm_per_h == pytest.approx(expected, rel=1e-6)


def test_kp_ordering_patch_gt_ointment():
    patch = _run_default(formulation_type="patch")
    ointment = _run_default(formulation_type="ointment")
    assert patch.kp_effective_cm_per_h > ointment.kp_effective_cm_per_h


def test_kp_ordering_ointment_gt_gel():
    ointment = _run_default(formulation_type="ointment")
    gel = _run_default(formulation_type="gel")
    assert ointment.kp_effective_cm_per_h > gel.kp_effective_cm_per_h


def test_kp_ordering_gel_gt_cream():
    gel = _run_default(formulation_type="gel")
    cream = _run_default(formulation_type="cream")
    assert gel.kp_effective_cm_per_h > cream.kp_effective_cm_per_h


def test_kp_ordering_cream_gt_lotion():
    cream = _run_default(formulation_type="cream")
    lotion = _run_default(formulation_type="lotion")
    assert cream.kp_effective_cm_per_h > lotion.kp_effective_cm_per_h


# ---------------------------------------------------------------------------
# logP effect
# ---------------------------------------------------------------------------


def test_higher_logp_higher_kp_in_moderate_range():
    low = _run_default(logP=1.0)
    high = _run_default(logP=3.0)
    assert high.kp_base_cm_per_h > low.kp_base_cm_per_h


# ---------------------------------------------------------------------------
# MW effect
# ---------------------------------------------------------------------------


def test_higher_mw_lower_kp():
    small = _run_default(mw_Da=200.0)
    large = _run_default(mw_Da=600.0)
    assert small.kp_base_cm_per_h > large.kp_base_cm_per_h


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


def test_bioavailability_between_0_and_100():
    r = _run_default()
    assert 0.0 <= r.systemic_bioavailability_pct <= 100.0


def test_bioavailability_capped_at_100():
    # Very permeable drug, long duration → should cap
    r = _run_default(logP=4.0, dose_mg=0.001, application_duration_h=100.0)
    assert r.systemic_bioavailability_pct <= 100.0


def test_flux_positive():
    r = _run_default()
    assert r.flux_mg_per_h >= 0.0


def test_total_absorbed_le_dose():
    r = _run_default()
    assert r.total_absorbed_mg <= r.dose_mg + 1e-9


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_local_class_is_valid():
    r = _run_default()
    assert r.local_tissue_concentration_class in {"high", "moderate", "low"}


def test_systemic_risk_is_valid():
    r = _run_default()
    assert r.systemic_risk in {"high", "moderate", "low"}


def test_local_class_high_when_f_above_50():
    # Craft conditions to yield >50% bioavailability
    r = _run_default(logP=4.0, mw_Da=100.0, dose_mg=0.001, application_duration_h=200.0)
    if r.systemic_bioavailability_pct > 50.0:
        assert r.local_tissue_concentration_class == "high"


def test_systemic_risk_low_when_f_below_5():
    # Low logP, low MW equivalency → minimal penetration
    r = _run_default(logP=-2.0, mw_Da=800.0)
    if r.systemic_bioavailability_pct < 5.0:
        assert r.systemic_risk == "low"


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_returns_5():
    results = compare_formulations(
        drug_name="TestDrug",
        mw_Da=300.0,
        logP=2.5,
        dose_mg=50.0,
        application_area_cm2=100.0,
        application_duration_h=8.0,
        vehicle_solubility_mg_mL=5.0,
    )
    assert len(results) == 5


def test_compare_formulations_sorted_descending():
    results = compare_formulations(
        drug_name="TestDrug",
        mw_Da=300.0,
        logP=2.5,
        dose_mg=50.0,
        application_area_cm2=100.0,
        application_duration_h=8.0,
        vehicle_solubility_mg_mL=5.0,
    )
    for i in range(len(results) - 1):
        assert (
            results[i].systemic_bioavailability_pct >= results[i + 1].systemic_bioavailability_pct
        )


def test_compare_formulations_patch_first():
    results = compare_formulations(
        drug_name="TestDrug",
        mw_Da=300.0,
        logP=2.5,
        dose_mg=50.0,
        application_area_cm2=100.0,
        application_duration_h=8.0,
        vehicle_solubility_mg_mL=5.0,
    )
    assert results[0].formulation_type == "patch"


def test_compare_formulations_lotion_last():
    results = compare_formulations(
        drug_name="TestDrug",
        mw_Da=300.0,
        logP=2.5,
        dose_mg=50.0,
        application_area_cm2=100.0,
        application_duration_h=8.0,
        vehicle_solubility_mg_mL=5.0,
    )
    assert results[-1].formulation_type == "lotion"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _run_default(mw_Da=0.0)


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _run_default(dose_mg=0.0)


def test_invalid_area_zero():
    with pytest.raises(ValueError, match="application_area"):
        _run_default(application_area_cm2=0.0)


def test_invalid_duration_zero():
    with pytest.raises(ValueError, match="application_duration"):
        _run_default(application_duration_h=0.0)


def test_invalid_vehicle_solubility_zero():
    with pytest.raises(ValueError, match="vehicle_solubility"):
        _run_default(vehicle_solubility_mg_mL=0.0)


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logP"):
        _run_default(logP=-6.0)


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logP"):
        _run_default(logP=11.0)


def test_invalid_formulation_type():
    with pytest.raises(ValueError, match="formulation_type"):
        _run_default(formulation_type="spray")
