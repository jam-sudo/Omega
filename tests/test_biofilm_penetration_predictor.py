"""
Tests for Phase 283 — Biofilm Penetration Predictor.
"""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.biofilm_penetration_predictor import (
    BiofilmPenetrationResult,
    predict_biofilm_penetration,
    screen_antibiotics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(**kwargs) -> BiofilmPenetrationResult:
    defaults = dict(
        drug_name="TestDrug",
        mw=350.0,
        logp=1.5,
        charge=0.0,
        bulk_conc_mg_L=4.0,
        biofilm_thickness_um=100.0,
        diffusion_reduction_factor=0.3,
        consumption_rate_per_h=0.1,
    )
    defaults.update(kwargs)
    return predict_biofilm_penetration(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, BiofilmPenetrationResult)


def test_field_preservation():
    result = make_result(
        drug_name="Ciprofloxacin",
        mw=331.3,
        logp=0.28,
        charge=0.0,
        bulk_conc_mg_L=2.0,
        biofilm_thickness_um=80.0,
        diffusion_reduction_factor=0.25,
        consumption_rate_per_h=0.05,
    )
    assert result.drug_name == "Ciprofloxacin"
    assert result.mw == 331.3
    assert result.logp == 0.28
    assert result.charge == 0.0
    assert result.bulk_conc_mg_L == 2.0
    assert result.biofilm_thickness_um == 80.0
    assert result.diffusion_reduction_factor == 0.25
    assert result.consumption_rate_per_h == 0.05


# ---------------------------------------------------------------------------
# Penetration fraction in [0, 1]
# ---------------------------------------------------------------------------


def test_penetration_fraction_in_range():
    result = make_result()
    assert 0.0 <= result.penetration_fraction <= 1.0


def test_penetration_fraction_zero_consumption():
    result = make_result(consumption_rate_per_h=0.0)
    # No consumption → phi=0 → 1/cosh(0)=1, only MW/charge penalties
    assert 0.0 <= result.penetration_fraction <= 1.0


def test_penetration_fraction_high_consumption():
    result = make_result(consumption_rate_per_h=100.0)
    assert 0.0 <= result.penetration_fraction <= 1.0


# ---------------------------------------------------------------------------
# Thicker biofilm → lower penetration
# ---------------------------------------------------------------------------


def test_thicker_biofilm_lower_penetration():
    thin = make_result(biofilm_thickness_um=50.0, consumption_rate_per_h=1.0)
    thick = make_result(biofilm_thickness_um=500.0, consumption_rate_per_h=1.0)
    assert thick.penetration_fraction < thin.penetration_fraction


# ---------------------------------------------------------------------------
# Higher consumption → lower penetration
# ---------------------------------------------------------------------------


def test_higher_consumption_lower_penetration():
    low = make_result(consumption_rate_per_h=0.01)
    high = make_result(consumption_rate_per_h=10.0)
    assert high.penetration_fraction < low.penetration_fraction


# ---------------------------------------------------------------------------
# Thiele modulus non-negative
# ---------------------------------------------------------------------------


def test_thiele_modulus_non_negative():
    result = make_result()
    assert result.thiele_modulus >= 0.0


def test_thiele_modulus_zero_when_no_consumption():
    result = make_result(consumption_rate_per_h=0.0)
    assert result.thiele_modulus == 0.0


# ---------------------------------------------------------------------------
# MW penalty
# ---------------------------------------------------------------------------


def test_large_mw_lower_penetration():
    small = make_result(mw=150.0, charge=0.0)
    large = make_result(mw=800.0, charge=0.0)
    assert large.penetration_fraction < small.penetration_fraction


def test_mw_factor_no_penalty_below_300():
    result = make_result(mw=200.0)
    assert result.mw_factor == pytest.approx(1.0, abs=1e-9)


def test_mw_factor_penalty_above_300():
    result = make_result(mw=800.0)
    assert result.mw_factor < 1.0


# ---------------------------------------------------------------------------
# Charge penalty
# ---------------------------------------------------------------------------


def test_high_charge_lower_penetration():
    neutral = make_result(charge=0.0)
    charged = make_result(charge=2.0)
    assert charged.penetration_fraction < neutral.penetration_fraction


def test_charge_factor_neutral():
    result = make_result(charge=1.0)
    assert result.charge_factor == pytest.approx(1.0)


def test_charge_factor_high_charge():
    result = make_result(charge=2.0)
    assert result.charge_factor == pytest.approx(0.7)


def test_charge_factor_high_negative_charge():
    result = make_result(charge=-3.0)
    assert result.charge_factor == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Effective concentration
# ---------------------------------------------------------------------------


def test_c_effective_proportional_to_bulk():
    result = make_result(bulk_conc_mg_L=8.0)
    result2 = make_result(bulk_conc_mg_L=4.0)
    # c_effective / bulk_conc should be same (same penetration_fraction)
    assert result.c_effective_mg_L == pytest.approx(result2.c_effective_mg_L * 2, rel=1e-9)


def test_c_effective_non_negative():
    result = make_result()
    assert result.c_effective_mg_L >= 0.0


# ---------------------------------------------------------------------------
# Penetration class
# ---------------------------------------------------------------------------


def test_penetration_class_good():
    # No consumption, small MW, no charge → high penetration
    result = make_result(
        mw=150.0, charge=0.0, consumption_rate_per_h=0.0, diffusion_reduction_factor=0.5
    )
    assert result.penetration_class == "good"


def test_penetration_class_poor():
    result = make_result(
        mw=900.0,
        charge=3.0,
        consumption_rate_per_h=50.0,
        biofilm_thickness_um=500.0,
        diffusion_reduction_factor=0.1,
    )
    assert result.penetration_class == "poor"


def test_penetration_class_valid_values():
    result = make_result()
    assert result.penetration_class in ("good", "moderate", "poor")


# ---------------------------------------------------------------------------
# screen_antibiotics
# ---------------------------------------------------------------------------


def make_drug_dict(drug_name, mw=300.0, logp=1.0, charge=0.0, **kwargs):
    defaults = dict(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        charge=charge,
        bulk_conc_mg_L=4.0,
        biofilm_thickness_um=100.0,
        diffusion_reduction_factor=0.3,
        consumption_rate_per_h=0.1,
    )
    defaults.update(kwargs)
    return defaults


def test_screen_antibiotics_sorted_descending():
    drugs = [
        make_drug_dict("DrugA", mw=150.0, charge=0.0, consumption_rate_per_h=0.01),
        make_drug_dict("DrugB", mw=600.0, charge=2.0, consumption_rate_per_h=5.0),
        make_drug_dict("DrugC", mw=300.0, charge=0.0, consumption_rate_per_h=0.5),
    ]
    results = screen_antibiotics(drugs)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].penetration_fraction >= results[i + 1].penetration_fraction


def test_screen_antibiotics_return_type():
    drugs = [make_drug_dict("DrugX")]
    results = screen_antibiotics(drugs)
    assert isinstance(results, list)
    assert isinstance(results[0], BiofilmPenetrationResult)


def test_screen_antibiotics_empty():
    results = screen_antibiotics([])
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        make_result(mw=0.0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        make_result(mw=-100.0)


def test_invalid_bulk_conc_zero():
    with pytest.raises(ValueError, match="bulk_conc_mg_L"):
        make_result(bulk_conc_mg_L=0.0)


def test_invalid_biofilm_thickness_zero():
    with pytest.raises(ValueError, match="biofilm_thickness_um"):
        make_result(biofilm_thickness_um=0.0)


def test_invalid_diffusion_factor_zero():
    with pytest.raises(ValueError, match="diffusion_reduction_factor"):
        make_result(diffusion_reduction_factor=0.0)


def test_invalid_diffusion_factor_above_one():
    with pytest.raises(ValueError, match="diffusion_reduction_factor"):
        make_result(diffusion_reduction_factor=1.5)


def test_invalid_consumption_rate_negative():
    with pytest.raises(ValueError, match="consumption_rate_per_h"):
        make_result(consumption_rate_per_h=-0.1)
