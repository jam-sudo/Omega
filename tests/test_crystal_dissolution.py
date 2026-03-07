"""Tests for biopharmaceutics/crystal_dissolution.py — Phase 887."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.crystal_dissolution import (
    CrystalDissolutionResult,
    compare_crystal_forms,
    predict_crystal_dissolution,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_basic_anhydrous():
    result = predict_crystal_dissolution("DrugA", dose_mg=100.0)
    assert isinstance(result, CrystalDissolutionResult)
    assert result.drug_name == "DrugA"
    assert result.crystal_form == "anhydrous"
    assert result.dose_mg == 100.0
    assert result.solubility_mg_mL > 0
    assert result.idr_mg_cm2_min > 0
    assert result.dissolution_rate_constant_per_min > 0


def test_amorphous_has_highest_solubility():
    r_amorphous = predict_crystal_dissolution(
        "DrugB", dose_mg=100.0, crystal_form="amorphous", logp=2.0
    )
    r_anhydrous = predict_crystal_dissolution(
        "DrugB", dose_mg=100.0, crystal_form="anhydrous", logp=2.0
    )
    assert r_amorphous.solubility_mg_mL > r_anhydrous.solubility_mg_mL


def test_amorphous_3x_solubility_vs_anhydrous():
    r_amorphous = predict_crystal_dissolution(
        "DrugC", dose_mg=100.0, crystal_form="amorphous", logp=2.0
    )
    r_anhydrous = predict_crystal_dissolution(
        "DrugC", dose_mg=100.0, crystal_form="anhydrous", logp=2.0
    )
    ratio = r_amorphous.solubility_mg_mL / r_anhydrous.solubility_mg_mL
    assert abs(ratio - 3.0) < 1e-9


def test_hydrate_lower_solubility_than_anhydrous():
    r_h = predict_crystal_dissolution("DrugD", dose_mg=100.0, crystal_form="hydrate", logp=2.0)
    r_a = predict_crystal_dissolution("DrugD", dose_mg=100.0, crystal_form="anhydrous", logp=2.0)
    assert r_h.solubility_mg_mL < r_a.solubility_mg_mL


def test_metastable_higher_solubility_than_anhydrous():
    r_m = predict_crystal_dissolution("DrugE", dose_mg=100.0, crystal_form="metastable", logp=2.0)
    r_a = predict_crystal_dissolution("DrugE", dose_mg=100.0, crystal_form="anhydrous", logp=2.0)
    assert r_m.solubility_mg_mL > r_a.solubility_mg_mL


def test_solvate_solubility():
    r_s = predict_crystal_dissolution("DrugF", dose_mg=100.0, crystal_form="solvate", logp=2.0)
    r_a = predict_crystal_dissolution("DrugF", dose_mg=100.0, crystal_form="anhydrous", logp=2.0)
    ratio = r_s.solubility_mg_mL / r_a.solubility_mg_mL
    assert abs(ratio - 1.2) < 1e-9


# ---------------------------------------------------------------------------
# IDR and rate constant
# ---------------------------------------------------------------------------


def test_idr_equals_solubility_times_005():
    r = predict_crystal_dissolution("DrugG", dose_mg=100.0)
    assert abs(r.idr_mg_cm2_min - r.solubility_mg_mL * 0.005) < 1e-12


def test_k_d_formula():
    r = predict_crystal_dissolution("DrugH", dose_mg=200.0, surface_area_cm2=50.0)
    expected_k_d = r.idr_mg_cm2_min * 50.0 / 200.0
    assert abs(r.dissolution_rate_constant_per_min - expected_k_d) < 1e-12


# ---------------------------------------------------------------------------
# Dissolution fractions and t50/t90
# ---------------------------------------------------------------------------


def test_t50_formula():
    r = predict_crystal_dissolution("DrugI", dose_mg=100.0)
    k_d = r.dissolution_rate_constant_per_min
    expected_t50 = math.log(2) / k_d
    assert abs(r.t50_dissolution_min - expected_t50) < 1e-9


def test_t90_approx_formula():
    r = predict_crystal_dissolution("DrugJ", dose_mg=100.0)
    k_d = r.dissolution_rate_constant_per_min
    expected_t90 = 2.302585 / k_d
    assert abs(r.t90_dissolution_min - expected_t90) < 1e-6


def test_fraction_at_30min():
    r = predict_crystal_dissolution("DrugK", dose_mg=100.0)
    k_d = r.dissolution_rate_constant_per_min
    expected = min(1.0, 1.0 - math.exp(-k_d * 30))
    assert abs(r.fraction_dissolved_at_30min - expected) < 1e-12


def test_fraction_at_60min():
    r = predict_crystal_dissolution("DrugL", dose_mg=100.0)
    k_d = r.dissolution_rate_constant_per_min
    expected = min(1.0, 1.0 - math.exp(-k_d * 60))
    assert abs(r.fraction_dissolved_at_60min - expected) < 1e-12


def test_f60_ge_f30():
    r = predict_crystal_dissolution("DrugM", dose_mg=100.0)
    assert r.fraction_dissolved_at_60min >= r.fraction_dissolved_at_30min


def test_f_abs_predicted_max_085():
    # When 100% dissolved, f_abs should cap at 0.85
    r = predict_crystal_dissolution(
        "DrugN", dose_mg=1.0, crystal_form="amorphous", surface_area_cm2=10000.0
    )
    assert r.f_abs_predicted <= 0.85


def test_dissolution_limited_flag():
    # High dose, low surface area, low solubility → dissolution limited
    r = predict_crystal_dissolution(
        "DrugO", dose_mg=1000.0, crystal_form="anhydrous", surface_area_cm2=1.0, logp=5.0
    )
    assert r.dissolution_limited is True


def test_not_dissolution_limited_for_amorphous_low_dose():
    # Very small dose with amorphous form should dissolve quickly
    r = predict_crystal_dissolution(
        "DrugP", dose_mg=1.0, crystal_form="amorphous", surface_area_cm2=500.0, logp=1.0
    )
    assert r.dissolution_limited is False


# ---------------------------------------------------------------------------
# Notes content
# ---------------------------------------------------------------------------


def test_amorphous_notes():
    r = predict_crystal_dissolution("DrugQ", dose_mg=100.0, crystal_form="amorphous")
    assert "instability" in r.notes.lower()


def test_dissolution_limited_notes():
    r = predict_crystal_dissolution(
        "DrugR", dose_mg=10000.0, crystal_form="anhydrous", surface_area_cm2=1.0, logp=6.0
    )
    assert r.dissolution_limited
    assert "dissolution-limited" in r.notes.lower()


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_crystal_dissolution("DrugS", dose_mg=-10.0)


def test_invalid_crystal_form():
    with pytest.raises(ValueError, match="crystal_form"):
        predict_crystal_dissolution("DrugT", dose_mg=100.0, crystal_form="invalid_form")


def test_invalid_surface_area():
    with pytest.raises(ValueError, match="surface_area_cm2"):
        predict_crystal_dissolution("DrugU", dose_mg=100.0, surface_area_cm2=0.0)


def test_invalid_volume():
    with pytest.raises(ValueError, match="volume_mL"):
        predict_crystal_dissolution("DrugV", dose_mg=100.0, volume_mL=-50.0)


# ---------------------------------------------------------------------------
# compare_crystal_forms
# ---------------------------------------------------------------------------


def test_compare_returns_five_results():
    results = compare_crystal_forms("DrugW", dose_mg=100.0)
    assert len(results) == 5


def test_compare_sorted_by_f_abs():
    results = compare_crystal_forms("DrugX", dose_mg=100.0)
    f_abs_values = [r.f_abs_predicted for r in results]
    assert f_abs_values == sorted(f_abs_values, reverse=True)


def test_compare_amorphous_first():
    results = compare_crystal_forms("DrugY", dose_mg=100.0, logp=3.0)
    # Amorphous should have highest f_abs due to 3x solubility boost
    assert results[0].crystal_form == "amorphous"


def test_compare_all_forms_present():
    results = compare_crystal_forms("DrugZ", dose_mg=100.0)
    forms = {r.crystal_form for r in results}
    assert forms == {"amorphous", "anhydrous", "hydrate", "solvate", "metastable"}


def test_custom_solubility_overrides_default():
    r = predict_crystal_dissolution("DrugAA", dose_mg=100.0, solubility_mg_mL=5.0)
    assert abs(r.solubility_mg_mL - 5.0) < 1e-12
