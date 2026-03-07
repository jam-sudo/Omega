"""Tests for Phase 239: patch_formulation_designer."""

import pytest

from omega_pbpk.prediction.patch_formulation_designer import (
    VALID_PATCH_TYPES,
    PatchDesignResult,
    compare_patch_types,
    design_patch,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
BASE_KWARGS = dict(
    drug_name=DRUG,
    target_daily_dose_mg=10.0,
    logp=2.5,
    mw=300.0,
    cl_L_per_h=5.0,
)


def make_result(**overrides):
    kw = dict(BASE_KWARGS)
    kw.update(overrides)
    return design_patch(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, PatchDesignResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.patch_area_cm2 = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Permeability
# ---------------------------------------------------------------------------


def test_permeability_positive():
    result = make_result()
    assert result.permeability_cm_h > 0


def test_higher_logp_higher_permeability():
    low = make_result(logp=1.0)
    high = make_result(logp=4.0)
    assert high.permeability_cm_h > low.permeability_cm_h


def test_higher_mw_lower_permeability():
    low_mw = make_result(mw=200.0)
    high_mw = make_result(mw=500.0)
    assert low_mw.permeability_cm_h > high_mw.permeability_cm_h


# ---------------------------------------------------------------------------
# Flux
# ---------------------------------------------------------------------------


def test_flux_positive():
    result = make_result()
    assert result.flux_mg_cm2_h > 0


def test_flux_varies_by_patch_type():
    reservoir = make_result(patch_type="reservoir")
    drug_in_adhesive = make_result(patch_type="drug_in_adhesive")
    # reservoir has higher flux_modifier (1.0) than drug_in_adhesive (0.6)
    assert reservoir.flux_mg_cm2_h > drug_in_adhesive.flux_mg_cm2_h


# ---------------------------------------------------------------------------
# Patch area
# ---------------------------------------------------------------------------


def test_patch_area_within_bounds():
    result = make_result()
    assert 1.0 <= result.patch_area_cm2 <= 100.0


def test_patch_area_clamped_max():
    # Very low permeability drug (high MW, low logP) → huge required area → clamped
    result = make_result(logp=-2.5, mw=700.0, target_daily_dose_mg=500.0)
    assert result.patch_area_cm2 == 100.0


def test_patch_area_clamped_min():
    # Very high permeability + tiny dose → area < 1 → clamped
    result = make_result(logp=7.0, mw=100.0, target_daily_dose_mg=0.001)
    assert result.patch_area_cm2 == 1.0


# ---------------------------------------------------------------------------
# Daily delivery
# ---------------------------------------------------------------------------


def test_daily_delivery_positive():
    result = make_result()
    assert result.daily_delivery_mg > 0


def test_daily_delivery_equals_flux_times_area_times_24():
    result = make_result()
    expected = result.flux_mg_cm2_h * result.patch_area_cm2 * 24.0
    assert abs(result.daily_delivery_mg - expected) < 1e-10


# ---------------------------------------------------------------------------
# Plasma concentration
# ---------------------------------------------------------------------------


def test_c_plasma_steady_positive():
    result = make_result()
    assert result.c_plasma_steady_mg_L > 0


def test_c_plasma_steady_formula():
    result = make_result(cl_L_per_h=5.0)
    expected = result.daily_delivery_mg / 24.0 / 5.0
    assert abs(result.c_plasma_steady_mg_L - expected) < 1e-10


# ---------------------------------------------------------------------------
# Therapeutic coverage
# ---------------------------------------------------------------------------


def test_therapeutic_coverage_adequate():
    # Use high logP, low MW to get meaningful permeability; use very wide range.
    # logp=5.0, mw=200 → perm~1e-4 cm/h; area clamped to 100 cm2; daily~0.2 mg
    # Css = 0.2/24/0.1 ~ 0.086 mg/L, inside [0.001, 100]
    result = design_patch(
        drug_name=DRUG,
        target_daily_dose_mg=1.0,
        logp=5.0,
        mw=200.0,
        cl_L_per_h=0.1,
        patch_type="matrix",
        therapeutic_range_mg_L=(0.001, 100.0),
    )
    assert result.therapeutic_coverage == "adequate"


def test_therapeutic_coverage_inadequate():
    # Very high CL → near-zero Css
    result = make_result(cl_L_per_h=1000.0, therapeutic_range_mg_L=(100.0, 200.0))
    assert result.therapeutic_coverage == "inadequate"


def test_therapeutic_coverage_valid_values():
    result = make_result()
    assert result.therapeutic_coverage in ("adequate", "inadequate")


# ---------------------------------------------------------------------------
# Steady-state time
# ---------------------------------------------------------------------------


def test_steady_state_time_positive():
    result = make_result()
    assert result.steady_state_time_h > 0


def test_steady_state_time_increases_with_lag():
    matrix = make_result(patch_type="matrix")  # lag=1.0
    reservoir = make_result(patch_type="reservoir")  # lag=2.0
    # reservoir has longer lag, so its steady_state_time should be longer
    assert reservoir.steady_state_time_h > matrix.steady_state_time_h


# ---------------------------------------------------------------------------
# compare_patch_types
# ---------------------------------------------------------------------------


def test_compare_returns_four():
    results = compare_patch_types(
        drug_name=DRUG,
        target_daily_dose_mg=10.0,
        logp=2.5,
        mw=300.0,
        cl_L_per_h=5.0,
    )
    assert len(results) == 4


def test_compare_all_patch_types_present():
    results = compare_patch_types(
        drug_name=DRUG,
        target_daily_dose_mg=10.0,
        logp=2.5,
        mw=300.0,
        cl_L_per_h=5.0,
    )
    types_in_result = {r.patch_type for r in results}
    assert types_in_result == set(VALID_PATCH_TYPES)


def test_compare_sorted_descending():
    results = compare_patch_types(
        drug_name=DRUG,
        target_daily_dose_mg=10.0,
        logp=2.5,
        mw=300.0,
        cl_L_per_h=5.0,
    )
    css_values = [r.c_plasma_steady_mg_L for r in results]
    assert css_values == sorted(css_values, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="target_daily_dose_mg"):
        make_result(target_daily_dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="target_daily_dose_mg"):
        make_result(target_daily_dose_mg=-5.0)


def test_validation_invalid_patch_type():
    with pytest.raises(ValueError, match="patch_type"):
        make_result(patch_type="nonexistent_type")


def test_validation_logp_out_of_range():
    with pytest.raises(ValueError, match="logp"):
        make_result(logp=10.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        make_result(cl_L_per_h=0.0)
