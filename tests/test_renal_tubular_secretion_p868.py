"""Tests for Phase 868: Renal Tubular Secretion Model."""

import pytest

from omega_pbpk.core.renal_tubular_secretion_p868 import (
    RenalTubularSecretionResult,
    screen_renal_impairment,
    simulate_renal_tubular_secretion,
)

# --- Type and field checks ---


def test_returns_correct_type():
    result = simulate_renal_tubular_secretion("DrugA", 100.0)
    assert isinstance(result, RenalTubularSecretionResult)


def test_fields_preserved():
    result = simulate_renal_tubular_secretion("DrugB", 50.0, fup=0.2, logp=2.0)
    assert result.drug_name == "DrugB"
    assert result.dose_mg == pytest.approx(50.0)
    assert result.fup == pytest.approx(0.2)
    assert result.logp == pytest.approx(2.0)


def test_result_is_frozen():
    result = simulate_renal_tubular_secretion("DrugA", 100.0)
    with pytest.raises(Exception):
        result.drug_name = "other"  # type: ignore


# --- Core clearance calculations ---


def test_cl_renal_positive():
    result = simulate_renal_tubular_secretion("DrugA", 100.0, fup=0.1)
    assert result.cl_renal_L_per_h > 0.0


def test_cl_filtration_equals_gfr_times_fup():
    result = simulate_renal_tubular_secretion("DrugA", 100.0, fup=0.2, gfr_L_per_h=7.5)
    assert result.cl_filtration_L_per_h == pytest.approx(7.5 * 0.2, rel=1e-6)


def test_cl_filtration_zero_when_gfr_zero():
    result = simulate_renal_tubular_secretion("DrugA", 100.0, fup=0.1, gfr_L_per_h=0.0)
    assert result.cl_filtration_L_per_h == pytest.approx(0.0)


def test_secretion_nonzero_when_gfr_zero():
    """Active secretion should remain even when GFR is 0."""
    result = simulate_renal_tubular_secretion(
        "DrugA", 100.0, fup=0.1, gfr_L_per_h=0.0, vmax_s_mg_per_h=10.0, km_s_mg_per_L=0.5
    )
    assert result.cl_secretion_L_per_h > 0.0


def test_cl_renal_positive_when_gfr_zero():
    """When GFR=0, secretion still drives positive CLrenal (if logP low)."""
    result = simulate_renal_tubular_secretion(
        "DrugA", 100.0, fup=0.5, gfr_L_per_h=0.0, vmax_s_mg_per_h=20.0, km_s_mg_per_L=0.1, logp=0.0
    )
    assert result.cl_renal_L_per_h > 0.0


# --- Reabsorption ---


def test_high_logp_high_reabsorption():
    result = simulate_renal_tubular_secretion("DrugA", 100.0, logp=20.0)
    assert result.reabsorption_fraction >= 0.8


def test_low_logp_low_reabsorption():
    result = simulate_renal_tubular_secretion("DrugA", 100.0, logp=0.0)
    assert result.reabsorption_fraction == pytest.approx(0.0)


def test_high_logp_lower_cl_renal_than_low_logp():
    r_high = simulate_renal_tubular_secretion("Drug", 100.0, logp=10.0, fup=0.3, gfr_L_per_h=7.5)
    r_low = simulate_renal_tubular_secretion("Drug", 100.0, logp=0.0, fup=0.3, gfr_L_per_h=7.5)
    assert r_high.cl_renal_L_per_h < r_low.cl_renal_L_per_h


# --- fe_urine ---


def test_fe_urine_in_valid_range():
    result = simulate_renal_tubular_secretion("DrugA", 100.0)
    assert 0.0 < result.fe_urine < 1.0


def test_fe_urine_increases_with_cl_renal():
    r1 = simulate_renal_tubular_secretion("Drug", 100.0, fup=0.5, logp=0.0, gfr_L_per_h=7.5)
    r2 = simulate_renal_tubular_secretion("Drug", 100.0, fup=0.01, logp=5.0, gfr_L_per_h=1.0)
    # r1 should have higher fe_urine
    assert r1.fe_urine > r2.fe_urine


# --- Secretion dominance ---


def test_secretion_dominant_when_vmax_large():
    result = simulate_renal_tubular_secretion(
        "Drug", 100.0, fup=0.1, gfr_L_per_h=7.5, vmax_s_mg_per_h=1000.0, km_s_mg_per_L=0.01
    )
    assert result.secretion_dominant is True


def test_secretion_not_dominant_when_vmax_zero():
    result = simulate_renal_tubular_secretion(
        "Drug", 100.0, fup=0.5, gfr_L_per_h=7.5, vmax_s_mg_per_h=0.0, km_s_mg_per_L=0.5
    )
    assert result.secretion_dominant is False


# --- Cumulative urine ---


def test_cumulative_urine_positive():
    result = simulate_renal_tubular_secretion("DrugA", 100.0, fup=0.3, logp=0.0)
    assert result.cumulative_urine_24h_mg > 0.0


def test_cumulative_urine_less_than_dose():
    result = simulate_renal_tubular_secretion("DrugA", 100.0)
    assert result.cumulative_urine_24h_mg <= 100.0


# --- screen_renal_impairment ---


def test_screen_renal_impairment_returns_list():
    results = screen_renal_impairment("DrugA", 100.0)
    assert isinstance(results, list)
    assert len(results) == 4


def test_screen_renal_impairment_sorted_descending():
    results = screen_renal_impairment("DrugA", 100.0)
    gfrs = [r.gfr_L_per_h for r in results]
    assert gfrs == sorted(gfrs, reverse=True)


def test_screen_renal_impairment_custom_levels():
    results = screen_renal_impairment("DrugA", 100.0, gfr_levels_L_per_h=[7.5, 2.0])
    assert len(results) == 2
    assert results[0].gfr_L_per_h >= results[1].gfr_L_per_h


# --- Validation ---


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_renal_tubular_secretion("D", 0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_renal_tubular_secretion("D", -5.0)


def test_validation_fup_zero():
    with pytest.raises(ValueError, match="fup must be in"):
        simulate_renal_tubular_secretion("D", 100.0, fup=0.0)


def test_validation_fup_negative():
    with pytest.raises(ValueError, match="fup must be in"):
        simulate_renal_tubular_secretion("D", 100.0, fup=-0.1)


def test_validation_gfr_negative():
    with pytest.raises(ValueError, match="gfr_L_per_h must be >= 0"):
        simulate_renal_tubular_secretion("D", 100.0, gfr_L_per_h=-1.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        simulate_renal_tubular_secretion("D", 100.0, vd_L=0.0)
