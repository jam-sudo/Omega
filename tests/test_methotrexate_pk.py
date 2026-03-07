"""Tests for Phase 975 — Methotrexate PK model."""

import pytest
from omega_pbpk.clinical.methotrexate_pk import (
    MethotrexatePKResult,
    simulate_methotrexate_pk,
)


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    assert isinstance(result, MethotrexatePKResult)


def test_drug_name():
    result = simulate_methotrexate_pk(dose_mg=500.0)
    assert result.drug_name == "Methotrexate"


def test_cmax_positive():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    assert result.cmax_mg_L > 0.0


def test_initial_concentration_zero():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    # First element should be ~0.0 (before infusion starts)
    assert result.c_central_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_auc_positive():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    assert result.auc_mg_h_per_L > 0.0


def test_times_list_nonempty():
    result = simulate_methotrexate_pk(dose_mg=500.0)
    assert len(result.times_h) > 0


def test_concentration_lists_same_length():
    result = simulate_methotrexate_pk(dose_mg=500.0)
    assert len(result.c_central_mg_L) == len(result.times_h)
    assert len(result.c_central_umol_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def test_umol_L_conversion():
    """µmol/L should equal mg/L / 0.4544."""
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    for mg_l, umol_l in zip(result.c_central_mg_L, result.c_central_umol_L):
        assert umol_l == pytest.approx(mg_l / 0.4544, rel=1e-6)


# ---------------------------------------------------------------------------
# Renal clearance effect
# ---------------------------------------------------------------------------


def test_reduced_gfr_higher_conc():
    """Reduced GFR (30 mL/min) → higher concentrations than normal GFR (100)."""
    normal = simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=100.0)
    reduced = simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=30.0)
    assert reduced.cmax_mg_L > normal.cmax_mg_L


def test_reduced_gfr_higher_c24():
    normal = simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=100.0)
    reduced = simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=30.0)
    assert reduced.c_at_24h_umol_L > normal.c_at_24h_umol_L


def test_reduced_gfr_higher_auc():
    normal = simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=100.0)
    reduced = simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=30.0)
    assert reduced.auc_mg_h_per_L > normal.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# Leucovorin rescue
# ---------------------------------------------------------------------------


def test_leucovorin_rescue_is_bool():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    assert isinstance(result.leucovorin_rescue_required, bool)


def test_high_dose_leucovorin_rescue():
    """5000 mg dose → leucovorin rescue required."""
    result = simulate_methotrexate_pk(dose_mg=5000.0, infusion_duration_h=24.0)
    assert result.leucovorin_rescue_required is True


def test_low_dose_no_leucovorin_rescue():
    """50 mg standard weekly RA dose → no rescue needed."""
    result = simulate_methotrexate_pk(
        dose_mg=50.0,
        infusion_duration_h=0.5,
        gfr_mL_min=100.0,
        t_end_h=120.0,
    )
    assert result.leucovorin_rescue_required is False


# ---------------------------------------------------------------------------
# Toxicity risk
# ---------------------------------------------------------------------------


def test_toxicity_risk_valid_values():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    assert result.toxicity_risk in {"low", "moderate", "high"}


def test_high_dose_high_toxicity():
    result = simulate_methotrexate_pk(dose_mg=5000.0, infusion_duration_h=24.0)
    assert result.toxicity_risk == "high"


def test_low_dose_low_toxicity():
    result = simulate_methotrexate_pk(
        dose_mg=50.0, infusion_duration_h=0.5, gfr_mL_min=100.0
    )
    assert result.toxicity_risk == "low"


# ---------------------------------------------------------------------------
# Concentration at 24h and 48h are floats
# ---------------------------------------------------------------------------


def test_c_at_24h_is_float():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    assert isinstance(result.c_at_24h_umol_L, float)


def test_c_at_48h_is_float():
    result = simulate_methotrexate_pk(dose_mg=1000.0)
    assert isinstance(result.c_at_48h_umol_L, float)


def test_c_at_48h_less_than_c_at_24h():
    """Concentration should decay: 48h < 24h for standard high dose."""
    result = simulate_methotrexate_pk(dose_mg=1000.0, infusion_duration_h=24.0)
    assert result.c_at_48h_umol_L < result.c_at_24h_umol_L


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_methotrexate_pk(dose_mg=0.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_methotrexate_pk(dose_mg=-100.0)


def test_invalid_gfr_zero_raises():
    with pytest.raises(ValueError, match="gfr_mL_min"):
        simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=0.0)


def test_invalid_gfr_negative_raises():
    with pytest.raises(ValueError, match="gfr_mL_min"):
        simulate_methotrexate_pk(dose_mg=1000.0, gfr_mL_min=-10.0)


def test_invalid_infusion_duration_raises():
    with pytest.raises(ValueError, match="infusion_duration_h"):
        simulate_methotrexate_pk(dose_mg=1000.0, infusion_duration_h=0.0)
