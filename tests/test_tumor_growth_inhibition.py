"""Tests for tumor growth inhibition (TGI) model — Phase 849."""

import math

import pytest

from omega_pbpk.core.tumor_growth_inhibition import (
    TumorGrowthResult,
    compare_dosing_regimens,
    simulate_tumor_growth,
)

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="Paclitaxel",
    dose_mg_per_kg=10.0,
    bw_kg=0.025,
    initial_tumor_volume_mm3=200.0,
    kg_growth_per_day=0.2,
    kd_kill_per_day_per_mg_L=0.1,
    cl_L_per_h_per_kg=5.0,
    vd_L_per_kg=50.0,
    tau_h=24.0,
    n_doses=14,
    model_type="exponential",
    t_end_days=28.0,
    dt_days=0.5 / 24.0,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_tumor_growth_result():
    result = simulate_tumor_growth(**_BASE)
    assert isinstance(result, TumorGrowthResult)


def test_result_fields_present():
    result = simulate_tumor_growth(**_BASE)
    assert hasattr(result, "drug_name")
    assert hasattr(result, "model_type")
    assert hasattr(result, "times_days")
    assert hasattr(result, "tumor_volume_mm3")
    assert hasattr(result, "drug_conc_mg_L")
    assert hasattr(result, "tumor_growth_inhibition_pct")
    assert hasattr(result, "doubling_time_days")
    assert hasattr(result, "complete_response")


# ---------------------------------------------------------------------------
# Time course consistency
# ---------------------------------------------------------------------------


def test_times_and_volumes_same_length():
    result = simulate_tumor_growth(**_BASE)
    assert len(result.times_days) == len(result.tumor_volume_mm3)


def test_times_and_concs_same_length():
    result = simulate_tumor_growth(**_BASE)
    assert len(result.times_days) == len(result.drug_conc_mg_L)


def test_times_start_at_zero():
    result = simulate_tumor_growth(**_BASE)
    assert result.times_days[0] == pytest.approx(0.0)


def test_final_volume_matches_last_element():
    result = simulate_tumor_growth(**_BASE)
    assert result.final_tumor_volume_mm3 == pytest.approx(result.tumor_volume_mm3[-1])


# ---------------------------------------------------------------------------
# No-drug baseline
# ---------------------------------------------------------------------------


def test_no_drug_tgi_is_zero():
    """dose_mg_per_kg=0 is invalid — use very tiny kd to approximate no drug effect."""
    # Use zero kd instead
    result = simulate_tumor_growth(**{**_BASE, "kd_kill_per_day_per_mg_L": 0.0})
    # With kd=0, no inhibition → TGI should be ~0
    assert abs(result.tumor_growth_inhibition_pct) < 1.0


def test_tumor_grows_without_drug_effect():
    """With kd=0, tumor should grow from initial volume."""
    result = simulate_tumor_growth(**{**_BASE, "kd_kill_per_day_per_mg_L": 0.0})
    assert result.final_tumor_volume_mm3 > result.initial_tumor_volume_mm3


# ---------------------------------------------------------------------------
# Drug reduces tumor vs control
# ---------------------------------------------------------------------------


def test_drug_reduces_final_volume():
    """High kd should reduce final volume vs very low kd (≈ control)."""
    result_drug = simulate_tumor_growth(**_BASE)
    result_ctrl = simulate_tumor_growth(**{**_BASE, "kd_kill_per_day_per_mg_L": 1e-9})
    assert result_drug.final_tumor_volume_mm3 < result_ctrl.final_tumor_volume_mm3


def test_tgi_positive_with_drug():
    result = simulate_tumor_growth(**_BASE)
    assert result.tumor_growth_inhibition_pct > 0.0


def test_tgi_bounded_approximately():
    """TGI can exceed 100 (tumor shrinks below initial) but should not be wildly negative."""
    result = simulate_tumor_growth(**_BASE)
    assert result.tumor_growth_inhibition_pct > -50.0


# ---------------------------------------------------------------------------
# Doubling time
# ---------------------------------------------------------------------------


def test_doubling_time_formula():
    kg = 0.2
    result = simulate_tumor_growth(**{**_BASE, "kg_growth_per_day": kg})
    expected = math.log(2.0) / kg
    assert result.doubling_time_days == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Complete response
# ---------------------------------------------------------------------------


def test_complete_response_high_dose():
    """Very high kill rate should produce complete response."""
    result = simulate_tumor_growth(**{**_BASE, "kd_kill_per_day_per_mg_L": 5.0})
    assert result.complete_response is True


def test_complete_response_false_low_dose():
    """Minimal kill rate should not produce complete response."""
    result = simulate_tumor_growth(**{**_BASE, "kd_kill_per_day_per_mg_L": 0.001})
    assert result.complete_response is False


# ---------------------------------------------------------------------------
# Model types
# ---------------------------------------------------------------------------


def test_logistic_model_runs():
    result = simulate_tumor_growth(**{**_BASE, "model_type": "logistic"})
    assert isinstance(result, TumorGrowthResult)
    assert len(result.times_days) > 0


def test_gompertz_model_runs():
    result = simulate_tumor_growth(**{**_BASE, "model_type": "gompertz"})
    assert isinstance(result, TumorGrowthResult)
    assert len(result.times_days) > 0


def test_simeoni_model_runs():
    result = simulate_tumor_growth(**{**_BASE, "model_type": "simeoni"})
    assert isinstance(result, TumorGrowthResult)
    assert len(result.times_days) > 0


# ---------------------------------------------------------------------------
# compare_dosing_regimens
# ---------------------------------------------------------------------------


def test_compare_regimens_returns_list():
    regimens = [
        {"dose_mg_per_kg": 5.0, "tau_h": 24.0, "n_doses": 14},
        {"dose_mg_per_kg": 10.0, "tau_h": 24.0, "n_doses": 14},
        {"dose_mg_per_kg": 20.0, "tau_h": 24.0, "n_doses": 14},
    ]
    results = compare_dosing_regimens(
        drug_name="Paclitaxel",
        bw_kg=0.025,
        initial_tumor_volume_mm3=200.0,
        kg=0.2,
        kd=0.1,
        cl=5.0,
        vd=50.0,
        regimens=regimens,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_regimens_sorted_desc():
    regimens = [
        {"dose_mg_per_kg": 5.0, "tau_h": 24.0, "n_doses": 14},
        {"dose_mg_per_kg": 20.0, "tau_h": 24.0, "n_doses": 14},
        {"dose_mg_per_kg": 10.0, "tau_h": 24.0, "n_doses": 14},
    ]
    results = compare_dosing_regimens(
        drug_name="Paclitaxel",
        bw_kg=0.025,
        initial_tumor_volume_mm3=200.0,
        kg=0.2,
        kd=0.1,
        cl=5.0,
        vd=50.0,
        regimens=regimens,
    )
    tgi_values = [r.tumor_growth_inhibition_pct for r in results]
    assert tgi_values == sorted(tgi_values, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        simulate_tumor_growth(**{**_BASE, "dose_mg_per_kg": -1.0})


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg_per_kg"):
        simulate_tumor_growth(**{**_BASE, "dose_mg_per_kg": 0.0})


def test_invalid_initial_volume_raises():
    with pytest.raises(ValueError, match="initial_tumor_volume_mm3"):
        simulate_tumor_growth(**{**_BASE, "initial_tumor_volume_mm3": -100.0})


def test_invalid_kg_raises():
    with pytest.raises(ValueError, match="kg_growth_per_day"):
        simulate_tumor_growth(**{**_BASE, "kg_growth_per_day": 0.0})


def test_invalid_n_doses_raises():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_tumor_growth(**{**_BASE, "n_doses": 0})


def test_invalid_model_type_raises():
    with pytest.raises(ValueError, match="model_type"):
        simulate_tumor_growth(**{**_BASE, "model_type": "nonexistent_model"})
