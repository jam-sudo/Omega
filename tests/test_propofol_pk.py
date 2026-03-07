"""Tests for Phase 999 — PropofolPK (Marsh 3-compartment model)."""
import pytest

from omega_pbpk.clinical.propofol_pk import PropofolPKResult, simulate_propofol_pk


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = simulate_propofol_pk()
    assert isinstance(result, PropofolPKResult)


def test_drug_name():
    result = simulate_propofol_pk()
    assert result.drug_name == "Propofol"


def test_cmax_central_positive():
    result = simulate_propofol_pk()
    assert result.cmax_central_mg_L > 0.0


def test_cmax_effect_positive():
    result = simulate_propofol_pk()
    assert result.cmax_effect_mg_L > 0.0


def test_tmax_effect_positive():
    result = simulate_propofol_pk()
    assert result.tmax_effect_min > 0.0


def test_auc_60min_positive():
    result = simulate_propofol_pk()
    assert result.auc_0_60min_mg_min_per_L > 0.0


def test_time_to_loc_positive():
    result = simulate_propofol_pk()
    assert result.time_to_loc_min > 0.0


def test_recovery_after_loc():
    """Time to recovery of consciousness must be after loss of consciousness."""
    result = simulate_propofol_pk()
    assert result.time_to_roc_min > result.time_to_loc_min


def test_bis_in_range():
    result = simulate_propofol_pk()
    assert 0.0 <= result.bispectral_index_at_peak <= 100.0


def test_effect_site_starts_at_zero():
    result = simulate_propofol_pk()
    assert result.c_effect_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_standard_induction_loc_under_10_min():
    """Standard induction: 200 mg over 10 min in 70 kg patient → LOC < 10 min.

    Threshold Ce > 1.0 mg/L (Schnider 1999 propofol Ce50 for LOC ≈ 1.0-1.7 mg/L).
    """
    result = simulate_propofol_pk(dose_mg=200.0, weight_kg=70.0, infusion_duration_min=10.0)
    assert result.time_to_loc_min < 10.0


def test_heavier_patient_lower_cmax():
    """Heavier patient → larger Vd → lower Cmax per same dose."""
    r70 = simulate_propofol_pk(dose_mg=200.0, weight_kg=70.0)
    r100 = simulate_propofol_pk(dose_mg=200.0, weight_kg=100.0)
    assert r70.cmax_central_mg_L > r100.cmax_central_mg_L


def test_maintenance_infusion_sustains_effect():
    """With high maintenance infusion, effect-site concentration stays elevated."""
    r_no_maint = simulate_propofol_pk(
        dose_mg=200.0, weight_kg=70.0, maintenance_mg_per_h=0.0, t_end_min=120.0
    )
    r_maint = simulate_propofol_pk(
        dose_mg=200.0, weight_kg=70.0, maintenance_mg_per_h=600.0, t_end_min=120.0
    )
    # The last effect-site concentration should be higher with maintenance
    assert r_maint.c_effect_mg_L[-1] > r_no_maint.c_effect_mg_L[-1]


def test_list_lengths_match():
    result = simulate_propofol_pk(t_end_min=60.0, dt_min=0.1)
    n = len(result.times_min)
    assert len(result.c_central_mg_L) == n
    assert len(result.c_fast_mg_L) == n
    assert len(result.c_slow_mg_L) == n
    assert len(result.c_effect_mg_L) == n


def test_times_start_at_zero():
    result = simulate_propofol_pk()
    assert result.times_min[0] == pytest.approx(0.0)


def test_notes_nonempty():
    result = simulate_propofol_pk()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        simulate_propofol_pk(dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        simulate_propofol_pk(dose_mg=-100.0)


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        simulate_propofol_pk(weight_kg=0.0)


def test_invalid_infusion_duration_raises():
    with pytest.raises(ValueError):
        simulate_propofol_pk(infusion_duration_min=0.0)


# ---------------------------------------------------------------------------
# Dose scaling test
# ---------------------------------------------------------------------------


def test_higher_dose_higher_cmax():
    r_low = simulate_propofol_pk(dose_mg=100.0, weight_kg=70.0)
    r_high = simulate_propofol_pk(dose_mg=300.0, weight_kg=70.0)
    assert r_high.cmax_central_mg_L > r_low.cmax_central_mg_L
