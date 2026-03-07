"""Tests for peritoneal PK simulation (Phase 711)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.peritoneal_pk import (
    PeritonealPKResult,
    compare_routes_ip_iv,
    simulate_peritoneal_pk,
)

# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_returns_peritoneal_pk_result():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert isinstance(result, PeritonealPKResult)


def test_route_is_ip():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert result.route == "ip"


def test_drug_name_preserved():
    result = simulate_peritoneal_pk("mycompound", dose_mg=50.0)
    assert result.drug_name == "mycompound"


def test_dose_preserved():
    result = simulate_peritoneal_pk("drug_a", dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_a_ip_initial_equals_dose():
    """a_ip_mg[0] should equal dose_mg."""
    dose = 150.0
    result = simulate_peritoneal_pk("drug_a", dose_mg=dose)
    assert result.a_ip_mg[0] == pytest.approx(dose, rel=1e-6)


def test_c_plasma_initial_is_zero():
    """c_plasma_mg_L[0] should be 0."""
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)


def test_cmax_positive():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_within_simulation_range():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0, t_end_h=24.0)
    assert 0.0 <= result.tmax_h <= 24.0


def test_t_half_positive():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert result.t_half_h > 0.0


# ---------------------------------------------------------------------------
# f_systemic_effective
# ---------------------------------------------------------------------------


def test_f_systemic_effective_formula():
    """f_effective = f_bypass + (1 - f_bypass) * fh."""
    f_bypass = 0.4
    fh = 0.7
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0, f_bypass=f_bypass, fh=fh)
    expected = f_bypass + (1.0 - f_bypass) * fh
    assert result.f_systemic_effective == pytest.approx(expected, rel=1e-9)


def test_f_systemic_effective_default():
    """Default: f_bypass=0.5, fh=0.8 -> f_eff = 0.5 + 0.5*0.8 = 0.9."""
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert result.f_systemic_effective == pytest.approx(0.9, rel=1e-9)


# ---------------------------------------------------------------------------
# Dialysis tests
# ---------------------------------------------------------------------------


def test_dialysis_flag_false_by_default():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert result.dialysis is False


def test_cl_dialysis_zero_without_dialysis():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0, dialysis=False)
    assert result.cl_dialysis_L_per_h == pytest.approx(0.0, abs=1e-10)


def test_cl_dialysis_positive_with_dialysis():
    result = simulate_peritoneal_pk(
        "drug_a", dose_mg=100.0, dialysis=True, dialysate_flow_mL_per_min=2000.0, fup=0.1
    )
    assert result.cl_dialysis_L_per_h > 0.0


def test_dialysis_reduces_auc():
    """AUC with dialysis should be lower than without dialysis."""
    result_no_dial = simulate_peritoneal_pk("drug_a", dose_mg=100.0, dialysis=False)
    result_dial = simulate_peritoneal_pk(
        "drug_a", dose_mg=100.0, dialysis=True, dialysate_flow_mL_per_min=2000.0, fup=0.5
    )
    assert result_dial.auc_mg_h_per_L < result_no_dial.auc_mg_h_per_L


def test_dialysis_flag_stored():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0, dialysis=True)
    assert result.dialysis is True


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    result1 = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    result2 = simulate_peritoneal_pk("drug_a", dose_mg=200.0)
    assert result2.cmax_mg_L == pytest.approx(2.0 * result1.cmax_mg_L, rel=1e-4)


def test_double_dose_doubles_auc():
    result1 = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    result2 = simulate_peritoneal_pk("drug_a", dose_mg=200.0)
    assert result2.auc_mg_h_per_L == pytest.approx(2.0 * result1.auc_mg_h_per_L, rel=1e-4)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_peritoneal_pk("drug_a", dose_mg=100.0)
    assert len(result.notes) > 0


def test_dialysis_notes_mention_dialysis():
    result = simulate_peritoneal_pk(
        "drug_a", dose_mg=100.0, dialysis=True, dialysate_flow_mL_per_min=2000.0, fup=0.1
    )
    assert "dialysis" in result.notes.lower() or "Dialysis" in result.notes


# ---------------------------------------------------------------------------
# compare_routes_ip_iv
# ---------------------------------------------------------------------------


def test_compare_routes_returns_dict():
    out = compare_routes_ip_iv("drug_a", dose_mg=100.0)
    assert isinstance(out, dict)


def test_compare_routes_has_ip_result():
    out = compare_routes_ip_iv("drug_a", dose_mg=100.0)
    assert "ip_result" in out
    assert isinstance(out["ip_result"], PeritonealPKResult)


def test_compare_routes_has_iv_result():
    out = compare_routes_ip_iv("drug_a", dose_mg=100.0)
    assert "iv_result" in out
    assert isinstance(out["iv_result"], dict)


def test_compare_routes_has_auc_ratio():
    out = compare_routes_ip_iv("drug_a", dose_mg=100.0)
    assert "auc_ratio" in out
    assert out["auc_ratio"] > 0.0


def test_compare_routes_has_notes():
    out = compare_routes_ip_iv("drug_a", dose_mg=100.0)
    assert "notes" in out
    assert len(out["notes"]) > 0


def test_compare_routes_iv_result_keys():
    out = compare_routes_ip_iv("drug_a", dose_mg=100.0)
    iv = out["iv_result"]
    assert "cmax_mg_L" in iv
    assert "auc_mg_h_per_L" in iv
    assert "t_half_h" in iv


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_peritoneal_pk("drug_a", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_peritoneal_pk("drug_a", dose_mg=-10.0)


def test_validation_f_bypass_ge_1_raises():
    with pytest.raises(ValueError):
        simulate_peritoneal_pk("drug_a", dose_mg=100.0, f_bypass=1.0)


def test_validation_f_bypass_zero_raises():
    with pytest.raises(ValueError):
        simulate_peritoneal_pk("drug_a", dose_mg=100.0, f_bypass=0.0)


def test_validation_fh_zero_raises():
    with pytest.raises(ValueError):
        simulate_peritoneal_pk("drug_a", dose_mg=100.0, fh=0.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError):
        simulate_peritoneal_pk("drug_a", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_fup_gt_1_raises():
    with pytest.raises(ValueError):
        simulate_peritoneal_pk("drug_a", dose_mg=100.0, fup=1.5)
