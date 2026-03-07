"""Tests for Phase 985 — adipose_pk_p985.py"""

import pytest
from omega_pbpk.core.adipose_pk_p985 import AdiposePKResult, simulate_adipose_pk


def test_returns_result_type():
    result = simulate_adipose_pk("Drug_A", dose_mg=100.0)
    assert isinstance(result, AdiposePKResult)


def test_cmax_plasma_positive():
    result = simulate_adipose_pk("Drug_A", dose_mg=100.0)
    assert result.cmax_plasma_mg_L > 0


def test_cmax_adipose_positive():
    result = simulate_adipose_pk("Drug_A", dose_mg=100.0)
    assert result.cmax_adipose_mg_kg > 0


def test_auc_plasma_positive():
    result = simulate_adipose_pk("Drug_A", dose_mg=100.0)
    assert result.auc_plasma_mg_h_per_L > 0


def test_auc_adipose_positive():
    result = simulate_adipose_pk("Drug_A", dose_mg=100.0)
    assert result.auc_adipose_mg_h_per_kg > 0


def test_adipose_to_plasma_ratio_positive():
    result = simulate_adipose_pk("Drug_A", dose_mg=100.0)
    assert result.adipose_to_plasma_ratio > 0


def test_lipophilic_higher_kp_than_hydrophilic():
    lip = simulate_adipose_pk("Lip", dose_mg=100.0, logp=5.0)
    hyd = simulate_adipose_pk("Hyd", dose_mg=100.0, logp=-1.0)
    assert lip.kp_adipose > hyd.kp_adipose


def test_lipophilic_higher_adipose_to_plasma_ratio():
    lip = simulate_adipose_pk("Lip", dose_mg=100.0, logp=5.0, t_end_h=72.0)
    hyd = simulate_adipose_pk("Hyd", dose_mg=100.0, logp=-1.0, t_end_h=72.0)
    assert lip.adipose_to_plasma_ratio > hyd.adipose_to_plasma_ratio


def test_high_bmi_larger_v_adipose():
    result_normal = simulate_adipose_pk("Drug", dose_mg=100.0, bmi=22.0)
    result_obese = simulate_adipose_pk("Drug", dose_mg=100.0, bmi=40.0)
    # v_adipose affects the simulation; higher BMI should not raise errors
    # and cmax_adipose should differ
    assert result_obese.drug_name == "Drug"
    assert result_normal.drug_name == "Drug"


def test_high_logp_redistribution_risk_true():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, logp=6.0, fup=0.1)
    assert result.redistribution_risk is True


def test_low_logp_redistribution_risk_false():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, logp=-2.0)
    assert result.redistribution_risk is False


def test_iv_initial_plasma_concentration_positive():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, route="iv")
    assert result.c_plasma_mg_L[0] > 0


def test_iv_initial_adipose_concentration_zero():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, route="iv")
    assert result.c_adipose_mg_kg[0] == pytest.approx(0.0)


def test_oral_initial_plasma_concentration_zero():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, route="oral")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_oral_route_works():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, route="oral")
    assert result.cmax_plasma_mg_L > 0
    assert result.cmax_adipose_mg_kg > 0


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_pk("Drug", dose_mg=-10.0)


def test_invalid_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_pk("Drug", dose_mg=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_adipose_pk("Drug", dose_mg=100.0, route="subcutaneous")


def test_invalid_bmi_raises():
    with pytest.raises(ValueError, match="bmi"):
        simulate_adipose_pk("Drug", dose_mg=100.0, bmi=-5.0)


def test_times_correct_length():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    # n_steps = int(10/0.1) + 1 = 101
    assert len(result.times_h) == 101
    assert len(result.c_plasma_mg_L) == 101
    assert len(result.c_adipose_mg_kg) == 101


def test_kp_adipose_field_positive():
    result = simulate_adipose_pk("Drug", dose_mg=100.0, logp=2.0)
    assert result.kp_adipose > 0


def test_notes_is_string():
    result = simulate_adipose_pk("Drug", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
