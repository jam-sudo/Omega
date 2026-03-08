"""Tests for nasal absorption PK simulation (Phase 1125)."""

import pytest

from omega_pbpk.core.nasal_absorption_pk import (
    NasalAbsorptionResult,
    _k_abs_nasal_from_logp,
    simulate_nasal_absorption_pk,
)

# ---------------------------------------------------------------------------
# k_abs_nasal helper
# ---------------------------------------------------------------------------


def test_k_abs_nasal_hydrophilic():
    """logP < 0 => 0.4 * 1.2 = 0.48."""
    k = _k_abs_nasal_from_logp(-1.0)
    assert abs(k - 0.48) < 1e-9


def test_k_abs_nasal_optimal():
    """0 <= logP <= 3 => 1.0 * 1.2 = 1.2."""
    assert abs(_k_abs_nasal_from_logp(0.0) - 1.2) < 1e-9
    assert abs(_k_abs_nasal_from_logp(1.5) - 1.2) < 1e-9
    assert abs(_k_abs_nasal_from_logp(3.0) - 1.2) < 1e-9


def test_k_abs_nasal_lipophilic():
    """logP > 3 => 0.7 * 1.2 = 0.84."""
    assert abs(_k_abs_nasal_from_logp(4.0) - 0.84) < 1e-9
    assert abs(_k_abs_nasal_from_logp(5.0) - 0.84) < 1e-9


# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_result_type():
    result = simulate_nasal_absorption_pk("TestDrug", 10.0, logP=1.0)
    assert isinstance(result, NasalAbsorptionResult)


def test_result_has_correct_drug_name():
    result = simulate_nasal_absorption_pk("Sumatriptan", 20.0, logP=1.5)
    assert result.drug_name == "Sumatriptan"


def test_result_has_correct_dose():
    result = simulate_nasal_absorption_pk("Drug", 15.0, logP=2.0)
    assert result.dose_mg == 15.0


def test_result_has_correct_logp():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=2.5)
    assert result.logP == 2.5


# ---------------------------------------------------------------------------
# Initial condition: C_plasma starts at 0
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_zero():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0)
    assert result.c_plasma_mg_L[0] == 0.0


def test_times_starts_at_zero():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0)
    assert result.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# PK shape: Cmax > 0, AUC > 0
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_in_valid_range():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0)
    assert 0.0 <= result.tmax_h <= 8.0


def test_bioavailability_positive():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0)
    assert result.f_nasal_bioavailability_pct > 0.0


# ---------------------------------------------------------------------------
# logP effect on Cmax
# ---------------------------------------------------------------------------


def test_optimal_logp_gives_higher_cmax_than_high_logp():
    """logP in [0, 3] (1.5) should give higher Cmax than logP > 3 (5.0)."""
    r_optimal = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5)
    r_lipophilic = simulate_nasal_absorption_pk("Drug", 10.0, logP=5.0)
    assert r_optimal.cmax_mg_L > r_lipophilic.cmax_mg_L


def test_optimal_logp_gives_higher_cmax_than_hydrophilic_logp():
    """logP in [0, 3] should give higher Cmax than logP < 0."""
    r_optimal = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5)
    r_hydrophilic = simulate_nasal_absorption_pk("Drug", 10.0, logP=-1.0)
    assert r_optimal.cmax_mg_L > r_hydrophilic.cmax_mg_L


def test_optimal_logp_gives_higher_auc_than_high_logp():
    r_optimal = simulate_nasal_absorption_pk("Drug", 10.0, logP=2.0)
    r_lipophilic = simulate_nasal_absorption_pk("Drug", 10.0, logP=4.0)
    assert r_optimal.auc_mg_h_per_L > r_lipophilic.auc_mg_h_per_L


# ---------------------------------------------------------------------------
# f_nasal_deposit effect
# ---------------------------------------------------------------------------


def test_higher_f_nasal_deposit_gives_higher_cmax():
    r_high = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5, f_nasal_deposit=0.9)
    r_low = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5, f_nasal_deposit=0.3)
    assert r_high.cmax_mg_L > r_low.cmax_mg_L


def test_higher_f_nasal_deposit_gives_higher_auc():
    r_high = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5, f_nasal_deposit=0.9)
    r_low = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5, f_nasal_deposit=0.3)
    assert r_high.auc_mg_h_per_L > r_low.auc_mg_h_per_L


def test_higher_dose_gives_proportionally_higher_auc():
    r_10 = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5)
    r_20 = simulate_nasal_absorption_pk("Drug", 20.0, logP=1.5)
    ratio = r_20.auc_mg_h_per_L / r_10.auc_mg_h_per_L
    assert abs(ratio - 2.0) < 0.01  # linear dose proportionality


# ---------------------------------------------------------------------------
# First-pass bypass advantage
# ---------------------------------------------------------------------------


def test_first_pass_bypass_advantage_is_bool():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5)
    assert isinstance(result.first_pass_bypass_advantage, bool)


def test_nasal_vs_oral_advantage_high_bioavailability():
    """With high nasal deposit and good logP, should exceed 0.6 oral fraction."""
    r = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5, cl_L_per_h=5.0, f_nasal_deposit=0.9)
    assert r.first_pass_bypass_advantage is True


# ---------------------------------------------------------------------------
# Bioavailability percentage
# ---------------------------------------------------------------------------


def test_bioavailability_pct_range():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5)
    # AUC cannot exceed theoretical IV AUC but should be positive
    assert 0.0 < result.f_nasal_bioavailability_pct <= 110.0


# ---------------------------------------------------------------------------
# Time series lengths
# ---------------------------------------------------------------------------


def test_time_series_same_length():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5, t_end_h=4.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_time_series_non_empty():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5)
    assert len(result.times_h) > 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nasal_absorption_pk("Drug", 0.0, logP=1.0)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nasal_absorption_pk("Drug", -5.0, logP=1.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0, cl_L_per_h=-1.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0, vd_L=0.0)


def test_invalid_logp_too_low_raises():
    with pytest.raises(ValueError, match="logP"):
        simulate_nasal_absorption_pk("Drug", 10.0, logP=-6.0)


def test_invalid_logp_too_high_raises():
    with pytest.raises(ValueError, match="logP"):
        simulate_nasal_absorption_pk("Drug", 10.0, logP=11.0)


def test_invalid_f_nasal_deposit_zero_raises():
    with pytest.raises(ValueError, match="f_nasal_deposit"):
        simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0, f_nasal_deposit=0.0)


def test_invalid_f_nasal_deposit_above_one_raises():
    with pytest.raises(ValueError, match="f_nasal_deposit"):
        simulate_nasal_absorption_pk("Drug", 10.0, logP=1.0, f_nasal_deposit=1.5)


# ---------------------------------------------------------------------------
# Notes string
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = simulate_nasal_absorption_pk("Drug", 10.0, logP=1.5)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
