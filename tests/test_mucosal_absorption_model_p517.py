"""Tests for Phase 517 — Mucosal Drug Absorption Model (core)."""

import math

import pytest

from omega_pbpk.core.mucosal_absorption_model_p517 import (
    MucosalAbsorptionResult,
    _compute_f_mucus,
    compare_mw_effect,
    simulate_mucosal_absorption,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    mw_Da=300.0,
    logP=1.5,
)


def default_result(**overrides):
    kw = dict(DEFAULT_KWARGS)
    kw.update(overrides)
    return simulate_mucosal_absorption(**kw)


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_return_type():
    result = default_result()
    assert isinstance(result, MucosalAbsorptionResult)


def test_fields_present():
    r = default_result()
    assert hasattr(r, "drug_name")
    assert hasattr(r, "dose_mg")
    assert hasattr(r, "mw_Da")
    assert hasattr(r, "logP")
    assert hasattr(r, "f_mucus")
    assert hasattr(r, "ka_apparent_per_h")
    assert hasattr(r, "e_hepatic")
    assert hasattr(r, "f_hepatic")
    assert hasattr(r, "times_h")
    assert hasattr(r, "c_systemic_mg_L")
    assert hasattr(r, "cmax_mg_L")
    assert hasattr(r, "tmax_h")
    assert hasattr(r, "auc_mg_h_per_L")
    assert hasattr(r, "bioavailability_pct")
    assert hasattr(r, "t_half_h")
    assert hasattr(r, "notes")


def test_drug_name_stored():
    r = default_result(drug_name="Aspirin")
    assert r.drug_name == "Aspirin"


def test_dose_stored():
    r = default_result(dose_mg=200.0)
    assert r.dose_mg == 200.0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_sys_starts_at_zero():
    r = default_result()
    assert r.c_systemic_mg_L[0] == 0.0


def test_times_start_at_zero():
    r = default_result()
    assert r.times_h[0] == 0.0


def test_c_sys_positive_after_time():
    r = default_result()
    # Systemic concentration should be positive sometime after t=0
    assert any(c > 0 for c in r.c_systemic_mg_L[1:])


# ---------------------------------------------------------------------------
# AUC and PK metrics
# ---------------------------------------------------------------------------


def test_auc_positive():
    r = default_result()
    assert r.auc_mg_h_per_L > 0.0


def test_cmax_positive():
    r = default_result()
    assert r.cmax_mg_L > 0.0


def test_tmax_non_negative():
    r = default_result()
    assert r.tmax_h >= 0.0


def test_t_half_positive():
    r = default_result()
    assert r.t_half_h > 0.0


def test_times_and_concentrations_same_length():
    r = default_result()
    assert len(r.times_h) == len(r.c_systemic_mg_L)


# ---------------------------------------------------------------------------
# f_mucus behaviour
# ---------------------------------------------------------------------------


def test_small_mw_higher_f_mucus_than_large_mw():
    r_small = simulate_mucosal_absorption("Drug", 100.0, mw_Da=150.0, logP=1.0)
    r_large = simulate_mucosal_absorption("Drug", 100.0, mw_Da=800.0, logP=1.0)
    assert r_small.f_mucus > r_large.f_mucus


def test_f_mucus_clamped_minimum():
    # Very large MW + very negative logP should still be >= 0.05
    f = _compute_f_mucus(mw_Da=10000.0, logP=-5.0)
    assert f >= 0.05


def test_f_mucus_clamped_maximum():
    f = _compute_f_mucus(mw_Da=50.0, logP=0.0)
    assert f <= 1.0


def test_f_mucus_in_range():
    r = default_result()
    assert 0.05 <= r.f_mucus <= 1.0


def test_logp_effect_on_f_mucus():
    # logP = 0 gives ratio 1.0; logP = -3 gives negative numerator => clamped
    f_neutral = _compute_f_mucus(mw_Da=300.0, logP=0.0)
    f_very_negative = _compute_f_mucus(mw_Da=300.0, logP=-3.0)
    assert f_neutral >= f_very_negative


def test_f_mucus_logp_positive_equals_one_ratio():
    # For logP > 0: (1+logP)/(1+logP) = 1, so f_mucus only depends on MW
    f = _compute_f_mucus(mw_Da=300.0, logP=2.0)
    expected = math.exp(-300.0 / 500.0)
    assert abs(f - expected) < 1e-9


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


def test_bioavailability_between_0_and_100():
    r = default_result()
    assert 0.0 <= r.bioavailability_pct <= 100.0


def test_bioavailability_formula():
    r = default_result()
    expected = r.f_mucus * 0.9 * r.f_hepatic * 100.0
    assert abs(r.bioavailability_pct - expected) < 0.01


def test_e_hepatic_f_hepatic_sum_to_one():
    r = default_result()
    assert abs(r.e_hepatic + r.f_hepatic - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compare_mw_effect
# ---------------------------------------------------------------------------


def test_compare_mw_effect_returns_list():
    results = compare_mw_effect("Drug", 100.0, logP=1.0, mw_values_Da=[200.0, 400.0, 600.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_mw_effect_sorted_descending():
    results = compare_mw_effect("Drug", 100.0, logP=1.0, mw_values_Da=[600.0, 200.0, 400.0])
    bav = [r.bioavailability_pct for r in results]
    assert bav == sorted(bav, reverse=True)


def test_compare_mw_single_value():
    results = compare_mw_effect("Drug", 100.0, logP=0.5, mw_values_Da=[300.0])
    assert len(results) == 1
    assert isinstance(results[0], MucosalAbsorptionResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mucosal_absorption("Drug", dose_mg=0.0, mw_Da=300.0, logP=1.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mucosal_absorption("Drug", dose_mg=-10.0, mw_Da=300.0, logP=1.0)


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_mucosal_absorption("Drug", dose_mg=100.0, mw_Da=0.0, logP=1.0)


def test_cl_hepatic_zero_raises():
    with pytest.raises(ValueError, match="cl_hepatic"):
        simulate_mucosal_absorption(
            "Drug", dose_mg=100.0, mw_Da=300.0, logP=1.0, cl_hepatic_L_per_h=0.0
        )


def test_vd_sys_zero_raises():
    with pytest.raises(ValueError, match="vd_sys"):
        simulate_mucosal_absorption("Drug", dose_mg=100.0, mw_Da=300.0, logP=1.0, vd_sys_L=0.0)
