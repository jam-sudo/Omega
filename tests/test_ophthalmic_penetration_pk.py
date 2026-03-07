"""Tests for ophthalmic drug penetration PK — Phase 733."""

from __future__ import annotations

import pytest

from omega_pbpk.core.ophthalmic_penetration_pk import (
    OphthalmicPenetrationResult,
    compare_ophthalmic_drugs,
    simulate_ophthalmic_penetration,
)


def _run(**kwargs):
    defaults = dict(drug_name="TestDrug", dose_mg=1.0)
    defaults.update(kwargs)
    return simulate_ophthalmic_penetration(**defaults)


# --- Basic result type ---


def test_returns_result_instance():
    result = _run()
    assert isinstance(result, OphthalmicPenetrationResult)


def test_cmax_aq_positive():
    result = _run()
    assert result.cmax_aq_mg_L > 0


def test_auc_positive():
    result = _run()
    assert result.auc_aq_mg_h_per_L > 0


def test_corneal_permeability_index_positive():
    result = _run()
    assert result.corneal_permeability_index > 0


def test_notes_nonempty():
    result = _run()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- Fraction checks ---


def test_fraction_drained_between_0_and_1():
    result = _run()
    assert 0.0 <= result.fraction_drained <= 1.0


def test_fraction_absorbed_between_0_and_1():
    result = _run()
    assert 0.0 <= result.fraction_absorbed <= 1.0


def test_fraction_drained_plus_absorbed_le_1():
    result = _run()
    assert result.fraction_drained + result.fraction_absorbed <= 1.0 + 1e-9


# --- logP effect ---


def test_low_logp_lower_penetration_than_optimal():
    low_logp = _run(logp=-1.0)  # very low logP
    optimal_logp = _run(logp=1.5)  # optimal
    assert optimal_logp.cmax_aq_mg_L > low_logp.cmax_aq_mg_L


# --- MW effect ---


def test_lower_mw_better_penetration():
    low_mw = _run(mw_da=150.0)
    high_mw = _run(mw_da=800.0)
    assert low_mw.cmax_aq_mg_L > high_mw.cmax_aq_mg_L


# --- Linear dose scaling ---


def test_linear_dose_scaling_cmax():
    r1 = _run(dose_mg=1.0)
    r2 = _run(dose_mg=2.0)
    ratio = r2.cmax_aq_mg_L / r1.cmax_aq_mg_L
    assert abs(ratio - 2.0) < 0.05


def test_linear_dose_scaling_auc():
    r1 = _run(dose_mg=1.0)
    r2 = _run(dose_mg=2.0)
    ratio = r2.auc_aq_mg_h_per_L / r1.auc_aq_mg_h_per_L
    assert abs(ratio - 2.0) < 0.05


# --- Time arrays ---


def test_times_and_conc_same_length():
    result = _run()
    assert len(result.times_h) == len(result.c_aq_mg_L)
    assert len(result.times_h) == len(result.a_surface_mg)
    assert len(result.times_h) == len(result.a_cornea_mg)


def test_surface_amount_decreases_monotonically():
    result = _run()
    # Surface amount should decrease (monotonically draining)
    assert result.a_surface_mg[0] >= result.a_surface_mg[-1]


def test_tmax_aq_within_simulation_time():
    result = _run(t_end_h=6.0)
    assert 0.0 <= result.tmax_aq_h <= 6.0


# --- Validation ---


def test_dose_le_zero_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_negative_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-5.0)


def test_zero_v_aq_raises_value_error():
    with pytest.raises(ValueError, match="v_aq_L"):
        _run(v_aq_L=0.0)


def test_zero_k_cornea_base_raises_value_error():
    with pytest.raises(ValueError, match="k_cornea_base_per_h"):
        _run(k_cornea_base_per_h=0.0)


def test_zero_k_drain_raises_value_error():
    with pytest.raises(ValueError, match="k_drain_per_h"):
        _run(k_drain_per_h=0.0)


# --- compare_ophthalmic_drugs ---


def test_compare_returns_list():
    drugs = [
        {"drug_name": "DrugA", "dose_mg": 1.0, "logp": 1.5, "mw_da": 200.0},
        {"drug_name": "DrugB", "dose_mg": 1.0, "logp": 3.5, "mw_da": 600.0},
    ]
    results = compare_ophthalmic_drugs(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_sorted_by_auc_descending():
    drugs = [
        {"drug_name": "DrugA", "dose_mg": 1.0, "logp": 1.5, "mw_da": 200.0},
        {"drug_name": "DrugB", "dose_mg": 1.0, "logp": 3.5, "mw_da": 600.0},
    ]
    results = compare_ophthalmic_drugs(drugs)
    aucs = [r.auc_aq_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)
