"""Tests for Phase 1109 — SC Bioavailability Prediction."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.subcutaneous_bioavailability import (
    SCBioavailabilityResult,
    compare_injection_sites,
    predict_sc_bioavailability,
)


def test_return_type():
    r = predict_sc_bioavailability("Drug", mw_Da=500.0, logP=1.0, dose_mg=10.0)
    assert isinstance(r, SCBioavailabilityResult)


def test_drug_name_preserved():
    r = predict_sc_bioavailability("Insulin", mw_Da=5800.0, logP=-1.0, dose_mg=1.0)
    assert r.drug_name == "Insulin"


def test_bioavailability_in_range():
    r = predict_sc_bioavailability("Drug", mw_Da=500.0, logP=1.0, dose_mg=10.0)
    assert 0.0 <= r.f_sc_bioavailability <= 1.0


def test_capillary_fraction_plus_lymph_equals_one():
    r = predict_sc_bioavailability("Drug", mw_Da=500.0, logP=1.0, dose_mg=10.0)
    assert abs(r.f_capillary_absorption + r.f_lymphatic_absorption - 1.0) < 1e-9


def test_small_molecule_capillary_dominant():
    r = predict_sc_bioavailability("SmallDrug", mw_Da=300.0, logP=1.0, dose_mg=10.0)
    assert r.absorption_route == "capillary_dominant"


def test_large_molecule_lymphatic_dominant():
    r = predict_sc_bioavailability("Biologic", mw_Da=50000.0, logP=-1.0, dose_mg=10.0)
    assert r.absorption_route == "lymphatic_dominant"


def test_high_logp_reduces_f():
    r_low = predict_sc_bioavailability("A", mw_Da=400.0, logP=1.0, dose_mg=10.0)
    r_high = predict_sc_bioavailability("B", mw_Da=400.0, logP=5.0, dose_mg=10.0)
    assert r_high.f_sc_bioavailability < r_low.f_sc_bioavailability


def test_ka_effective_positive():
    r = predict_sc_bioavailability("Drug", mw_Da=500.0, logP=1.0, dose_mg=10.0)
    assert r.ka_effective_per_h > 0.0


def test_tmax_positive():
    r = predict_sc_bioavailability("Drug", mw_Da=500.0, logP=1.0, dose_mg=10.0)
    assert r.tmax_predicted_h > 0.0


def test_abdomen_highest_f_among_sites():
    results = compare_injection_sites("Drug", mw_Da=400.0, logP=1.0, dose_mg=10.0)
    sites = [r.injection_site for r in results]
    assert results[0].injection_site == "abdomen"


def test_large_volume_feasibility():
    r = predict_sc_bioavailability("Drug", mw_Da=400.0, logP=1.0, dose_mg=10.0,
                                    injection_volume_mL=5.0)
    assert r.dose_feasibility == "challenging"


def test_feasible_volume():
    r = predict_sc_bioavailability("Drug", mw_Da=400.0, logP=1.0, dose_mg=10.0,
                                    injection_volume_mL=1.0)
    assert r.dose_feasibility == "feasible"


def test_insoluble_drug_penalty():
    r_sol = predict_sc_bioavailability("Drug", mw_Da=400.0, logP=1.0, dose_mg=10.0,
                                        solubility_mg_mL=100.0,
                                        injection_volume_mL=1.0)
    r_insol = predict_sc_bioavailability("Drug", mw_Da=400.0, logP=1.0, dose_mg=100.0,
                                          solubility_mg_mL=0.05,
                                          injection_volume_mL=1.0)
    assert r_insol.f_sc_bioavailability <= r_sol.f_sc_bioavailability


def test_compare_returns_three_results():
    results = compare_injection_sites("Drug", mw_Da=400.0, logP=1.0, dose_mg=10.0)
    assert len(results) == 3


def test_compare_sorted_descending():
    results = compare_injection_sites("Drug", mw_Da=400.0, logP=1.0, dose_mg=10.0)
    fs = [r.f_sc_bioavailability for r in results]
    assert fs == sorted(fs, reverse=True)


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_sc_bioavailability("X", mw_Da=0.0, logP=1.0, dose_mg=10.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_sc_bioavailability("X", mw_Da=400.0, logP=1.0, dose_mg=0.0)


def test_invalid_volume_raises():
    with pytest.raises(ValueError, match="injection_volume_mL"):
        predict_sc_bioavailability("X", mw_Da=400.0, logP=1.0, dose_mg=10.0,
                                    injection_volume_mL=0.0)


def test_invalid_site_raises():
    with pytest.raises(ValueError, match="injection_site"):
        predict_sc_bioavailability("X", mw_Da=400.0, logP=1.0, dose_mg=10.0,
                                    injection_site="back")


def test_invalid_logp_raises():
    with pytest.raises(ValueError, match="logP"):
        predict_sc_bioavailability("X", mw_Da=400.0, logP=15.0, dose_mg=10.0)
