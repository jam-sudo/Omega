"""Tests for Phase 1049 — Drug Cardiac Toxicity Biomarker."""

import pytest

from omega_pbpk.prediction.cardiac_toxicity_biomarker import (
    CardiacToxicityResult,
    simulate_cardiac_toxicity,
)


def default_result():
    return simulate_cardiac_toxicity(
        drug_name="TestDrug",
        dose_mg=100.0,
        herg_ic50_mg_L=10.0,
        cardiac_ic50_mg_L=50.0,
        cl_L_per_h=5.0,
        vd_L=30.0,
        baseline_qt_ms=400.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


def test_return_type():
    result = default_result()
    assert isinstance(result, CardiacToxicityResult)


def test_initial_concentration():
    result = default_result()
    # c_plasma[0] = dose/vd = 100/30 ~ 3.333
    expected = 100.0 / 30.0
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-6


def test_troponin_initial_near_baseline():
    result = default_result()
    assert abs(result.troponin_ng_mL[0] - 0.01) < 1e-6


def test_bnp_initial_near_baseline():
    result = default_result()
    assert abs(result.bnp_ng_mL[0] - 0.01) < 1e-6


def test_qt_initial_formula():
    # QT at t=0: baseline + 100 * c0 / (herg_ic50 + c0)
    result = default_result()
    c0 = 100.0 / 30.0  # dose/vd
    herg_ic50 = 10.0
    expected_qt = 400.0 + 100.0 * c0 / (herg_ic50 + c0)
    assert abs(result.qt_interval_ms[0] - expected_qt) < 0.01


def test_auc_plasma_positive():
    result = default_result()
    assert result.auc_plasma > 0.0


def test_peak_troponin_positive():
    result = default_result()
    assert result.peak_troponin > 0.0


def test_peak_bnp_positive():
    result = default_result()
    assert result.peak_bnp > 0.0


def test_max_qt_greater_than_baseline():
    result = default_result()
    assert result.max_qt_ms > result.baseline_qt_ms


def test_delta_qt_positive():
    result = default_result()
    assert result.delta_qt_ms > 0.0


def test_cardiac_risk_score_range():
    result = default_result()
    assert 0.0 <= result.cardiac_risk_score <= 100.0


def test_risk_category_valid():
    result = default_result()
    valid = {"safe", "monitor", "warning", "contraindicated"}
    assert result.risk_category in valid


def test_lower_herg_ic50_higher_delta_qt():
    r_low = simulate_cardiac_toxicity(
        drug_name="A", dose_mg=100.0, herg_ic50_mg_L=1.0, cardiac_ic50_mg_L=50.0
    )
    r_high = simulate_cardiac_toxicity(
        drug_name="A", dose_mg=100.0, herg_ic50_mg_L=100.0, cardiac_ic50_mg_L=50.0
    )
    assert r_low.delta_qt_ms > r_high.delta_qt_ms


def test_higher_dose_higher_risk_score():
    r_low = simulate_cardiac_toxicity(drug_name="A", dose_mg=10.0)
    r_high = simulate_cardiac_toxicity(drug_name="A", dose_mg=1000.0)
    assert r_high.cardiac_risk_score >= r_low.cardiac_risk_score


def test_notes_nonempty():
    result = default_result()
    assert len(result.notes) > 0


def test_baseline_qt_stored():
    result = default_result()
    assert result.baseline_qt_ms == 400.0


def test_drug_name_stored():
    result = default_result()
    assert result.drug_name == "TestDrug"


def test_dose_stored():
    result = default_result()
    assert result.dose_mg == 100.0


def test_list_lengths_consistent():
    result = default_result()
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.troponin_ng_mL) == n
    assert len(result.bnp_ng_mL) == n
    assert len(result.qt_interval_ms) == n


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_cardiac_toxicity(drug_name="A", dose_mg=0.0)


def test_validation_herg_ic50_zero():
    with pytest.raises(ValueError):
        simulate_cardiac_toxicity(drug_name="A", dose_mg=100.0, herg_ic50_mg_L=0.0)


def test_validation_baseline_qt_out_of_range():
    with pytest.raises(ValueError):
        simulate_cardiac_toxicity(drug_name="A", dose_mg=100.0, baseline_qt_ms=250.0)


def test_risk_safe_low_dose():
    result = simulate_cardiac_toxicity(
        drug_name="Safe",
        dose_mg=1.0,
        herg_ic50_mg_L=1000.0,
        cardiac_ic50_mg_L=10000.0,
        t_end_h=24.0,
    )
    assert result.cardiac_risk_score < 40.0
