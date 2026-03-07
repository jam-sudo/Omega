"""Tests for Phase 758: Drug concentration prediction from dose."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.dose_to_concentration import (
    ConcentrationPredictionResult,
    predict_concentration,
    predict_concentration_profile,
)

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_concentration_prediction_result():
    result = predict_concentration("DrugA", dose_mg=100.0, t_h=2.0, cl_L_per_h=5.0, vd_L=50.0)
    assert isinstance(result, ConcentrationPredictionResult)


# ---------------------------------------------------------------------------
# IV bolus: C at t=0 equals Dose/Vd
# ---------------------------------------------------------------------------


def test_iv_bolus_c_at_t0():
    dose = 100.0
    vd = 50.0
    result = predict_concentration(
        "DrugA", dose_mg=dose, t_h=0.0, cl_L_per_h=5.0, vd_L=vd, route="iv_bolus"
    )
    expected = dose / vd
    assert abs(result.c_predicted_mg_L - expected) < 1e-6


# ---------------------------------------------------------------------------
# IV bolus: C decreases over time
# ---------------------------------------------------------------------------


def test_iv_bolus_c_decreases_over_time():
    c_early = predict_concentration(
        "DrugA", dose_mg=100.0, t_h=1.0, cl_L_per_h=5.0, vd_L=50.0, route="iv_bolus"
    )
    c_late = predict_concentration(
        "DrugA", dose_mg=100.0, t_h=5.0, cl_L_per_h=5.0, vd_L=50.0, route="iv_bolus"
    )
    assert c_early.c_predicted_mg_L > c_late.c_predicted_mg_L


# ---------------------------------------------------------------------------
# Oral: C starts at 0 at t=0
# ---------------------------------------------------------------------------


def test_oral_c_at_t0_is_zero():
    result = predict_concentration(
        "DrugA", dose_mg=100.0, t_h=0.0, cl_L_per_h=5.0, vd_L=50.0, route="oral"
    )
    assert result.c_predicted_mg_L == 0.0


# ---------------------------------------------------------------------------
# Oral: C peaks then decreases
# ---------------------------------------------------------------------------


def test_oral_peaks_then_decreases():
    c_values = []
    for t in [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]:
        r = predict_concentration(
            "DrugA", dose_mg=100.0, t_h=t, cl_L_per_h=5.0, vd_L=50.0, route="oral", ka_per_h=1.0
        )
        c_values.append(r.c_predicted_mg_L)
    # Concentration should first rise then fall — check max is not at first or last point
    max_idx = c_values.index(max(c_values))
    assert 0 < max_idx < len(c_values) - 1


# ---------------------------------------------------------------------------
# Multi-dose > single dose (accumulation)
# ---------------------------------------------------------------------------


def test_multi_dose_greater_than_single():
    single = predict_concentration(
        "DrugA",
        dose_mg=100.0,
        t_h=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="oral",
        n_doses=1,
        dosing_interval_h=8.0,
    )
    multi = predict_concentration(
        "DrugA",
        dose_mg=100.0,
        t_h=10.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="oral",
        n_doses=3,
        dosing_interval_h=8.0,
    )
    assert multi.c_predicted_mg_L > single.c_predicted_mg_L


# ---------------------------------------------------------------------------
# css_avg = F * Dose / (CL * tau)
# ---------------------------------------------------------------------------


def test_css_avg_oral():
    dose = 200.0
    cl = 10.0
    tau = 12.0
    f = 0.7
    result = predict_concentration(
        "DrugA",
        dose_mg=dose,
        t_h=0.0,
        cl_L_per_h=cl,
        vd_L=80.0,
        route="oral",
        f_oral=f,
        dosing_interval_h=tau,
    )
    expected_css_avg = f * dose / (cl * tau)
    assert abs(result.css_avg_mg_L - expected_css_avg) < 1e-6


# ---------------------------------------------------------------------------
# css_max > css_avg > css_min
# ---------------------------------------------------------------------------


def test_css_ordering():
    result = predict_concentration(
        "DrugA",
        dose_mg=100.0,
        t_h=1.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="oral",
        dosing_interval_h=8.0,
    )
    assert result.css_max_mg_L > result.css_avg_mg_L > result.css_min_mg_L


# ---------------------------------------------------------------------------
# ke = CL / Vd
# ---------------------------------------------------------------------------


def test_ke_equals_cl_over_vd():
    cl = 6.0
    vd = 40.0
    result = predict_concentration(
        "DrugA", dose_mg=100.0, t_h=1.0, cl_L_per_h=cl, vd_L=vd, route="iv_bolus"
    )
    expected_ke = cl / vd
    assert abs(result.ke_per_h - expected_ke) < 1e-8


# ---------------------------------------------------------------------------
# t_half = ln2 / ke
# ---------------------------------------------------------------------------


def test_t_half_from_ke():
    cl = 6.0
    vd = 40.0
    result = predict_concentration(
        "DrugA", dose_mg=100.0, t_h=1.0, cl_L_per_h=cl, vd_L=vd, route="iv_bolus"
    )
    ke = cl / vd
    expected_t_half = _LN2 / ke
    assert abs(result.t_half_h - expected_t_half) < 1e-6


# ---------------------------------------------------------------------------
# IV infusion: C at end of infusion > 0
# ---------------------------------------------------------------------------


def test_iv_infusion_c_at_end_of_infusion():
    duration = 2.0
    result = predict_concentration(
        "DrugA",
        dose_mg=100.0,
        t_h=duration,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="iv_infusion",
        infusion_duration_h=duration,
    )
    assert result.c_predicted_mg_L > 0.0


# ---------------------------------------------------------------------------
# predict_concentration_profile returns list of (t, C) tuples
# ---------------------------------------------------------------------------


def test_profile_returns_list_of_tuples():
    t_points = [0.0, 1.0, 2.0, 4.0, 8.0]
    profile = predict_concentration_profile(
        "DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, t_points=t_points, route="oral"
    )
    assert isinstance(profile, list)
    assert len(profile) == len(t_points)
    for t, c in profile:
        assert isinstance(t, float)
        assert isinstance(c, float)


# ---------------------------------------------------------------------------
# Linear dose scaling: 2x dose → 2x concentration
# ---------------------------------------------------------------------------


def test_linear_dose_scaling_iv_bolus():
    c1 = predict_concentration(
        "DrugA", dose_mg=100.0, t_h=2.0, cl_L_per_h=5.0, vd_L=50.0, route="iv_bolus"
    )
    c2 = predict_concentration(
        "DrugA", dose_mg=200.0, t_h=2.0, cl_L_per_h=5.0, vd_L=50.0, route="iv_bolus"
    )
    assert abs(c2.c_predicted_mg_L - 2.0 * c1.c_predicted_mg_L) < 1e-6


def test_linear_dose_scaling_oral():
    c1 = predict_concentration(
        "DrugA", dose_mg=100.0, t_h=2.0, cl_L_per_h=5.0, vd_L=50.0, route="oral"
    )
    c2 = predict_concentration(
        "DrugA", dose_mg=200.0, t_h=2.0, cl_L_per_h=5.0, vd_L=50.0, route="oral"
    )
    assert abs(c2.c_predicted_mg_L - 2.0 * c1.c_predicted_mg_L) < 1e-6


# ---------------------------------------------------------------------------
# Validation: dose_mg <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_dose_mg_zero():
    with pytest.raises(ValueError):
        predict_concentration("DrugA", dose_mg=0.0, t_h=1.0, cl_L_per_h=5.0, vd_L=50.0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError):
        predict_concentration("DrugA", dose_mg=-10.0, t_h=1.0, cl_L_per_h=5.0, vd_L=50.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError):
        predict_concentration("DrugA", dose_mg=100.0, t_h=1.0, cl_L_per_h=0.0, vd_L=50.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError):
        predict_concentration("DrugA", dose_mg=100.0, t_h=1.0, cl_L_per_h=5.0, vd_L=0.0)


def test_validation_n_doses_zero():
    with pytest.raises(ValueError):
        predict_concentration("DrugA", dose_mg=100.0, t_h=1.0, cl_L_per_h=5.0, vd_L=50.0, n_doses=0)


# ---------------------------------------------------------------------------
# notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = predict_concentration("DrugA", dose_mg=100.0, t_h=2.0, cl_L_per_h=5.0, vd_L=50.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# IV bolus: concentration follows exact exponential decay
# ---------------------------------------------------------------------------


def test_iv_bolus_exact_exponential_decay():
    dose = 100.0
    vd = 50.0
    cl = 5.0
    ke = cl / vd
    t = 3.0
    result = predict_concentration(
        "DrugA", dose_mg=dose, t_h=t, cl_L_per_h=cl, vd_L=vd, route="iv_bolus"
    )
    expected = (dose / vd) * math.exp(-ke * t)
    assert abs(result.c_predicted_mg_L - expected) < 1e-8


# ---------------------------------------------------------------------------
# IV infusion: during infusion C is monotonically increasing
# ---------------------------------------------------------------------------


def test_iv_infusion_rising_during_infusion():
    c_early = predict_concentration(
        "DrugA",
        dose_mg=100.0,
        t_h=0.5,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="iv_infusion",
        infusion_duration_h=2.0,
    )
    c_late = predict_concentration(
        "DrugA",
        dose_mg=100.0,
        t_h=1.5,
        cl_L_per_h=5.0,
        vd_L=50.0,
        route="iv_infusion",
        infusion_duration_h=2.0,
    )
    assert c_late.c_predicted_mg_L > c_early.c_predicted_mg_L


# ---------------------------------------------------------------------------
# profile time ordering preserved
# ---------------------------------------------------------------------------


def test_profile_preserves_time_order():
    t_points = [1.0, 3.0, 5.0, 7.0]
    profile = predict_concentration_profile(
        "DrugA", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, t_points=t_points, route="iv_bolus"
    )
    times = [p[0] for p in profile]
    assert times == t_points
