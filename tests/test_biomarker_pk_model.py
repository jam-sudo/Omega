"""Tests for biomarker-linked indirect response PK/PD model — Phase 493."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.clinical.biomarker_pk_model import (
    IndirectResponseResult,
    calculate_baseline_biomarker,
    calculate_hysteresis_area,
    predict_effect_onset,
    simulate_indirect_response,
)

# ---------------------------------------------------------------------------
# calculate_baseline_biomarker
# ---------------------------------------------------------------------------


def test_baseline_equals_kin_over_kout():
    assert calculate_baseline_biomarker(2.0, 0.5) == pytest.approx(4.0)


def test_baseline_kin_zero_raises():
    with pytest.raises(ValueError, match="kin"):
        calculate_baseline_biomarker(0.0, 1.0)


def test_baseline_kout_zero_raises():
    with pytest.raises(ValueError, match="kout"):
        calculate_baseline_biomarker(1.0, 0.0)


def test_baseline_negative_kin_raises():
    with pytest.raises(ValueError, match="kin"):
        calculate_baseline_biomarker(-1.0, 1.0)


# ---------------------------------------------------------------------------
# predict_effect_onset
# ---------------------------------------------------------------------------


def test_predict_onset_cmax_above_ec50():
    # When Cmax >> EC50, onset = ln(cmax/ec50) / kout
    onset = predict_effect_onset(ec50_mg_L=1.0, cmax_mg_L=10.0, kout=0.5)
    expected = math.log(10.0) / 0.5
    assert onset == pytest.approx(expected, rel=1e-6)


def test_predict_onset_cmax_equal_ec50_returns_1_over_kout():
    onset = predict_effect_onset(ec50_mg_L=2.0, cmax_mg_L=2.0, kout=0.4)
    assert onset == pytest.approx(1.0 / 0.4, rel=1e-6)


def test_predict_onset_cmax_below_ec50_returns_floor():
    onset = predict_effect_onset(ec50_mg_L=5.0, cmax_mg_L=1.0, kout=0.5)
    assert onset == pytest.approx(1.0 / 0.5, rel=1e-6)


def test_predict_onset_ec50_zero_raises():
    with pytest.raises(ValueError, match="ec50"):
        predict_effect_onset(0.0, 5.0, 0.5)


def test_predict_onset_cmax_zero_raises():
    with pytest.raises(ValueError, match="cmax"):
        predict_effect_onset(1.0, 0.0, 0.5)


def test_predict_onset_kout_zero_raises():
    with pytest.raises(ValueError, match="kout"):
        predict_effect_onset(1.0, 5.0, 0.0)


# ---------------------------------------------------------------------------
# calculate_hysteresis_area
# ---------------------------------------------------------------------------


def test_hysteresis_area_non_negative():
    times = [0.0, 1.0, 2.0, 3.0]
    conc = [10.0, 5.0, 2.0, 0.5]
    effect = [1.0, 2.0, 1.5, 1.1]
    area = calculate_hysteresis_area(times, conc, times, effect)
    assert area >= 0.0


def test_hysteresis_area_no_loop_is_small():
    # Perfectly correlated → no hysteresis
    times = [0.0, 1.0, 2.0]
    conc = [5.0, 3.0, 1.0]
    effect = [5.0, 3.0, 1.0]  # effect == conc → degenerate loop
    area = calculate_hysteresis_area(times, conc, times, effect)
    assert area == pytest.approx(0.0, abs=1e-10)


def test_hysteresis_mismatched_lengths_raises():
    # conc has different length from pk_times_h → should raise
    with pytest.raises(ValueError):
        calculate_hysteresis_area([0, 1, 2], [1, 2], [0, 1, 2], [1, 2, 3])


def test_hysteresis_single_point_pk_raises():
    with pytest.raises(ValueError):
        calculate_hysteresis_area([0.0], [1.0], [0.0, 1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# simulate_indirect_response — result type and fields
# ---------------------------------------------------------------------------


def _default_sim(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        kin=2.0,
        kout=0.5,
        ec50_mg_L=1.0,
        emax=1.0,
        route="iv",
        t_end_h=24.0,
        dt_h=0.5,
        model_type="inhibit_kin",
    )
    defaults.update(kwargs)
    return simulate_indirect_response(**defaults)


def test_returns_indirect_response_result():
    r = _default_sim()
    assert isinstance(r, IndirectResponseResult)


def test_result_fields_present():
    r = _default_sim()
    assert hasattr(r, "times_h")
    assert hasattr(r, "c_plasma_mg_L")
    assert hasattr(r, "biomarker")
    assert hasattr(r, "baseline_biomarker")
    assert hasattr(r, "max_effect_pct_change")
    assert hasattr(r, "time_to_max_effect_h")
    assert hasattr(r, "auc_effect_above_baseline")


def test_baseline_matches_kin_over_kout():
    r = _default_sim(kin=3.0, kout=0.6)
    assert r.baseline_biomarker == pytest.approx(3.0 / 0.6, rel=1e-3)


def test_time_grid_length_matches_conc_and_biomarker():
    r = _default_sim()
    assert len(r.times_h) == len(r.c_plasma_mg_L)
    assert len(r.times_h) == len(r.biomarker)


# ---------------------------------------------------------------------------
# simulate_indirect_response — model-type behaviour
# ---------------------------------------------------------------------------


def test_inhibit_kin_biomarker_decreases():
    r = _default_sim(model_type="inhibit_kin", emax=1.0)
    # Minimum biomarker should be below baseline
    assert min(r.biomarker) < r.baseline_biomarker


def test_stimulate_kin_biomarker_increases():
    r = _default_sim(model_type="stimulate_kin", emax=1.0)
    assert max(r.biomarker) > r.baseline_biomarker


def test_inhibit_kout_biomarker_increases():
    r = _default_sim(model_type="inhibit_kout", emax=1.0)
    assert max(r.biomarker) > r.baseline_biomarker


def test_stimulate_kout_biomarker_decreases():
    r = _default_sim(model_type="stimulate_kout", emax=1.0)
    assert min(r.biomarker) < r.baseline_biomarker


def test_emax_zero_biomarker_stays_at_baseline():
    r = _default_sim(emax=0.0, model_type="inhibit_kin", t_end_h=48.0)
    tol = r.baseline_biomarker * 0.005  # allow 0.5% numerical drift
    for b in r.biomarker:
        assert abs(b - r.baseline_biomarker) <= tol, (
            f"biomarker {b} deviated from baseline {r.baseline_biomarker}"
        )


def test_max_effect_occurs_after_cmax():
    # For IV bolus with inhibit_kin the Cmax is at t=0, but the biomarker
    # lags → time_to_max_effect > 0.
    r = _default_sim(model_type="inhibit_kin", route="iv")
    # Cmax at t=0, biomarker effect is delayed
    assert r.time_to_max_effect_h > 0.0


def test_oral_route_cmax_not_at_t0():
    r = _default_sim(route="oral", ka_per_h=1.0)
    # For oral route the first plasma concentration is 0 (dose in gut)
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-10)
    # Plasma peaks later
    cmax_idx = r.c_plasma_mg_L.index(max(r.c_plasma_mg_L))
    assert cmax_idx > 0


def test_higher_dose_higher_effect():
    r_low = _default_sim(dose_mg=10.0, model_type="inhibit_kin")
    r_high = _default_sim(dose_mg=200.0, model_type="inhibit_kin")
    # Higher dose → larger absolute max % change
    assert abs(r_high.max_effect_pct_change) > abs(r_low.max_effect_pct_change)


def test_auc_effect_non_negative():
    r = _default_sim()
    assert r.auc_effect_above_baseline >= 0.0


def test_model_type_in_result():
    for mt in ("inhibit_kin", "stimulate_kin", "inhibit_kout", "stimulate_kout"):
        r = _default_sim(model_type=mt)
        assert r.model_type == mt


# ---------------------------------------------------------------------------
# simulate_indirect_response — input validation
# ---------------------------------------------------------------------------


def test_invalid_model_type_raises():
    with pytest.raises(ValueError, match="model_type"):
        _default_sim(model_type="invalid_model")


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _default_sim(dose_mg=0.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _default_sim(cl_L_per_h=0.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _default_sim(vd_L=0.0)


def test_kin_zero_raises():
    with pytest.raises(ValueError, match="kin"):
        _default_sim(kin=0.0)


def test_kout_zero_raises():
    with pytest.raises(ValueError, match="kout"):
        _default_sim(kout=0.0)


def test_ec50_zero_raises():
    with pytest.raises(ValueError, match="ec50"):
        _default_sim(ec50_mg_L=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        _default_sim(route="subcutaneous")
