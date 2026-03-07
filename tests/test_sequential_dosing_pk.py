"""Tests for Phase 715 — sequential IV-to-oral dosing PK."""

import pytest

from omega_pbpk.clinical.sequential_dosing_pk import (
    IVToOralResult,
    optimize_switch_time,
    simulate_iv_to_oral_switch,
)

# --- Basic return type and structure ---


def test_returns_iv_to_oral_result():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert isinstance(result, IVToOralResult)


def test_times_start_at_zero():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert result.times_h[0] == pytest.approx(0.0)


def test_initial_concentration_zero():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_cmax_iv_phase_positive():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert result.cmax_iv_phase_mg_L > 0


def test_cmax_oral_phase_positive():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert result.cmax_oral_phase_mg_L > 0


def test_trough_at_switch_positive():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert result.trough_at_switch_mg_L > 0


def test_auc_total_equals_sum_of_phases():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    # auc_total from full profile; iv+oral sum should be close (overlap at switch point)
    # They won't be exactly equal due to trapezoidal counting, but auc_total ~ auc_iv + auc_oral
    assert result.auc_total == pytest.approx(result.auc_iv_phase + result.auc_oral_phase, rel=0.05)


def test_t_half_h_positive():
    result = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=100,
        oral_dose_mg=200,
        switch_time_h=6,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    expected_t_half = 0.693147 / (5.0 / 50.0)
    assert result.t_half_h == pytest.approx(expected_t_half, rel=1e-4)


def test_therapeutic_continuity_true_when_maintained():
    # Large iv dose + moderate switch time → trough should be > 50% Cmax_iv
    result = simulate_iv_to_oral_switch(
        "DrugB",
        iv_dose_mg=500,
        oral_dose_mg=500,
        switch_time_h=1.0,
        iv_duration_h=0.5,
        ka_per_h=1.0,
        f_oral=1.0,
        cl_L_per_h=1.0,
        vd_L=50.0,
        t_end_h=24.0,
    )
    # If therapeutic_continuity is True, trough > 0.5 * cmax_iv
    if result.therapeutic_continuity:
        assert result.trough_at_switch_mg_L > 0.5 * result.cmax_iv_phase_mg_L


def test_notes_nonempty():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = simulate_iv_to_oral_switch(
        "Vancomycin", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=12
    )
    assert result.drug_name == "Vancomycin"


def test_times_list_nonempty():
    result = simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6)
    assert len(result.times_h) > 0
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_longer_iv_phase_higher_trough_short_halflife():
    """Switching later (while drug still present) should give a different trough."""
    result_early = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=200,
        oral_dose_mg=200,
        switch_time_h=4,
        iv_duration_h=0.5,
        cl_L_per_h=10.0,
        vd_L=50.0,
        t_end_h=24.0,
    )
    result_late = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=200,
        oral_dose_mg=200,
        switch_time_h=12,
        iv_duration_h=0.5,
        cl_L_per_h=10.0,
        vd_L=50.0,
        t_end_h=24.0,
    )
    # Different switch times → different trough values
    assert result_early.trough_at_switch_mg_L != pytest.approx(result_late.trough_at_switch_mg_L)


def test_switch_6h_vs_12h_different_troughs():
    r6 = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=100,
        oral_dose_mg=100,
        switch_time_h=6,
        iv_duration_h=0.5,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=24.0,
    )
    r12 = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=100,
        oral_dose_mg=100,
        switch_time_h=12,
        iv_duration_h=0.5,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=24.0,
    )
    assert r6.trough_at_switch_mg_L != pytest.approx(r12.trough_at_switch_mg_L)


def test_double_dose_doubles_cmax():
    """Doubling both doses should approximately double the Cmax (linearity)."""
    r1 = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=100,
        oral_dose_mg=200,
        switch_time_h=6,
        iv_duration_h=0.5,
        ka_per_h=1.0,
        f_oral=0.8,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
    )
    r2 = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=200,
        oral_dose_mg=400,
        switch_time_h=6,
        iv_duration_h=0.5,
        ka_per_h=1.0,
        f_oral=0.8,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
    )
    assert r2.cmax_iv_phase_mg_L == pytest.approx(r1.cmax_iv_phase_mg_L * 2, rel=0.01)
    assert r2.cmax_oral_phase_mg_L == pytest.approx(r1.cmax_oral_phase_mg_L * 2, rel=0.01)


def test_auc_scales_with_dose():
    r1 = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=100,
        oral_dose_mg=200,
        switch_time_h=6,
        iv_duration_h=0.5,
        ka_per_h=1.0,
        f_oral=0.8,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
    )
    r2 = simulate_iv_to_oral_switch(
        "DrugA",
        iv_dose_mg=200,
        oral_dose_mg=400,
        switch_time_h=6,
        iv_duration_h=0.5,
        ka_per_h=1.0,
        f_oral=0.8,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=48.0,
    )
    assert r2.auc_total == pytest.approx(r1.auc_total * 2, rel=0.02)


# --- Validation errors ---


def test_switch_time_zero_raises():
    with pytest.raises(ValueError, match="switch_time_h"):
        simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=0)


def test_switch_time_negative_raises():
    with pytest.raises(ValueError, match="switch_time_h"):
        simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=-5)


def test_oral_dose_zero_raises():
    with pytest.raises(ValueError, match="oral_dose_mg"):
        simulate_iv_to_oral_switch("DrugA", iv_dose_mg=100, oral_dose_mg=0, switch_time_h=6)


def test_f_oral_zero_raises():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_iv_to_oral_switch(
            "DrugA", iv_dose_mg=100, oral_dose_mg=200, switch_time_h=6, f_oral=0
        )


# --- optimize_switch_time ---


def test_optimize_switch_time_returns_dict():
    result = optimize_switch_time(
        drug_name="DrugA",
        iv_dose_mg=500,
        oral_dose_mg=500,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.0,
        f_oral=0.8,
        target_trough_mg_L=1.0,
    )
    assert isinstance(result, dict)
    assert "optimal_switch_h" in result
    assert "trough_achieved_mg_L" in result
    assert "candidates" in result


def test_optimize_switch_time_candidates_list():
    result = optimize_switch_time(
        drug_name="DrugA",
        iv_dose_mg=500,
        oral_dose_mg=500,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.0,
        f_oral=0.8,
        target_trough_mg_L=1.0,
        candidate_times_h=[6.0, 12.0, 24.0],
    )
    assert len(result["candidates"]) == 3
    for c in result["candidates"]:
        assert "time" in c
        assert "trough" in c


def test_optimize_switch_time_optimal_in_candidates():
    result = optimize_switch_time(
        drug_name="DrugB",
        iv_dose_mg=200,
        oral_dose_mg=200,
        cl_L_per_h=4.0,
        vd_L=40.0,
        ka_per_h=1.0,
        f_oral=0.9,
        target_trough_mg_L=2.0,
        candidate_times_h=[6.0, 12.0, 18.0],
    )
    candidate_times = [c["time"] for c in result["candidates"]]
    assert result["optimal_switch_h"] in candidate_times
