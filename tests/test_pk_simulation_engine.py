"""Tests for core/pk_simulation_engine.py — Phase 496."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.core.pk_simulation_engine import (
    PKSimResult,
    apply_dose,
    calculate_summary_stats,
    create_dosing_regimen,
    simulate_pk,
)


# ---------------------------------------------------------------------------
# create_dosing_regimen
# ---------------------------------------------------------------------------


class TestCreateDosingRegimen:
    def test_correct_number_of_doses(self):
        regimen = create_dosing_regimen(100.0, 8.0, 3)
        assert len(regimen) == 3

    def test_timing_is_correct(self):
        regimen = create_dosing_regimen(100.0, 8.0, 3)
        assert regimen[0]["time_h"] == pytest.approx(0.0)
        assert regimen[1]["time_h"] == pytest.approx(8.0)
        assert regimen[2]["time_h"] == pytest.approx(16.0)

    def test_dose_mg_stored(self):
        regimen = create_dosing_regimen(50.0, 12.0, 2)
        assert all(e["dose_mg"] == 50.0 for e in regimen)

    def test_start_h_offset(self):
        regimen = create_dosing_regimen(100.0, 6.0, 2, start_h=2.0)
        assert regimen[0]["time_h"] == pytest.approx(2.0)
        assert regimen[1]["time_h"] == pytest.approx(8.0)

    def test_single_dose(self):
        regimen = create_dosing_regimen(100.0, 24.0, 1)
        assert len(regimen) == 1
        assert regimen[0]["time_h"] == pytest.approx(0.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            create_dosing_regimen(0.0, 8.0, 3)

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError):
            create_dosing_regimen(100.0, 0.0, 3)

    def test_zero_doses_raises(self):
        with pytest.raises(ValueError):
            create_dosing_regimen(100.0, 8.0, 0)


# ---------------------------------------------------------------------------
# apply_dose
# ---------------------------------------------------------------------------


class TestApplyDose:
    def test_iv_dose_increases_c1(self):
        state = {"c1": 0.0, "a2": 0.0, "gut": 0.0}
        new = apply_dose(state, 100.0, "iv", 10.0)
        assert new["c1"] == pytest.approx(10.0)

    def test_oral_dose_goes_to_gut(self):
        state = {"c1": 0.0, "a2": 0.0, "gut": 0.0}
        new = apply_dose(state, 100.0, "oral", 10.0, f_oral=1.0)
        assert new["gut"] == pytest.approx(100.0)

    def test_f_oral_fraction_applied(self):
        state = {"c1": 0.0, "a2": 0.0, "gut": 0.0}
        new = apply_dose(state, 100.0, "oral", 10.0, f_oral=0.5)
        assert new["gut"] == pytest.approx(50.0)

    def test_immutable_original_state(self):
        state = {"c1": 5.0, "a2": 0.0, "gut": 0.0}
        _ = apply_dose(state, 100.0, "iv", 10.0)
        assert state["c1"] == 5.0  # original unchanged

    def test_invalid_route_raises(self):
        state = {"c1": 0.0, "a2": 0.0, "gut": 0.0}
        with pytest.raises(ValueError, match="route"):
            apply_dose(state, 100.0, "sublingual", 10.0)

    def test_iv_accumulates_on_existing_c1(self):
        state = {"c1": 3.0, "a2": 0.0, "gut": 0.0}
        new = apply_dose(state, 100.0, "iv", 10.0)
        assert new["c1"] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# simulate_pk — 1-compartment IV
# ---------------------------------------------------------------------------


class TestSimulatePK1CptIV:
    def test_returns_result(self):
        r = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv")
        assert isinstance(r, PKSimResult)

    def test_monoexponential_decay(self):
        """ln(C) vs t should be linear for 1-cpt IV."""
        r = simulate_pk("Drug", 100.0, 2.0, 20.0, route="iv", t_end_h=24.0, dt_h=0.1)
        # Check C(t2)/C(t1) ≈ exp(-ke*(t2-t1)) at t > 0
        ke = 2.0 / 20.0
        idx_5 = int(5.0 / 0.1)
        idx_10 = int(10.0 / 0.1)
        c5 = r.c_plasma_mg_L[idx_5]
        c10 = r.c_plasma_mg_L[idx_10]
        ratio = c10 / c5
        expected_ratio = math.exp(-ke * 5.0)
        assert ratio == pytest.approx(expected_ratio, rel=0.05)

    def test_t_half_matches_analytical(self):
        cl, vd = 2.0, 20.0
        r = simulate_pk("Drug", 100.0, cl, vd, route="iv", t_end_h=48.0, dt_h=0.1)
        expected_t_half = math.log(2.0) * vd / cl
        assert r.t_half_h == pytest.approx(expected_t_half, rel=0.10)

    def test_double_dose_doubles_cmax(self):
        r1 = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv")
        r2 = simulate_pk("Drug", 200.0, 5.0, 10.0, route="iv")
        assert r2.cmax_mg_L == pytest.approx(2 * r1.cmax_mg_L, rel=0.05)

    def test_double_dose_doubles_auc(self):
        r1 = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv", t_end_h=48.0)
        r2 = simulate_pk("Drug", 200.0, 5.0, 10.0, route="iv", t_end_h=48.0)
        assert r2.auc_mg_h_L == pytest.approx(2 * r1.auc_mg_h_L, rel=0.05)

    def test_vss_equals_vd_for_1cpt(self):
        r = simulate_pk("Drug", 100.0, 5.0, 15.0, route="iv")
        assert r.vss_L == pytest.approx(15.0)

    def test_peripheral_conc_empty_for_1cpt(self):
        r = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv")
        assert r.c_peripheral_mg_L == []


# ---------------------------------------------------------------------------
# simulate_pk — 1-compartment oral
# ---------------------------------------------------------------------------


class TestSimulatePK1CptOral:
    def test_tmax_delayed_vs_iv(self):
        r_iv = simulate_pk("Drug", 100.0, 2.0, 10.0, route="iv")
        r_oral = simulate_pk("Drug", 100.0, 2.0, 10.0, route="oral", ka_per_h=1.0)
        assert r_oral.tmax_h > 0.0

    def test_same_t_half_as_iv(self):
        cl, vd = 2.0, 20.0
        r_oral = simulate_pk("Drug", 100.0, cl, vd, route="oral",
                              ka_per_h=1.0, f_oral=1.0, t_end_h=48.0, dt_h=0.1)
        expected_t_half = math.log(2.0) * vd / cl
        assert r_oral.t_half_h == pytest.approx(expected_t_half, rel=0.15)

    def test_f_oral_reduces_cmax(self):
        r_full = simulate_pk("Drug", 100.0, 2.0, 10.0, route="oral", f_oral=1.0)
        r_half = simulate_pk("Drug", 100.0, 2.0, 10.0, route="oral", f_oral=0.5)
        assert r_half.cmax_mg_L < r_full.cmax_mg_L

    def test_tmax_nonzero_for_oral(self):
        r = simulate_pk("Drug", 100.0, 2.0, 10.0, route="oral", ka_per_h=0.5)
        assert r.tmax_h > 0.5


# ---------------------------------------------------------------------------
# simulate_pk — 2-compartment
# ---------------------------------------------------------------------------


class TestSimulatePK2Cpt:
    def test_peripheral_conc_nonzero(self):
        r = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv",
                        n_compartments=2, q12_L_per_h=2.0, vd2_L=20.0,
                        t_end_h=24.0)
        assert any(c > 0 for c in r.c_peripheral_mg_L)

    def test_vss_equals_vd1_plus_vd2(self):
        r = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv",
                        n_compartments=2, q12_L_per_h=2.0, vd2_L=20.0)
        assert r.vss_L == pytest.approx(30.0)

    def test_2cpt_double_dose_doubles_cmax(self):
        r1 = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv",
                         n_compartments=2, q12_L_per_h=2.0, vd2_L=20.0, t_end_h=48.0)
        r2 = simulate_pk("Drug", 200.0, 5.0, 10.0, route="iv",
                         n_compartments=2, q12_L_per_h=2.0, vd2_L=20.0, t_end_h=48.0)
        assert r2.cmax_mg_L == pytest.approx(2 * r1.cmax_mg_L, rel=0.05)

    def test_2cpt_peripheral_conc_list_matches_time_length(self):
        r = simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv",
                        n_compartments=2, q12_L_per_h=2.0, vd2_L=20.0)
        assert len(r.c_peripheral_mg_L) == len(r.times_h)

    def test_2cpt_missing_vd2_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("Drug", 100.0, 5.0, 10.0, route="iv",
                        n_compartments=2, q12_L_per_h=2.0, vd2_L=0.0)


# ---------------------------------------------------------------------------
# Multiple dosing
# ---------------------------------------------------------------------------


class TestMultipleDosing:
    def test_accumulation_at_steady_state(self):
        """After many doses, Cmax should be higher than after the first dose."""
        # Use 1h interval so drug accumulates noticeably before 10 doses
        regimen = create_dosing_regimen(100.0, 1.0, 10)
        r = simulate_pk("Drug", 100.0, 5.0, 10.0, route="oral",
                        ka_per_h=2.0, dosing_regimen=regimen, t_end_h=12.0, dt_h=0.05)
        # Cmax over doses 8-10 window should exceed Cmax in first dose interval
        # First dose interval: t in [0, 1h)
        idx_1h = int(1.0 / 0.05)
        # Late doses: t in [7, 10h)
        idx_7h = int(7.0 / 0.05)
        idx_10h = int(10.0 / 0.05)
        cmax_first = max(r.c_plasma_mg_L[:idx_1h])
        cmax_late = max(r.c_plasma_mg_L[idx_7h:idx_10h])
        assert cmax_late > cmax_first

    def test_create_regimen_and_simulate_consistent(self):
        regimen = create_dosing_regimen(100.0, 12.0, 3)
        assert len(regimen) == 3
        r = simulate_pk("Drug", 100.0, 2.0, 10.0, route="oral",
                        ka_per_h=1.0, dosing_regimen=regimen, t_end_h=36.0)
        assert r.cmax_mg_L > 0


# ---------------------------------------------------------------------------
# calculate_summary_stats
# ---------------------------------------------------------------------------


class TestCalculateSummaryStats:
    def test_returns_dict_with_required_keys(self):
        times = [0.0, 1.0, 2.0, 3.0]
        concs = [10.0, 7.0, 5.0, 3.0]
        stats = calculate_summary_stats(times, concs, 100.0)
        assert "cmax" in stats
        assert "tmax" in stats
        assert "auc" in stats
        assert "t_half" in stats

    def test_cmax_correct(self):
        times = [0.0, 1.0, 2.0]
        concs = [5.0, 10.0, 3.0]
        stats = calculate_summary_stats(times, concs, 100.0)
        assert stats["cmax"] == pytest.approx(10.0)

    def test_tmax_correct(self):
        times = [0.0, 1.0, 2.0]
        concs = [5.0, 10.0, 3.0]
        stats = calculate_summary_stats(times, concs, 100.0)
        assert stats["tmax"] == pytest.approx(1.0)

    def test_auc_trapezoidal(self):
        times = [0.0, 1.0, 2.0]
        concs = [0.0, 2.0, 0.0]
        stats = calculate_summary_stats(times, concs, 100.0)
        assert stats["auc"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("", 100.0, 5.0, 10.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("Drug", -100.0, 5.0, 10.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("Drug", 100.0, 0.0, 10.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("Drug", 100.0, 5.0, 0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("Drug", 100.0, 5.0, 10.0, route="im")

    def test_invalid_n_compartments_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("Drug", 100.0, 5.0, 10.0, n_compartments=3)

    def test_f_oral_gt_1_raises(self):
        with pytest.raises(ValueError):
            simulate_pk("Drug", 100.0, 5.0, 10.0, route="oral", f_oral=1.5)
