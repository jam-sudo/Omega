"""Tests for biopharmaceutics/lag_time_models.py — Phase 354."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.lag_time_models import (
    LagTimeResult,
    calculate_lag_times,
    compare_formulations,
    simulate_absorption_with_lag,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BASE_SIM_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=500.0,
    cl_L_per_h=10.0,
    vd_L=60.0,
    ka_per_h=1.5,
    f_oral=0.9,
    t_end_h=24.0,
    dt_h=0.1,
)


def _sim(**override):
    kw = dict(BASE_SIM_KWARGS)
    kw.update(override)
    return simulate_absorption_with_lag(**kw)


# ---------------------------------------------------------------------------
# calculate_lag_times
# ---------------------------------------------------------------------------


class TestCalculateLagTimes:
    def test_fasted_faster_than_fed(self):
        fasted = calculate_lag_times("immediate_release", fasted_state=True)
        fed = calculate_lag_times("immediate_release", fasted_state=False)
        assert fasted["gastric_emptying_h"] < fed["gastric_emptying_h"]

    def test_fasted_gastric_emptying(self):
        lag = calculate_lag_times("immediate_release", fasted_state=True)
        assert lag["gastric_emptying_h"] == pytest.approx(0.25)

    def test_fed_gastric_emptying(self):
        lag = calculate_lag_times("immediate_release", fasted_state=False)
        assert lag["gastric_emptying_h"] == pytest.approx(1.0)

    def test_enteric_coated_has_coating_dissolution(self):
        lag = calculate_lag_times("enteric_coated", coating_thickness_um=50.0)
        assert lag["coating_dissolution_h"] > 0.0

    def test_immediate_release_no_coating(self):
        lag = calculate_lag_times("immediate_release")
        assert lag["coating_dissolution_h"] == pytest.approx(0.0)

    def test_tablet_no_coating(self):
        lag = calculate_lag_times("tablet")
        assert lag["coating_dissolution_h"] == pytest.approx(0.0)

    def test_capsule_no_coating(self):
        lag = calculate_lag_times("capsule")
        assert lag["coating_dissolution_h"] == pytest.approx(0.0)

    def test_enteric_coated_no_disintegration(self):
        lag = calculate_lag_times("enteric_coated")
        assert lag["disintegration_h"] == pytest.approx(0.0)

    def test_total_equals_sum(self):
        for ft in ["immediate_release", "enteric_coated", "tablet", "capsule"]:
            lag = calculate_lag_times(ft)
            expected = (
                lag["gastric_emptying_h"]
                + lag["disintegration_h"]
                + lag["coating_dissolution_h"]
            )
            assert lag["total_h"] == pytest.approx(expected, rel=1e-6)

    def test_larger_tablet_longer_disintegration(self):
        small = calculate_lag_times("tablet", tablet_size_mg=100.0)
        large = calculate_lag_times("tablet", tablet_size_mg=1000.0)
        assert large["disintegration_h"] > small["disintegration_h"]

    def test_thicker_coating_longer_dissolution(self):
        thin = calculate_lag_times("enteric_coated", coating_thickness_um=10.0)
        thick = calculate_lag_times("enteric_coated", coating_thickness_um=200.0)
        assert thick["coating_dissolution_h"] > thin["coating_dissolution_h"]

    def test_invalid_formulation_raises(self):
        with pytest.raises(ValueError, match="formulation_type"):
            calculate_lag_times("suppository")


# ---------------------------------------------------------------------------
# simulate_absorption_with_lag
# ---------------------------------------------------------------------------


class TestSimulateAbsorptionWithLag:
    def test_returns_dataclass(self):
        r = _sim()
        assert isinstance(r, LagTimeResult)

    def test_frozen_dataclass(self):
        r = _sim()
        with pytest.raises((AttributeError, TypeError)):
            r.tlag_total_h = 0.0  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = _sim(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_formulation_type_preserved(self):
        r = _sim(formulation_type="tablet")
        assert r.formulation_type == "tablet"

    def test_tmax_with_lag_gt_without(self):
        r = _sim(formulation_type="enteric_coated")
        assert r.tmax_with_lag_h > r.tmax_without_lag_h

    def test_tmax_lag_approx_tmax_plus_tlag(self):
        r = _sim(formulation_type="immediate_release")
        assert r.tmax_with_lag_h == pytest.approx(
            r.tmax_without_lag_h + r.tlag_total_h, rel=1e-3
        )

    def test_c_plasma_all_nonneg(self):
        r = _sim()
        assert all(c >= 0.0 for c in r.c_plasma_mg_L)

    def test_c_plasma_no_lag_all_nonneg(self):
        r = _sim()
        assert all(c >= 0.0 for c in r.c_plasma_no_lag_mg_L)

    def test_times_and_concentrations_same_length(self):
        r = _sim()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_plasma_no_lag_mg_L)

    def test_auc_positive(self):
        r = _sim()
        assert r.auc_mg_h_per_L > 0.0

    def test_enteric_coated_highest_lag(self):
        ec = _sim(formulation_type="enteric_coated")
        ir = _sim(formulation_type="immediate_release")
        assert ec.tlag_total_h > ir.tlag_total_h

    def test_capsule_lowest_lag(self):
        """Capsule disintegrates fastest (3 min) so has the smallest total lag."""
        cap = _sim(formulation_type="capsule")
        for ft in ["enteric_coated", "tablet", "immediate_release"]:
            other = _sim(formulation_type=ft)
            assert cap.tlag_total_h <= other.tlag_total_h

    def test_cmax_nonneg(self):
        r = _sim()
        assert r.cmax_mg_L >= 0.0

    def test_time_to_cmax_pct_positive(self):
        r = _sim(formulation_type="enteric_coated")
        assert r.time_to_cmax_pct_of_ref > 100.0  # lag makes tmax longer

    def test_cmax_reduction_pct_nonneg(self):
        r = _sim()
        assert r.cmax_reduction_pct >= 0.0

    def test_notes_is_string(self):
        r = _sim()
        assert isinstance(r.notes, str)

    def test_auc_similar_with_and_without_lag_long_window(self):
        """With a sufficiently long window, AUC should be similar for IR vs EC."""
        ir = simulate_absorption_with_lag(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            formulation_type="immediate_release",
            t_end_h=48.0,
        )
        ec = simulate_absorption_with_lag(
            drug_name="Drug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            formulation_type="enteric_coated",
            t_end_h=48.0,
        )
        # AUC within 10% — tlag shifts profile but same total exposure
        ratio = ec.auc_mg_h_per_L / ir.auc_mg_h_per_L
        assert 0.90 <= ratio <= 1.10


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=-1.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _sim(cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _sim(vd_L=0.0)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            _sim(ka_per_h=0.0)

    def test_f_oral_gt1_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _sim(f_oral=1.5)

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _sim(f_oral=0.0)


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_returns_4_results(self):
        results = compare_formulations("Drug", 500.0, 10.0, 50.0)
        assert len(results) == 4

    def test_sorted_by_lag_ascending(self):
        results = compare_formulations("Drug", 500.0, 10.0, 50.0)
        lags = [r.tlag_total_h for r in results]
        assert lags == sorted(lags)

    def test_all_results_are_dataclass(self):
        results = compare_formulations("Drug", 500.0, 10.0, 50.0)
        assert all(isinstance(r, LagTimeResult) for r in results)

    def test_first_has_smallest_lag(self):
        results = compare_formulations("Drug", 500.0, 10.0, 50.0)
        assert results[0].tlag_total_h == min(r.tlag_total_h for r in results)

    def test_last_has_largest_lag(self):
        results = compare_formulations("Drug", 500.0, 10.0, 50.0)
        assert results[-1].tlag_total_h == max(r.tlag_total_h for r in results)

    def test_kwargs_forwarded(self):
        results = compare_formulations("Drug", 500.0, 10.0, 50.0, ka_per_h=2.0)
        assert all(isinstance(r, LagTimeResult) for r in results)
