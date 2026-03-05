"""Tests for circadian pharmacokinetics module."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.core.chronopk import (
    CYP3A4_RHYTHM,
    GFR_RHYTHM,
    HEPATIC_FLOW_RHYTHM,
    ChronoPKResult,
    CircadianPhysiology,
    CircadianRhythm,
    compare_dosing_times,
    simulate_chronopk,
)

# ── CircadianRhythm ────────────────────────────────────────────────


class TestCircadianRhythm:
    def test_value_at_acrophase_equals_peak(self):
        r = CircadianRhythm(mesor=100.0, amplitude=20.0, acrophase_h=14.0)
        assert r.value_at(14.0) == pytest.approx(120.0)

    def test_value_at_nadir(self):
        r = CircadianRhythm(mesor=100.0, amplitude=20.0, acrophase_h=14.0)
        # Nadir at acrophase + 12h = 02:00
        assert r.value_at(2.0) == pytest.approx(80.0)

    def test_mesor_is_24h_mean(self):
        r = CircadianRhythm(mesor=100.0, amplitude=20.0, acrophase_h=14.0)
        times = np.linspace(0, 24, 10000, endpoint=False)
        vals = r.values_at(times)
        assert float(np.mean(vals)) == pytest.approx(100.0, abs=0.1)

    def test_values_at_vectorized(self):
        r = CircadianRhythm(mesor=50.0, amplitude=10.0, acrophase_h=8.0)
        times = np.array([8.0, 20.0])
        vals = r.values_at(times)
        assert vals[0] == pytest.approx(60.0)  # peak
        assert vals[1] == pytest.approx(40.0)  # nadir

    def test_zero_amplitude_constant(self):
        r = CircadianRhythm(mesor=100.0, amplitude=0.0, acrophase_h=14.0)
        for t in [0, 6, 12, 18]:
            assert r.value_at(float(t)) == pytest.approx(100.0)

    def test_custom_period(self):
        r = CircadianRhythm(mesor=100.0, amplitude=20.0, acrophase_h=0.0, period_h=12.0)
        assert r.value_at(0.0) == pytest.approx(120.0)
        assert r.value_at(6.0) == pytest.approx(80.0)  # half period = nadir


# ── Default rhythms ────────────────────────────────────────────────


class TestDefaultRhythms:
    def test_gfr_rhythm_mesor(self):
        assert GFR_RHYTHM.mesor == pytest.approx(120.0)
        assert GFR_RHYTHM.acrophase_h == pytest.approx(14.0)

    def test_hepatic_flow_rhythm(self):
        assert HEPATIC_FLOW_RHYTHM.mesor == pytest.approx(90.0)

    def test_cyp3a4_rhythm(self):
        assert CYP3A4_RHYTHM.mesor == pytest.approx(1.0)
        assert CYP3A4_RHYTHM.amplitude == pytest.approx(0.3)


# ── simulate_chronopk ──────────────────────────────────────────────


class TestSimulateChronoPK:
    def test_returns_result_type(self):
        result = simulate_chronopk(
            dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0, route="oral"
        )
        assert isinstance(result, ChronoPKResult)

    def test_oral_cmax_positive(self):
        result = simulate_chronopk(
            dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.5, route="oral"
        )
        assert result.cmax > 0

    def test_iv_cmax_at_t0(self):
        result = simulate_chronopk(dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, route="iv")
        # IV Cmax should be at or very near t=0
        assert result.tmax_h < 0.1
        assert result.cmax == pytest.approx(100.0 / 50.0, rel=0.05)

    def test_auc_positive(self):
        result = simulate_chronopk(
            dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0, route="oral"
        )
        assert result.auc_0_24 > 0

    def test_time_length_matches_conc(self):
        result = simulate_chronopk(
            dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0, route="oral"
        )
        assert len(result.times_h) == len(result.plasma_conc)

    def test_physiology_profiles_present(self):
        result = simulate_chronopk(
            dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0, route="oral"
        )
        assert len(result.gfr_profile) == len(result.times_h)
        assert len(result.hepatic_flow_profile) == len(result.times_h)
        assert len(result.cyp3a4_profile) == len(result.times_h)

    def test_dosing_time_stored(self):
        result = simulate_chronopk(
            dose_mg=100,
            cl_base_L_per_h=5.0,
            vd_L=50.0,
            ka_per_h=1.0,
            dosing_time_h=22.0,
            route="oral",
        )
        assert result.dosing_time_h == pytest.approx(22.0)

    def test_different_dosing_times_different_auc(self):
        r1 = simulate_chronopk(
            dose_mg=100,
            cl_base_L_per_h=5.0,
            vd_L=50.0,
            ka_per_h=1.0,
            dosing_time_h=8.0,
            route="oral",
        )
        r2 = simulate_chronopk(
            dose_mg=100,
            cl_base_L_per_h=5.0,
            vd_L=50.0,
            ka_per_h=1.0,
            dosing_time_h=22.0,
            route="oral",
        )
        # AUC should differ due to circadian CL variation
        assert r1.auc_0_24 != pytest.approx(r2.auc_0_24, rel=0.001)

    def test_no_circadian_constant_cl(self):
        flat = CircadianPhysiology(
            gfr=CircadianRhythm(120.0, 0.0, 0.0),
            hepatic_flow=CircadianRhythm(90.0, 0.0, 0.0),
            cyp3a4=CircadianRhythm(1.0, 0.0, 0.0),
        )
        r1 = simulate_chronopk(
            dose_mg=100,
            cl_base_L_per_h=5.0,
            vd_L=50.0,
            ka_per_h=1.0,
            dosing_time_h=8.0,
            route="oral",
            physiology=flat,
        )
        r2 = simulate_chronopk(
            dose_mg=100,
            cl_base_L_per_h=5.0,
            vd_L=50.0,
            ka_per_h=1.0,
            dosing_time_h=22.0,
            route="oral",
            physiology=flat,
        )
        # With flat rhythms, dosing time shouldn't matter
        assert r1.auc_0_24 == pytest.approx(r2.auc_0_24, rel=0.01)

    def test_cl_effective_near_base(self):
        result = simulate_chronopk(
            dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0, route="oral"
        )
        # Effective CL should be close to base CL (within ~20% due to circadian)
        assert result.cl_effective_L_per_h == pytest.approx(5.0, rel=0.3)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_chronopk(dose_mg=0, cl_base_L_per_h=5.0, vd_L=50.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError):
            simulate_chronopk(dose_mg=100, cl_base_L_per_h=-1.0, vd_L=50.0)


# ── compare_dosing_times ───────────────────────────────────────────


class TestCompareDoseTimes:
    def test_default_five_times(self):
        results = compare_dosing_times(dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0)
        assert len(results) == 5
        assert all(isinstance(r, ChronoPKResult) for r in results)

    def test_custom_times(self):
        results = compare_dosing_times(
            dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0, dosing_times_h=[6.0, 18.0]
        )
        assert len(results) == 2
        assert results[0].dosing_time_h == pytest.approx(6.0)
        assert results[1].dosing_time_h == pytest.approx(18.0)

    def test_all_results_have_positive_auc(self):
        results = compare_dosing_times(dose_mg=100, cl_base_L_per_h=5.0, vd_L=50.0, ka_per_h=1.0)
        for r in results:
            assert r.auc_0_24 > 0
