"""Tests for Phase 665 — Drug Cartilage Penetration."""

import pytest

from omega_pbpk.core.cartilage_penetration_p665 import (
    CartilagePenetrationResult,
    simulate_cartilage_penetration,
)

# Default parameters for convenience
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=400.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_cartilage_penetration(**params)


class TestReturnType:
    def test_returns_result(self):
        r = _run()
        assert isinstance(r, CartilagePenetrationResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0


class TestListLengths:
    def test_times_length(self):
        r = _run(t_end_h=10.0, dt_h=0.1)
        expected = int(10.0 / 0.1) + 1
        assert len(r.times_h) == expected

    def test_synovial_length(self):
        r = _run()
        assert len(r.c_synovial_mg_L) == len(r.times_h)

    def test_surface_length(self):
        r = _run()
        assert len(r.c_cartilage_surface_mg_g) == len(r.times_h)

    def test_deep_length(self):
        r = _run()
        assert len(r.c_cartilage_deep_mg_g) == len(r.times_h)


class TestNonNegative:
    def test_synovial_non_negative(self):
        r = _run()
        assert all(v >= 0 for v in r.c_synovial_mg_L)

    def test_surface_non_negative(self):
        r = _run()
        assert all(v >= 0 for v in r.c_cartilage_surface_mg_g)

    def test_deep_non_negative(self):
        r = _run()
        assert all(v >= 0 for v in r.c_cartilage_deep_mg_g)

    def test_diffusion_coefficient_positive(self):
        r = _run()
        assert r.diffusion_coefficient_cm2_h > 0


class TestPenetrationBehavior:
    def test_surface_leads_deep_early(self):
        """Surface concentration should be higher than deep early on."""
        r = _run()
        # Check at ~10% into simulation
        idx = len(r.times_h) // 10
        if idx > 0:
            assert r.c_cartilage_surface_mg_g[idx] >= r.c_cartilage_deep_mg_g[idx]

    def test_deep_lower_than_surface_overall(self):
        """Deep cartilage AUC should be less than surface AUC."""
        r = _run()
        assert r.cartilage_auc_deep <= r.cartilage_auc_surface

    def test_penetration_depth_range(self):
        r = _run()
        assert 0 <= r.penetration_depth_pct <= 100

    def test_auc_surface_positive(self):
        r = _run()
        assert r.cartilage_auc_surface > 0

    def test_auc_deep_positive(self):
        r = _run()
        assert r.cartilage_auc_deep > 0


class TestMWEffect:
    def test_lower_mw_higher_diffusion(self):
        """Lower MW should give higher diffusion coefficient."""
        r_low = _run(mw_Da=200.0)
        r_high = _run(mw_Da=800.0)
        assert r_low.diffusion_coefficient_cm2_h > r_high.diffusion_coefficient_cm2_h

    def test_lower_mw_better_penetration(self):
        """Lower MW should give better penetration (higher surface AUC)."""
        r_low = _run(mw_Da=200.0)
        r_high = _run(mw_Da=800.0)
        assert r_low.cartilage_auc_surface > r_high.cartilage_auc_surface


class TestDoseProportionality:
    def test_double_dose_doubles_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cartilage_auc_surface / r1.cartilage_auc_surface
        assert 1.8 < ratio < 2.2

    def test_double_dose_doubles_deep_auc(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cartilage_auc_deep / r1.cartilage_auc_deep
        assert 1.8 < ratio < 2.2


class TestSteadyState:
    def test_steady_state_surface_non_negative(self):
        r = _run()
        assert r.steady_state_surface_mg_g >= 0

    def test_steady_state_deep_non_negative(self):
        r = _run()
        assert r.steady_state_deep_mg_g >= 0

    def test_surface_to_synovial_ratio_positive(self):
        r = _run()
        assert r.surface_to_synovial_ratio > 0


class TestTimeTo50:
    def test_time_to_50_positive(self):
        r = _run()
        assert r.time_to_50pct_ss_h > 0

    def test_time_to_50_less_than_end(self):
        r = _run()
        assert r.time_to_50pct_ss_h <= 72.0


class TestNotes:
    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0


class TestValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_dose_negative(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-10)

    def test_cl_zero(self):
        with pytest.raises(ValueError):
            _run(cl_sys_L_per_h=0)

    def test_vd_zero(self):
        with pytest.raises(ValueError):
            _run(vd_sys_L=0)

    def test_mw_zero(self):
        with pytest.raises(ValueError):
            _run(mw_Da=0)
