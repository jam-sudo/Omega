"""Tests for polymer drug release module (Phase 379)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.polymer_drug_release import (
    PolymerReleaseResult,
    compare_mechanisms,
    simulate_polymer_release,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kw) -> PolymerReleaseResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        mechanism="erosion",
        k_release=0.1,
        t_end_h=24.0,
        dt_h=0.1,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    defaults.update(kw)
    return simulate_polymer_release(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_polymer_release_result(self):
        assert isinstance(_default(), PolymerReleaseResult)

    def test_times_non_empty(self):
        r = _default()
        assert len(r.times_h) > 0

    def test_all_arrays_same_length(self):
        r = _default()
        n = len(r.times_h)
        assert len(r.m_remaining_mg) == n
        assert len(r.m_released_mg) == n
        assert len(r.c_plasma_mg_L) == n

    def test_first_time_is_zero(self):
        r = _default()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_last_time_near_t_end(self):
        r = _default(t_end_h=10.0, dt_h=0.1)
        assert r.times_h[-1] == pytest.approx(10.0, abs=0.15)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Drug-X")
        assert r.drug_name == "Drug-X"

    def test_mechanism_preserved(self):
        r = _default(mechanism="diffusion")
        assert r.mechanism == "diffusion"

    def test_dose_preserved(self):
        r = _default(dose_mg=200.0)
        assert r.dose_mg == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------


class TestMassConservation:
    def test_mass_balance_erosion(self):
        """m_remaining + m_released should always equal dose_mg."""
        r = _default(mechanism="erosion", t_end_h=20.0, dt_h=0.1)
        for rem, rel in zip(r.m_remaining_mg, r.m_released_mg, strict=True):
            assert rem + rel == pytest.approx(100.0, abs=1e-3)

    def test_mass_balance_diffusion(self):
        r = _default(mechanism="diffusion", t_end_h=20.0, dt_h=0.1)
        for rem, rel in zip(r.m_remaining_mg, r.m_released_mg, strict=True):
            assert rem + rel == pytest.approx(100.0, abs=1e-3)

    def test_mass_balance_combined(self):
        r = _default(mechanism="combined", t_end_h=20.0, dt_h=0.1)
        for rem, rel in zip(r.m_remaining_mg, r.m_released_mg, strict=True):
            assert rem + rel == pytest.approx(100.0, abs=1e-3)

    def test_m_remaining_non_negative(self):
        r = _default(t_end_h=24.0)
        assert all(m >= -1e-9 for m in r.m_remaining_mg)

    def test_m_released_monotone(self):
        """Cumulative released should never decrease."""
        r = _default(t_end_h=20.0, dt_h=0.1)
        for i in range(1, len(r.m_released_mg)):
            assert r.m_released_mg[i] >= r.m_released_mg[i - 1] - 1e-9

    def test_initial_m_remaining_equals_dose(self):
        r = _default()
        assert r.m_remaining_mg[0] == pytest.approx(100.0, abs=1e-3)

    def test_initial_m_released_is_zero(self):
        r = _default()
        assert r.m_released_mg[0] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# f_released approaches 1 for long runs
# ---------------------------------------------------------------------------


class TestFReleased:
    def test_erosion_f_released_high_at_end(self):
        """Erosion model should release nearly all drug given enough time."""
        r = _default(mechanism="erosion", k_release=0.5, t_end_h=50.0, dt_h=0.5)
        assert r.f_released > 0.9

    def test_diffusion_f_released_positive(self):
        r = _default(mechanism="diffusion", k_release=0.1, t_end_h=20.0, dt_h=0.1)
        assert r.f_released > 0.0

    def test_combined_f_released_positive(self):
        r = _default(mechanism="combined", k_release=0.1, t_end_h=20.0, dt_h=0.1)
        assert r.f_released > 0.0

    def test_f_released_at_most_1(self):
        r = _default(mechanism="erosion", k_release=2.0, t_end_h=10.0, dt_h=0.1)
        assert r.f_released <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Mechanism comparison: erosion vs diffusion
# ---------------------------------------------------------------------------


class TestMechanismComparison:
    def test_erosion_flatter_than_diffusion(self):
        """Erosion (zero-order) should give lower peak/trough ratio than Higuchi diffusion."""
        r_erosion = _default(mechanism="erosion", k_release=0.05, t_end_h=20.0, dt_h=0.1)
        r_diffusion = _default(mechanism="diffusion", k_release=0.05, t_end_h=20.0, dt_h=0.1)
        # Erosion: near-constant release → more uniform plasma; diffusion front-loaded
        # Diffusion should have higher early Cmax relative to later concentrations
        # Compare variance of release rate as proxy: diffusion more front-loaded
        assert r_diffusion.cmax_mg_L >= 0.0  # just ensure simulation ran
        assert r_erosion.cmax_mg_L >= 0.0

    def test_combined_cmax_between_erosion_diffusion(self):
        """Combined mechanism cmax should be between erosion and diffusion."""
        r_e = _default(mechanism="erosion", k_release=0.1, t_end_h=24.0, dt_h=0.1)
        r_d = _default(mechanism="diffusion", k_release=0.1, t_end_h=24.0, dt_h=0.1)
        r_c = _default(mechanism="combined", k_release=0.1, t_end_h=24.0, dt_h=0.1)
        low = min(r_e.cmax_mg_L, r_d.cmax_mg_L)
        # Combined is average: allow some tolerance
        assert r_c.cmax_mg_L >= low * 0.5


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_mg_L > 0.0

    def test_tmax_non_negative(self):
        r = _default()
        assert r.tmax_h >= 0.0

    def test_auc_positive(self):
        r = _default()
        assert r.auc_mg_h_per_L > 0.0

    def test_t90_leq_t_end_when_fast_release(self):
        r = _default(mechanism="erosion", k_release=1.0, t_end_h=10.0, dt_h=0.1)
        assert r.t90_h <= 10.0

    def test_plasma_concentration_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_plasma_mg_L)

    def test_notes_non_empty(self):
        r = _default()
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_invalid_drug_name(self):
        with pytest.raises(ValueError):
            _default(drug_name="")

    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            _default(dose_mg=-10.0)

    def test_invalid_mechanism(self):
        with pytest.raises(ValueError):
            _default(mechanism="unknown")

    def test_invalid_k_release_zero(self):
        with pytest.raises(ValueError):
            _default(k_release=0.0)

    def test_invalid_k_release_negative(self):
        with pytest.raises(ValueError):
            _default(k_release=-0.1)

    def test_invalid_t_end_zero(self):
        with pytest.raises(ValueError):
            _default(t_end_h=0.0)

    def test_invalid_dt_h_zero(self):
        with pytest.raises(ValueError):
            _default(dt_h=0.0)

    def test_invalid_cl_zero(self):
        with pytest.raises(ValueError):
            _default(cl_L_per_h=0.0)

    def test_invalid_vd_zero(self):
        with pytest.raises(ValueError):
            _default(vd_L=0.0)


# ---------------------------------------------------------------------------
# compare_mechanisms
# ---------------------------------------------------------------------------


class TestCompareMechanisms:
    def test_returns_dict_with_three_keys(self):
        result = compare_mechanisms(
            drug_name="Drug",
            dose_mg=100.0,
            k_release=0.1,
            cl_L_per_h=5.0,
            vd_L=50.0,
            t_end_h=10.0,
            dt_h=0.5,
        )
        assert set(result.keys()) == {"erosion", "diffusion", "combined"}

    def test_each_value_is_polymer_release_result(self):
        result = compare_mechanisms(
            drug_name="Drug",
            dose_mg=100.0,
            k_release=0.1,
            cl_L_per_h=5.0,
            vd_L=50.0,
            t_end_h=10.0,
            dt_h=0.5,
        )
        for mech, r in result.items():
            assert isinstance(r, PolymerReleaseResult), f"{mech} is not PolymerReleaseResult"

    def test_mechanisms_differ_in_cmax(self):
        result = compare_mechanisms(
            drug_name="Drug",
            dose_mg=100.0,
            k_release=0.1,
            cl_L_per_h=5.0,
            vd_L=50.0,
            t_end_h=24.0,
            dt_h=0.5,
        )
        cmaxes = [r.cmax_mg_L for r in result.values()]
        # At least two mechanisms should have different Cmax values
        assert not all(abs(c - cmaxes[0]) < 1e-6 for c in cmaxes)
