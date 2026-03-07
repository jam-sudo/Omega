"""Tests for biopharmaceutics/dissolution_rate_limited.py — Phase 535."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.dissolution_rate_limited import (
    DissolutionAbsorptionResult,
    dissolution_rate,
    particle_size_effect,
    simulate_dissolution_absorption,
)

# ---------------------------------------------------------------------------
# dissolution_rate tests
# ---------------------------------------------------------------------------


class TestDissolutionRate:
    def test_basic_rate_positive(self):
        rate = dissolution_rate(
            cs_mg_mL=1.0,
            c_dissolved_mg_mL=0.0,
            surface_area_cm2=10.0,
            D_cm2_h=0.018,
            h_cm=0.005,
        )
        assert rate > 0.0

    def test_no_driving_force_zero_rate(self):
        """At saturation (C == Cs) the rate should be zero."""
        rate = dissolution_rate(
            cs_mg_mL=1.0,
            c_dissolved_mg_mL=1.0,
            surface_area_cm2=10.0,
            D_cm2_h=0.018,
        )
        assert rate == 0.0

    def test_supersaturation_zero_rate(self):
        """When C > Cs the rate should be zero (non-negative)."""
        rate = dissolution_rate(
            cs_mg_mL=0.5,
            c_dissolved_mg_mL=1.0,
            surface_area_cm2=10.0,
            D_cm2_h=0.018,
        )
        assert rate == 0.0

    def test_zero_surface_area(self):
        rate = dissolution_rate(
            cs_mg_mL=1.0,
            c_dissolved_mg_mL=0.0,
            surface_area_cm2=0.0,
            D_cm2_h=0.018,
        )
        assert rate == 0.0

    def test_rate_proportional_to_surface_area(self):
        r1 = dissolution_rate(1.0, 0.0, 5.0, 0.018)
        r2 = dissolution_rate(1.0, 0.0, 10.0, 0.018)
        assert math.isclose(r2, 2.0 * r1, rel_tol=1e-6)

    def test_rate_proportional_to_diffusion_coeff(self):
        r1 = dissolution_rate(1.0, 0.0, 10.0, 0.018)
        r2 = dissolution_rate(1.0, 0.0, 10.0, 0.036)
        assert math.isclose(r2, 2.0 * r1, rel_tol=1e-6)

    def test_default_h_used(self):
        """Calling without h_cm should use 0.005 cm default."""
        r_default = dissolution_rate(1.0, 0.0, 10.0, 0.018)
        r_explicit = dissolution_rate(1.0, 0.0, 10.0, 0.018, h_cm=0.005)
        assert math.isclose(r_default, r_explicit, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# simulate_dissolution_absorption tests
# ---------------------------------------------------------------------------


class TestSimulateDissolutionAbsorption:
    def _run_default(self, **kwargs):
        defaults = dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            cs_mg_mL=1.0,
            particle_radius_um=50.0,
            peff_cm_s=1e-4,
            vd_L=50.0,
            cl_L_per_h=5.0,
            t_end_h=12.0,
            dt_h=0.1,
        )
        defaults.update(kwargs)
        return simulate_dissolution_absorption(**defaults)

    def test_returns_result_type(self):
        res = self._run_default()
        assert isinstance(res, DissolutionAbsorptionResult)

    def test_fa_in_unit_interval(self):
        res = self._run_default()
        assert 0.0 <= res.fa <= 1.0

    def test_cmax_positive(self):
        res = self._run_default()
        assert res.cmax >= 0.0

    def test_auc_positive(self):
        res = self._run_default()
        assert res.auc >= 0.0

    def test_tmax_in_range(self):
        res = self._run_default()
        assert 0.0 <= res.tmax_h <= 12.0

    def test_times_list_length(self):
        res = self._run_default(t_end_h=6.0, dt_h=0.1)
        # n_steps = 60; times has 61 entries
        assert len(res.times_h) == 61
        assert len(res.m_undissolved_mg) == 61
        assert len(res.c_lumen_mg_mL) == 61
        assert len(res.c_plasma_mg_L) == 61

    def test_small_particles_higher_fa_than_large(self):
        """Smaller particles dissolve faster → higher fa."""
        res_small = self._run_default(particle_radius_um=5.0, cs_mg_mL=0.5, dose_mg=50.0)
        res_large = self._run_default(particle_radius_um=100.0, cs_mg_mL=0.5, dose_mg=50.0)
        assert res_small.fa >= res_large.fa

    def test_high_cs_not_dissolution_limited(self):
        """Very high solubility → dissolves quickly → fa close to 1 → not dissolution limited."""
        res = self._run_default(
            cs_mg_mL=100.0,
            particle_radius_um=10.0,
            dose_mg=10.0,
            peff_cm_s=1e-3,
            t_end_h=24.0,
        )
        assert not res.dissolution_limited

    def test_low_cs_dissolution_limited(self):
        """Low solubility + large particles + limited time → dissolution limited."""
        res = self._run_default(
            cs_mg_mL=0.01,
            particle_radius_um=200.0,
            dose_mg=200.0,
            t_end_h=8.0,
        )
        assert res.dissolution_limited

    def test_low_peff_absorption_limited(self):
        """Very low permeability: drug dissolves but is not absorbed → still low fa."""
        res = self._run_default(
            cs_mg_mL=50.0,
            particle_radius_um=5.0,
            peff_cm_s=1e-9,
            dose_mg=20.0,
            t_end_h=12.0,
        )
        assert res.fa < 0.5

    def test_dissolution_limited_flag_consistency(self):
        """dissolution_limited should equal fa < 0.8."""
        res = self._run_default()
        assert res.dissolution_limited == (res.fa < 0.8)

    def test_double_dose_not_double_auc_when_limited(self):
        """If dissolution-limited, 2x dose should NOT give 2x AUC."""
        res1 = self._run_default(cs_mg_mL=0.05, particle_radius_um=150.0, dose_mg=50.0)
        res2 = self._run_default(cs_mg_mL=0.05, particle_radius_um=150.0, dose_mg=100.0)
        if res1.dissolution_limited and res2.dissolution_limited:
            # AUC should increase but not double
            assert res2.auc < 2.0 * res1.auc + 1e-6

    def test_validation_cs_zero(self):
        with pytest.raises(ValueError, match="cs_mg_mL"):
            simulate_dissolution_absorption("X", 100.0, 0.0, 50.0)

    def test_validation_cs_negative(self):
        with pytest.raises(ValueError, match="cs_mg_mL"):
            simulate_dissolution_absorption("X", 100.0, -1.0, 50.0)

    def test_validation_particle_radius_zero(self):
        with pytest.raises(ValueError, match="particle_radius_um"):
            simulate_dissolution_absorption("X", 100.0, 1.0, 0.0)

    def test_validation_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_dissolution_absorption("X", 0.0, 1.0, 50.0)

    def test_notes_nonempty(self):
        res = self._run_default()
        assert len(res.notes) > 0


# ---------------------------------------------------------------------------
# particle_size_effect tests
# ---------------------------------------------------------------------------


class TestParticleSizeEffect:
    def test_returns_list_of_dicts(self):
        results = particle_size_effect("Drug", 50.0, 0.5, [5.0, 50.0, 200.0])
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_dict_keys(self):
        results = particle_size_effect("Drug", 50.0, 0.5, [10.0, 50.0])
        for r in results:
            assert {"radius_um", "fa", "cmax", "auc", "dissolution_limited"} == set(r.keys())

    def test_sorted_by_radius_ascending(self):
        radii = [200.0, 5.0, 50.0]
        results = particle_size_effect("Drug", 50.0, 0.5, radii)
        returned_radii = [r["radius_um"] for r in results]
        assert returned_radii == sorted(returned_radii)

    def test_smaller_radius_higher_auc(self):
        """Smaller particles → more dissolution → higher AUC."""
        results = particle_size_effect("Drug", 30.0, 0.3, [10.0, 200.0])
        auc_small = next(r["auc"] for r in results if r["radius_um"] == 10.0)
        auc_large = next(r["auc"] for r in results if r["radius_um"] == 200.0)
        assert auc_small >= auc_large

    def test_fa_values_in_unit_interval(self):
        results = particle_size_effect("Drug", 50.0, 1.0, [5.0, 10.0, 50.0])
        for r in results:
            assert 0.0 <= r["fa"] <= 1.0

    def test_dissolution_limited_flag_consistent(self):
        results = particle_size_effect("Drug", 50.0, 0.5, [5.0, 50.0, 200.0])
        for r in results:
            assert r["dissolution_limited"] == (r["fa"] < 0.8)
