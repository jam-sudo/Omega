"""Tests for particle size-dependent dissolution (Phase 102)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.particle_dissolution import (
    ParticleResult,
    compare_particle_sizes,
    simulate_particle_dissolution,
)


def _default(**kw):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        particle_radius_um=10.0,
        solubility_mg_mL=0.5,
        t_end_h=6.0,
        dt_h=0.05,
    )
    defaults.update(kw)
    return simulate_particle_dissolution(**defaults)


class TestParticleResult:
    def test_returns_result_type(self):
        assert isinstance(_default(), ParticleResult)

    def test_dissolved_fraction_between_0_and_1(self):
        r = _default()
        assert all(0.0 <= f <= 1.0 for f in r.dissolved_fraction)

    def test_times_and_fraction_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.dissolved_fraction)

    def test_f_dissolved_1h_between_0_and_1(self):
        r = _default()
        assert 0.0 <= r.f_dissolved_1h <= 1.0

    def test_f_dissolved_4h_geq_1h(self):
        r = _default()
        assert r.f_dissolved_4h >= r.f_dissolved_1h - 1e-6

    def test_dissolution_rate_positive(self):
        r = _default()
        assert r.dissolution_rate_mg_h > 0.0

    def test_t63_within_simulation_time(self):
        r = _default()
        assert 0.0 <= r.t63_h <= 6.0

    def test_diffusion_layer_thickness_positive(self):
        r = _default()
        assert r.diffusion_layer_thickness_um > 0.0


class TestDissolutionPhysics:
    def test_smaller_particle_dissolves_faster(self):
        r_small = _default(particle_radius_um=2.0)
        r_large = _default(particle_radius_um=20.0)
        assert r_small.f_dissolved_4h >= r_large.f_dissolved_4h - 1e-6

    def test_higher_solubility_faster(self):
        r_high = _default(solubility_mg_mL=5.0)
        r_low = _default(solubility_mg_mL=0.1)
        assert r_high.f_dissolved_4h >= r_low.f_dissolved_4h - 1e-6

    def test_dissolution_fraction_non_decreasing(self):
        r = _default()
        fracs = r.dissolved_fraction
        for i in range(1, len(fracs)):
            assert fracs[i] >= fracs[i - 1] - 1e-9


class TestValidation:
    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            _default(particle_radius_um=-1.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            _default(solubility_mg_mL=0.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)


class TestCompareParticleSizes:
    def test_returns_list_of_results(self):
        results = compare_particle_sizes("Drug", 100.0, [5.0, 10.0, 20.0], 0.5)
        assert isinstance(results, list)
        assert all(isinstance(r, ParticleResult) for r in results)

    def test_length_matches_radii(self):
        results = compare_particle_sizes("Drug", 100.0, [5.0, 10.0, 20.0], 0.5)
        assert len(results) == 3
