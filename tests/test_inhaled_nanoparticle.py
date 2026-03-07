"""Tests for inhaled nanoparticle lung deposition model (Phase 335)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.inhaled_nanoparticle import (
    InhaledNanoparticleResult,
    calculate_lung_deposition,
    compare_particle_sizes,
    simulate_inhaled_nanoparticle_pk,
)

# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError, match="particle_diameter_um"):
            calculate_lung_deposition("Drug", 0.0, 1.0)

    def test_negative_diameter_raises(self):
        with pytest.raises(ValueError, match="particle_diameter_um"):
            calculate_lung_deposition("Drug", -1.0, 1.0)

    def test_diameter_too_large_raises(self):
        with pytest.raises(ValueError, match="particle_diameter_um"):
            calculate_lung_deposition("Drug", 101.0, 1.0)

    def test_diameter_at_boundary_100_ok(self):
        dep = calculate_lung_deposition("Drug", 100.0, 1.0)
        assert dep["f_lung_total"] >= 0.0

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_inhaled_mg"):
            calculate_lung_deposition("Drug", 1.0, 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_inhaled_mg"):
            calculate_lung_deposition("Drug", 1.0, -5.0)

    def test_zero_flow_rate_raises(self):
        with pytest.raises(ValueError, match="flow_rate_L_min"):
            calculate_lung_deposition("Drug", 1.0, 1.0, flow_rate_L_min=0.0)

    def test_negative_flow_rate_raises(self):
        with pytest.raises(ValueError, match="flow_rate_L_min"):
            calculate_lung_deposition("Drug", 1.0, 1.0, flow_rate_L_min=-5.0)

    def test_simulate_invalid_diameter_raises(self):
        with pytest.raises(ValueError):
            simulate_inhaled_nanoparticle_pk("Drug", 0.0, 1.0)


# ---------------------------------------------------------------------------
# Deposition fraction sanity tests
# ---------------------------------------------------------------------------


class TestDepositionFractions:
    def test_fractions_sum_to_one_or_less_small_particle(self):
        dep = calculate_lung_deposition("Drug", 1.0, 10.0)
        total = (
            dep["f_oropharyngeal"]
            + dep["f_tracheobronchial"]
            + dep["f_alveolar"]
            + dep["f_exhaled"]
        )
        assert abs(total - 1.0) < 1e-9

    def test_fractions_sum_to_one_or_less_large_particle(self):
        dep = calculate_lung_deposition("Drug", 10.0, 10.0)
        total = (
            dep["f_oropharyngeal"]
            + dep["f_tracheobronchial"]
            + dep["f_alveolar"]
            + dep["f_exhaled"]
        )
        assert abs(total - 1.0) < 1e-9

    def test_all_fractions_non_negative(self):
        for d in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            dep = calculate_lung_deposition("Drug", d, 1.0)
            assert dep["f_oropharyngeal"] >= 0.0
            assert dep["f_tracheobronchial"] >= 0.0
            assert dep["f_alveolar"] >= 0.0
            assert dep["f_exhaled"] >= 0.0

    def test_f_exhaled_non_negative(self):
        for d in [0.5, 1.0, 2.0, 5.0, 10.0]:
            dep = calculate_lung_deposition("Drug", d, 1.0)
            assert dep["f_exhaled"] >= 0.0

    def test_large_particle_high_oropharyngeal(self):
        # 10 µm particles should have notably higher ORP deposition than 1 µm
        dep_large = calculate_lung_deposition("Drug", 10.0, 1.0)
        dep_small = calculate_lung_deposition("Drug", 1.0, 1.0)
        assert dep_large["f_oropharyngeal"] > dep_small["f_oropharyngeal"]

    def test_small_particle_higher_alveolar_than_large(self):
        dep_small = calculate_lung_deposition("Drug", 1.0, 1.0)
        dep_large = calculate_lung_deposition("Drug", 10.0, 1.0)
        assert dep_small["f_alveolar"] > dep_large["f_alveolar"]

    def test_oropharyngeal_near_zero_for_tiny_particle(self):
        dep = calculate_lung_deposition("Drug", 0.1, 1.0)
        assert dep["f_oropharyngeal"] < 0.01


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------


class TestSimulationResult:
    def test_result_is_inhaled_nanoparticle_result(self):
        result = simulate_inhaled_nanoparticle_pk("TestDrug", 1.0, 10.0)
        assert isinstance(result, InhaledNanoparticleResult)

    def test_bioavailability_positive(self):
        result = simulate_inhaled_nanoparticle_pk("Drug", 1.0, 10.0)
        assert result.bioavailability_pct > 0.0

    def test_bioavailability_at_most_100(self):
        for d in [0.5, 1.0, 2.0, 5.0, 10.0]:
            result = simulate_inhaled_nanoparticle_pk("Drug", d, 10.0)
            assert result.bioavailability_pct <= 100.0

    def test_m_absorbed_total_consistent(self):
        result = simulate_inhaled_nanoparticle_pk("Drug", 1.0, 10.0)
        expected = (
            result.m_absorbed_oropharyngeal_mg
            + result.m_absorbed_tracheobronchial_mg
            + result.m_absorbed_alveolar_mg
        )
        assert abs(result.m_absorbed_total_mg - expected) < 1e-10

    def test_f_systemic_consistent_with_absorbed_mass(self):
        dose = 20.0
        result = simulate_inhaled_nanoparticle_pk("Drug", 1.0, dose)
        assert abs(result.f_systemic - result.m_absorbed_total_mg / dose) < 1e-10

    def test_notes_contains_drug_name(self):
        result = simulate_inhaled_nanoparticle_pk("Salbutamol", 1.0, 5.0)
        assert "Salbutamol" in result.notes

    def test_dose_linearity(self):
        # 2x dose → 2x absorbed mass
        r1 = simulate_inhaled_nanoparticle_pk("Drug", 2.0, 10.0)
        r2 = simulate_inhaled_nanoparticle_pk("Drug", 2.0, 20.0)
        assert abs(r2.m_absorbed_total_mg - 2.0 * r1.m_absorbed_total_mg) < 1e-9

    def test_f_lung_total_in_0_1(self):
        result = simulate_inhaled_nanoparticle_pk("Drug", 2.0, 10.0)
        assert 0.0 <= result.f_lung_total <= 1.0

    def test_dataclass_is_frozen(self):
        result = simulate_inhaled_nanoparticle_pk("Drug", 1.0, 10.0)
        with pytest.raises((AttributeError, TypeError)):
            result.bioavailability_pct = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compare_particle_sizes tests
# ---------------------------------------------------------------------------


class TestCompareParticleSizes:
    def test_default_returns_5_results(self):
        results = compare_particle_sizes("Drug", 10.0)
        assert len(results) == 5

    def test_custom_diameters(self):
        results = compare_particle_sizes("Drug", 10.0, diameters_um=[1.0, 5.0, 10.0])
        assert len(results) == 3

    def test_sorted_by_bioavailability_descending(self):
        results = compare_particle_sizes("Drug", 10.0)
        ba_vals = [r.bioavailability_pct for r in results]
        assert ba_vals == sorted(ba_vals, reverse=True)

    def test_all_results_are_correct_type(self):
        results = compare_particle_sizes("Drug", 10.0)
        for r in results:
            assert isinstance(r, InhaledNanoparticleResult)

    def test_drug_name_preserved(self):
        results = compare_particle_sizes("Budesonide", 5.0)
        for r in results:
            assert r.drug_name == "Budesonide"
