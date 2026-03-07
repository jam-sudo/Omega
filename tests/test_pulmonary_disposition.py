"""Tests for core/pulmonary_disposition.py — Phase 324."""

from __future__ import annotations

import pytest

from omega_pbpk.core.pulmonary_disposition import (
    PulmonaryPKResult,
    compare_particle_sizes,
    pulmonary_targeting_index,
    simulate_inhaled_pk,
)


# ---------------------------------------------------------------------------
# simulate_inhaled_pk — basic behaviour
# ---------------------------------------------------------------------------


class TestSimulateInhaledPK:
    def test_returns_pulmonary_pk_result(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0)
        assert isinstance(result, PulmonaryPKResult)

    def test_result_is_frozen(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.cmax_systemic = 0.0  # type: ignore[misc]

    def test_times_h_length_matches_c_systemic(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0, t_end_h=4.0, dt_h=0.1)
        assert len(result.times_h) == len(result.c_systemic)

    def test_times_start_at_zero(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0)
        assert result.times_h[0] == pytest.approx(0.0)

    def test_c_systemic_non_negative(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=5.0)
        assert all(c >= 0.0 for c in result.c_systemic)

    def test_cmax_positive(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0)
        assert result.cmax_systemic > 0.0

    def test_auc_positive(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0)
        assert result.auc_systemic > 0.0

    def test_auc_lung_positive(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0)
        assert result.auc_lung_exposure > 0.0

    def test_dose_stored(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=3.0)
        assert result.dose_mg == pytest.approx(3.0)

    def test_particle_size_stored(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0, particle_size_um=5.0)
        assert result.particle_size_um == pytest.approx(5.0)

    def test_notes_non_empty(self):
        result = simulate_inhaled_pk("DrugA", dose_mg=1.0)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Deposition fractions by particle size
# ---------------------------------------------------------------------------


class TestLungDepositionFraction:
    def test_small_particle_deposition_fraction(self):
        """< 1 µm should give 40% deposition."""
        result = simulate_inhaled_pk("D", dose_mg=1.0, particle_size_um=0.5)
        assert result.lung_deposition_fraction == pytest.approx(0.4, rel=1e-6)

    def test_optimal_particle_deposition_fraction(self):
        """1–5 µm should give 60% deposition."""
        result = simulate_inhaled_pk("D", dose_mg=1.0, particle_size_um=2.0)
        assert result.lung_deposition_fraction == pytest.approx(0.6, rel=1e-6)

    def test_large_particle_deposition_fraction(self):
        """> 5 µm should give 30% deposition."""
        result = simulate_inhaled_pk("D", dose_mg=1.0, particle_size_um=10.0)
        assert result.lung_deposition_fraction == pytest.approx(0.3, rel=1e-6)

    def test_higher_dose_gives_higher_cmax(self):
        r1 = simulate_inhaled_pk("D", dose_mg=1.0)
        r2 = simulate_inhaled_pk("D", dose_mg=2.0)
        assert r2.cmax_systemic > r1.cmax_systemic

    def test_optimal_particle_cmax_vs_large(self):
        """2 µm particles deposit more → higher systemic exposure than 10 µm."""
        r_opt = simulate_inhaled_pk("D", dose_mg=1.0, particle_size_um=2.0)
        r_large = simulate_inhaled_pk("D", dose_mg=1.0, particle_size_um=10.0)
        assert r_opt.cmax_systemic > r_large.cmax_systemic


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestSimulateInhaledPKValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_inhaled_pk("D", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_inhaled_pk("D", dose_mg=-1.0)

    def test_zero_particle_size_raises(self):
        with pytest.raises(ValueError, match="particle_size_um"):
            simulate_inhaled_pk("D", dose_mg=1.0, particle_size_um=0.0)

    def test_zero_mucociliary_cl_raises(self):
        with pytest.raises(ValueError, match="cl_mucociliary_per_h"):
            simulate_inhaled_pk("D", dose_mg=1.0, cl_mucociliary_per_h=0.0)

    def test_zero_alveolar_cl_raises(self):
        with pytest.raises(ValueError, match="cl_alveolar_per_h"):
            simulate_inhaled_pk("D", dose_mg=1.0, cl_alveolar_per_h=0.0)

    def test_zero_papp_raises(self):
        with pytest.raises(ValueError, match="papp_epithelium_cm_s"):
            simulate_inhaled_pk("D", dose_mg=1.0, papp_epithelium_cm_s=0.0)

    def test_zero_vd_lung_raises(self):
        with pytest.raises(ValueError, match="vd_lung_L"):
            simulate_inhaled_pk("D", dose_mg=1.0, vd_lung_L=0.0)

    def test_zero_cl_systemic_raises(self):
        with pytest.raises(ValueError, match="cl_systemic_L_per_h"):
            simulate_inhaled_pk("D", dose_mg=1.0, cl_systemic_L_per_h=0.0)

    def test_zero_vd_systemic_raises(self):
        with pytest.raises(ValueError, match="vd_systemic_L"):
            simulate_inhaled_pk("D", dose_mg=1.0, vd_systemic_L=0.0)


# ---------------------------------------------------------------------------
# pulmonary_targeting_index
# ---------------------------------------------------------------------------


class TestPulmonaryTargetingIndex:
    def test_basic_ratio(self):
        pti = pulmonary_targeting_index(10.0, 5.0)
        assert pti == pytest.approx(2.0, rel=1e-6)

    def test_equal_auc_gives_one(self):
        pti = pulmonary_targeting_index(5.0, 5.0)
        assert pti == pytest.approx(1.0, rel=1e-6)

    def test_systemic_greater_pti_less_than_one(self):
        pti = pulmonary_targeting_index(2.0, 10.0)
        assert pti == pytest.approx(0.2, rel=1e-6)

    def test_zero_lung_auc_raises(self):
        with pytest.raises(ValueError, match="auc_lung"):
            pulmonary_targeting_index(0.0, 5.0)

    def test_zero_systemic_auc_raises(self):
        with pytest.raises(ValueError, match="auc_systemic"):
            pulmonary_targeting_index(5.0, 0.0)


# ---------------------------------------------------------------------------
# compare_particle_sizes
# ---------------------------------------------------------------------------


class TestCompareParticleSizes:
    def test_returns_four_results(self):
        results = compare_particle_sizes("DrugB", 1.0, cl_systemic=10.0, vd_systemic=50.0)
        assert len(results) == 4

    def test_keys_are_correct(self):
        results = compare_particle_sizes("DrugB", 1.0, cl_systemic=10.0, vd_systemic=50.0)
        assert set(results.keys()) == {"0.5um", "2.0um", "5.0um", "10.0um"}

    def test_all_results_are_pulmonary_pk_result(self):
        results = compare_particle_sizes("DrugB", 1.0, cl_systemic=10.0, vd_systemic=50.0)
        assert all(isinstance(v, PulmonaryPKResult) for v in results.values())

    def test_optimal_size_has_highest_auc(self):
        """2 µm particle should give highest systemic AUC (60% deposition)."""
        results = compare_particle_sizes("DrugB", 1.0, cl_systemic=5.0, vd_systemic=30.0)
        auc_2um = results["2.0um"].auc_systemic
        auc_10um = results["10.0um"].auc_systemic
        assert auc_2um > auc_10um

    def test_pti_field_populated(self):
        results = compare_particle_sizes("DrugB", 1.0, cl_systemic=10.0, vd_systemic=50.0)
        for r in results.values():
            assert r.pulmonary_targeting_index > 0
