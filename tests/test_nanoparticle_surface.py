"""Tests for biopharmaceutics/nanoparticle_surface.py — Phase 471."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.nanoparticle_surface import (
    ProteinCoronaResult,
    cellular_uptake_efficiency,
    estimate_circulation_time,
    nanoparticle_stability_score,
    predict_protein_corona,
)


# ---------------------------------------------------------------------------
# predict_protein_corona — basic structure
# ---------------------------------------------------------------------------


class TestPredictProteinCorona:
    def test_returns_protein_corona_result(self):
        result = predict_protein_corona(100.0, -20.0)
        assert isinstance(result, ProteinCoronaResult)

    def test_fields_preserved(self):
        result = predict_protein_corona(150.0, -15.0, peg_density_chains_per_nm2=0.5, material="gold")
        assert result.diameter_nm == 150.0
        assert result.surface_charge_mV == -15.0
        assert result.peg_density == 0.5
        assert result.material == "gold"

    def test_opsonization_between_0_and_1(self):
        result = predict_protein_corona(100.0, -20.0)
        assert 0.0 <= result.opsonization_index <= 1.0

    def test_adsorption_fractions_between_0_and_1(self):
        result = predict_protein_corona(100.0, -20.0)
        assert 0.0 <= result.albumin_adsorption <= 1.0
        assert 0.0 <= result.igg_adsorption <= 1.0
        assert 0.0 <= result.apolipoprotein_adsorption <= 1.0

    def test_mps_clearance_rate_positive(self):
        result = predict_protein_corona(100.0, -20.0)
        assert result.mps_clearance_rate >= 0.0

    def test_notes_is_string(self):
        result = predict_protein_corona(100.0, -20.0)
        assert isinstance(result.notes, str)

    # PEGylation effect
    def test_pegylated_lower_opsonization_than_bare(self):
        bare = predict_protein_corona(100.0, -20.0, peg_density_chains_per_nm2=0.0)
        pegylated = predict_protein_corona(100.0, -20.0, peg_density_chains_per_nm2=1.0)
        assert pegylated.opsonization_index < bare.opsonization_index

    def test_high_peg_density_dramatically_reduces_opsonization(self):
        bare = predict_protein_corona(100.0, 0.0)
        high_peg = predict_protein_corona(100.0, 0.0, peg_density_chains_per_nm2=3.0)
        assert high_peg.opsonization_index < bare.opsonization_index * 0.5

    # Charge effect
    def test_negative_charge_lower_opsonization_than_positive(self):
        negative = predict_protein_corona(100.0, -20.0)
        positive = predict_protein_corona(100.0, +20.0)
        assert negative.opsonization_index < positive.opsonization_index

    def test_positive_charge_higher_opsonization(self):
        neutral = predict_protein_corona(100.0, -2.0)
        positive = predict_protein_corona(100.0, +30.0)
        assert positive.opsonization_index > neutral.opsonization_index

    # Size effect
    def test_large_nps_higher_mps_clearance(self):
        small = predict_protein_corona(50.0, -20.0)
        large = predict_protein_corona(500.0, -20.0)
        assert large.mps_clearance_rate > small.mps_clearance_rate

    # Material effect
    def test_plga_vs_gold_different_profiles(self):
        plga = predict_protein_corona(100.0, -20.0, material="PLGA")
        gold = predict_protein_corona(100.0, -20.0, material="gold")
        # Different materials give different IgG adsorption
        assert plga.igg_adsorption != gold.igg_adsorption

    def test_liposome_lower_opsonization_than_gold(self):
        liposome = predict_protein_corona(100.0, -20.0, material="liposome")
        gold = predict_protein_corona(100.0, -20.0, material="gold")
        assert liposome.opsonization_index < gold.opsonization_index

    # Input validation
    def test_invalid_diameter_raises(self):
        with pytest.raises(ValueError):
            predict_protein_corona(-10.0, -20.0)

    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError):
            predict_protein_corona(0.0, -20.0)

    def test_negative_peg_density_raises(self):
        with pytest.raises(ValueError):
            predict_protein_corona(100.0, -20.0, peg_density_chains_per_nm2=-0.1)

    def test_unknown_material_does_not_crash(self):
        result = predict_protein_corona(100.0, -20.0, material="unknown_polymer")
        assert 0.0 <= result.opsonization_index <= 1.0


# ---------------------------------------------------------------------------
# estimate_circulation_time
# ---------------------------------------------------------------------------


class TestEstimateCirculationTime:
    def test_returns_positive_float(self):
        result = estimate_circulation_time(100.0, 0.0, -20.0, 0.5)
        assert result > 0.0

    def test_pegylated_longer_circulation(self):
        bare = estimate_circulation_time(100.0, 0.0, -20.0, 0.5)
        peg = estimate_circulation_time(100.0, 1.5, -20.0, 0.5)
        assert peg > bare

    def test_larger_particles_shorter_circulation(self):
        small = estimate_circulation_time(50.0, 0.5, -20.0, 0.3)
        large = estimate_circulation_time(500.0, 0.5, -20.0, 0.3)
        assert large < small

    def test_high_opsonization_shorter_circulation(self):
        low_ops = estimate_circulation_time(100.0, 0.5, -20.0, 0.1)
        high_ops = estimate_circulation_time(100.0, 0.5, -20.0, 0.9)
        assert high_ops < low_ops

    def test_invalid_diameter_raises(self):
        with pytest.raises(ValueError):
            estimate_circulation_time(0.0, 0.5, -20.0, 0.5)

    def test_invalid_opsonization_raises(self):
        with pytest.raises(ValueError):
            estimate_circulation_time(100.0, 0.5, -20.0, 1.5)

    def test_negative_peg_density_raises(self):
        with pytest.raises(ValueError):
            estimate_circulation_time(100.0, -0.1, -20.0, 0.5)


# ---------------------------------------------------------------------------
# cellular_uptake_efficiency
# ---------------------------------------------------------------------------


class TestCellularUptakeEfficiency:
    def test_returns_fraction_0_to_1(self):
        result = cellular_uptake_efficiency(100.0, -20.0)
        assert 0.0 <= result <= 1.0

    def test_macrophage_higher_uptake_than_cancer(self):
        mac = cellular_uptake_efficiency(100.0, -20.0, cell_type="macrophage")
        cancer = cellular_uptake_efficiency(100.0, -20.0, cell_type="cancer")
        assert mac > cancer

    def test_positive_charge_higher_uptake_than_negative(self):
        pos = cellular_uptake_efficiency(100.0, +30.0)
        neg = cellular_uptake_efficiency(100.0, -20.0)
        assert pos > neg

    def test_negative_charge_lower_macrophage_uptake_than_positive(self):
        neg = cellular_uptake_efficiency(100.0, -20.0, cell_type="macrophage")
        pos = cellular_uptake_efficiency(100.0, +20.0, cell_type="macrophage")
        assert neg < pos

    def test_cancer_vs_endothelial_differ(self):
        cancer = cellular_uptake_efficiency(100.0, -20.0, cell_type="cancer")
        endo = cellular_uptake_efficiency(100.0, -20.0, cell_type="endothelial")
        assert cancer != endo

    def test_invalid_diameter_raises(self):
        with pytest.raises(ValueError):
            cellular_uptake_efficiency(-10.0, -20.0)

    def test_unknown_cell_type_does_not_crash(self):
        result = cellular_uptake_efficiency(100.0, -20.0, cell_type="neuron")
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# nanoparticle_stability_score
# ---------------------------------------------------------------------------


class TestNanoparticleStabilityScore:
    def test_returns_0_to_100(self):
        score = nanoparticle_stability_score(100.0, -30.0)
        assert 0.0 <= score <= 100.0

    def test_high_zeta_potential_better_stability(self):
        high_zeta = nanoparticle_stability_score(100.0, -35.0)
        low_zeta = nanoparticle_stability_score(100.0, -5.0)
        assert high_zeta > low_zeta

    def test_peg_improves_stability(self):
        no_peg = nanoparticle_stability_score(100.0, -20.0, peg_density_chains_per_nm2=0.0)
        with_peg = nanoparticle_stability_score(100.0, -20.0, peg_density_chains_per_nm2=1.0)
        assert with_peg > no_peg

    def test_physiological_ph_best_stability(self):
        optimal = nanoparticle_stability_score(100.0, -30.0, pH=7.4)
        acidic = nanoparticle_stability_score(100.0, -30.0, pH=4.0)
        assert optimal > acidic

    def test_neutral_ph_moderate_zeta_good_score(self):
        score = nanoparticle_stability_score(100.0, -20.0, pH=7.4)
        assert score >= 40.0

    def test_small_particles_more_stable(self):
        small = nanoparticle_stability_score(50.0, -30.0)
        large = nanoparticle_stability_score(600.0, -30.0)
        assert small > large

    def test_invalid_diameter_raises(self):
        with pytest.raises(ValueError):
            nanoparticle_stability_score(0.0, -20.0)

    def test_negative_peg_density_raises(self):
        with pytest.raises(ValueError):
            nanoparticle_stability_score(100.0, -20.0, peg_density_chains_per_nm2=-1.0)

    def test_positive_and_negative_same_magnitude_similar_score(self):
        pos = nanoparticle_stability_score(100.0, +30.0)
        neg = nanoparticle_stability_score(100.0, -30.0)
        # Both have |zeta| = 30, should give same zeta_score
        assert abs(pos - neg) < 5.0
