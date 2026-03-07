"""Tests for gut microbiome drug metabolism module (Phase 239)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.gut_microbiome_metabolism import (
    ColonicPKResult,
    MicrobiomeMetabResult,
    assess_microbiome_metabolism,
    dysbiosis_impact,
    simulate_colonic_pk,
)


# ---------------------------------------------------------------------------
# assess_microbiome_metabolism tests
# ---------------------------------------------------------------------------


class TestAssessMicrobiomeMetabolism:
    def test_azo_group_detected(self):
        result = assess_microbiome_metabolism("sulfasalazine", "CC-N=N-CC", 1000.0)
        assert "azo_reduction" in result.reactions_detected

    def test_nitro_group_detected_charged(self):
        result = assess_microbiome_metabolism("metronidazole", "c[N+](=O)[O-]", 500.0)
        assert "nitro_reduction" in result.reactions_detected

    def test_nitro_group_detected_uncharged(self):
        result = assess_microbiome_metabolism("nitrofurantoin", "CN(=O)=O", 100.0)
        assert "nitro_reduction" in result.reactions_detected

    def test_glucuronide_detected_large_mw(self):
        # large MW with ester linkage → glucuronidase
        result = assess_microbiome_metabolism("morphine_glucuronide", "C(=O)OCC", 500.0, mw=500.0)
        assert "glucuronidase" in result.reactions_detected

    def test_glucuronide_not_detected_small_mw(self):
        # Same SMILES but MW < 400 → no glucuronidase, but ester_hydrolysis
        result = assess_microbiome_metabolism("small_ester", "C(=O)OCC", 200.0, mw=200.0)
        assert "glucuronidase" not in result.reactions_detected

    def test_ester_hydrolysis_detected(self):
        result = assess_microbiome_metabolism("aspirin", "C(=O)OCC", 500.0, mw=200.0)
        assert "ester_hydrolysis" in result.reactions_detected

    def test_no_reactions_no_susceptible_groups(self):
        result = assess_microbiome_metabolism("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 200.0)
        assert result.n_reactions == 0
        assert result.f_micro == 0.0

    def test_f_micro_capped_at_0_8(self):
        # azo + nitro + glucuronide (large MW) + 5x density
        smiles = "N=N[N+](=O)[O-]C(=O)O"
        result = assess_microbiome_metabolism(
            "poly_drug", smiles, 100.0, mw=500.0, microbiome_density_relative=10.0
        )
        assert result.f_micro <= 0.8

    def test_f_micro_scales_with_density(self):
        r_normal = assess_microbiome_metabolism(
            "azo_drug", "CC-N=N-CC", 100.0, microbiome_density_relative=1.0
        )
        r_dense = assess_microbiome_metabolism(
            "azo_drug", "CC-N=N-CC", 100.0, microbiome_density_relative=2.0
        )
        assert r_dense.f_micro >= r_normal.f_micro

    def test_iv_route_reduces_f_micro(self):
        r_oral = assess_microbiome_metabolism("azo_drug", "CC-N=N-CC", 100.0, route="oral")
        r_iv = assess_microbiome_metabolism("azo_drug", "CC-N=N-CC", 100.0, route="iv")
        assert r_iv.f_micro < r_oral.f_micro

    def test_result_type(self):
        result = assess_microbiome_metabolism("test", "CC-N=N-CC", 100.0)
        assert isinstance(result, MicrobiomeMetabResult)

    def test_reactions_detected_is_list(self):
        result = assess_microbiome_metabolism("test", "CC-N=N-CC", 100.0)
        assert isinstance(result.reactions_detected, list)

    def test_n_reactions_matches_detected(self):
        result = assess_microbiome_metabolism("test", "CC-N=N-CC", 100.0)
        assert result.n_reactions == len(result.reactions_detected)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="daily_dose_mg"):
            assess_microbiome_metabolism("test", "CC", 0.0)

    def test_invalid_density_raises(self):
        with pytest.raises(ValueError, match="microbiome_density_relative"):
            assess_microbiome_metabolism("test", "CC", 100.0, microbiome_density_relative=-1.0)


# ---------------------------------------------------------------------------
# simulate_colonic_pk tests
# ---------------------------------------------------------------------------


class TestSimulateColonicPK:
    def test_active_metabolite_formed_when_f_micro_positive(self):
        result = simulate_colonic_pk("sulfasalazine", dose_mg=1000.0, f_micro=0.6)
        assert result.cmax_active > 0.0

    def test_no_active_when_f_micro_zero(self):
        result = simulate_colonic_pk("control_drug", dose_mg=500.0, f_micro=0.0)
        # Active metabolite should be negligible (exactly 0 or tiny float error)
        assert result.cmax_active < 1e-10

    def test_parent_concentration_positive(self):
        result = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3)
        assert result.cmax_parent > 0.0

    def test_colonic_conversion_fraction_in_range(self):
        result = simulate_colonic_pk(
            "drug", dose_mg=500.0, f_micro=0.5, k_micro_per_h=0.2, t_end_h=72.0
        )
        assert 0.0 <= result.colonic_conversion_fraction <= 1.0

    def test_auc_parent_positive(self):
        result = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3)
        assert result.auc_parent > 0.0

    def test_auc_active_positive_when_f_micro_nonzero(self):
        result = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.5)
        assert result.auc_active > 0.0

    def test_result_type(self):
        result = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3)
        assert isinstance(result, ColonicPKResult)

    def test_times_list(self):
        result = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3)
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_c_parent_list(self):
        result = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3)
        assert isinstance(result.c_parent_mg_L, list)
        assert len(result.c_parent_mg_L) == len(result.times_h)

    def test_c_active_list(self):
        result = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3)
        assert isinstance(result.c_active_mg_L, list)
        assert len(result.c_active_mg_L) == len(result.times_h)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_colonic_pk("drug", dose_mg=0.0, f_micro=0.3)

    def test_invalid_f_micro_raises(self):
        with pytest.raises(ValueError, match="f_micro"):
            simulate_colonic_pk("drug", dose_mg=100.0, f_micro=1.5)

    def test_invalid_cl_parent_raises(self):
        with pytest.raises(ValueError, match="cl_parent"):
            simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3, cl_parent_L_per_h=0.0)


# ---------------------------------------------------------------------------
# dysbiosis_impact tests
# ---------------------------------------------------------------------------


class TestDysbiosisImpact:
    def test_lower_k_micro_yields_less_active_metabolite(self):
        normal = simulate_colonic_pk(
            "drug", dose_mg=500.0, f_micro=0.6, k_micro_per_h=0.1, t_end_h=48.0
        )
        dysbiotic = dysbiosis_impact(normal, dysbiosis_factor=0.2)
        # Reduced microbial activity → less active metabolite formed
        assert dysbiotic.cmax_active <= normal.cmax_active

    def test_dysbiosis_result_type(self):
        normal = simulate_colonic_pk("drug", dose_mg=500.0, f_micro=0.5)
        result = dysbiosis_impact(normal)
        assert isinstance(result, ColonicPKResult)

    def test_invalid_dysbiosis_factor_raises(self):
        normal = simulate_colonic_pk("drug", dose_mg=100.0, f_micro=0.3)
        with pytest.raises(ValueError, match="dysbiosis_factor"):
            dysbiosis_impact(normal, dysbiosis_factor=0.0)

    def test_dysbiosis_preserves_drug_name(self):
        normal = simulate_colonic_pk("sulfasalazine", dose_mg=500.0, f_micro=0.5)
        result = dysbiosis_impact(normal)
        assert result.drug_name == "sulfasalazine"
