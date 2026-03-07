"""
Tests for Phase 782 — Tissue Binding PK
"""

import pytest

from omega_pbpk.core.tissue_binding_pk import (
    TissueBindingPKResult,
    simulate_tissue_binding_pk,
)


class TestTissueBindingPKBasic:
    def setup_method(self):
        self.result = simulate_tissue_binding_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            route="iv",
            t_end_h=48.0,
        )

    def test_returns_correct_type(self):
        assert isinstance(self.result, TissueBindingPKResult)

    def test_drug_name_preserved(self):
        assert self.result.drug_name == "TestDrug"

    def test_dose_preserved(self):
        assert self.result.dose_mg == 100.0

    def test_route_preserved(self):
        assert self.result.route == "iv"

    def test_cmax_plasma_positive(self):
        assert self.result.cmax_plasma > 0.0

    def test_auc_plasma_positive(self):
        assert self.result.auc_plasma > 0.0

    def test_lists_same_length(self):
        n = len(self.result.times_h)
        assert len(self.result.c_plasma_total_mg_L) == n
        assert len(self.result.c_tissue_free_nmol_L) == n
        assert len(self.result.c_tissue_bound_nmol_L) == n

    def test_concentrations_nonnegative(self):
        assert all(c >= 0.0 for c in self.result.c_plasma_total_mg_L)
        assert all(c >= 0.0 for c in self.result.c_tissue_free_nmol_L)
        assert all(c >= 0.0 for c in self.result.c_tissue_bound_nmol_L)

    def test_occupancy_in_valid_range(self):
        assert 0.0 <= self.result.occupancy_at_cmax_pct <= 100.0

    def test_notes_is_string(self):
        assert isinstance(self.result.notes, str)
        assert len(self.result.notes) > 0

    def test_tmax_within_simulation_time(self):
        assert 0.0 <= self.result.tmax_plasma_h <= 48.0


class TestTissueBindingPKOral:
    def test_oral_route_cmax_positive(self):
        r = simulate_tissue_binding_pk(
            drug_name="OralDrug",
            dose_mg=200.0,
            route="oral",
            f_oral=0.7,
            ka_per_h=0.5,
            t_end_h=48.0,
        )
        assert r.cmax_plasma > 0.0

    def test_oral_route_tmax_greater_than_zero(self):
        r = simulate_tissue_binding_pk(
            drug_name="OralDrug",
            dose_mg=200.0,
            route="oral",
            ka_per_h=0.5,
            t_end_h=48.0,
        )
        assert r.tmax_plasma_h > 0.0


class TestTissueBindingPKPhysiology:
    def test_higher_bmax_greater_tissue_bound(self):
        r_low = simulate_tissue_binding_pk("D", 100.0, bmax_nmol_L=50.0, t_end_h=48.0)
        r_high = simulate_tissue_binding_pk("D", 100.0, bmax_nmol_L=500.0, t_end_h=48.0)
        max_bound_low = max(r_low.c_tissue_bound_nmol_L)
        max_bound_high = max(r_high.c_tissue_bound_nmol_L)
        assert max_bound_high > max_bound_low

    def test_occupancy_increases_with_dose(self):
        r_low = simulate_tissue_binding_pk("D", 10.0, bmax_nmol_L=100.0, kd_nmol_L=10.0)
        r_high = simulate_tissue_binding_pk("D", 1000.0, bmax_nmol_L=100.0, kd_nmol_L=10.0)
        assert r_high.occupancy_at_cmax_pct >= r_low.occupancy_at_cmax_pct

    def test_tissue_to_plasma_ratio_nonnegative(self):
        r = simulate_tissue_binding_pk("D", 100.0)
        assert r.tissue_to_plasma_ratio_free >= 0.0

    def test_higher_cl_lower_auc(self):
        r_low_cl = simulate_tissue_binding_pk("D", 100.0, cl_L_per_h=1.0, t_end_h=48.0)
        r_high_cl = simulate_tissue_binding_pk("D", 100.0, cl_L_per_h=20.0, t_end_h=48.0)
        assert r_high_cl.auc_plasma < r_low_cl.auc_plasma


class TestTissueBindingPKValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_tissue_binding_pk("D", dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_tissue_binding_pk("D", dose_mg=-50.0)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_tissue_binding_pk("D", 100.0, fup=0.0)

    def test_fup_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            simulate_tissue_binding_pk("D", 100.0, fup=1.5)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError):
            simulate_tissue_binding_pk("D", 100.0, route="inhalation")

    def test_kd_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_tissue_binding_pk("D", 100.0, kd_nmol_L=0.0)
