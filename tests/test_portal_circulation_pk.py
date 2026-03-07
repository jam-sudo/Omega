"""
Tests for Phase 781 — Portal Circulation PK
"""

import pytest

from omega_pbpk.core.portal_circulation_pk import (
    PortalCirculationResult,
    compute_hepatic_extraction,
    simulate_portal_pk,
)

# ---------------------------------------------------------------------------
# compute_hepatic_extraction tests
# ---------------------------------------------------------------------------


class TestComputeHepaticExtraction:
    def test_returns_float(self):
        er = compute_hepatic_extraction(30.0, 0.1)
        assert isinstance(er, float)

    def test_range_zero_to_one(self):
        er = compute_hepatic_extraction(30.0, 0.1)
        assert 0.0 <= er < 1.0

    def test_zero_clint_gives_zero_extraction(self):
        er = compute_hepatic_extraction(0.0, 0.1)
        assert er == 0.0

    def test_high_clint_gives_high_extraction(self):
        er_low = compute_hepatic_extraction(1.0, 0.1)
        er_high = compute_hepatic_extraction(1000.0, 0.1)
        assert er_high > er_low

    def test_higher_fup_gives_higher_extraction(self):
        er_low_fup = compute_hepatic_extraction(30.0, 0.01)
        er_high_fup = compute_hepatic_extraction(30.0, 0.5)
        assert er_high_fup > er_low_fup

    def test_well_stirred_formula(self):
        cl_int = 30.0
        fup = 0.1
        q = 75.0
        expected = fup * cl_int / (q + fup * cl_int)
        assert abs(compute_hepatic_extraction(cl_int, fup, q) - expected) < 1e-10

    def test_invalid_fup_zero_raises(self):
        with pytest.raises(ValueError):
            compute_hepatic_extraction(30.0, 0.0)

    def test_invalid_fup_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            compute_hepatic_extraction(30.0, 1.5)

    def test_custom_portal_flow(self):
        er1 = compute_hepatic_extraction(30.0, 0.1, q_portal_L_per_h=75.0)
        er2 = compute_hepatic_extraction(30.0, 0.1, q_portal_L_per_h=50.0)
        # Lower flow → higher extraction ratio
        assert er2 > er1


# ---------------------------------------------------------------------------
# simulate_portal_pk tests
# ---------------------------------------------------------------------------


class TestSimulatePortalPK:
    def setup_method(self):
        self.result = simulate_portal_pk(
            drug_name="TestDrug",
            dose_mg=100.0,
            t_end_h=24.0,
        )

    def test_returns_portal_circulation_result(self):
        assert isinstance(self.result, PortalCirculationResult)

    def test_drug_name_preserved(self):
        assert self.result.drug_name == "TestDrug"

    def test_dose_preserved(self):
        assert self.result.dose_mg == 100.0

    def test_cmax_systemic_positive(self):
        assert self.result.cmax_systemic > 0.0

    def test_auc_systemic_positive(self):
        assert self.result.auc_systemic > 0.0

    def test_lists_same_length(self):
        n = len(self.result.times_h)
        assert len(self.result.c_portal_mg_L) == n
        assert len(self.result.c_liver_mg_L) == n
        assert len(self.result.c_systemic_mg_L) == n

    def test_f_hepatic_in_range(self):
        assert 0.0 <= self.result.f_hepatic <= 1.0

    def test_hepatic_extraction_ratio_in_range(self):
        assert 0.0 <= self.result.hepatic_extraction_ratio < 1.0

    def test_f_oral_overall_consistent(self):
        expected = self.result.f_gut_wall * self.result.f_hepatic
        assert abs(self.result.f_oral_overall - expected) < 1e-10

    def test_linear_dose_scaling_cmax(self):
        r1 = simulate_portal_pk("D", 100.0, t_end_h=24.0)
        r2 = simulate_portal_pk("D", 200.0, t_end_h=24.0)
        ratio = r2.cmax_systemic / r1.cmax_systemic
        assert abs(ratio - 2.0) < 0.01

    def test_linear_dose_scaling_auc(self):
        r1 = simulate_portal_pk("D", 100.0, t_end_h=24.0)
        r2 = simulate_portal_pk("D", 200.0, t_end_h=24.0)
        ratio = r2.auc_systemic / r1.auc_systemic
        assert abs(ratio - 2.0) < 0.05

    def test_higher_clint_lower_systemic_auc(self):
        r_low = simulate_portal_pk("D", 100.0, cl_int_L_per_h=5.0, t_end_h=24.0)
        r_high = simulate_portal_pk("D", 100.0, cl_int_L_per_h=100.0, t_end_h=24.0)
        assert r_high.auc_systemic < r_low.auc_systemic

    def test_higher_clint_higher_extraction(self):
        r_low = simulate_portal_pk("D", 100.0, cl_int_L_per_h=5.0)
        r_high = simulate_portal_pk("D", 100.0, cl_int_L_per_h=200.0)
        assert r_high.hepatic_extraction_ratio > r_low.hepatic_extraction_ratio

    def test_zero_gut_wall_f_gives_low_systemic(self):
        r = simulate_portal_pk("D", 100.0, f_gut_wall=0.0)
        assert r.cmax_systemic == 0.0 or r.auc_systemic < 1e-6

    def test_notes_is_string(self):
        assert isinstance(self.result.notes, str)
        assert len(self.result.notes) > 0

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_portal_pk("D", dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_portal_pk("D", dose_mg=-10.0)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_portal_pk("D", 100.0, fup=0.0)

    def test_fup_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            simulate_portal_pk("D", 100.0, fup=1.5)

    def test_tmax_systemic_within_simulation_time(self):
        # Allow small floating-point overshoot from step accumulation
        assert 0.0 <= self.result.tmax_systemic_h <= 24.0 + 0.1

    def test_concentrations_nonnegative(self):
        assert all(c >= 0.0 for c in self.result.c_systemic_mg_L)
        assert all(c >= 0.0 for c in self.result.c_portal_mg_L)
        assert all(c >= 0.0 for c in self.result.c_liver_mg_L)
