"""Tests for Phase 1127: Plasma Protein Binding Kinetics."""

import math
import pytest

from omega_pbpk.core.plasma_protein_binding_kinetics import (
    ProteinBindingKineticsResult,
    simulate_protein_binding_kinetics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(drug_name="Warfarin", dose_mg=5.0, **kwargs):
    return simulate_protein_binding_kinetics(drug_name, dose_mg, **kwargs)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_result_dataclass(self):
        res = _run()
        assert isinstance(res, ProteinBindingKineticsResult)

    def test_drug_name_preserved(self):
        res = _run(drug_name="TestDrug")
        assert res.drug_name == "TestDrug"

    def test_dose_preserved(self):
        res = _run(dose_mg=10.0)
        assert res.dose_mg == 10.0

    def test_fu_plasma_preserved(self):
        res = _run(fu_plasma=0.05)
        assert res.fu_plasma == pytest.approx(0.05)

    def test_times_length_consistent(self):
        res = _run(t_end_h=12.0, dt_h=0.1)
        expected_n = int(12.0 / 0.1) + 1
        assert len(res.times_h) == expected_n

    def test_concentration_lists_same_length(self):
        res = _run(t_end_h=12.0, dt_h=0.1)
        n = len(res.times_h)
        assert len(res.c_free_mg_L) == n
        assert len(res.c_bound_mg_L) == n
        assert len(res.c_tissue_mg_L) == n
        assert len(res.c_total_plasma_mg_L) == n


# ---------------------------------------------------------------------------
# Mass balance and consistency tests
# ---------------------------------------------------------------------------

class TestMassBalance:
    def test_c_total_is_free_plus_bound(self):
        res = _run()
        for cf, cb, ct in zip(res.c_free_mg_L, res.c_bound_mg_L, res.c_total_plasma_mg_L):
            assert ct == pytest.approx(cf + cb, abs=1e-10)

    def test_non_negative_concentrations(self):
        res = _run()
        assert all(c >= 0.0 for c in res.c_free_mg_L)
        assert all(c >= 0.0 for c in res.c_bound_mg_L)
        assert all(c >= 0.0 for c in res.c_tissue_mg_L)
        assert all(c >= 0.0 for c in res.c_total_plasma_mg_L)

    def test_concentrations_decline_over_time(self):
        """Total plasma concentration should generally decline."""
        res = _run(t_end_h=24.0)
        # Compare first 10% vs last 10% of simulation
        n = len(res.c_total_plasma_mg_L)
        early = res.c_total_plasma_mg_L[0]
        late = res.c_total_plasma_mg_L[n - 1]
        assert late < early

    def test_tissue_rises_then_falls(self):
        """Tissue should start at 0 and rise."""
        res = _run()
        assert res.c_tissue_mg_L[0] == pytest.approx(0.0)
        # Should rise from zero
        assert max(res.c_tissue_mg_L) > 0.0


# ---------------------------------------------------------------------------
# Pharmacokinetic metric tests
# ---------------------------------------------------------------------------

class TestPKMetrics:
    def test_cmax_free_positive(self):
        res = _run()
        assert res.cmax_free_mg_L > 0

    def test_cmax_total_positive(self):
        res = _run()
        assert res.cmax_total_mg_L > 0

    def test_cmax_total_geq_cmax_free(self):
        res = _run(fu_plasma=0.1)
        assert res.cmax_total_mg_L >= res.cmax_free_mg_L

    def test_auc_free_positive(self):
        res = _run()
        assert res.auc_free_mg_h_per_L > 0

    def test_auc_total_positive(self):
        res = _run()
        assert res.auc_total_mg_h_per_L > 0

    def test_auc_free_less_than_auc_total(self):
        """Free drug AUC should be less than total AUC for fu < 1."""
        res = _run(fu_plasma=0.1)
        assert res.auc_free_mg_h_per_L < res.auc_total_mg_h_per_L

    def test_auc_scales_with_dose(self):
        """Doubling dose should approximately double AUC."""
        res1 = _run(dose_mg=5.0)
        res2 = _run(dose_mg=10.0)
        ratio = res2.auc_total_mg_h_per_L / res1.auc_total_mg_h_per_L
        assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Binding kinetics tests
# ---------------------------------------------------------------------------

class TestBindingKinetics:
    def test_koff_computed_correctly(self):
        """koff = kon * P_total * fu / (1 - fu)."""
        fu = 0.1
        kon = 100.0
        P_total = 4500.0
        expected_koff = kon * P_total * fu / (1.0 - fu)
        res = _run(fu_plasma=fu, kon_L_per_mg_per_h=kon)
        assert res.koff_per_h == pytest.approx(expected_koff, rel=1e-6)

    def test_binding_equilibration_h_positive(self):
        res = _run()
        assert res.binding_equilibration_h > 0

    def test_binding_equilibration_h_formula(self):
        """binding_equilibration_h ≈ 1 / (kon*P_total + koff)."""
        fu = 0.1
        kon = 100.0
        P_total = 4500.0
        koff = kon * P_total * fu / (1.0 - fu)
        expected = 1.0 / (kon * P_total + koff)
        res = _run(fu_plasma=fu, kon_L_per_mg_per_h=kon)
        assert res.binding_equilibration_h == pytest.approx(expected, rel=1e-6)

    def test_lower_fu_higher_bound_fraction(self):
        """Lower fu (higher affinity) → higher fraction of drug is bound."""
        res_high_fu = _run(fu_plasma=0.5)
        res_low_fu = _run(fu_plasma=0.05)
        # At t=0, c_bound / c_total = 1 - fu
        assert res_low_fu.c_bound_mg_L[0] / res_low_fu.c_total_plasma_mg_L[0] > \
               res_high_fu.c_bound_mg_L[0] / res_high_fu.c_total_plasma_mg_L[0]

    def test_higher_fu_more_free_drug(self):
        """Higher fu → more free drug in plasma."""
        res_high = _run(fu_plasma=0.5)
        res_low = _run(fu_plasma=0.1)
        assert res_high.c_free_mg_L[0] > res_low.c_free_mg_L[0]

    def test_fu_at_tmax_near_nominal(self):
        """At tmax (t=0 for IV), fu_at_tmax ≈ fu_plasma."""
        fu = 0.1
        res = _run(fu_plasma=fu)
        assert res.fu_at_tmax == pytest.approx(fu, rel=0.01)


# ---------------------------------------------------------------------------
# Parameter sensitivity tests
# ---------------------------------------------------------------------------

class TestParameterSensitivity:
    def test_higher_cl_lower_auc(self):
        """Higher clearance → lower AUC."""
        res_low_cl = _run(cl_L_per_h=2.0)
        res_high_cl = _run(cl_L_per_h=20.0)
        assert res_high_cl.auc_total_mg_h_per_L < res_low_cl.auc_total_mg_h_per_L

    def test_higher_kpt_more_tissue_accumulation(self):
        """Higher plasma-to-tissue transfer → more tissue drug."""
        res_low = _run(kpt_per_h=0.05)
        res_high = _run(kpt_per_h=0.5)
        assert max(res_high.c_tissue_mg_L) > max(res_low.c_tissue_mg_L)

    def test_kon_affects_koff(self):
        """Changing kon should scale koff proportionally."""
        res1 = _run(kon_L_per_mg_per_h=50.0)
        res2 = _run(kon_L_per_mg_per_h=100.0)
        assert res2.koff_per_h == pytest.approx(res1.koff_per_h * 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------

class TestNotesField:
    def test_notes_is_string(self):
        res = _run()
        assert isinstance(res.notes, str)

    def test_notes_contains_key_info(self):
        res = _run(fu_plasma=0.1)
        assert "P_total" in res.notes or "koff" in res.notes


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_protein_binding_kinetics("Drug", 0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_protein_binding_kinetics("Drug", -5.0)

    def test_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_protein_binding_kinetics("Drug", 5.0, fu_plasma=0.0)

    def test_fu_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_protein_binding_kinetics("Drug", 5.0, fu_plasma=1.0)

    def test_fu_gt_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_protein_binding_kinetics("Drug", 5.0, fu_plasma=1.5)

    def test_kon_zero_raises(self):
        with pytest.raises(ValueError, match="kon"):
            simulate_protein_binding_kinetics("Drug", 5.0, kon_L_per_mg_per_h=0.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_protein_binding_kinetics("Drug", 5.0, cl_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_total_L"):
            simulate_protein_binding_kinetics("Drug", 5.0, vd_total_L=0.0)
