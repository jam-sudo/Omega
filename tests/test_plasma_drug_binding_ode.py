"""Tests for Phase 214 — dynamic plasma drug-protein binding ODE simulation."""

import pytest

from omega_pbpk.core.plasma_drug_binding_ode import (
    ProteinBindingODEResult,
    assess_binding_saturation,
    simulate_protein_binding,
)

# Convenience defaults
DRUG = "TestDrug"
DOSE = 100.0  # mg
VD = 10.0  # L
CL = 1.0  # L/h


def _sim(**kwargs):
    defaults = dict(drug_name=DRUG, dose_mg=DOSE, vd_L=VD, cl_L_per_h=CL, t_end_h=12.0)
    defaults.update(kwargs)
    return simulate_protein_binding(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_protein_binding_ode_result(self):
        r = _sim()
        assert isinstance(r, ProteinBindingODEResult)

    def test_drug_name_preserved(self):
        r = _sim(drug_name="Warfarin")
        assert r.drug_name == "Warfarin"

    def test_dose_preserved(self):
        r = _sim(dose_mg=50.0)
        assert r.dose_mg == pytest.approx(50.0)

    def test_notes_preserved(self):
        r = _sim(notes="run 1")
        assert r.notes == "run 1"

    def test_times_non_empty(self):
        r = _sim()
        assert len(r.times_h) > 0

    def test_lists_same_length(self):
        r = _sim()
        n = len(r.times_h)
        assert len(r.c_free_mg_L) == n
        assert len(r.c_bound_alb_mg_L) == n
        assert len(r.c_bound_aag_mg_L) == n
        assert len(r.c_total_mg_L) == n

    def test_times_starts_at_zero(self):
        r = _sim()
        assert r.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mass balance / compartmental consistency
# ---------------------------------------------------------------------------


class TestMassBalance:
    def test_c_total_equals_sum_of_compartments(self):
        r = _sim()
        for cf, ca, cb, ct in zip(
            r.c_free_mg_L, r.c_bound_alb_mg_L, r.c_bound_aag_mg_L, r.c_total_mg_L
        ):
            assert ct == pytest.approx(cf + ca + cb, rel=1e-6)

    def test_c_free_nonnegative(self):
        r = _sim()
        assert all(c >= 0.0 for c in r.c_free_mg_L)

    def test_c_bound_alb_nonnegative(self):
        r = _sim()
        assert all(c >= 0.0 for c in r.c_bound_alb_mg_L)

    def test_c_bound_aag_nonnegative(self):
        r = _sim()
        assert all(c >= 0.0 for c in r.c_bound_aag_mg_L)

    def test_initial_condition_free_equals_dose_over_vd(self):
        r = _sim(dose_mg=200.0, vd_L=20.0)
        assert r.c_free_mg_L[0] == pytest.approx(200.0 / 20.0, rel=1e-6)

    def test_concentrations_decline_over_time(self):
        r = _sim()
        assert r.c_total_mg_L[-1] < r.c_total_mg_L[0]


# ---------------------------------------------------------------------------
# fu and AUC
# ---------------------------------------------------------------------------


class TestFuAndAuc:
    def test_fu_at_cmax_in_0_1(self):
        r = _sim()
        assert 0.0 < r.fu_at_cmax <= 1.0

    def test_fu_at_trough_in_0_1(self):
        r = _sim()
        assert 0.0 < r.fu_at_trough <= 1.0

    def test_auc_total_greater_than_auc_free(self):
        r = _sim()
        assert r.auc_total_mg_h_per_L > r.auc_free_mg_h_per_L

    def test_auc_free_positive(self):
        r = _sim()
        assert r.auc_free_mg_h_per_L > 0.0

    def test_auc_total_positive(self):
        r = _sim()
        assert r.auc_total_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# High-dose saturation
# ---------------------------------------------------------------------------


class TestSaturation:
    def test_high_dose_lower_fu_than_low_dose(self):
        r_low = _sim(dose_mg=10.0, vd_L=2.0)  # C0 = 5 mg/L — below Kd
        r_high = _sim(dose_mg=2000.0, vd_L=2.0)  # C0 = 1000 mg/L — saturating
        # At very high dose binding proteins saturate → lower bound fraction
        # → fu_at_cmax is HIGHER for high dose (more free)
        assert r_high.fu_at_cmax >= r_low.fu_at_cmax


# ---------------------------------------------------------------------------
# assess_binding_saturation
# ---------------------------------------------------------------------------


class TestAssessBindingSaturation:
    def test_returns_dict(self):
        r = _sim()
        sat = assess_binding_saturation(r)
        assert isinstance(sat, dict)

    def test_has_expected_keys(self):
        r = _sim()
        sat = assess_binding_saturation(r)
        assert "alb_saturated" in sat
        assert "aag_saturated" in sat
        assert "fu_at_cmax" in sat
        assert "fu_at_trough" in sat

    def test_alb_saturated_is_bool(self):
        r = _sim()
        sat = assess_binding_saturation(r)
        assert isinstance(sat["alb_saturated"], bool)

    def test_aag_saturated_is_bool(self):
        r = _sim()
        sat = assess_binding_saturation(r)
        assert isinstance(sat["aag_saturated"], bool)

    def test_fu_at_cmax_matches_result(self):
        r = _sim()
        sat = assess_binding_saturation(r)
        assert sat["fu_at_cmax"] == pytest.approx(r.fu_at_cmax)

    def test_fu_at_trough_matches_result(self):
        r = _sim()
        sat = assess_binding_saturation(r)
        assert sat["fu_at_trough"] == pytest.approx(r.fu_at_trough)

    def test_high_dose_aag_saturated(self):
        # Very high dose should saturate AAG (capacity = 5 mg/L)
        r = _sim(dose_mg=10000.0, vd_L=2.0, t_end_h=1.0)
        sat = assess_binding_saturation(r)
        assert sat["aag_saturated"] is True


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_protein_binding(DRUG, dose_mg=0.0, vd_L=VD, cl_L_per_h=CL)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_protein_binding(DRUG, dose_mg=-10.0, vd_L=VD, cl_L_per_h=CL)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_protein_binding(DRUG, dose_mg=DOSE, vd_L=0.0, cl_L_per_h=CL)

    def test_vd_negative_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_protein_binding(DRUG, dose_mg=DOSE, vd_L=-5.0, cl_L_per_h=CL)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_protein_binding(DRUG, dose_mg=DOSE, vd_L=VD, cl_L_per_h=0.0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_protein_binding(DRUG, dose_mg=DOSE, vd_L=VD, cl_L_per_h=-1.0)
