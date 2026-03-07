"""Tests for clinical PK decision support — Phase 504."""

import math
import pytest

from omega_pbpk.clinical.clinical_pk_support import (
    TherapeuticWindowResult,
    assess_therapeutic_window,
    bayesian_dose_update,
    population_adjustment,
    recommend_dose_timing,
)


# ---------------------------------------------------------------------------
# assess_therapeutic_window
# ---------------------------------------------------------------------------


class TestAssessTherapeuticWindow:
    def test_in_window_when_cmin_above_mec_and_cmax_below_mtc(self):
        result = assess_therapeutic_window(
            cmin_mg_L=2.0, cmax_mg_L=8.0, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert result.in_window is True
        assert result.subtherapeutic is False
        assert result.supratherapeutic is False

    def test_subtherapeutic_when_cmax_below_mec(self):
        result = assess_therapeutic_window(
            cmin_mg_L=0.1, cmax_mg_L=0.5, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert result.subtherapeutic is True
        assert result.in_window is False

    def test_supratherapeutic_when_cmin_above_mtc(self):
        result = assess_therapeutic_window(
            cmin_mg_L=12.0, cmax_mg_L=20.0, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert result.supratherapeutic is True
        assert result.in_window is False

    def test_not_supratherapeutic_when_cmax_above_mtc_but_cmin_below(self):
        result = assess_therapeutic_window(
            cmin_mg_L=5.0, cmax_mg_L=15.0, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert result.supratherapeutic is False

    def test_fluctuation_pct_formula(self):
        """fluctuation_pct = (Cmax - Cmin) / Cavg * 100."""
        cmin, cmax = 2.0, 8.0
        cavg = (cmax + cmin) / 2.0
        expected_fluct = (cmax - cmin) / cavg * 100.0
        result = assess_therapeutic_window(
            cmin_mg_L=cmin, cmax_mg_L=cmax, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert abs(result.fluctuation_pct - expected_fluct) < 1e-10

    def test_fluctuation_zero_when_cmin_equals_cmax(self):
        result = assess_therapeutic_window(
            cmin_mg_L=5.0, cmax_mg_L=5.0, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert result.fluctuation_pct == 0.0

    def test_therapeutic_range_pct_full_overlap(self):
        """When [Cmin, Cmax] is fully within [MEC, MTC] → 100%."""
        result = assess_therapeutic_window(
            cmin_mg_L=2.0, cmax_mg_L=8.0, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert result.therapeutic_range_pct == 100.0

    def test_therapeutic_range_pct_no_overlap(self):
        """When [Cmin, Cmax] is entirely below MEC → 0%."""
        result = assess_therapeutic_window(
            cmin_mg_L=0.1, cmax_mg_L=0.5, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert result.therapeutic_range_pct == 0.0

    def test_recommendation_contains_increase_when_subtherapeutic(self):
        result = assess_therapeutic_window(
            cmin_mg_L=0.1, cmax_mg_L=0.5, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert "subtherapeutic" in result.recommendation.lower() or "dose" in result.recommendation.lower()

    def test_recommendation_contains_reduce_when_supratherapeutic(self):
        result = assess_therapeutic_window(
            cmin_mg_L=12.0, cmax_mg_L=20.0, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert "reduce" in result.recommendation.lower() or "exceed" in result.recommendation.lower()

    def test_invalid_cmin_negative_raises(self):
        with pytest.raises(ValueError, match="cmin"):
            assess_therapeutic_window(
                cmin_mg_L=-1.0, cmax_mg_L=5.0, mec_mg_L=1.0, mtc_mg_L=10.0
            )

    def test_invalid_cmax_less_than_cmin_raises(self):
        with pytest.raises(ValueError, match="cmax"):
            assess_therapeutic_window(
                cmin_mg_L=5.0, cmax_mg_L=3.0, mec_mg_L=1.0, mtc_mg_L=10.0
            )

    def test_invalid_mtc_le_mec_raises(self):
        with pytest.raises(ValueError, match="mtc"):
            assess_therapeutic_window(
                cmin_mg_L=2.0, cmax_mg_L=5.0, mec_mg_L=5.0, mtc_mg_L=5.0
            )

    def test_returns_dataclass(self):
        result = assess_therapeutic_window(
            cmin_mg_L=2.0, cmax_mg_L=8.0, mec_mg_L=1.0, mtc_mg_L=10.0
        )
        assert isinstance(result, TherapeuticWindowResult)

    def test_narrow_therapeutic_index_note(self):
        """MTC/MEC < 2 → narrow TI note."""
        result = assess_therapeutic_window(
            cmin_mg_L=2.0, cmax_mg_L=3.0, mec_mg_L=2.0, mtc_mg_L=3.5
        )
        assert "narrow" in result.notes.lower()


# ---------------------------------------------------------------------------
# recommend_dose_timing
# ---------------------------------------------------------------------------


class TestRecommendDoseTiming:
    def test_returns_dict_with_required_keys(self):
        result = recommend_dose_timing(
            t_half_h=6.0,
            target_trough_mg_L=1.0,
            dose_mg=500.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
        )
        assert "recommended_interval_h" in result
        assert "predicted_trough_mg_L" in result
        assert "predicted_peak_mg_L" in result
        assert "interval_evaluations" in result
        assert "rationale" in result

    def test_shorter_interval_lower_trough_for_same_dose(self):
        """Shorter interval → trough closer to peak; longer → lower trough."""
        result = recommend_dose_timing(
            t_half_h=6.0,
            target_trough_mg_L=1.0,
            dose_mg=500.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            interval_options_h=[6.0, 24.0],
        )
        evals = {e["interval_h"]: e for e in result["interval_evaluations"]}
        # Shorter interval → higher trough (more accumulation)
        assert evals[6.0]["predicted_trough_mg_L"] > evals[24.0]["predicted_trough_mg_L"]

    def test_custom_intervals_respected(self):
        result = recommend_dose_timing(
            t_half_h=8.0,
            target_trough_mg_L=0.5,
            dose_mg=200.0,
            cl_L_per_h=3.0,
            vd_L=30.0,
            interval_options_h=[12.0, 24.0],
        )
        assert len(result["interval_evaluations"]) == 2

    def test_invalid_t_half_raises(self):
        with pytest.raises(ValueError, match="t_half_h"):
            recommend_dose_timing(
                t_half_h=0.0, target_trough_mg_L=1.0,
                dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0
            )

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            recommend_dose_timing(
                t_half_h=6.0, target_trough_mg_L=1.0,
                dose_mg=-100.0, cl_L_per_h=5.0, vd_L=50.0
            )

    def test_all_default_intervals_evaluated(self):
        result = recommend_dose_timing(
            t_half_h=6.0, target_trough_mg_L=1.0,
            dose_mg=500.0, cl_L_per_h=5.0, vd_L=50.0,
        )
        intervals = [e["interval_h"] for e in result["interval_evaluations"]]
        assert set(intervals) == {6.0, 8.0, 12.0, 24.0}


# ---------------------------------------------------------------------------
# bayesian_dose_update
# ---------------------------------------------------------------------------


class TestBayesianDoseUpdate:
    def test_no_change_when_observed_equals_predicted(self):
        """When C_obs == C_pred, CL should remain unchanged."""
        cl = 5.0
        vd = 50.0
        dose = 500.0
        t = 4.0
        # Compute predicted concentration for these parameters
        c_pred = (dose / vd) * math.exp(-cl / vd * t)

        result = bayesian_dose_update(
            prior_cl_L_per_h=cl,
            prior_vd_L=vd,
            observed_conc_mg_L=c_pred,
            observed_time_h=t,
            dose_mg=dose,
            n_iter=5,
        )
        assert abs(result["fold_change_cl"] - 1.0) < 1e-6

    def test_cl_increases_when_observed_lower_than_predicted(self):
        """If C_obs < C_pred, then drug clears faster → CL should increase."""
        cl = 5.0
        vd = 50.0
        dose = 500.0
        t = 4.0
        c_pred = (dose / vd) * math.exp(-cl / vd * t)
        # Observe lower concentration than predicted (higher clearance)
        c_obs = c_pred * 0.5

        result = bayesian_dose_update(
            prior_cl_L_per_h=cl,
            prior_vd_L=vd,
            observed_conc_mg_L=c_obs,
            observed_time_h=t,
            dose_mg=dose,
            n_iter=3,
        )
        assert result["posterior_cl_L_per_h"] > cl

    def test_cl_decreases_when_observed_higher_than_predicted(self):
        """If C_obs > C_pred, drug clears slower → CL should decrease."""
        cl = 5.0
        vd = 50.0
        dose = 500.0
        t = 4.0
        c_pred = (dose / vd) * math.exp(-cl / vd * t)
        c_obs = c_pred * 2.0

        result = bayesian_dose_update(
            prior_cl_L_per_h=cl,
            prior_vd_L=vd,
            observed_conc_mg_L=c_obs,
            observed_time_h=t,
            dose_mg=dose,
            n_iter=3,
        )
        assert result["posterior_cl_L_per_h"] < cl

    def test_vd_unchanged(self):
        """Bayesian update only modifies CL, not Vd."""
        result = bayesian_dose_update(
            prior_cl_L_per_h=5.0,
            prior_vd_L=50.0,
            observed_conc_mg_L=1.5,
            observed_time_h=4.0,
            dose_mg=500.0,
        )
        assert result["posterior_vd_L"] == 50.0

    def test_returns_required_keys(self):
        result = bayesian_dose_update(
            prior_cl_L_per_h=5.0,
            prior_vd_L=50.0,
            observed_conc_mg_L=2.0,
            observed_time_h=4.0,
            dose_mg=500.0,
        )
        assert "posterior_cl_L_per_h" in result
        assert "posterior_vd_L" in result
        assert "predicted_conc_mg_L" in result
        assert "fold_change_cl" in result

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="prior_cl"):
            bayesian_dose_update(
                prior_cl_L_per_h=0.0, prior_vd_L=50.0,
                observed_conc_mg_L=2.0, observed_time_h=4.0, dose_mg=500.0,
            )

    def test_invalid_observed_conc_raises(self):
        with pytest.raises(ValueError, match="observed_conc"):
            bayesian_dose_update(
                prior_cl_L_per_h=5.0, prior_vd_L=50.0,
                observed_conc_mg_L=0.0, observed_time_h=4.0, dose_mg=500.0,
            )

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            bayesian_dose_update(
                prior_cl_L_per_h=5.0, prior_vd_L=50.0,
                observed_conc_mg_L=2.0, observed_time_h=4.0, dose_mg=500.0,
                route="oral",
            )

    def test_n_iter_reflected_in_output(self):
        result = bayesian_dose_update(
            prior_cl_L_per_h=5.0, prior_vd_L=50.0,
            observed_conc_mg_L=2.0, observed_time_h=4.0, dose_mg=500.0,
            n_iter=7,
        )
        assert result["n_iterations"] == 7


# ---------------------------------------------------------------------------
# population_adjustment
# ---------------------------------------------------------------------------


class TestPopulationAdjustment:
    def test_heavier_patient_higher_cl(self):
        result_70 = population_adjustment(cl_typical=5.0, vd_typical=50.0, weight_kg=70.0)
        result_100 = population_adjustment(cl_typical=5.0, vd_typical=50.0, weight_kg=100.0)
        assert result_100["adjusted_cl_L_per_h"] > result_70["adjusted_cl_L_per_h"]

    def test_heavier_patient_higher_vd(self):
        result_70 = population_adjustment(cl_typical=5.0, vd_typical=50.0, weight_kg=70.0)
        result_100 = population_adjustment(cl_typical=5.0, vd_typical=50.0, weight_kg=100.0)
        assert result_100["adjusted_vd_L"] > result_70["adjusted_vd_L"]

    def test_elderly_lower_cl(self):
        result_young = population_adjustment(cl_typical=5.0, vd_typical=50.0, age_years=30.0)
        result_old = population_adjustment(cl_typical=5.0, vd_typical=50.0, age_years=75.0)
        assert result_old["adjusted_cl_L_per_h"] < result_young["adjusted_cl_L_per_h"]

    def test_female_lower_cl_than_male(self):
        result_male = population_adjustment(cl_typical=5.0, vd_typical=50.0, sex="male")
        result_female = population_adjustment(cl_typical=5.0, vd_typical=50.0, sex="female")
        assert result_female["adjusted_cl_L_per_h"] < result_male["adjusted_cl_L_per_h"]

    def test_high_creatinine_lower_renal_factor(self):
        result_normal = population_adjustment(
            cl_typical=5.0, vd_typical=50.0, creatinine_umol_L=80.0
        )
        result_high = population_adjustment(
            cl_typical=5.0, vd_typical=50.0, creatinine_umol_L=200.0
        )
        assert result_high["renal_factor"] < result_normal["renal_factor"]

    def test_reference_patient_near_typical(self):
        """70 kg, 35 years, male, Cr=80 → adjusted ≈ typical."""
        result = population_adjustment(
            cl_typical=5.0, vd_typical=50.0,
            weight_kg=70.0, age_years=35.0, sex="male", creatinine_umol_L=80.0,
        )
        # Combined factor should be close to 1.0 for reference patient
        assert 0.5 < result["combined_factor"] < 2.0

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_typical"):
            population_adjustment(cl_typical=0.0, vd_typical=50.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_typical"):
            population_adjustment(cl_typical=5.0, vd_typical=-10.0)

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            population_adjustment(cl_typical=5.0, vd_typical=50.0, sex="unknown")

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg"):
            population_adjustment(cl_typical=5.0, vd_typical=50.0, weight_kg=0.0)

    def test_returns_required_keys(self):
        result = population_adjustment(cl_typical=5.0, vd_typical=50.0)
        assert "adjusted_cl_L_per_h" in result
        assert "adjusted_vd_L" in result
        assert "weight_factor" in result
        assert "age_factor" in result
        assert "sex_factor" in result
        assert "renal_factor" in result
        assert "combined_factor" in result
        assert "notes" in result
