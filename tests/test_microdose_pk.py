"""Tests for Phase 553: Microdose PK Extrapolation."""

import math

import pytest

from omega_pbpk.clinical.microdose_pk import (
    MicrodosePKResult,
    auc_from_sparse_sampling,
    extrapolate_from_microdose,
    microdose_clinical_utility,
    validate_linearity,
)

# ---------------------------------------------------------------------------
# extrapolate_from_microdose
# ---------------------------------------------------------------------------


class TestExtrapolateFromMicrodose:
    def test_100x_dose_ratio_scales_cmax(self):
        result = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,  # 10000 ug → ratio 100
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
        )
        assert result["dose_ratio"] == pytest.approx(100.0)
        assert result["cmax_predicted_ng_mL"] == pytest.approx(100.0)

    def test_100x_dose_ratio_scales_auc(self):
        result = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
        )
        assert result["auc_predicted_ng_h_mL"] == pytest.approx(500.0)

    def test_cl_computed_correctly(self):
        # microdose_ng = 100 * 1000 = 100000 ng; auc = 5 ng*h/mL
        # cl_mL_per_h = 100000 / 5 = 20000 mL/h = 20 L/h
        result = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
        )
        assert result["cl_L_per_h"] == pytest.approx(20.0)

    def test_assumption_valid_for_low_ratio(self):
        result = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
        )
        assert result["assumption_valid"] is True

    def test_assumption_invalid_for_extreme_ratio(self):
        # ratio > 10000: 100 ug microdose, 2000 mg therapeutic → 2000*1000/100 = 20000
        result = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=2000.0,
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
        )
        assert result["assumption_valid"] is False

    def test_bioavailability_correction_applied(self):
        # f correction doubles Cmax
        result_no_f = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
        )
        result_with_f = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
            f_microdose=0.3,
            f_therapeutic=0.6,
        )
        assert result_with_f["cmax_predicted_ng_mL"] == pytest.approx(
            result_no_f["cmax_predicted_ng_mL"] * 2.0
        )

    def test_invalid_microdose_raises(self):
        with pytest.raises(ValueError):
            extrapolate_from_microdose(
                microdose_ug=-1.0,
                therapeutic_dose_mg=10.0,
                microdose_cmax_ng_mL=1.0,
                microdose_auc_ng_h_mL=5.0,
            )

    def test_invalid_therapeutic_dose_raises(self):
        with pytest.raises(ValueError):
            extrapolate_from_microdose(
                microdose_ug=100.0,
                therapeutic_dose_mg=0.0,
                microdose_cmax_ng_mL=1.0,
                microdose_auc_ng_h_mL=5.0,
            )

    def test_route_stored_in_notes(self):
        result = extrapolate_from_microdose(
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            microdose_cmax_ng_mL=1.0,
            microdose_auc_ng_h_mL=5.0,
            route="iv",
        )
        assert "iv" in result["notes"].lower()

    def test_dose_ratio_calculation(self):
        # 50 ug microdose, 1 mg therapeutic → ratio = 1000/50 = 20
        result = extrapolate_from_microdose(
            microdose_ug=50.0,
            therapeutic_dose_mg=1.0,
            microdose_cmax_ng_mL=2.0,
            microdose_auc_ng_h_mL=10.0,
        )
        assert result["dose_ratio"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# validate_linearity
# ---------------------------------------------------------------------------


class TestValidateLinearity:
    def test_proportional_doses_linear_true(self):
        # dose2 = 2 * dose1, AUC2 = 2 * AUC1 → perfectly proportional
        result = validate_linearity(
            dose1_mg=10.0,
            dose2_mg=20.0,
            auc1_ng_h_mL=100.0,
            auc2_ng_h_mL=200.0,
        )
        assert result["linear"] is True
        assert result["deviation_pct"] == pytest.approx(0.0)

    def test_non_proportional_auc_linear_false(self):
        # expected AUC2 = 200, observed = 300 → 50% deviation
        result = validate_linearity(
            dose1_mg=10.0,
            dose2_mg=20.0,
            auc1_ng_h_mL=100.0,
            auc2_ng_h_mL=300.0,
        )
        assert result["linear"] is False
        assert result["deviation_pct"] == pytest.approx(50.0)

    def test_expected_auc2_computed_correctly(self):
        result = validate_linearity(
            dose1_mg=5.0,
            dose2_mg=15.0,
            auc1_ng_h_mL=50.0,
            auc2_ng_h_mL=150.0,
        )
        assert result["expected_auc2"] == pytest.approx(150.0)

    def test_tolerance_respected(self):
        # 20% deviation with 25% tolerance → linear
        result = validate_linearity(
            dose1_mg=10.0,
            dose2_mg=20.0,
            auc1_ng_h_mL=100.0,
            auc2_ng_h_mL=240.0,  # expected 200, deviation 20%
            tolerance_pct=25.0,
        )
        assert result["linear"] is True

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            validate_linearity(
                dose1_mg=0.0,
                dose2_mg=10.0,
                auc1_ng_h_mL=100.0,
                auc2_ng_h_mL=200.0,
            )

    def test_observed_auc2_returned(self):
        result = validate_linearity(
            dose1_mg=10.0,
            dose2_mg=20.0,
            auc1_ng_h_mL=100.0,
            auc2_ng_h_mL=200.0,
        )
        assert result["observed_auc2"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# auc_from_sparse_sampling
# ---------------------------------------------------------------------------


class TestAucFromSparseSampling:
    def test_known_two_point_profile(self):
        # Simple 2-point: t=[0, 1], C=[10, 5]
        # AUC(0-1) = 0.5*(10+5)*1 = 7.5
        # ke = (ln10 - ln5)/1 = ln2 ≈ 0.693
        # AUC_extrap = 5 / 0.693 ≈ 7.21
        result = auc_from_sparse_sampling(
            times_h=[0.0, 1.0],
            concentrations_ng_mL=[10.0, 5.0],
        )
        assert result["auc_0_last"] == pytest.approx(7.5)
        assert result["ke_terminal_per_h"] == pytest.approx(math.log(2), rel=1e-3)
        assert result["t_half_h"] == pytest.approx(1.0, rel=1e-2)

    def test_auc_total_is_sum(self):
        result = auc_from_sparse_sampling(
            times_h=[0.0, 1.0, 2.0],
            concentrations_ng_mL=[100.0, 50.0, 25.0],
        )
        assert result["auc_total_ng_h_mL"] == pytest.approx(
            result["auc_0_last"] + result["auc_extrap"]
        )

    def test_pct_extrap_small_for_well_sampled(self):
        # Longer observation → less extrapolation needed
        times = [0.0, 1.0, 2.0, 4.0, 8.0]
        concs = [100.0, 50.0, 25.0, 6.25, 0.39]
        result = auc_from_sparse_sampling(times, concs)
        # At 8h most AUC is already covered
        assert result["pct_extrap"] < 20.0

    def test_single_timepoint_raises(self):
        with pytest.raises(ValueError):
            auc_from_sparse_sampling(times_h=[1.0], concentrations_ng_mL=[10.0])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            auc_from_sparse_sampling(
                times_h=[0.0, 1.0],
                concentrations_ng_mL=[10.0, 5.0, 2.0],
            )

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError):
            auc_from_sparse_sampling(
                times_h=[0.0, 1.0],
                concentrations_ng_mL=[10.0, -1.0],
            )


# ---------------------------------------------------------------------------
# microdose_clinical_utility
# ---------------------------------------------------------------------------


class TestMicrodoseClinicalUtility:
    def _make_result(self, dose_ratio_target: float) -> MicrodosePKResult:
        # microdose=100ug, pick therapeutic to get desired ratio
        therapeutic_mg = dose_ratio_target * 100.0 / 1000.0
        return microdose_clinical_utility(
            drug_name="TestDrug",
            microdose_ug=100.0,
            therapeutic_dose_mg=therapeutic_mg,
            f_microdose=0.5,
            cl_L_per_h=10.0,
            vd_L=50.0,
            route="oral",
        )

    def test_high_utility_rating(self):
        result = self._make_result(500.0)  # ratio 500 → high
        assert result.utility_rating == "high"

    def test_moderate_utility_rating(self):
        result = self._make_result(2000.0)  # ratio 2000 → moderate
        assert result.utility_rating == "moderate"

    def test_low_utility_rating(self):
        result = self._make_result(6000.0)  # ratio 6000 → low
        assert result.utility_rating == "low"

    def test_t_half_computed(self):
        result = microdose_clinical_utility(
            drug_name="Drug",
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            f_microdose=0.5,
            cl_L_per_h=10.0,
            vd_L=50.0,
        )
        expected_t_half = math.log(2) * 50.0 / 10.0
        assert result.t_half_h == pytest.approx(expected_t_half, rel=1e-4)

    def test_result_is_frozen_dataclass(self):
        result = microdose_clinical_utility(
            drug_name="Drug",
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            f_microdose=0.5,
            cl_L_per_h=10.0,
            vd_L=50.0,
        )
        assert isinstance(result, MicrodosePKResult)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_invalid_f_microdose_raises(self):
        with pytest.raises(ValueError):
            microdose_clinical_utility(
                drug_name="Drug",
                microdose_ug=100.0,
                therapeutic_dose_mg=10.0,
                f_microdose=0.0,
                cl_L_per_h=10.0,
                vd_L=50.0,
            )

    def test_dose_ratio_in_result(self):
        result = microdose_clinical_utility(
            drug_name="Drug",
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,  # 10000 ug → ratio 100
            f_microdose=0.5,
            cl_L_per_h=10.0,
            vd_L=50.0,
        )
        assert result.dose_ratio == pytest.approx(100.0)

    def test_auc_positive(self):
        result = microdose_clinical_utility(
            drug_name="Drug",
            microdose_ug=100.0,
            therapeutic_dose_mg=10.0,
            f_microdose=0.5,
            cl_L_per_h=10.0,
            vd_L=50.0,
        )
        assert result.auc_predicted_ng_h_mL > 0
