"""Tests for clinical/target_engagement_predictor.py — Phase 479."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.target_engagement_predictor import (
    TargetEngagementResult,
    duration_above_threshold,
    minimum_effective_conc,
    predict_target_engagement,
    te_time_course,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MW = 500.0  # g/mol — default


def conc_nM_to_mg_L(conc_nM: float, mw: float = MW) -> float:
    """Inverse of _mg_L_to_nM: nM -> mg/L."""
    return conc_nM * mw / 1_000_000.0 * 1_000.0


# ---------------------------------------------------------------------------
# predict_target_engagement — occupancy math
# ---------------------------------------------------------------------------


class TestPredictTargetEngagement:
    def test_returns_dataclass(self):
        result = predict_target_engagement("DrugA", 1.0, 0.5, 10.0)
        assert isinstance(result, TargetEngagementResult)

    def test_free_conc_equals_kd_gives_50pct(self):
        """At Cu = Kd, occupancy = 50%."""
        kd_nM = 100.0
        # We want free_plasma_nM == kd_nM
        # free_plasma_nM = total_nM * fu_plasma
        # total_nM = plasma_conc_mg_L * 1000 / mw * 1000
        fu = 1.0
        plasma_mg_L = conc_nM_to_mg_L(kd_nM, MW)
        result = predict_target_engagement("DrugA", plasma_mg_L, fu, kd_nM, MW)
        assert abs(result.target_occupancy_pct - 50.0) < 0.01

    def test_free_conc_much_greater_than_kd_approaches_100pct(self):
        """At Cu >> Kd, occupancy -> ~100%."""
        kd_nM = 1.0
        plasma_mg_L = conc_nM_to_mg_L(10_000.0, MW)  # 10 000x Kd
        result = predict_target_engagement("DrugX", plasma_mg_L, 1.0, kd_nM, MW)
        assert result.target_occupancy_pct > 99.0

    def test_free_conc_much_less_than_kd_approaches_0pct(self):
        """At Cu << Kd, occupancy -> ~0%."""
        kd_nM = 1000.0
        plasma_mg_L = conc_nM_to_mg_L(0.001, MW)  # 1/1 000 000 * Kd
        result = predict_target_engagement("DrugX", plasma_mg_L, 1.0, kd_nM, MW)
        assert result.target_occupancy_pct < 1.0

    def test_is_above_ec50_false_when_occupancy_below_50(self):
        kd_nM = 100.0
        plasma_mg_L = conc_nM_to_mg_L(10.0, MW)  # Cu << Kd -> occ < 50%
        result = predict_target_engagement("DrugA", plasma_mg_L, 1.0, kd_nM, MW)
        assert not result.is_above_ec50

    def test_is_above_ec50_true_when_occupancy_above_50(self):
        kd_nM = 10.0
        plasma_mg_L = conc_nM_to_mg_L(1000.0, MW)  # Cu >> Kd -> occ > 50%
        result = predict_target_engagement("DrugA", plasma_mg_L, 1.0, kd_nM, MW)
        assert result.is_above_ec50

    def test_safety_margin_equals_free_conc_over_kd(self):
        kd_nM = 50.0
        free_nM = 200.0
        fu = 1.0
        plasma_mg_L = conc_nM_to_mg_L(free_nM, MW)
        result = predict_target_engagement("DrugA", plasma_mg_L, fu, kd_nM, MW)
        expected_margin = free_nM / kd_nM
        assert abs(result.safety_margin - expected_margin) < 0.01

    def test_higher_fu_gives_higher_occupancy(self):
        kd_nM = 50.0
        plasma_mg_L = conc_nM_to_mg_L(50.0, MW)
        r_low_fu = predict_target_engagement("D", plasma_mg_L, 0.1, kd_nM, MW)
        r_high_fu = predict_target_engagement("D", plasma_mg_L, 1.0, kd_nM, MW)
        assert r_high_fu.target_occupancy_pct > r_low_fu.target_occupancy_pct

    def test_free_plasma_conc_nM_field(self):
        fu = 0.5
        kd_nM = 100.0
        plasma_mg_L = conc_nM_to_mg_L(200.0, MW)
        result = predict_target_engagement("D", plasma_mg_L, fu, kd_nM, MW)
        assert abs(result.free_plasma_conc_nM - 100.0) < 0.01

    def test_kp_brain_scales_free_conc(self):
        """With kp_brain, free brain conc = free_plasma * kp_brain."""
        kd_nM = 100.0
        plasma_mg_L = conc_nM_to_mg_L(200.0, MW)
        fu = 1.0
        # No brain: free = 200 nM -> occ high
        r_no_brain = predict_target_engagement("D", plasma_mg_L, fu, kd_nM, MW)
        # With kp_brain=0.1: free_brain = 20 nM -> occ lower
        r_brain = predict_target_engagement("D", plasma_mg_L, fu, kd_nM, MW, kp_brain=0.1)
        assert r_brain.target_occupancy_pct < r_no_brain.target_occupancy_pct

    def test_drug_name_preserved(self):
        result = predict_target_engagement("Aspirin", 1.0, 0.5, 10.0)
        assert result.drug_name == "Aspirin"

    def test_zero_plasma_conc_gives_zero_occupancy(self):
        result = predict_target_engagement("D", 0.0, 0.5, 10.0)
        assert result.target_occupancy_pct == 0.0

    def test_notes_is_string(self):
        result = predict_target_engagement("D", 1.0, 0.5, 10.0)
        assert isinstance(result.notes, str)

    # --- Input validation ---

    def test_invalid_fu_below_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_target_engagement("D", 1.0, -0.1, 10.0)

    def test_invalid_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_target_engagement("D", 1.0, 0.0, 10.0)

    def test_fu_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_target_engagement("D", 1.0, 1.1, 10.0)

    def test_kd_zero_raises(self):
        with pytest.raises(ValueError, match="kd_nM"):
            predict_target_engagement("D", 1.0, 0.5, 0.0)

    def test_kd_negative_raises(self):
        with pytest.raises(ValueError, match="kd_nM"):
            predict_target_engagement("D", 1.0, 0.5, -5.0)

    def test_negative_plasma_conc_raises(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            predict_target_engagement("D", -1.0, 0.5, 10.0)


# ---------------------------------------------------------------------------
# te_time_course
# ---------------------------------------------------------------------------


class TestTeTimeCourse:
    def test_returns_dict_with_required_keys(self):
        result = te_time_course([1.0, 2.0, 0.5], [0, 1, 2], 0.5, 10.0)
        for key in (
            "times_h",
            "plasma_concs_mg_L",
            "free_concs_nM",
            "occupancy_pct",
            "above_ec50",
            "peak_occupancy_pct",
            "trough_occupancy_pct",
        ):
            assert key in result

    def test_same_length_as_input(self):
        concs = [1.0, 2.0, 3.0, 4.0, 5.0]
        times = [0, 1, 2, 3, 4]
        result = te_time_course(concs, times, 0.5, 10.0)
        assert len(result["occupancy_pct"]) == len(concs)
        assert len(result["free_concs_nM"]) == len(concs)

    def test_peak_and_trough(self):
        concs = [1.0, 5.0, 0.1]
        times = [0, 1, 2]
        result = te_time_course(concs, times, 1.0, 10.0)
        assert result["peak_occupancy_pct"] >= result["trough_occupancy_pct"]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            te_time_course([1.0, 2.0], [0, 1, 2], 0.5, 10.0)

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            te_time_course([1.0], [0], 0.0, 10.0)

    def test_above_ec50_flag(self):
        kd_nM = 10.0
        fu = 1.0
        # At Kd: 50% -> at EC50
        conc_at_kd = conc_nM_to_mg_L(kd_nM, MW)
        result = te_time_course([conc_at_kd], [0], fu, kd_nM, MW)
        # Occupancy == 50 -> borderline; ">=" check means True
        assert result["above_ec50"][0] is True


# ---------------------------------------------------------------------------
# minimum_effective_conc
# ---------------------------------------------------------------------------


class TestMinimumEffectiveConc:
    def test_80pct_occupancy_requires_4x_kd(self):
        """At 80% occupancy, Cu = 4 * Kd."""
        kd_nM = 100.0
        fu = 1.0
        mec = minimum_effective_conc(kd_nM, fu, 80.0, MW)
        # Cu = 4*Kd nM; total = Cu/fu = 400 nM; mg/L = 400 * 500 / 1e6 * 1000
        expected_mg_L = 4.0 * kd_nM * MW / 1_000_000.0 * 1_000.0
        assert abs(mec - expected_mg_L) < 1e-9

    def test_50pct_occupancy_requires_1x_kd(self):
        """At 50% occupancy, Cu = Kd."""
        kd_nM = 50.0
        fu = 1.0
        mec = minimum_effective_conc(kd_nM, fu, 50.0, MW)
        expected_mg_L = kd_nM * MW / 1_000_000.0 * 1_000.0
        assert abs(mec - expected_mg_L) < 1e-9

    def test_lower_fu_requires_higher_total_conc(self):
        kd_nM = 100.0
        mec_high_fu = minimum_effective_conc(kd_nM, 1.0, 80.0, MW)
        mec_low_fu = minimum_effective_conc(kd_nM, 0.1, 80.0, MW)
        assert mec_low_fu > mec_high_fu

    def test_invalid_kd_raises(self):
        with pytest.raises(ValueError, match="kd_nM"):
            minimum_effective_conc(0.0, 0.5, 80.0, MW)

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            minimum_effective_conc(10.0, 0.0, 80.0, MW)

    def test_occupancy_100_raises(self):
        with pytest.raises(ValueError, match="target_occupancy_pct"):
            minimum_effective_conc(10.0, 0.5, 100.0, MW)

    def test_occupancy_0_raises(self):
        with pytest.raises(ValueError, match="target_occupancy_pct"):
            minimum_effective_conc(10.0, 0.5, 0.0, MW)

    def test_returns_positive_float(self):
        result = minimum_effective_conc(50.0, 0.5, 80.0, MW)
        assert result > 0


# ---------------------------------------------------------------------------
# duration_above_threshold
# ---------------------------------------------------------------------------


class TestDurationAboveThreshold:
    def test_all_above_returns_total_duration(self):
        times = [0.0, 1.0, 2.0, 3.0]
        profile = [80.0, 90.0, 85.0, 75.0]
        dur = duration_above_threshold(profile, times, threshold_pct=50.0)
        assert abs(dur - 3.0) < 1e-9

    def test_all_below_returns_zero(self):
        times = [0.0, 1.0, 2.0]
        profile = [10.0, 20.0, 30.0]
        dur = duration_above_threshold(profile, times, threshold_pct=50.0)
        assert dur == 0.0

    def test_partial_crossing(self):
        # Rising from 0 to 100 linearly; crosses 50 at t=0.5
        times = [0.0, 1.0]
        profile = [0.0, 100.0]
        dur = duration_above_threshold(profile, times, threshold_pct=50.0)
        assert abs(dur - 0.5) < 1e-9

    def test_single_point_returns_zero(self):
        dur = duration_above_threshold([80.0], [0.0], threshold_pct=50.0)
        assert dur == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            duration_above_threshold([80.0, 90.0], [0.0], threshold_pct=50.0)

    def test_falling_crossing(self):
        # Falls from 100 to 0; crosses 50 at t=0.5
        times = [0.0, 1.0]
        profile = [100.0, 0.0]
        dur = duration_above_threshold(profile, times, threshold_pct=50.0)
        assert abs(dur - 0.5) < 1e-9
