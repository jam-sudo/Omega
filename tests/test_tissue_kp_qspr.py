"""Tests for prediction/tissue_distribution.py — Phase 294."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.tissue_distribution import (
    TissueKpResult,
    predict_tissue_kp_qspr,
    predict_vss_from_kp,
    tissue_distribution_profile,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TISSUES_13 = {
    "adipose",
    "bone",
    "brain",
    "gut",
    "heart",
    "kidney",
    "liver",
    "lung",
    "muscle",
    "skin",
    "spleen",
    "thymus",
    "gonads",
}


# ---------------------------------------------------------------------------
# predict_tissue_kp_qspr
# ---------------------------------------------------------------------------


class TestPredictTissueKpQspr:
    def test_returns_tissue_kp_result(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        assert isinstance(result, TissueKpResult)

    def test_all_13_tissues_present(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        assert set(result.kp_values.keys()) == TISSUES_13

    def test_kp_values_positive(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        for tissue, kp in result.kp_values.items():
            assert kp > 0.0, f"Kp for {tissue} should be positive"

    def test_adipose_highest_logP_positive(self):
        """Adipose has highest lipid fraction -> highest Kp for lipophilic drug."""
        result = predict_tissue_kp_qspr(3.0, None, None, 0.1)
        assert result.highest_kp_tissue == "adipose"

    def test_logP_increases_kp_adipose(self):
        low = predict_tissue_kp_qspr(1.0, None, None, 0.1)
        high = predict_tissue_kp_qspr(5.0, None, None, 0.1)
        assert high.kp_values["adipose"] > low.kp_values["adipose"]

    def test_fu_plasma_decreases_kp(self):
        low_fu = predict_tissue_kp_qspr(2.0, None, None, 0.01)
        high_fu = predict_tissue_kp_qspr(2.0, None, None, 0.9)
        assert low_fu.kp_values["liver"] > high_fu.kp_values["liver"]

    def test_logP_field_stored(self):
        result = predict_tissue_kp_qspr(2.5, None, None, 0.1)
        assert result.logP == pytest.approx(2.5)

    def test_fu_plasma_field_stored(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.2)
        assert result.fu_plasma == pytest.approx(0.2)

    def test_bp_ratio_field_stored(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1, bp_ratio=1.5)
        assert result.bp_ratio == pytest.approx(1.5)

    def test_vss_positive(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        assert result.vss_L_per_kg > 0.0

    def test_highest_kp_tissue_in_dict(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        assert result.highest_kp_tissue in result.kp_values

    def test_lowest_kp_tissue_in_dict(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        assert result.lowest_kp_tissue in result.kp_values

    def test_highest_is_max(self):
        result = predict_tissue_kp_qspr(3.0, None, None, 0.1)
        assert result.kp_values[result.highest_kp_tissue] == max(result.kp_values.values())

    def test_lowest_is_min(self):
        result = predict_tissue_kp_qspr(3.0, None, None, 0.1)
        assert result.kp_values[result.lowest_kp_tissue] == min(result.kp_values.values())

    def test_pka_acid_stored_in_notes(self):
        result = predict_tissue_kp_qspr(2.0, pka_acid=4.5, pka_base=None, fu_plasma=0.1)
        assert "4.5" in result.notes

    def test_pka_base_stored_in_notes(self):
        result = predict_tissue_kp_qspr(2.0, pka_acid=None, pka_base=9.0, fu_plasma=0.1)
        assert "9.0" in result.notes

    def test_invalid_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_tissue_kp_qspr(2.0, None, None, 0.0)

    def test_invalid_fu_plasma_negative_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_tissue_kp_qspr(2.0, None, None, -0.1)

    def test_invalid_fu_plasma_gt1_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_tissue_kp_qspr(2.0, None, None, 1.5)

    def test_invalid_bp_ratio_raises(self):
        with pytest.raises(ValueError, match="bp_ratio"):
            predict_tissue_kp_qspr(2.0, None, None, 0.1, bp_ratio=-1.0)

    def test_negative_logP_kp_still_positive(self):
        result = predict_tissue_kp_qspr(-3.0, None, None, 0.5)
        for kp in result.kp_values.values():
            assert kp > 0.0

    def test_frozen_dataclass(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        with pytest.raises((AttributeError, TypeError)):
            result.logP = 999.0  # type: ignore[misc]

    def test_vss_in_result_matches_standalone(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        standalone = predict_vss_from_kp(result.kp_values, 0.1)
        assert result.vss_L_per_kg == pytest.approx(standalone, rel=1e-4)


# ---------------------------------------------------------------------------
# predict_vss_from_kp
# ---------------------------------------------------------------------------


class TestPredictVssFromKp:
    def test_returns_float(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        vss = predict_vss_from_kp(result.kp_values, 0.1)
        assert isinstance(vss, float)

    def test_vss_positive(self):
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        vss = predict_vss_from_kp(result.kp_values, 0.1)
        assert vss > 0.0

    def test_vss_increases_with_logP(self):
        low = predict_tissue_kp_qspr(1.0, None, None, 0.1)
        high = predict_tissue_kp_qspr(5.0, None, None, 0.1)
        vss_low = predict_vss_from_kp(low.kp_values, 0.1)
        vss_high = predict_vss_from_kp(high.kp_values, 0.1)
        assert vss_high > vss_low

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError):
            predict_vss_from_kp({"adipose": 1.0}, fu_plasma=0.0)

    def test_vss_at_least_plasma_volume(self):
        # Minimum Vss = plasma volume = 0.04 L/kg + tissue contributions
        result = predict_tissue_kp_qspr(2.0, None, None, 0.1)
        vss = predict_vss_from_kp(result.kp_values, 0.1)
        assert vss >= 0.04

    def test_invalid_bp_ratio_raises(self):
        with pytest.raises(ValueError, match="bp_ratio"):
            predict_vss_from_kp({"adipose": 1.0}, fu_plasma=0.1, bp_ratio=0.0)


# ---------------------------------------------------------------------------
# tissue_distribution_profile
# ---------------------------------------------------------------------------


class TestTissueDistributionProfile:
    def test_returns_list_of_results(self):
        results = tissue_distribution_profile([1.0, 2.0, 3.0], 0.1)
        assert isinstance(results, list)
        assert all(isinstance(r, TissueKpResult) for r in results)

    def test_length_matches_logP_range(self):
        logP_range = [0.0, 1.0, 2.0, 3.0, 4.0]
        results = tissue_distribution_profile(logP_range, 0.1)
        assert len(results) == len(logP_range)

    def test_logP_values_stored_correctly(self):
        logP_range = [1.0, 3.0, 5.0]
        results = tissue_distribution_profile(logP_range, 0.1)
        for r, lp in zip(results, logP_range):
            assert r.logP == pytest.approx(lp)

    def test_empty_range_raises(self):
        with pytest.raises(ValueError):
            tissue_distribution_profile([], 0.1)

    def test_adipose_kp_monotonic_with_logP(self):
        logP_range = [1.0, 2.0, 3.0, 4.0]
        results = tissue_distribution_profile(logP_range, 0.1)
        kps = [r.kp_values["adipose"] for r in results]
        assert all(kps[i] < kps[i + 1] for i in range(len(kps) - 1))

    def test_fu_plasma_consistent_across_profile(self):
        fu = 0.3
        results = tissue_distribution_profile([1.0, 2.0], fu)
        for r in results:
            assert r.fu_plasma == pytest.approx(fu)
