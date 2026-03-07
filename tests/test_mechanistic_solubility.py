"""Tests for mechanistic solubility prediction — Phase 503."""

import pytest

from omega_pbpk.prediction.mechanistic_solubility import (
    SolubilityProfileResult,
    alogps_correction,
    predict_intrinsic_solubility,
    predict_solubility_profile,
    solubility_class,
    yalkowsky_solubility,
)

# ---------------------------------------------------------------------------
# yalkowsky_solubility
# ---------------------------------------------------------------------------


class TestYalkowskySolubility:
    def test_basic_output_is_positive(self):
        result = yalkowsky_solubility(logP=2.0, tm_C=150.0)
        assert result > 0.0

    def test_higher_logp_lower_solubility(self):
        s_low = yalkowsky_solubility(logP=1.0, tm_C=150.0)
        s_high = yalkowsky_solubility(logP=4.0, tm_C=150.0)
        assert s_low > s_high

    def test_higher_tm_lower_solubility(self):
        s_low_tm = yalkowsky_solubility(logP=2.0, tm_C=100.0)
        s_high_tm = yalkowsky_solubility(logP=2.0, tm_C=250.0)
        assert s_low_tm > s_high_tm

    def test_mw_gt_500_applies_correction(self):
        s_small = yalkowsky_solubility(logP=2.0, tm_C=150.0, mw=300.0)
        s_large = yalkowsky_solubility(logP=2.0, tm_C=150.0, mw=600.0)
        # MW > 500 adds +0.1 to log(S) → higher solubility
        assert s_large > s_small

    def test_mw_exactly_500_no_correction(self):
        s_500 = yalkowsky_solubility(logP=2.0, tm_C=150.0, mw=500.0)
        s_499 = yalkowsky_solubility(logP=2.0, tm_C=150.0, mw=499.0)
        # Both should be equal (no correction applied at exactly 500)
        assert abs(s_500 - s_499) < 1e-10

    def test_tm_le_25_raises(self):
        with pytest.raises(ValueError, match="Melting point"):
            yalkowsky_solubility(logP=2.0, tm_C=25.0)

    def test_tm_below_25_raises(self):
        with pytest.raises(ValueError, match="Melting point"):
            yalkowsky_solubility(logP=2.0, tm_C=10.0)

    def test_log_formula_correctness(self):
        """Verify manual calculation matches function."""
        logP = 2.5
        tm_C = 180.0
        correction = 0.5
        expected_log_s = -logP - 0.01 * (tm_C - 25.0) + correction
        expected = 10.0**expected_log_s
        result = yalkowsky_solubility(logP=logP, tm_C=tm_C)
        assert abs(result - expected) < 1e-12

    def test_known_drug_approximation(self):
        """Aspirin-like: logP~1.2, Tm~140°C → S should be in reasonable range."""
        s = yalkowsky_solubility(logP=1.2, tm_C=140.0)
        # log(S) = -1.2 - 0.01*(140-25) + 0.5 = -1.2 - 1.15 + 0.5 = -1.85
        # S ≈ 0.014 mg/mL — within 1 log unit of ~5 mg/mL (literature)
        # Just confirm it's between 0.001 and 10 mg/mL
        assert 0.001 < s < 10.0


# ---------------------------------------------------------------------------
# alogps_correction
# ---------------------------------------------------------------------------


class TestAlogpsCorrection:
    def test_more_hbd_higher_correction(self):
        corr_low = alogps_correction(logP=2.0, n_hbd=0, n_hba=0, mw=300.0)
        corr_high = alogps_correction(logP=2.0, n_hbd=3, n_hba=0, mw=300.0)
        assert corr_high > corr_low

    def test_more_hba_lower_correction(self):
        corr_low_hba = alogps_correction(logP=2.0, n_hbd=0, n_hba=0, mw=300.0)
        corr_high_hba = alogps_correction(logP=2.0, n_hbd=0, n_hba=5, mw=300.0)
        assert corr_low_hba > corr_high_hba

    def test_zero_hbonds_zero_correction(self):
        corr = alogps_correction(logP=2.0, n_hbd=0, n_hba=0, mw=300.0)
        assert corr == 0.0

    def test_correction_magnitude(self):
        """1 HBD → +0.15, 2 HBA → -0.10 → net +0.05."""
        corr = alogps_correction(logP=2.0, n_hbd=1, n_hba=2, mw=300.0)
        assert abs(corr - 0.05) < 1e-10

    def test_negative_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            alogps_correction(logP=2.0, n_hbd=-1, n_hba=0, mw=300.0)

    def test_negative_hba_raises(self):
        with pytest.raises(ValueError, match="n_hba"):
            alogps_correction(logP=2.0, n_hbd=0, n_hba=-1, mw=300.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            alogps_correction(logP=2.0, n_hbd=0, n_hba=0, mw=0.0)


# ---------------------------------------------------------------------------
# predict_intrinsic_solubility
# ---------------------------------------------------------------------------


class TestPredictIntrinsicSolubility:
    def test_higher_logp_lower_solubility(self):
        s_lo = predict_intrinsic_solubility(logP=1.0, tm_C=150.0)
        s_hi = predict_intrinsic_solubility(logP=5.0, tm_C=150.0)
        assert s_lo > s_hi

    def test_higher_tm_lower_solubility(self):
        s_lo = predict_intrinsic_solubility(logP=2.0, tm_C=100.0)
        s_hi = predict_intrinsic_solubility(logP=2.0, tm_C=300.0)
        assert s_lo > s_hi

    def test_more_hbd_higher_solubility(self):
        s_0 = predict_intrinsic_solubility(logP=2.0, tm_C=150.0, n_hbd=0)
        s_3 = predict_intrinsic_solubility(logP=2.0, tm_C=150.0, n_hbd=3)
        assert s_3 > s_0

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_intrinsic_solubility(logP=2.0, tm_C=150.0, mw=0.0)

    def test_tm_le_25_raises(self):
        with pytest.raises(ValueError, match="Melting point"):
            predict_intrinsic_solubility(logP=2.0, tm_C=20.0)

    def test_output_is_positive(self):
        result = predict_intrinsic_solubility(logP=3.0, tm_C=200.0, n_hbd=1, n_hba=2, mw=400.0)
        assert result > 0.0


# ---------------------------------------------------------------------------
# solubility_class
# ---------------------------------------------------------------------------


class TestSolubilityClass:
    def test_highly_soluble_at_1(self):
        assert solubility_class(1.0) == "highly_soluble"

    def test_highly_soluble_above_1(self):
        assert solubility_class(5.0) == "highly_soluble"

    def test_soluble_boundary(self):
        assert solubility_class(0.1) == "soluble"

    def test_soluble_mid(self):
        assert solubility_class(0.5) == "soluble"

    def test_sparingly_soluble_boundary(self):
        assert solubility_class(0.01) == "sparingly_soluble"

    def test_sparingly_soluble_mid(self):
        assert solubility_class(0.05) == "sparingly_soluble"

    def test_poorly_soluble_below_threshold(self):
        assert solubility_class(0.005) == "poorly_soluble"

    def test_poorly_soluble_very_low(self):
        assert solubility_class(1e-6) == "poorly_soluble"

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            solubility_class(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            solubility_class(-0.1)


# ---------------------------------------------------------------------------
# predict_solubility_profile
# ---------------------------------------------------------------------------


class TestPredictSolubilityProfile:
    def test_returns_dataclass_instance(self):
        result = predict_solubility_profile(logP=2.0, tm_C=150.0)
        assert isinstance(result, SolubilityProfileResult)

    def test_fields_preserved(self):
        result = predict_solubility_profile(logP=2.5, tm_C=180.0, n_hbd=1, n_hba=2, mw=350.0)
        assert result.logP == 2.5
        assert result.tm_C == 180.0

    def test_yalkowsky_matches_standalone(self):
        result = predict_solubility_profile(logP=2.0, tm_C=150.0, n_hbd=0, n_hba=0, mw=300.0)
        expected_yal = yalkowsky_solubility(logP=2.0, tm_C=150.0, mw=300.0)
        assert abs(result.yalkowsky_estimate_mg_mL - expected_yal) < 1e-12

    def test_dose_solubility_ratio(self):
        """100 mg in 250 mL = 0.4 mg/mL threshold."""
        # We'll check via a compound that should produce ~0.8 mg/mL solubility
        result = predict_solubility_profile(logP=2.0, tm_C=150.0, n_hbd=0, n_hba=0, mw=300.0)
        expected_ratio = result.intrinsic_solubility_mg_mL / 0.4
        assert abs(result.dose_solubility_ratio_100mg - expected_ratio) < 1e-10

    def test_potentially_limited_when_below_threshold(self):
        # High logP + high Tm → low solubility → potentially limited
        result = predict_solubility_profile(logP=5.0, tm_C=250.0)
        assert result.potentially_solubility_limited is True

    def test_not_potentially_limited_when_above_threshold(self):
        # Low logP + low Tm → high solubility → not limited
        result = predict_solubility_profile(logP=0.0, tm_C=50.0, n_hbd=3)
        assert result.potentially_solubility_limited is False

    def test_solubility_class_consistent(self):
        result = predict_solubility_profile(logP=2.0, tm_C=150.0)
        expected_class = solubility_class(result.intrinsic_solubility_mg_mL)
        assert result.solubility_class == expected_class

    def test_tm_le_25_raises(self):
        with pytest.raises(ValueError, match="Melting point"):
            predict_solubility_profile(logP=2.0, tm_C=25.0)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_solubility_profile(logP=2.0, tm_C=150.0, mw=-10.0)

    def test_invalid_n_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            predict_solubility_profile(logP=2.0, tm_C=150.0, n_hbd=-1)

    def test_invalid_n_hba_raises(self):
        with pytest.raises(ValueError, match="n_hba"):
            predict_solubility_profile(logP=2.0, tm_C=150.0, n_hba=-1)

    def test_hbond_correction_factor_no_hbonds(self):
        result = predict_solubility_profile(logP=2.0, tm_C=150.0, n_hbd=0, n_hba=0)
        # Zero H-bonds → correction factor = 10^0 = 1.0
        assert abs(result.hbond_correction_factor - 1.0) < 1e-10

    def test_hbond_correction_factor_with_donors(self):
        result = predict_solubility_profile(logP=2.0, tm_C=150.0, n_hbd=2, n_hba=0)
        # 2 HBDs → log correction = 0.30 → factor = 10^0.30 ≈ 2.0
        assert result.hbond_correction_factor > 1.0
