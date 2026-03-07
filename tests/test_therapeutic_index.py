"""Tests for Phase 570 — Therapeutic Index and Safety Margin."""

import pytest

from omega_pbpk.clinical.therapeutic_index import (
    certain_safety_factor,
    dose_range_to_achieve_window,
    nti_flag,
    pk_pd_safety_window,
    safety_margin,
    therapeutic_index,
)

# ---------------------------------------------------------------------------
# therapeutic_index
# ---------------------------------------------------------------------------


class TestTherapeuticIndex:
    def test_basic_ratio(self):
        assert therapeutic_index(100.0, 10.0) == pytest.approx(10.0)

    def test_ti_equals_1_when_equal(self):
        assert therapeutic_index(50.0, 50.0) == pytest.approx(1.0)

    def test_fractional_doses(self):
        assert therapeutic_index(0.5, 0.1) == pytest.approx(5.0, rel=1e-9)

    def test_td50_zero_raises(self):
        with pytest.raises(ValueError, match="td50"):
            therapeutic_index(0.0, 10.0)

    def test_ed50_zero_raises(self):
        with pytest.raises(ValueError, match="ed50"):
            therapeutic_index(100.0, 0.0)

    def test_negative_td50_raises(self):
        with pytest.raises(ValueError):
            therapeutic_index(-10.0, 5.0)

    def test_negative_ed50_raises(self):
        with pytest.raises(ValueError):
            therapeutic_index(50.0, -5.0)


# ---------------------------------------------------------------------------
# certain_safety_factor
# ---------------------------------------------------------------------------


class TestCertainSafetyFactor:
    def test_basic_ratio(self):
        assert certain_safety_factor(10.0, 5.0) == pytest.approx(2.0)

    def test_csf_more_conservative_than_ti(self):
        """CSF uses LD1 and ED99, which are closer together than TD50 and ED50,
        so CSF < TI for a typical drug with a normal-distributed dose-response."""
        # Simulate: TD50=100, ED50=10 → TI=10
        # LD1 < TD50, ED99 > ED50 → CSF < TI
        ti = therapeutic_index(100.0, 10.0)
        # Use LD1=30 (below TD50), ED99=20 (above ED50) → CSF=1.5 < 10
        csf = certain_safety_factor(30.0, 20.0)
        assert csf < ti

    def test_ld1_zero_raises(self):
        with pytest.raises(ValueError, match="ld1"):
            certain_safety_factor(0.0, 5.0)

    def test_ed99_zero_raises(self):
        with pytest.raises(ValueError, match="ed99"):
            certain_safety_factor(10.0, 0.0)


# ---------------------------------------------------------------------------
# safety_margin
# ---------------------------------------------------------------------------


class TestSafetyMargin:
    def test_margin_pct_formula(self):
        # NOAEL=100, efficacy=10 → margin = (100-10)/10*100 = 900%
        result = safety_margin(100.0, 10.0)
        assert result["safety_margin_pct"] == pytest.approx(900.0)

    def test_adequate_classification(self):
        # ratio=10 > 10 → adequate
        result = safety_margin(100.0, 10.0)
        assert result["classification"] == "adequate"

    def test_narrow_classification(self):
        # ratio=5 (2–10) → narrow
        result = safety_margin(50.0, 10.0)
        assert result["classification"] == "narrow"

    def test_inadequate_classification(self):
        # ratio=1.5 < 2 → inadequate
        result = safety_margin(15.0, 10.0)
        assert result["classification"] == "inadequate"

    def test_returns_required_keys(self):
        result = safety_margin(100.0, 10.0)
        assert set(result.keys()) == {"safety_margin_pct", "classification"}

    def test_noael_zero_raises(self):
        with pytest.raises(ValueError):
            safety_margin(0.0, 10.0)

    def test_efficacy_zero_raises(self):
        with pytest.raises(ValueError):
            safety_margin(100.0, 0.0)


# ---------------------------------------------------------------------------
# pk_pd_safety_window
# ---------------------------------------------------------------------------


class TestPkPdSafetyWindow:
    def test_large_separation_low_risk(self):
        result = pk_pd_safety_window(
            cmax_efficacy=1.0,
            cmax_toxicity=20.0,
            auc_efficacy=10.0,
            auc_toxicity=200.0,
        )
        assert result["risk_level"] == "low"
        assert result["therapeutic_window"] == pytest.approx(20.0)

    def test_moderate_risk(self):
        result = pk_pd_safety_window(
            cmax_efficacy=1.0,
            cmax_toxicity=3.0,
            auc_efficacy=10.0,
            auc_toxicity=40.0,
        )
        assert result["risk_level"] == "moderate"

    def test_high_risk(self):
        result = pk_pd_safety_window(
            cmax_efficacy=1.0,
            cmax_toxicity=1.5,
            auc_efficacy=10.0,
            auc_toxicity=15.0,
        )
        assert result["risk_level"] == "high"

    def test_therapeutic_window_is_min(self):
        result = pk_pd_safety_window(
            cmax_efficacy=1.0,
            cmax_toxicity=6.0,
            auc_efficacy=10.0,
            auc_toxicity=30.0,
        )
        # cmax_window=6, auc_window=3 → min=3
        assert result["therapeutic_window"] == pytest.approx(3.0)

    def test_required_keys(self):
        result = pk_pd_safety_window(1.0, 10.0, 5.0, 50.0)
        assert set(result.keys()) == {
            "cmax_window",
            "auc_window",
            "therapeutic_window",
            "risk_level",
        }

    def test_zero_efficacy_raises(self):
        with pytest.raises(ValueError):
            pk_pd_safety_window(0.0, 10.0, 5.0, 50.0)


# ---------------------------------------------------------------------------
# nti_flag
# ---------------------------------------------------------------------------


class TestNtiFlag:
    def test_nti_true_below_threshold(self):
        assert nti_flag(1.5, threshold=2.0) is True

    def test_nti_true_at_threshold(self):
        assert nti_flag(2.0, threshold=2.0) is True

    def test_nti_false_above_threshold(self):
        assert nti_flag(2.1, threshold=2.0) is False

    def test_default_threshold_2(self):
        assert nti_flag(1.9) is True
        assert nti_flag(2.5) is False

    def test_custom_threshold(self):
        assert nti_flag(3.0, threshold=3.5) is True
        assert nti_flag(4.0, threshold=3.5) is False


# ---------------------------------------------------------------------------
# dose_range_to_achieve_window
# ---------------------------------------------------------------------------


class TestDoseRangeToAchieveWindow:
    def test_e_equals_half_emax_at_ec50_when_hill1(self):
        """At Hill=1 and E=50% Emax, inversion should give C=EC50."""
        result = dose_range_to_achieve_window(
            ed50=10.0,
            hill=1.0,
            emax=100.0,
            min_effect=50.0,
            max_safe_effect=80.0,
        )
        assert result["c_min_mg_L"] == pytest.approx(10.0, rel=1e-6)

    def test_c_max_greater_than_c_min(self):
        result = dose_range_to_achieve_window(
            ed50=10.0,
            hill=1.0,
            emax=100.0,
            min_effect=20.0,
            max_safe_effect=80.0,
        )
        assert result["c_max_mg_L"] > result["c_min_mg_L"]

    def test_therapeutic_range_fold(self):
        result = dose_range_to_achieve_window(
            ed50=10.0,
            hill=1.0,
            emax=100.0,
            min_effect=20.0,
            max_safe_effect=80.0,
        )
        expected_fold = result["c_max_mg_L"] / result["c_min_mg_L"]
        assert result["therapeutic_range_fold"] == pytest.approx(expected_fold, rel=1e-6)

    def test_required_keys(self):
        result = dose_range_to_achieve_window(
            ed50=5.0,
            hill=2.0,
            emax=100.0,
            min_effect=25.0,
            max_safe_effect=75.0,
        )
        assert set(result.keys()) == {"c_min_mg_L", "c_max_mg_L", "therapeutic_range_fold", "notes"}

    def test_min_effect_zero_raises(self):
        with pytest.raises(ValueError):
            dose_range_to_achieve_window(10.0, 1.0, 100.0, 0.0, 50.0)

    def test_min_effect_ge_emax_raises(self):
        with pytest.raises(ValueError):
            dose_range_to_achieve_window(10.0, 1.0, 100.0, 100.0, 100.0)

    def test_max_safe_effect_le_min_raises(self):
        with pytest.raises(ValueError):
            dose_range_to_achieve_window(10.0, 1.0, 100.0, 50.0, 30.0)

    def test_ed50_zero_raises(self):
        with pytest.raises(ValueError, match="ed50"):
            dose_range_to_achieve_window(0.0, 1.0, 100.0, 25.0, 75.0)

    def test_hill_zero_raises(self):
        with pytest.raises(ValueError, match="hill"):
            dose_range_to_achieve_window(10.0, 0.0, 100.0, 25.0, 75.0)
