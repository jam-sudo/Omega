"""Tests for QSAR-based clearance predictor (Phase 563)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.qsar_clearance_predictor import (
    clearance_sensitivity,
    predict_cl_ckd,
    predict_cl_from_properties,
    predict_renal_cl,
    vd_from_cl_thalf,
)

# ---------------------------------------------------------------------------
# predict_cl_from_properties
# ---------------------------------------------------------------------------


class TestPredictClFromProperties:
    """Tests for the QSAR-based total CL prediction."""

    _base_kwargs = dict(logP=2.0, mw=350.0, psa=60.0, fu_plasma=0.5)

    def test_returns_dict(self):
        result = predict_cl_from_properties(**self._base_kwargs)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = predict_cl_from_properties(**self._base_kwargs)
        assert "cl_predicted_L_per_h" in result
        assert "cl_category" in result
        assert "hepatic_fraction_estimate" in result
        assert "notes" in result

    def test_cl_positive(self):
        result = predict_cl_from_properties(**self._base_kwargs)
        assert result["cl_predicted_L_per_h"] > 0.0

    def test_higher_fu_higher_cl(self):
        """Higher unbound fraction -> higher predicted CL."""
        r_high = predict_cl_from_properties(logP=2.0, mw=350.0, psa=60.0, fu_plasma=0.9)
        r_low = predict_cl_from_properties(logP=2.0, mw=350.0, psa=60.0, fu_plasma=0.1)
        assert r_high["cl_predicted_L_per_h"] > r_low["cl_predicted_L_per_h"]

    def test_higher_psa_lower_cl(self):
        """Higher PSA -> lower CL (reduces lipophilicity-driven metabolism)."""
        r_high = predict_cl_from_properties(logP=2.0, mw=350.0, psa=120.0, fu_plasma=0.5)
        r_low = predict_cl_from_properties(logP=2.0, mw=350.0, psa=30.0, fu_plasma=0.5)
        assert r_high["cl_predicted_L_per_h"] < r_low["cl_predicted_L_per_h"]

    def test_cyp3a4_substrate_increases_cl(self):
        """CYP3A4 substrate -> 50% higher CL."""
        r_sub = predict_cl_from_properties(
            logP=2.0, mw=350.0, psa=60.0, fu_plasma=0.5, cyp3a4_substrate=True
        )
        r_nonsub = predict_cl_from_properties(
            logP=2.0, mw=350.0, psa=60.0, fu_plasma=0.5, cyp3a4_substrate=False
        )
        assert abs(r_sub["cl_predicted_L_per_h"] / r_nonsub["cl_predicted_L_per_h"] - 1.5) < 1e-9

    def test_low_logP_reduces_cl(self):
        """logP < 0 -> 0.7x multiplier."""
        r_neg = predict_cl_from_properties(logP=-1.0, mw=350.0, psa=60.0, fu_plasma=0.5)
        r_mid = predict_cl_from_properties(logP=1.0, mw=350.0, psa=60.0, fu_plasma=0.5)
        assert r_neg["cl_predicted_L_per_h"] < r_mid["cl_predicted_L_per_h"]

    def test_high_logP_increases_cl(self):
        """logP > 3 -> 1.2x multiplier."""
        r_high = predict_cl_from_properties(logP=4.0, mw=350.0, psa=60.0, fu_plasma=0.5)
        r_mid = predict_cl_from_properties(logP=1.5, mw=350.0, psa=60.0, fu_plasma=0.5)
        assert r_high["cl_predicted_L_per_h"] > r_mid["cl_predicted_L_per_h"]

    def test_cl_category_low(self):
        """Very low fu -> low CL category."""
        r = predict_cl_from_properties(logP=0.0, mw=500.0, psa=100.0, fu_plasma=0.01)
        assert r["cl_category"] == "low"

    def test_cl_category_valid_set(self):
        r = predict_cl_from_properties(**self._base_kwargs)
        assert r["cl_category"] in {"low", "moderate", "high"}

    def test_hepatic_fraction_clamped(self):
        """hepatic_fraction_estimate must be between 0.3 and 0.95."""
        r = predict_cl_from_properties(logP=10.0, mw=350.0, psa=60.0, fu_plasma=0.5)
        assert r["hepatic_fraction_estimate"] <= 0.95
        r2 = predict_cl_from_properties(logP=-10.0, mw=350.0, psa=60.0, fu_plasma=0.5)
        assert r2["hepatic_fraction_estimate"] >= 0.3

    def test_notes_non_empty(self):
        r = predict_cl_from_properties(**self._base_kwargs)
        assert len(r["notes"]) > 0

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError):
            predict_cl_from_properties(logP=2.0, mw=350.0, psa=60.0, fu_plasma=1.5)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            predict_cl_from_properties(logP=2.0, mw=0.0, psa=60.0, fu_plasma=0.5)


# ---------------------------------------------------------------------------
# predict_renal_cl
# ---------------------------------------------------------------------------


class TestPredictRenalCl:
    """Tests for renal clearance prediction."""

    def test_fu1_gfr120_reabs0_gives_7_2(self):
        """fu=1, GFR=120, reabsorption=0 -> CL_renal = 7.2 L/h."""
        r = predict_renal_cl(fu_plasma=1.0, gfr_mL_min=120.0, reabsorption_fraction=0.0)
        assert abs(r["cl_renal_L_per_h"] - 7.2) < 1e-9

    def test_returns_dict_with_keys(self):
        r = predict_renal_cl(fu_plasma=0.5)
        assert "cl_renal_L_per_h" in r
        assert "fe_renal_fraction" in r
        assert "notes" in r

    def test_reabsorption_reduces_cl(self):
        r_no = predict_renal_cl(fu_plasma=1.0, gfr_mL_min=120.0, reabsorption_fraction=0.0)
        r_half = predict_renal_cl(fu_plasma=1.0, gfr_mL_min=120.0, reabsorption_fraction=0.5)
        assert r_no["cl_renal_L_per_h"] > r_half["cl_renal_L_per_h"]

    def test_fe_renal_fraction_correct(self):
        r = predict_renal_cl(fu_plasma=0.5, reabsorption_fraction=0.3)
        assert abs(r["fe_renal_fraction"] - 0.7) < 1e-9

    def test_zero_fu_gives_zero_cl(self):
        r = predict_renal_cl(fu_plasma=0.0)
        assert r["cl_renal_L_per_h"] == 0.0

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError):
            predict_renal_cl(fu_plasma=1.5)

    def test_negative_gfr_raises(self):
        with pytest.raises(ValueError):
            predict_renal_cl(fu_plasma=0.5, gfr_mL_min=-10.0)


# ---------------------------------------------------------------------------
# predict_cl_ckd
# ---------------------------------------------------------------------------


class TestPredictClCkd:
    """Tests for CKD clearance adjustment."""

    def test_normal_egfr_no_change(self):
        """At eGFR=120 (normal), CL_CKD == cl_normal."""
        r = predict_cl_ckd(cl_normal_L_per_h=10.0, egfr_mL_min=120.0, renal_fraction=0.4)
        assert abs(r["cl_ckd_L_per_h"] - 10.0) < 1e-9

    def test_zero_egfr_nonrenal_only(self):
        """At eGFR=0, CL_CKD = cl_normal * (1 - renal_fraction)."""
        r = predict_cl_ckd(cl_normal_L_per_h=10.0, egfr_mL_min=0.0, renal_fraction=0.5)
        assert abs(r["cl_ckd_L_per_h"] - 5.0) < 1e-9

    def test_cl_ratio_vs_normal_correct(self):
        r = predict_cl_ckd(cl_normal_L_per_h=10.0, egfr_mL_min=60.0, renal_fraction=0.5)
        # CL_CKD = 10*(0.5) + 10*0.5*(60/120) = 5 + 2.5 = 7.5
        assert abs(r["cl_ckd_L_per_h"] - 7.5) < 1e-9
        assert abs(r["cl_ratio_vs_normal"] - 0.75) < 1e-9

    def test_dose_adjustment_le_1_for_ckd(self):
        """dose_adjustment_factor <= 1 for CKD (reduced eGFR)."""
        r = predict_cl_ckd(cl_normal_L_per_h=10.0, egfr_mL_min=30.0, renal_fraction=0.6)
        assert r["dose_adjustment_factor"] <= 1.0

    def test_dose_adjustment_1_for_normal(self):
        r = predict_cl_ckd(cl_normal_L_per_h=10.0, egfr_mL_min=120.0, renal_fraction=0.5)
        assert abs(r["dose_adjustment_factor"] - 1.0) < 1e-9

    def test_zero_renal_fraction_no_effect(self):
        """No renal fraction -> CKD doesn't affect CL."""
        r = predict_cl_ckd(cl_normal_L_per_h=10.0, egfr_mL_min=10.0, renal_fraction=0.0)
        assert abs(r["cl_ckd_L_per_h"] - 10.0) < 1e-9

    def test_invalid_cl_normal_raises(self):
        with pytest.raises(ValueError):
            predict_cl_ckd(cl_normal_L_per_h=0.0, egfr_mL_min=60.0, renal_fraction=0.5)


# ---------------------------------------------------------------------------
# vd_from_cl_thalf
# ---------------------------------------------------------------------------


class TestVdFromClThalf:
    """Tests for Vd calculation from CL and t1/2."""

    def test_correct_calculation(self):
        """Vd = CL * t_half / ln(2)."""
        cl = 5.0
        t_half = 10.0
        expected = cl * t_half / math.log(2.0)
        assert abs(vd_from_cl_thalf(cl, t_half) - expected) < 1e-9

    def test_positive_result(self):
        assert vd_from_cl_thalf(3.0, 8.0) > 0.0

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            vd_from_cl_thalf(0.0, 5.0)

    def test_zero_thalf_raises(self):
        with pytest.raises(ValueError):
            vd_from_cl_thalf(5.0, 0.0)


# ---------------------------------------------------------------------------
# clearance_sensitivity
# ---------------------------------------------------------------------------


class TestClearanceSensitivity:
    """Tests for clearance sensitivity analysis."""

    _fu_range = [0.1, 0.3, 0.5, 0.7, 0.9]

    def test_length_matches_fu_range(self):
        result = clearance_sensitivity(self._fu_range, mw=350.0, psa=60.0, logP=2.0)
        assert len(result) == len(self._fu_range)

    def test_each_entry_has_required_keys(self):
        result = clearance_sensitivity(self._fu_range, mw=350.0, psa=60.0, logP=2.0)
        for entry in result:
            assert "fu_plasma" in entry
            assert "cl_predicted_L_per_h" in entry

    def test_fu_values_preserved(self):
        result = clearance_sensitivity(self._fu_range, mw=350.0, psa=60.0, logP=2.0)
        for i, entry in enumerate(result):
            assert abs(entry["fu_plasma"] - self._fu_range[i]) < 1e-9

    def test_cl_increases_with_fu(self):
        """Higher fu -> higher CL (monotonic)."""
        result = clearance_sensitivity(self._fu_range, mw=350.0, psa=60.0, logP=2.0)
        cls = [r["cl_predicted_L_per_h"] for r in result]
        for i in range(len(cls) - 1):
            assert cls[i] < cls[i + 1]

    def test_empty_list_returns_empty(self):
        result = clearance_sensitivity([], mw=350.0, psa=60.0, logP=2.0)
        assert result == []

    def test_single_fu_value(self):
        result = clearance_sensitivity([0.5], mw=350.0, psa=60.0, logP=2.0)
        assert len(result) == 1
