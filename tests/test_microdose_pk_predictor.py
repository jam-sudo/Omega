"""Tests for Phase 394 - Microdose PK Predictor."""

import pytest

from omega_pbpk.prediction.microdose_pk_predictor import (
    MicrodosePKResult,
    compare_extraction_ratios,
    predict_microdose_pk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _base_kwargs():
    """Standard microdose scenario: linear PK expected."""
    return dict(
        drug_name="TestDrug",
        microdose_ug=100.0,
        therapeutic_dose_mg=10.0,  # dose_ratio = 10*1000/100 = 100
        cmax_microdose_ng_mL=5.0,
        tmax_microdose_h=1.0,
        auc_microdose_ng_h_mL=50.0,
        t_half_microdose_h=4.0,
        estimated_fa=0.9,
        extraction_ratio=0.3,
    )


def _high_er_kwargs():
    """High extraction ratio scenario -> first-pass saturation risk."""
    kw = _base_kwargs()
    kw["extraction_ratio"] = 0.9
    kw["therapeutic_dose_mg"] = 200.0  # dose_ratio = 200*1000/100 = 2000 >> 100
    return kw


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_microdose_pk_result(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert isinstance(result, MicrodosePKResult)

    def test_result_is_frozen(self):
        result = predict_microdose_pk(**_base_kwargs())
        with pytest.raises(Exception):
            result.drug_name = "other"  # type: ignore

    def test_drug_name_preserved(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert result.drug_name == "TestDrug"

    def test_microdose_ug_preserved(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert result.microdose_ug == pytest.approx(100.0)

    def test_therapeutic_dose_preserved(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert result.therapeutic_dose_mg == pytest.approx(10.0)

    def test_notes_is_string(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Dose ratio tests
# ---------------------------------------------------------------------------


class TestDoseRatio:
    def test_dose_ratio_correct(self):
        # dose_ratio = 10 mg * 1000 / 100 ug = 100
        result = predict_microdose_pk(**_base_kwargs())
        assert result.dose_ratio == pytest.approx(100.0)

    def test_dose_ratio_small(self):
        kw = _base_kwargs()
        kw["microdose_ug"] = 100.0
        kw["therapeutic_dose_mg"] = 1.0  # ratio = 1000/100 = 10
        result = predict_microdose_pk(**kw)
        assert result.dose_ratio == pytest.approx(10.0)

    def test_dose_ratio_large(self):
        kw = _base_kwargs()
        kw["therapeutic_dose_mg"] = 100.0  # ratio = 100*1000/100 = 1000
        result = predict_microdose_pk(**kw)
        assert result.dose_ratio == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# AUC scaling tests
# ---------------------------------------------------------------------------


class TestAUCScaling:
    def test_auc_scales_linearly_for_proportional_pk(self):
        """For dose-proportional PK, AUC should scale by dose_ratio."""
        result = predict_microdose_pk(**_base_kwargs())
        if (
            result.dose_proportionality_flag
            and result.f_microdose == result.f_predicted_therapeutic
        ):
            expected_auc = result.auc_microdose_ng_h_per_mL * result.dose_ratio
            assert result.auc_predicted_therapeutic_ng_h_per_mL == pytest.approx(
                expected_auc, rel=1e-6
            )

    def test_auc_therapeutic_larger_than_microdose(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert result.auc_predicted_therapeutic_ng_h_per_mL > result.auc_microdose_ng_h_per_mL

    def test_auc_ratio_equals_dose_ratio_for_linear(self):
        kw = _base_kwargs()
        kw["extraction_ratio"] = 0.3
        kw["estimated_fa"] = 0.9
        result = predict_microdose_pk(**kw)
        if result.dose_proportionality_flag:
            auc_ratio = (
                result.auc_predicted_therapeutic_ng_h_per_mL / result.auc_microdose_ng_h_per_mL
            )
            assert auc_ratio == pytest.approx(result.dose_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# Saturation and dose proportionality tests
# ---------------------------------------------------------------------------


class TestSaturationRisk:
    def test_no_saturation_for_base_case(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert result.saturation_risk == "none"

    def test_dose_proportionality_true_for_base(self):
        result = predict_microdose_pk(**_base_kwargs())
        assert result.dose_proportionality_flag is True

    def test_first_pass_saturation_high_er(self):
        """High ER and high dose_ratio -> first_pass_saturation."""
        result = predict_microdose_pk(**_high_er_kwargs())
        assert "first_pass_saturation" in result.saturation_risk

    def test_dose_proportionality_false_for_high_er(self):
        result = predict_microdose_pk(**_high_er_kwargs())
        assert result.dose_proportionality_flag is False

    def test_absorption_saturation_flag(self):
        """Low fa and high dose_ratio -> absorption_saturation."""
        kw = _base_kwargs()
        kw["estimated_fa"] = 0.3  # < 0.5
        kw["therapeutic_dose_mg"] = 50.0  # dose_ratio=500, > 10
        result = predict_microdose_pk(**kw)
        assert "absorption_saturation" in result.saturation_risk

    def test_both_saturation_risk(self):
        """Both conditions met -> 'both'."""
        kw = _base_kwargs()
        kw["extraction_ratio"] = 0.9
        kw["estimated_fa"] = 0.3
        kw["therapeutic_dose_mg"] = 200.0  # dose_ratio=2000
        result = predict_microdose_pk(**kw)
        assert result.saturation_risk == "both"


# ---------------------------------------------------------------------------
# Confidence tests
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_high_confidence_small_dose_ratio(self):
        kw = _base_kwargs()
        kw["therapeutic_dose_mg"] = 5.0  # dose_ratio=50 < 100, no saturation
        result = predict_microdose_pk(**kw)
        assert result.prediction_confidence == "high"

    def test_medium_confidence_medium_dose_ratio(self):
        kw = _base_kwargs()
        kw["therapeutic_dose_mg"] = 50.0  # dose_ratio=500, < 1000
        result = predict_microdose_pk(**kw)
        # No saturation → medium confidence (dose_ratio >= 100)
        assert result.prediction_confidence in ("medium", "low")

    def test_low_confidence_very_large_dose_ratio_with_saturation(self):
        result = predict_microdose_pk(**_high_er_kwargs())
        # dose_ratio=2000, first_pass saturation
        assert result.prediction_confidence == "low"


# ---------------------------------------------------------------------------
# Compare extraction ratios tests
# ---------------------------------------------------------------------------


class TestCompareExtractionRatios:
    def test_compare_returns_five_results(self):
        kw = _base_kwargs()
        results = compare_extraction_ratios(
            drug_name=kw["drug_name"],
            microdose_ug=kw["microdose_ug"],
            therapeutic_dose_mg=kw["therapeutic_dose_mg"],
            cmax_ng_mL=kw["cmax_microdose_ng_mL"],
            tmax_h=kw["tmax_microdose_h"],
            auc_ng_h_mL=kw["auc_microdose_ng_h_mL"],
            t_half_h=kw["t_half_microdose_h"],
        )
        assert len(results) == 5

    def test_compare_sorted_descending_auc(self):
        kw = _base_kwargs()
        results = compare_extraction_ratios(
            drug_name=kw["drug_name"],
            microdose_ug=kw["microdose_ug"],
            therapeutic_dose_mg=kw["therapeutic_dose_mg"],
            cmax_ng_mL=kw["cmax_microdose_ng_mL"],
            tmax_h=kw["tmax_microdose_h"],
            auc_ng_h_mL=kw["auc_microdose_ng_h_mL"],
            t_half_h=kw["t_half_microdose_h"],
        )
        aucs = [r.auc_predicted_therapeutic_ng_h_per_mL for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_compare_returns_microdose_pk_results(self):
        kw = _base_kwargs()
        results = compare_extraction_ratios(
            drug_name=kw["drug_name"],
            microdose_ug=kw["microdose_ug"],
            therapeutic_dose_mg=kw["therapeutic_dose_mg"],
            cmax_ng_mL=kw["cmax_microdose_ng_mL"],
            tmax_h=kw["tmax_microdose_h"],
            auc_ng_h_mL=kw["auc_microdose_ng_h_mL"],
            t_half_h=kw["t_half_microdose_h"],
        )
        for r in results:
            assert isinstance(r, MicrodosePKResult)

    def test_compare_custom_ers(self):
        kw = _base_kwargs()
        custom_ers = [0.1, 0.5, 0.9]
        results = compare_extraction_ratios(
            drug_name=kw["drug_name"],
            microdose_ug=kw["microdose_ug"],
            therapeutic_dose_mg=kw["therapeutic_dose_mg"],
            cmax_ng_mL=kw["cmax_microdose_ng_mL"],
            tmax_h=kw["tmax_microdose_h"],
            auc_ng_h_mL=kw["auc_microdose_ng_h_mL"],
            t_half_h=kw["t_half_microdose_h"],
            ers=custom_ers,
        )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_microdose_above_100_ug_raises(self):
        kw = _base_kwargs()
        kw["microdose_ug"] = 101.0
        with pytest.raises(ValueError, match="microdose_ug"):
            predict_microdose_pk(**kw)

    def test_microdose_zero_raises(self):
        kw = _base_kwargs()
        kw["microdose_ug"] = 0.0
        with pytest.raises(ValueError, match="microdose_ug"):
            predict_microdose_pk(**kw)

    def test_microdose_negative_raises(self):
        kw = _base_kwargs()
        kw["microdose_ug"] = -1.0
        with pytest.raises(ValueError, match="microdose_ug"):
            predict_microdose_pk(**kw)

    def test_auc_zero_raises(self):
        kw = _base_kwargs()
        kw["auc_microdose_ng_h_mL"] = 0.0
        with pytest.raises(ValueError, match="auc_microdose"):
            predict_microdose_pk(**kw)

    def test_therapeutic_dose_zero_raises(self):
        kw = _base_kwargs()
        kw["therapeutic_dose_mg"] = 0.0
        with pytest.raises(ValueError, match="therapeutic_dose_mg"):
            predict_microdose_pk(**kw)

    def test_t_half_zero_raises(self):
        kw = _base_kwargs()
        kw["t_half_microdose_h"] = 0.0
        with pytest.raises(ValueError, match="t_half"):
            predict_microdose_pk(**kw)

    def test_fa_above_one_raises(self):
        kw = _base_kwargs()
        kw["estimated_fa"] = 1.5
        with pytest.raises(ValueError, match="estimated_fa"):
            predict_microdose_pk(**kw)

    def test_fa_zero_raises(self):
        kw = _base_kwargs()
        kw["estimated_fa"] = 0.0
        with pytest.raises(ValueError, match="estimated_fa"):
            predict_microdose_pk(**kw)

    def test_er_above_one_raises(self):
        kw = _base_kwargs()
        kw["extraction_ratio"] = 1.5
        with pytest.raises(ValueError, match="extraction_ratio"):
            predict_microdose_pk(**kw)

    def test_er_negative_raises(self):
        kw = _base_kwargs()
        kw["extraction_ratio"] = -0.1
        with pytest.raises(ValueError, match="extraction_ratio"):
            predict_microdose_pk(**kw)
