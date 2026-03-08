"""Tests for Phase 312 — Drug-Induced QT Prolongation Risk Calculator."""

import pytest

from omega_pbpk.risk.qt_prolongation_risk_p312 import (
    QTRiskResult,
    calculate_qt_risk,
    compare_dose_levels,
)

VALID_RISK_CATEGORIES = {"high", "moderate", "low"}
VALID_REGULATORY_CONCERNS = {"black_box_warning", "precaution", "none"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_qt(**kwargs):
    defaults = dict(
        drug_name="DrugA",
        ic50_herg_uM=1.0,
        cmax_free_nM=10.0,
        logp=2.0,
        mw=400.0,
        clinical_dose_mg=100.0,
        has_sodium_channel_block=False,
    )
    defaults.update(kwargs)
    return calculate_qt_risk(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = make_qt()
        assert isinstance(result, QTRiskResult)

    def test_drug_name_preserved(self):
        result = make_qt(drug_name="Cisapride")
        assert result.drug_name == "Cisapride"

    def test_ic50_herg_um_preserved(self):
        result = make_qt(ic50_herg_uM=0.5)
        assert result.ic50_herg_uM == 0.5

    def test_cmax_free_nm_preserved(self):
        result = make_qt(cmax_free_nM=50.0)
        assert result.cmax_free_nM == 50.0

    def test_logp_preserved(self):
        result = make_qt(logp=1.5)
        assert result.logp == 1.5

    def test_mw_preserved(self):
        result = make_qt(mw=350.0)
        assert result.mw == 350.0

    def test_clinical_dose_preserved(self):
        result = make_qt(clinical_dose_mg=200.0)
        assert result.clinical_dose_mg == 200.0


# ---------------------------------------------------------------------------
# IC50 conversion
# ---------------------------------------------------------------------------


class TestIC50Conversion:
    def test_ic50_nM_conversion(self):
        result = make_qt(ic50_herg_uM=2.0)
        assert abs(result.ic50_herg_nM - 2000.0) < 1e-9

    def test_ic50_nM_conversion_small(self):
        result = make_qt(ic50_herg_uM=0.01)
        assert abs(result.ic50_herg_nM - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# Safety margin
# ---------------------------------------------------------------------------


class TestSafetyMargin:
    def test_safety_margin_positive(self):
        result = make_qt()
        assert result.safety_margin > 0

    def test_safety_margin_formula(self):
        result = make_qt(ic50_herg_uM=1.0, cmax_free_nM=10.0)
        expected = 1000.0 / 10.0  # 100
        assert abs(result.safety_margin - expected) < 1e-9

    def test_higher_cmax_lower_safety_margin(self):
        low_cmax = make_qt(cmax_free_nM=1.0)
        high_cmax = make_qt(cmax_free_nM=100.0)
        assert high_cmax.safety_margin < low_cmax.safety_margin

    def test_higher_ic50_higher_safety_margin(self):
        low_ic50 = make_qt(ic50_herg_uM=0.1)
        high_ic50 = make_qt(ic50_herg_uM=10.0)
        assert high_ic50.safety_margin > low_ic50.safety_margin


# ---------------------------------------------------------------------------
# hERG inhibition percentage
# ---------------------------------------------------------------------------


class TestHERGInhibition:
    def test_inhibition_pct_in_range(self):
        result = make_qt()
        assert 0.0 <= result.herg_inhibition_pct <= 100.0

    def test_inhibition_increases_with_cmax(self):
        low = make_qt(cmax_free_nM=1.0)
        high = make_qt(cmax_free_nM=1000.0)
        assert high.herg_inhibition_pct > low.herg_inhibition_pct

    def test_inhibition_at_ic50_is_50_pct(self):
        # Cmax == IC50 → 50% inhibition
        result = make_qt(ic50_herg_uM=1.0, cmax_free_nM=1000.0)
        assert abs(result.herg_inhibition_pct - 50.0) < 1e-9


# ---------------------------------------------------------------------------
# Sodium channel block (INa) modifier
# ---------------------------------------------------------------------------


class TestSodiumChannelBlock:
    def test_ina_block_reduces_delta_qtc(self):
        without = make_qt(has_sodium_channel_block=False, logp=2.0, mw=400.0)
        with_block = make_qt(has_sodium_channel_block=True, logp=2.0, mw=400.0)
        assert with_block.delta_qtc_ms_estimate < without.delta_qtc_ms_estimate

    def test_ina_block_reduces_by_30_pct(self):
        """Compare two results where only has_sodium_channel_block differs."""
        without = make_qt(has_sodium_channel_block=False, logp=2.0, mw=400.0)
        with_block = make_qt(has_sodium_channel_block=True, logp=2.0, mw=400.0)
        ratio = with_block.delta_qtc_ms_estimate / without.delta_qtc_ms_estimate
        assert abs(ratio - 0.70) < 1e-9


# ---------------------------------------------------------------------------
# Risk category
# ---------------------------------------------------------------------------


class TestRiskCategory:
    def test_risk_category_valid_value(self):
        result = make_qt()
        assert result.risk_category in VALID_RISK_CATEGORIES

    def test_low_safety_margin_is_high_risk(self):
        # safety_margin < 10 → high risk
        result = make_qt(ic50_herg_uM=0.001, cmax_free_nM=1000.0)
        assert result.risk_category == "high"

    def test_high_safety_margin_is_low_risk(self):
        # safety_margin >> 30, delta_qtc small
        result = make_qt(ic50_herg_uM=100.0, cmax_free_nM=1.0, logp=1.0, mw=400.0)
        assert result.risk_category == "low"


# ---------------------------------------------------------------------------
# Regulatory concern
# ---------------------------------------------------------------------------


class TestRegulatoryConcern:
    def test_regulatory_concern_valid_value(self):
        result = make_qt()
        assert result.regulatory_concern in VALID_REGULATORY_CONCERNS

    def test_safe_drug_no_concern(self):
        result = make_qt(ic50_herg_uM=100.0, cmax_free_nM=1.0, logp=1.0, mw=400.0)
        assert result.regulatory_concern == "none"

    def test_high_qtc_black_box_warning(self):
        # Need delta_qtc > 30; very low IC50 and high Cmax with logp>3 and mw<300
        result = make_qt(
            ic50_herg_uM=0.01,
            cmax_free_nM=5000.0,
            logp=4.0,
            mw=250.0,
            has_sodium_channel_block=False,
        )
        assert result.regulatory_concern == "black_box_warning"


# ---------------------------------------------------------------------------
# Thorough QT study required
# ---------------------------------------------------------------------------


class TestTQTStudy:
    def test_tqt_required_when_safety_margin_low(self):
        result = make_qt(ic50_herg_uM=0.01, cmax_free_nM=100.0)
        assert result.thorough_qt_study_required is True

    def test_tqt_not_required_for_safe_drug(self):
        result = make_qt(ic50_herg_uM=100.0, cmax_free_nM=1.0, logp=1.0, mw=400.0)
        assert result.thorough_qt_study_required is False

    def test_tqt_required_is_bool(self):
        result = make_qt()
        assert isinstance(result.thorough_qt_study_required, bool)


# ---------------------------------------------------------------------------
# compare_dose_levels
# ---------------------------------------------------------------------------


class TestCompareDoseLevels:
    def test_returns_list(self):
        results = compare_dose_levels("Drug", 1.0, [10.0, 50.0, 200.0], 2.0, 400.0, False)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_ascending_by_safety_margin(self):
        results = compare_dose_levels("Drug", 1.0, [5.0, 50.0, 500.0], 2.0, 400.0, False)
        margins = [r.safety_margin for r in results]
        assert margins == sorted(margins)

    def test_all_results_correct_type(self):
        results = compare_dose_levels("Drug", 1.0, [10.0, 100.0], 2.0, 400.0, False)
        for r in results:
            assert isinstance(r, QTRiskResult)

    def test_single_dose_level(self):
        results = compare_dose_levels("Drug", 1.0, [10.0], 2.0, 400.0, False)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_ic50_zero(self):
        with pytest.raises(ValueError, match="ic50_herg_uM"):
            make_qt(ic50_herg_uM=0.0)

    def test_invalid_ic50_negative(self):
        with pytest.raises(ValueError, match="ic50_herg_uM"):
            make_qt(ic50_herg_uM=-1.0)

    def test_invalid_cmax_zero(self):
        with pytest.raises(ValueError, match="cmax_free_nM"):
            make_qt(cmax_free_nM=0.0)

    def test_invalid_cmax_negative(self):
        with pytest.raises(ValueError, match="cmax_free_nM"):
            make_qt(cmax_free_nM=-5.0)

    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw"):
            make_qt(mw=0.0)

    def test_invalid_clinical_dose_zero(self):
        with pytest.raises(ValueError, match="clinical_dose_mg"):
            make_qt(clinical_dose_mg=0.0)
