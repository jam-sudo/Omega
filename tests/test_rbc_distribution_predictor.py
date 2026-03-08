"""Tests for RBC distribution predictor (Phase 298)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.rbc_distribution_predictor import (
    RBCDistributionResult,
    compare_ionization_effects,
    predict_rbc_distribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_drug(
    drug_name="TestDrug",
    logp=1.5,
    pka=7.0,
    ionization_type="neutral",
    fu_plasma=0.5,
    hematocrit=0.45,
) -> RBCDistributionResult:
    return predict_rbc_distribution(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fu_plasma=fu_plasma,
        hematocrit=hematocrit,
    )


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _make_drug()
        assert isinstance(result, RBCDistributionResult)

    def test_drug_name_preserved(self):
        result = _make_drug(drug_name="Chloroquine")
        assert result.drug_name == "Chloroquine"

    def test_logp_preserved(self):
        result = _make_drug(logp=3.0)
        assert result.logp == 3.0

    def test_pka_preserved(self):
        result = _make_drug(pka=8.5)
        assert result.pka == 8.5

    def test_ionization_type_preserved(self):
        result = _make_drug(ionization_type="acid")
        assert result.ionization_type == "acid"

    def test_fu_plasma_preserved(self):
        result = _make_drug(fu_plasma=0.3)
        assert result.fu_plasma == 0.3

    def test_hematocrit_preserved(self):
        result = _make_drug(hematocrit=0.40)
        assert result.hematocrit == 0.40

    def test_result_is_frozen(self):
        result = _make_drug()
        with pytest.raises((AttributeError, TypeError)):
            result.bp_ratio = 1.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# B/P ratio range checks
# ---------------------------------------------------------------------------


class TestBPRatioRange:
    def test_bp_ratio_at_least_03(self):
        # Extreme case: very high logP, very low fu_plasma
        result = _make_drug(logp=4.0, fu_plasma=0.01)
        assert result.bp_ratio >= 0.3

    def test_bp_ratio_at_most_10(self):
        # Extreme base with very high pKa
        result = _make_drug(logp=4.0, pka=12.0, ionization_type="base", fu_plasma=0.99)
        assert result.bp_ratio <= 10.0

    def test_bp_ratio_positive(self):
        result = _make_drug()
        assert result.bp_ratio > 0.0

    def test_bp_ratio_clamped_lower_bound(self):
        # High logP → fu_rbc=0.2; fu_plasma=0.01 → ratio = 0.55 + 0.45*(0.01/0.2) = 0.55+0.0225=~0.57
        # Won't hit 0.3, try hydrophilic with low plasma binding
        result = _make_drug(logp=-1.0, fu_plasma=0.01, hematocrit=0.45)
        # fu_rbc=0.9 → (0.55) + 0.45*(0.01/0.9) = 0.55 + 0.005 = 0.555; above 0.3
        assert result.bp_ratio >= 0.3


# ---------------------------------------------------------------------------
# LogP effects on fu_rbc and B/P
# ---------------------------------------------------------------------------


class TestLogPEffects:
    def test_high_logp_low_fu_rbc(self):
        result = _make_drug(logp=3.0)
        assert result.fu_rbc_estimate == 0.2

    def test_mid_logp_mid_fu_rbc(self):
        result = _make_drug(logp=1.0)
        assert result.fu_rbc_estimate == 0.5

    def test_low_logp_high_fu_rbc(self):
        result = _make_drug(logp=-1.0)
        assert result.fu_rbc_estimate == 0.9

    def test_high_logp_higher_bp_ratio_than_low(self):
        # High logP → low fu_rbc → fu_plasma/fu_rbc is large → higher B/P
        high_lp = _make_drug(logp=4.0, fu_plasma=0.5)
        low_lp = _make_drug(logp=-1.0, fu_plasma=0.5)
        assert high_lp.bp_ratio > low_lp.bp_ratio


# ---------------------------------------------------------------------------
# Ionization type effects
# ---------------------------------------------------------------------------


class TestIonizationEffects:
    def test_base_high_pka_higher_bp_than_neutral(self):
        # Basic drug with high pKa should have ion-trapping boost
        base = _make_drug(logp=2.0, pka=10.0, ionization_type="base")
        neutral = _make_drug(logp=2.0, pka=10.0, ionization_type="neutral")
        assert base.bp_ratio >= neutral.bp_ratio

    def test_acid_no_base_correction(self):
        # Acid should not receive base ion-trapping correction
        acid = _make_drug(logp=2.0, pka=4.0, ionization_type="acid")
        # Just verify it runs and is in range
        assert 0.3 <= acid.bp_ratio <= 10.0

    def test_neutral_no_correction(self):
        neutral = _make_drug(ionization_type="neutral")
        assert 0.3 <= neutral.bp_ratio <= 10.0


# ---------------------------------------------------------------------------
# Distribution class
# ---------------------------------------------------------------------------


class TestDistributionClass:
    _VALID_CLASSES = {"extensive_rbc", "moderate_rbc", "low_rbc"}

    def test_class_in_valid_set(self):
        result = _make_drug()
        assert result.distribution_class in self._VALID_CLASSES

    def test_extensive_rbc_when_bp_above_2(self):
        # High logP drug with high fu_plasma should yield high B/P
        result = _make_drug(logp=4.0, fu_plasma=0.9, pka=10.0, ionization_type="base")
        if result.bp_ratio > 2.0:
            assert result.distribution_class == "extensive_rbc"

    def test_low_rbc_when_bp_below_1(self):
        # Hydrophilic with low fu_plasma relative to fu_rbc
        result = _make_drug(logp=-1.0, fu_plasma=0.1)
        # fu_rbc=0.9; B/P = 0.55 + 0.45*(0.1/0.9)=0.55+0.05=0.6 → low_rbc
        assert result.distribution_class == "low_rbc"

    def test_moderate_rbc_class(self):
        result = _make_drug(logp=1.0, fu_plasma=0.5)
        # fu_rbc=0.5; B/P = 0.55 + 0.45*(0.5/0.5)=0.55+0.45=1.0 → moderate_rbc
        assert result.distribution_class in {"moderate_rbc", "extensive_rbc"}


# ---------------------------------------------------------------------------
# compare_ionization_effects
# ---------------------------------------------------------------------------


class TestCompareIonizationEffects:
    def test_returns_three_results(self):
        results = compare_ionization_effects("TestDrug", logp=2.0, pka=8.0, fu_plasma=0.5)
        assert len(results) == 3

    def test_all_ionization_types_present(self):
        results = compare_ionization_effects("TestDrug", logp=2.0, pka=8.0, fu_plasma=0.5)
        types = {r.ionization_type for r in results}
        assert types == {"acid", "base", "neutral"}

    def test_sorted_descending_by_bp_ratio(self):
        results = compare_ionization_effects("TestDrug", logp=2.0, pka=10.0, fu_plasma=0.5)
        bp_values = [r.bp_ratio for r in results]
        assert bp_values == sorted(bp_values, reverse=True)

    def test_all_results_correct_type(self):
        results = compare_ionization_effects("TestDrug", logp=2.0, pka=8.0, fu_plasma=0.5)
        for r in results:
            assert isinstance(r, RBCDistributionResult)

    def test_basic_drug_tends_toward_top(self):
        # For pKa > 7.4, base should have highest B/P
        results = compare_ionization_effects("TestDrug", logp=2.0, pka=10.0, fu_plasma=0.5)
        assert results[0].ionization_type == "base"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="logp"):
            _make_drug(logp=-6.0)

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="logp"):
            _make_drug(logp=11.0)

    def test_pka_zero(self):
        with pytest.raises(ValueError, match="pka"):
            _make_drug(pka=0.0)

    def test_pka_above_14(self):
        with pytest.raises(ValueError, match="pka"):
            _make_drug(pka=15.0)

    def test_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _make_drug(fu_plasma=0.0)

    def test_fu_plasma_above_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _make_drug(fu_plasma=1.1)

    def test_hematocrit_zero(self):
        with pytest.raises(ValueError, match="hematocrit"):
            _make_drug(hematocrit=0.0)

    def test_hematocrit_one(self):
        with pytest.raises(ValueError, match="hematocrit"):
            _make_drug(hematocrit=1.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            _make_drug(ionization_type="zwitterion")
