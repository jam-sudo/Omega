"""Tests for Phase 492 — Drug Efflux Pump Inhibition Effect."""

import pytest

from omega_pbpk.clinical.efflux_pump_inhibition import (
    EffluxPumpInhibitionResult,
    compare_inhibitors,
    predict_efflux_pump_inhibition,
)


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = predict_efflux_pump_inhibition("Digoxin", "verapamil", 0.3, 1.0)
        assert isinstance(result, EffluxPumpInhibitionResult)

    def test_result_is_frozen(self):
        result = predict_efflux_pump_inhibition("Digoxin", "verapamil", 0.3, 1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = predict_efflux_pump_inhibition("Digoxin", "verapamil", 0.3, 1.0)
        assert result.drug_name == "Digoxin"

    def test_inhibitor_preserved(self):
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.2, 0.1)
        assert result.inhibitor == "tariquidar"

    def test_pump_target_assigned(self):
        result = predict_efflux_pump_inhibition("Drug", "verapamil", 0.3, 1.0)
        assert result.pump_target == "P-gp"


class TestNoInhibition:
    def test_zero_conc_fa_unchanged(self):
        """At inhibitor_conc=0, fa_inhibited must equal base_fa."""
        result = predict_efflux_pump_inhibition("Drug", "verapamil", 0.4, 0.0)
        assert result.fa_inhibited == pytest.approx(result.fa_baseline)

    def test_zero_conc_auc_ratio_is_one(self):
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.2, 0.0)
        assert result.auc_ratio == pytest.approx(1.0)

    def test_zero_conc_inhibition_fraction_zero(self):
        result = predict_efflux_pump_inhibition("Drug", "Ko143", 0.5, 0.0)
        assert result.inhibition_fraction == pytest.approx(0.0)

    def test_zero_conc_no_bbb_increase(self):
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.3, 0.0)
        assert result.bbb_penetration_ratio == pytest.approx(1.0)


class TestTariquidarHighInhibition:
    def test_fi_near_complete_at_1uM(self):
        """tariquidar Ki=0.01 uM; at 1 uM, fi = 1/(1+0.01) ≈ 0.99."""
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.2, 1.0)
        assert result.inhibition_fraction > 0.99

    def test_tariquidar_high_auc_ratio(self):
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.1, 1.0)
        assert result.auc_ratio > 5.0  # max_fold=10, fi~0.99 => large ratio

    def test_tariquidar_significant_benefit(self):
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.1, 1.0)
        assert result.clinical_benefit == "significant"


class TestAbsorptionLimits:
    def test_fa_inhibited_not_exceed_one(self):
        for inhibitor in ["verapamil", "elacridar", "tariquidar", "Ko143", "MK571"]:
            result = predict_efflux_pump_inhibition("Drug", inhibitor, 0.8, 100.0)
            assert result.fa_inhibited <= 1.0

    def test_auc_ratio_at_least_one(self):
        for inhibitor in ["verapamil", "elacridar", "tariquidar", "Ko143", "MK571"]:
            result = predict_efflux_pump_inhibition("Drug", inhibitor, 0.3, 1.0)
            assert result.auc_ratio >= 1.0

    def test_fa_inhibited_increases_with_conc(self):
        low = predict_efflux_pump_inhibition("Drug", "verapamil", 0.3, 0.5)
        high = predict_efflux_pump_inhibition("Drug", "verapamil", 0.3, 5.0)
        assert high.fa_inhibited >= low.fa_inhibited


class TestBBBPenetration:
    def test_pgp_inhibitor_increases_bbb(self):
        result = predict_efflux_pump_inhibition("Drug", "verapamil", 0.3, 1.0)
        assert result.bbb_penetration_ratio > 1.0

    def test_dual_pgp_bcrp_inhibitor_increases_bbb(self):
        result = predict_efflux_pump_inhibition("Drug", "elacridar", 0.3, 1.0)
        assert result.bbb_penetration_ratio > 1.0

    def test_bcrp_only_no_bbb_increase(self):
        """Ko143 targets BCRP only, not P-gp => bbb_ratio = 1.0."""
        result = predict_efflux_pump_inhibition("Drug", "Ko143", 0.3, 1.0)
        assert result.bbb_penetration_ratio == pytest.approx(1.0)

    def test_mrp_no_bbb_increase(self):
        result = predict_efflux_pump_inhibition("Drug", "MK571", 0.3, 1.0)
        assert result.bbb_penetration_ratio == pytest.approx(1.0)

    def test_bbb_ratio_increases_with_conc(self):
        low = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.3, 0.001)
        high = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.3, 10.0)
        assert high.bbb_penetration_ratio > low.bbb_penetration_ratio


class TestClinicalBenefit:
    def test_benefit_none_at_zero_conc(self):
        result = predict_efflux_pump_inhibition("Drug", "verapamil", 0.5, 0.0)
        assert result.clinical_benefit == "none"

    def test_benefit_significant_for_potent_inhibitor(self):
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.1, 1.0)
        assert result.clinical_benefit == "significant"

    def test_benefit_values_are_valid(self):
        for inhibitor in ["verapamil", "elacridar", "tariquidar", "Ko143", "MK571"]:
            result = predict_efflux_pump_inhibition("Drug", inhibitor, 0.3, 1.0)
            assert result.clinical_benefit in ("none", "minor", "significant")


class TestCombinationRecommended:
    def test_combination_recommended_for_tariquidar(self):
        result = predict_efflux_pump_inhibition("Drug", "tariquidar", 0.2, 1.0)
        assert result.combination_recommended is True

    def test_combination_not_recommended_at_zero_conc(self):
        for inhibitor in ["verapamil", "elacridar", "tariquidar", "Ko143", "MK571"]:
            result = predict_efflux_pump_inhibition("Drug", inhibitor, 0.5, 0.0)
            assert result.combination_recommended is False

    def test_combination_recommended_is_bool(self):
        result = predict_efflux_pump_inhibition("Drug", "Ko143", 0.3, 1.0)
        assert isinstance(result.combination_recommended, bool)


class TestCompareInhibitors:
    def test_returns_list(self):
        results = compare_inhibitors("Drug", 0.3)
        assert isinstance(results, list)

    def test_returns_five_results(self):
        results = compare_inhibitors("Drug", 0.3)
        assert len(results) == 5

    def test_sorted_by_auc_ratio_descending(self):
        results = compare_inhibitors("Drug", 0.3, inhibitor_conc_uM=1.0)
        for i in range(1, len(results)):
            assert results[i - 1].auc_ratio >= results[i].auc_ratio

    def test_all_results_are_correct_type(self):
        results = compare_inhibitors("Drug", 0.3)
        for r in results:
            assert isinstance(r, EffluxPumpInhibitionResult)

    def test_tariquidar_is_top_pgp_inhibitor(self):
        """tariquidar (Ki=0.01) should beat verapamil (Ki=2.0) in potency."""
        results = compare_inhibitors("Drug", 0.1, inhibitor_conc_uM=1.0)
        names = [r.inhibitor for r in results]
        assert names.index("tariquidar") < names.index("verapamil")


class TestValidation:
    def test_invalid_inhibitor(self):
        with pytest.raises(ValueError, match="inhibitor must be one of"):
            predict_efflux_pump_inhibition("Drug", "unknown_drug", 0.3, 1.0)

    def test_base_fa_zero(self):
        with pytest.raises(ValueError, match="base_fa must be in"):
            predict_efflux_pump_inhibition("Drug", "verapamil", 0.0, 1.0)

    def test_base_fa_negative(self):
        with pytest.raises(ValueError, match="base_fa must be in"):
            predict_efflux_pump_inhibition("Drug", "verapamil", -0.1, 1.0)

    def test_base_fa_exceeds_one(self):
        with pytest.raises(ValueError, match="base_fa must be in"):
            predict_efflux_pump_inhibition("Drug", "verapamil", 1.1, 1.0)

    def test_negative_inhibitor_conc(self):
        with pytest.raises(ValueError, match="inhibitor_conc_uM must be >= 0"):
            predict_efflux_pump_inhibition("Drug", "verapamil", 0.3, -1.0)
