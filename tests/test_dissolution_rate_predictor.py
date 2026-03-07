"""Tests for Phase 241: dissolution_rate_predictor."""

import pytest

from omega_pbpk.prediction.dissolution_rate_predictor import (
    DissolutionRateResult,
    compare_crystal_forms,
    predict_dissolution_rate,
)

_VALID_DISSOLUTION_CLASSES = {"class1", "class2", "class3"}


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = predict_dissolution_rate("DrugA", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        assert isinstance(result, DissolutionRateResult)

    def test_fields_present(self):
        result = predict_dissolution_rate("DrugA", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        assert hasattr(result, "drug_name")
        assert hasattr(result, "intrinsic_dissolution_rate_mg_cm2_min")
        assert hasattr(result, "dissolution_class")
        assert hasattr(result, "estimated_t90_min")


class TestIDR:
    def test_idr_positive(self):
        result = predict_dissolution_rate("DrugA", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        assert result.intrinsic_dissolution_rate_mg_cm2_min > 0

    def test_idr_increases_with_solubility(self):
        r_low = predict_dissolution_rate("X", solubility_mg_mL=0.1, mw=300.0, logp=2.0)
        r_high = predict_dissolution_rate("X", solubility_mg_mL=10.0, mw=300.0, logp=2.0)
        assert (
            r_high.intrinsic_dissolution_rate_mg_cm2_min
            > r_low.intrinsic_dissolution_rate_mg_cm2_min
        )

    def test_idr_amorphous_greater_than_crystalline(self):
        r_am = predict_dissolution_rate(
            "X", solubility_mg_mL=1.0, mw=300.0, logp=2.0, crystal_form="amorphous"
        )
        r_cr = predict_dissolution_rate(
            "X", solubility_mg_mL=1.0, mw=300.0, logp=2.0, crystal_form="crystalline"
        )
        assert (
            r_am.intrinsic_dissolution_rate_mg_cm2_min > r_cr.intrinsic_dissolution_rate_mg_cm2_min
        )

    def test_apparent_dissolution_rate_positive(self):
        result = predict_dissolution_rate("DrugA", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        assert result.apparent_dissolution_rate_mg_min > 0

    def test_apparent_rate_equals_idr_times_area(self):
        area = 5.0
        result = predict_dissolution_rate(
            "X", solubility_mg_mL=1.0, mw=300.0, logp=2.0, surface_area_cm2=area
        )
        expected = result.intrinsic_dissolution_rate_mg_cm2_min * area
        assert abs(result.apparent_dissolution_rate_mg_min - expected) < 1e-10


class TestDissolutionClass:
    def test_dissolution_class_valid(self):
        for sol in [0.01, 0.5, 5.0]:
            result = predict_dissolution_rate("X", solubility_mg_mL=sol, mw=300.0, logp=2.0)
            assert result.dissolution_class in _VALID_DISSOLUTION_CLASSES

    def test_high_solubility_class1(self):
        # IDR = D*Cs/h = 4.44e-4 * sol / 25e-4 = 0.1776 * sol
        # For class1: IDR > 1.0 => sol > 1.0/0.1776 ~ 5.63
        result = predict_dissolution_rate("X", solubility_mg_mL=20.0, mw=300.0, logp=2.0)
        assert result.dissolution_class == "class1"

    def test_low_solubility_class3(self):
        # class3: IDR < 0.1 => sol < 0.1/0.1776 ~ 0.563
        # amorphous modifier=1.0, so sol=0.01 gives IDR << 0.1
        result = predict_dissolution_rate("X", solubility_mg_mL=0.01, mw=300.0, logp=2.0)
        assert result.dissolution_class == "class3"

    def test_improvement_needed_class3(self):
        result = predict_dissolution_rate("X", solubility_mg_mL=0.01, mw=300.0, logp=2.0)
        assert result.improvement_needed is True

    def test_no_improvement_needed_class1(self):
        result = predict_dissolution_rate("X", solubility_mg_mL=20.0, mw=300.0, logp=2.0)
        assert result.improvement_needed is False


class TestT90:
    def test_t90_positive(self):
        result = predict_dissolution_rate("X", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        assert result.estimated_t90_min > 0

    def test_t90_clamped_at_999(self):
        # Extremely low solubility -> t90 clamped
        result = predict_dissolution_rate(
            "X", solubility_mg_mL=1e-8, mw=300.0, logp=2.0, surface_area_cm2=0.001
        )
        assert result.estimated_t90_min == 999.0

    def test_higher_solubility_lower_t90(self):
        r_low = predict_dissolution_rate("X", solubility_mg_mL=0.5, mw=300.0, logp=2.0)
        r_high = predict_dissolution_rate("X", solubility_mg_mL=5.0, mw=300.0, logp=2.0)
        assert r_high.estimated_t90_min < r_low.estimated_t90_min


class TestCompareCrystalForms:
    def test_returns_5_results(self):
        results = compare_crystal_forms("X", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        assert len(results) == 5

    def test_all_are_dataclass_instances(self):
        results = compare_crystal_forms("X", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        for r in results:
            assert isinstance(r, DissolutionRateResult)

    def test_sorted_by_idr_descending(self):
        results = compare_crystal_forms("X", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        idrs = [r.intrinsic_dissolution_rate_mg_cm2_min for r in results]
        assert idrs == sorted(idrs, reverse=True)

    def test_amorphous_in_results(self):
        results = compare_crystal_forms("X", solubility_mg_mL=1.0, mw=300.0, logp=2.0)
        forms = {r.crystal_form for r in results}
        assert "amorphous" in forms


class TestValidation:
    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            predict_dissolution_rate("X", solubility_mg_mL=-1.0, mw=300.0, logp=2.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            predict_dissolution_rate("X", solubility_mg_mL=0.0, mw=300.0, logp=2.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_dissolution_rate("X", solubility_mg_mL=1.0, mw=-300.0, logp=2.0)

    def test_negative_surface_area_raises(self):
        with pytest.raises(ValueError, match="surface_area"):
            predict_dissolution_rate(
                "X", solubility_mg_mL=1.0, mw=300.0, logp=2.0, surface_area_cm2=-5.0
            )

    def test_invalid_crystal_form_raises(self):
        with pytest.raises(ValueError, match="crystal_form"):
            predict_dissolution_rate(
                "X", solubility_mg_mL=1.0, mw=300.0, logp=2.0, crystal_form="invalid_form"
            )
