"""Tests for Phase 282 — Excretion Balance Estimator."""

import pytest

from omega_pbpk.prediction.excretion_balance_estimator import (
    ExcretionBalanceResult,
    estimate_excretion_balance,
    screen_excretion_routes,
)

# Default parameters
DEFAULT_PARAMS = dict(
    drug_name="TestDrug",
    cl_renal_L_per_h=3.0,
    cl_hepatic_L_per_h=2.0,
    cl_pulmonary_L_per_h=0.5,
    cl_other_L_per_h=0.5,
    dose_mg=200.0,
    vd_L=40.0,
    fu_plasma=0.3,
)


def make_result(**overrides) -> ExcretionBalanceResult:
    params = {**DEFAULT_PARAMS, **overrides}
    return estimate_excretion_balance(**params)


class TestReturnType:
    def test_returns_excretion_balance_result(self):
        result = make_result()
        assert isinstance(result, ExcretionBalanceResult)

    def test_drug_name_preserved(self):
        result = make_result(drug_name="Metformin")
        assert result.drug_name == "Metformin"

    def test_dose_mg_preserved(self):
        result = make_result(dose_mg=500.0)
        assert result.dose_mg == 500.0

    def test_fu_plasma_preserved(self):
        result = make_result(fu_plasma=0.8)
        assert result.fu_plasma == pytest.approx(0.8)


class TestFractionsSumToOne:
    def test_fe_sum_to_one(self):
        result = make_result()
        total = result.fe_renal + result.fe_hepatic + result.fe_pulmonary + result.fe_other
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_fe_sum_renal_dominant(self):
        result = make_result(cl_renal_L_per_h=10.0, cl_hepatic_L_per_h=1.0)
        total = result.fe_renal + result.fe_hepatic + result.fe_pulmonary + result.fe_other
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_fe_all_zero_except_renal(self):
        result = make_result(
            cl_renal_L_per_h=5.0,
            cl_hepatic_L_per_h=0.0,
            cl_pulmonary_L_per_h=0.0,
            cl_other_L_per_h=0.0,
        )
        assert result.fe_renal == pytest.approx(1.0)
        assert result.fe_hepatic == pytest.approx(0.0)


class TestClearanceAndHalfLife:
    def test_cl_total_correct(self):
        result = make_result()
        expected = 3.0 + 2.0 + 0.5 + 0.5
        assert result.cl_total_L_per_h == pytest.approx(expected)

    def test_ke_correct(self):
        result = make_result()
        expected_ke = result.cl_total_L_per_h / DEFAULT_PARAMS["vd_L"]
        assert result.ke_per_h == pytest.approx(expected_ke)

    def test_t_half_positive(self):
        result = make_result()
        assert result.t_half_h > 0.0

    def test_t_half_formula(self):
        result = make_result()
        expected = 0.693 / result.ke_per_h
        assert result.t_half_h == pytest.approx(expected, rel=1e-6)


class TestFractionExcreted:
    def test_higher_cl_renal_higher_fe_renal(self):
        low = make_result(cl_renal_L_per_h=1.0)
        high = make_result(cl_renal_L_per_h=8.0)
        assert high.fe_renal > low.fe_renal

    def test_fe_renal_between_0_and_1(self):
        result = make_result()
        assert 0.0 <= result.fe_renal <= 1.0

    def test_fe_hepatic_between_0_and_1(self):
        result = make_result()
        assert 0.0 <= result.fe_hepatic <= 1.0

    def test_fe_pulmonary_between_0_and_1(self):
        result = make_result()
        assert 0.0 <= result.fe_pulmonary <= 1.0


class TestAmountsExcreted:
    def test_amount_renal_correct(self):
        result = make_result()
        expected = result.dose_mg * result.fe_renal
        assert result.amount_excreted_renal_mg == pytest.approx(expected)

    def test_amount_hepatic_correct(self):
        result = make_result()
        expected = result.dose_mg * result.fe_hepatic
        assert result.amount_excreted_hepatic_mg == pytest.approx(expected)

    def test_amount_pulmonary_correct(self):
        result = make_result()
        expected = result.dose_mg * result.fe_pulmonary
        assert result.amount_excreted_pulmonary_mg == pytest.approx(expected)


class TestGFREstimate:
    def test_cl_renal_gfr_estimate_positive(self):
        result = make_result()
        assert result.cl_renal_gfr_estimate_L_per_h > 0.0

    def test_cl_renal_gfr_estimate_formula(self):
        result = make_result(fu_plasma=0.5)
        expected = 0.5 * 7.5
        assert result.cl_renal_gfr_estimate_L_per_h == pytest.approx(expected)


class TestDominantRoute:
    def test_dominant_route_renal_when_highest(self):
        result = make_result(
            cl_renal_L_per_h=10.0,
            cl_hepatic_L_per_h=1.0,
            cl_pulmonary_L_per_h=0.1,
            cl_other_L_per_h=0.1,
        )
        assert result.dominant_route == "renal"

    def test_dominant_route_hepatic_when_highest(self):
        result = make_result(
            cl_renal_L_per_h=1.0,
            cl_hepatic_L_per_h=10.0,
            cl_pulmonary_L_per_h=0.1,
            cl_other_L_per_h=0.1,
        )
        assert result.dominant_route == "hepatic"

    def test_dominant_route_is_string(self):
        result = make_result()
        assert isinstance(result.dominant_route, str)


class TestExcretionClass:
    def test_renally_cleared_when_fe_renal_gt_half(self):
        result = make_result(
            cl_renal_L_per_h=10.0,
            cl_hepatic_L_per_h=0.5,
            cl_pulmonary_L_per_h=0.0,
            cl_other_L_per_h=0.0,
        )
        assert result.excretion_class == "renally_cleared"

    def test_hepatically_cleared_when_fe_hepatic_gt_half(self):
        result = make_result(
            cl_renal_L_per_h=0.5,
            cl_hepatic_L_per_h=10.0,
            cl_pulmonary_L_per_h=0.0,
            cl_other_L_per_h=0.0,
        )
        assert result.excretion_class == "hepatically_cleared"

    def test_mixed_when_no_route_dominates(self):
        result = make_result(
            cl_renal_L_per_h=3.0,
            cl_hepatic_L_per_h=3.0,
            cl_pulmonary_L_per_h=0.0,
            cl_other_L_per_h=0.0,
        )
        assert result.excretion_class == "mixed"

    def test_excretion_class_valid_values(self):
        result = make_result()
        assert result.excretion_class in {
            "renally_cleared",
            "hepatically_cleared",
            "mixed",
        }


class TestScreenExcretionRoutes:
    def test_returns_list(self):
        results = screen_excretion_routes("TestDrug", 100.0, 40.0, 0.3, [1.0, 5.0], [2.0, 4.0])
        assert isinstance(results, list)

    def test_sorted_descending_by_fe_renal(self):
        results = screen_excretion_routes("TestDrug", 100.0, 40.0, 0.3, [1.0, 3.0, 8.0], [2.0, 4.0])
        for i in range(len(results) - 1):
            assert results[i].fe_renal >= results[i + 1].fe_renal

    def test_length_matches_combinations(self):
        cl_r = [1.0, 5.0]
        cl_h = [2.0, 4.0, 6.0]
        results = screen_excretion_routes("Drug", 100.0, 40.0, 0.3, cl_r, cl_h)
        assert len(results) == len(cl_r) * len(cl_h)


class TestValidation:
    def test_all_cl_zero_raises(self):
        with pytest.raises(ValueError, match="least one"):
            estimate_excretion_balance(
                "Drug",
                cl_renal_L_per_h=0.0,
                cl_hepatic_L_per_h=0.0,
                cl_pulmonary_L_per_h=0.0,
                cl_other_L_per_h=0.0,
                dose_mg=100.0,
                vd_L=40.0,
                fu_plasma=0.3,
            )

    def test_negative_cl_renal_raises(self):
        with pytest.raises(ValueError, match="cl_renal"):
            make_result(cl_renal_L_per_h=-1.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=-50.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            make_result(vd_L=0.0)

    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            make_result(fu_plasma=0.0)

    def test_fu_plasma_above_1_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            make_result(fu_plasma=1.5)
