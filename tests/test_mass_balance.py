"""Tests for omega_pbpk.core.mass_balance module."""

import pytest

from omega_pbpk.core.mass_balance import (
    MassBalanceResult,
    check_mass_balance,
    compare_mass_balance,
    track_simulation_balance,
)

# ---------------------------------------------------------------------------
# MassBalanceResult dataclass structure
# ---------------------------------------------------------------------------


class TestMassBalanceResultDataclass:
    def test_has_required_fields(self):
        result = check_mass_balance("DrugA", dose_mg=100, absorbed_mg=95)
        assert hasattr(result, "drug_name")
        assert hasattr(result, "dose_mg")
        assert hasattr(result, "total_recovered_mg")
        assert hasattr(result, "mass_balance_pct")
        assert hasattr(result, "absorbed_mg")
        assert hasattr(result, "distributed_mg")
        assert hasattr(result, "metabolized_mg")
        assert hasattr(result, "excreted_mg")
        assert hasattr(result, "remaining_mg")
        assert hasattr(result, "is_balanced")
        assert hasattr(result, "tolerance_pct")
        assert hasattr(result, "error_message")

    def test_return_type_is_mass_balance_result(self):
        result = check_mass_balance("DrugB", dose_mg=50, absorbed_mg=50)
        assert isinstance(result, MassBalanceResult)

    def test_drug_name_stored_correctly(self):
        result = check_mass_balance("Warfarin", dose_mg=10, absorbed_mg=9)
        assert result.drug_name == "Warfarin"

    def test_dose_mg_stored_correctly(self):
        result = check_mass_balance("DrugC", dose_mg=200, absorbed_mg=180)
        assert result.dose_mg == 200


# ---------------------------------------------------------------------------
# check_mass_balance — balanced cases
# ---------------------------------------------------------------------------


class TestCheckMassBalancePerfectBalance:
    def test_perfect_balance_is_balanced_true(self):
        result = check_mass_balance(
            "DrugD",
            dose_mg=100,
            absorbed_mg=60,
            distributed_mg=20,
            metabolized_mg=30,
            excreted_mg=10,
        )
        assert result.is_balanced is True

    def test_perfect_balance_pct_near_100(self):
        result = check_mass_balance(
            "DrugE",
            dose_mg=100,
            absorbed_mg=60,
            distributed_mg=20,
            metabolized_mg=30,
            excreted_mg=10,
        )
        assert abs(result.mass_balance_pct - 100.0) < 1.0

    def test_error_message_none_when_balanced(self):
        result = check_mass_balance(
            "DrugF",
            dose_mg=100,
            absorbed_mg=100,
            tolerance_pct=1.0,
        )
        assert result.error_message is None or result.error_message == ""

    def test_within_tolerance_is_balanced(self):
        # 99.5 mg recovered out of 100 mg => 99.5% — within 1% tolerance
        result = check_mass_balance(
            "DrugG",
            dose_mg=100,
            absorbed_mg=50,
            metabolized_mg=49.5,
        )
        assert result.is_balanced is True

    def test_total_recovered_mg_matches_components(self):
        result = check_mass_balance(
            "DrugH",
            dose_mg=100,
            absorbed_mg=40,
            metabolized_mg=35,
            excreted_mg=20,
            remaining_mg=5,
        )
        expected_total = 40 + 35 + 20 + 5
        assert abs(result.total_recovered_mg - expected_total) < 1e-9


# ---------------------------------------------------------------------------
# check_mass_balance — unbalanced cases
# ---------------------------------------------------------------------------


class TestCheckMassBalanceUnbalanced:
    def test_under_recovery_is_not_balanced(self):
        # Only 50 mg recovered from 100 mg dose => 50%
        result = check_mass_balance(
            "DrugI",
            dose_mg=100,
            absorbed_mg=50,
            tolerance_pct=1.0,
        )
        assert result.is_balanced is False

    def test_over_recovery_is_not_balanced(self):
        # 120 mg recovered from 100 mg dose => 120%
        result = check_mass_balance(
            "DrugJ",
            dose_mg=100,
            absorbed_mg=120,
            tolerance_pct=1.0,
        )
        assert result.is_balanced is False

    def test_error_message_non_empty_when_unbalanced(self):
        result = check_mass_balance(
            "DrugK",
            dose_mg=100,
            absorbed_mg=50,
            tolerance_pct=1.0,
        )
        assert result.error_message is not None and len(result.error_message) > 0

    def test_mass_balance_pct_correct_under_recovery(self):
        result = check_mass_balance(
            "DrugL",
            dose_mg=200,
            absorbed_mg=100,
        )
        assert abs(result.mass_balance_pct - 50.0) < 1e-6

    def test_mass_balance_pct_correct_over_recovery(self):
        result = check_mass_balance(
            "DrugM",
            dose_mg=100,
            absorbed_mg=150,
        )
        assert abs(result.mass_balance_pct - 150.0) < 1e-6

    def test_custom_tolerance_affects_balanced_flag(self):
        # 95% recovery — unbalanced at 1% tolerance, balanced at 10% tolerance
        result_tight = check_mass_balance("DrugN", dose_mg=100, absorbed_mg=95, tolerance_pct=1.0)
        result_loose = check_mass_balance("DrugN", dose_mg=100, absorbed_mg=95, tolerance_pct=10.0)
        assert result_tight.is_balanced is False
        assert result_loose.is_balanced is True


# ---------------------------------------------------------------------------
# check_mass_balance — input validation
# ---------------------------------------------------------------------------


class TestCheckMassBalanceValidation:
    def test_zero_dose_raises_value_error(self):
        with pytest.raises(ValueError):
            check_mass_balance("DrugO", dose_mg=0, absorbed_mg=0)

    def test_negative_dose_raises_value_error(self):
        with pytest.raises(ValueError):
            check_mass_balance("DrugP", dose_mg=-10, absorbed_mg=5)

    def test_tolerance_stored_in_result(self):
        result = check_mass_balance("DrugQ", dose_mg=100, absorbed_mg=99, tolerance_pct=2.5)
        assert result.tolerance_pct == 2.5


# ---------------------------------------------------------------------------
# track_simulation_balance
# ---------------------------------------------------------------------------


class TestTrackSimulationBalance:
    def _iv_times_and_conc(self):
        """Simple IV bolus profile: C(t) = C0 * exp(-ke*t)."""
        import math

        dose_mg = 100.0
        vd_L = 10.0
        cl_L_per_h = 2.0
        ke = cl_L_per_h / vd_L  # 0.2 h^-1
        times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        c0 = dose_mg / vd_L
        conc = [c0 * math.exp(-ke * t) for t in times]
        return times, conc, dose_mg, vd_L, cl_L_per_h

    def test_returns_mass_balance_result_type(self):
        times, conc, dose_mg, vd_L, cl_L_per_h = self._iv_times_and_conc()
        result = track_simulation_balance(times, conc, dose_mg, vd_L, cl_L_per_h, route="iv")
        assert isinstance(result, MassBalanceResult)

    def test_mass_balance_pct_positive(self):
        times, conc, dose_mg, vd_L, cl_L_per_h = self._iv_times_and_conc()
        result = track_simulation_balance(times, conc, dose_mg, vd_L, cl_L_per_h, route="iv")
        assert result.mass_balance_pct > 0

    def test_total_recovered_mg_positive(self):
        times, conc, dose_mg, vd_L, cl_L_per_h = self._iv_times_and_conc()
        result = track_simulation_balance(times, conc, dose_mg, vd_L, cl_L_per_h, route="iv")
        assert result.total_recovered_mg > 0

    def test_oral_route_accepted(self):
        import math

        dose_mg = 100.0
        vd_L = 10.0
        cl_L_per_h = 1.0
        ka = 1.5
        f = 0.8
        ke = cl_L_per_h / vd_L
        times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        conc = [
            (f * dose_mg * ka) / (vd_L * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))
            for t in times
        ]
        result = track_simulation_balance(
            times, conc, dose_mg, vd_L, cl_L_per_h, route="oral", ka_per_h=ka, f=f
        )
        assert isinstance(result, MassBalanceResult)
        assert result.total_recovered_mg > 0


# ---------------------------------------------------------------------------
# compare_mass_balance
# ---------------------------------------------------------------------------


class TestCompareMassBalance:
    def _make_results(self):
        balanced1 = check_mass_balance("DrugR", dose_mg=100, absorbed_mg=100, tolerance_pct=1.0)
        balanced2 = check_mass_balance(
            "DrugS",
            dose_mg=100,
            absorbed_mg=40,
            metabolized_mg=35,
            excreted_mg=25,
            tolerance_pct=1.0,
        )
        unbalanced1 = check_mass_balance("DrugT", dose_mg=100, absorbed_mg=60, tolerance_pct=1.0)
        return balanced1, balanced2, unbalanced1

    def test_returns_dict(self):
        b1, b2, u1 = self._make_results()
        summary = compare_mass_balance([b1, b2, u1])
        assert isinstance(summary, dict)

    def test_n_balanced_count(self):
        b1, b2, u1 = self._make_results()
        summary = compare_mass_balance([b1, b2, u1])
        assert summary["n_balanced"] == 2

    def test_n_unbalanced_count(self):
        b1, b2, u1 = self._make_results()
        summary = compare_mass_balance([b1, b2, u1])
        assert summary["n_unbalanced"] == 1

    def test_unbalanced_drugs_list_contains_unbalanced_name(self):
        b1, b2, u1 = self._make_results()
        summary = compare_mass_balance([b1, b2, u1])
        assert "DrugT" in summary["unbalanced_drugs"]

    def test_mean_mass_balance_pct_present(self):
        b1, b2, u1 = self._make_results()
        summary = compare_mass_balance([b1, b2, u1])
        assert "mean_mass_balance_pct" in summary

    def test_min_max_pct_present(self):
        b1, b2, u1 = self._make_results()
        summary = compare_mass_balance([b1, b2, u1])
        assert "min_pct" in summary
        assert "max_pct" in summary

    def test_all_balanced_gives_zero_unbalanced(self):
        b1 = check_mass_balance("DrugU", dose_mg=100, absorbed_mg=100, tolerance_pct=1.0)
        b2 = check_mass_balance("DrugV", dose_mg=50, absorbed_mg=50, tolerance_pct=1.0)
        summary = compare_mass_balance([b1, b2])
        assert summary["n_unbalanced"] == 0
        assert summary["unbalanced_drugs"] == []

    def test_min_pct_less_than_or_equal_max_pct(self):
        b1, b2, u1 = self._make_results()
        summary = compare_mass_balance([b1, b2, u1])
        assert summary["min_pct"] <= summary["max_pct"]
