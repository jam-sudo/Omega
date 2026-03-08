"""Tests for Phase 557 - Hepatic Enzyme Induction Kinetics."""

import math

import pytest

from omega_pbpk.clinical.enzyme_induction_kinetics_p557 import (
    InductionKineticsResult,
    compare_inducers,
    simulate_induction_kinetics,
)


class TestSimulateInductionKinetics:
    """Tests for simulate_induction_kinetics."""

    def test_return_type(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0)
        assert isinstance(result, InductionKineticsResult)

    def test_drug_name(self):
        result = simulate_induction_kinetics("Rifampin", 5.0, 1.0, 1.0)
        assert result.drug_name == "Rifampin"

    def test_list_lengths_match(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0)
        n = len(result.times_days)
        assert len(result.enzyme_activity_relative) == n
        assert len(result.cl_relative) == n
        assert len(result.auc_ratio_vs_victim) == n

    def test_times_start_at_zero(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0)
        assert result.times_days[0] == 0.0

    def test_enzyme_starts_at_one(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0)
        assert result.enzyme_activity_relative[0] == 1.0

    def test_enzyme_increases_monotonically(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0, t_end_days=20.0)
        for i in range(1, len(result.enzyme_activity_relative)):
            assert result.enzyme_activity_relative[i] >= result.enzyme_activity_relative[i - 1]

    def test_enzyme_approaches_steady_state(self):
        # E_ss = 1 + emax * C / (ec50 + C) = 1 + 3*1/(1+1) = 2.5
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0, t_end_days=30.0, dt_days=0.05)
        e_ss = 2.5
        final_e = result.enzyme_activity_relative[-1]
        assert abs(final_e - e_ss) < 0.1

    def test_higher_emax_higher_max_fold(self):
        r1 = simulate_induction_kinetics("Drug1", 2.0, 1.0, 1.0, t_end_days=30.0)
        r2 = simulate_induction_kinetics("Drug2", 6.0, 1.0, 1.0, t_end_days=30.0)
        assert r2.max_induction_fold > r1.max_induction_fold

    def test_higher_kdeg_faster_induction(self):
        r_slow = simulate_induction_kinetics(
            "Slow", 4.0, 1.0, 1.0, kdeg_per_day=0.3, t_end_days=30.0
        )
        r_fast = simulate_induction_kinetics(
            "Fast", 4.0, 1.0, 1.0, kdeg_per_day=1.0, t_end_days=30.0
        )
        assert r_fast.t50_induction_days < r_slow.t50_induction_days

    def test_auc_ratio_is_inverse_of_cl(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0)
        for cl, auc in zip(result.cl_relative, result.auc_ratio_vs_victim):
            assert abs(auc - 1.0 / cl) < 1e-10

    def test_cl_relative_equals_enzyme_activity(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0)
        for e, cl in zip(result.enzyme_activity_relative, result.cl_relative):
            assert e == cl

    def test_induction_class_weak(self):
        # emax_fold=0.5, C=1, ec50=1 → E_ss = 1 + 0.5*0.5 = 1.25 < 2
        result = simulate_induction_kinetics("Weak", 0.5, 1.0, 1.0, t_end_days=30.0)
        assert result.induction_class == "weak"

    def test_induction_class_moderate(self):
        # E_ss = 1 + 4*1/(1+1) = 3.0
        result = simulate_induction_kinetics("Mod", 4.0, 1.0, 1.0, t_end_days=30.0)
        assert result.induction_class == "moderate"

    def test_induction_class_strong(self):
        # E_ss = 1 + 10*1/(1+1) = 6.0
        result = simulate_induction_kinetics("Strong", 10.0, 1.0, 1.0, t_end_days=30.0)
        assert result.induction_class == "strong"

    def test_days_to_ss_greater_than_t50(self):
        result = simulate_induction_kinetics("DrugA", 4.0, 1.0, 1.0, t_end_days=30.0)
        assert result.days_to_steady_state > result.t50_induction_days

    def test_t50_is_finite(self):
        result = simulate_induction_kinetics("DrugA", 4.0, 1.0, 1.0, t_end_days=30.0)
        assert math.isfinite(result.t50_induction_days)

    def test_days_to_ss_is_finite(self):
        result = simulate_induction_kinetics("DrugA", 4.0, 1.0, 1.0, t_end_days=30.0)
        assert math.isfinite(result.days_to_steady_state)

    def test_notes_contains_info(self):
        result = simulate_induction_kinetics("DrugA", 3.0, 1.0, 1.0)
        assert "Forward Euler" in result.notes

    def test_validation_emax_zero(self):
        with pytest.raises(ValueError, match="emax_fold"):
            simulate_induction_kinetics("Bad", 0.0, 1.0, 1.0)

    def test_validation_ec50_negative(self):
        with pytest.raises(ValueError, match="ec50_mg_L"):
            simulate_induction_kinetics("Bad", 3.0, -1.0, 1.0)

    def test_validation_plasma_conc_zero(self):
        with pytest.raises(ValueError, match="plasma_conc_mg_L"):
            simulate_induction_kinetics("Bad", 3.0, 1.0, 0.0)


class TestCompareInducers:
    """Tests for compare_inducers."""

    def test_sorted_descending(self):
        inducers = [
            {"drug_name": "Weak", "emax_fold": 1.0, "ec50_mg_L": 1.0},
            {"drug_name": "Strong", "emax_fold": 10.0, "ec50_mg_L": 1.0},
            {"drug_name": "Moderate", "emax_fold": 4.0, "ec50_mg_L": 1.0},
        ]
        results = compare_inducers(inducers, plasma_conc_mg_L=1.0)
        folds = [r.max_induction_fold for r in results]
        assert folds == sorted(folds, reverse=True)

    def test_returns_list_of_results(self):
        inducers = [
            {"drug_name": "A", "emax_fold": 2.0, "ec50_mg_L": 1.0},
        ]
        results = compare_inducers(inducers)
        assert len(results) == 1
        assert isinstance(results[0], InductionKineticsResult)
