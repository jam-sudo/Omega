"""Tests for Phase 667 - Tissue Binding Competition."""

import pytest

from omega_pbpk.core.tissue_binding_competition_p667 import (
    TissueBindingCompetitionResult,
    simulate_binding_competition,
)


@pytest.fixture
def default_result():
    """Default competition result with equal drugs."""
    return simulate_binding_competition(
        drug_name_a="DrugA",
        drug_name_b="DrugB",
        tissue="liver",
        bmax_mg_L=10.0,
        kd_a_mg_L=1.0,
        kd_b_mg_L=1.0,
        initial_free_a_mg_L=5.0,
        initial_free_b_mg_L=5.0,
    )


@pytest.fixture
def a_wins_result():
    """Drug A has lower Kd (higher affinity)."""
    return simulate_binding_competition(
        drug_name_a="DrugA",
        drug_name_b="DrugB",
        tissue="liver",
        bmax_mg_L=10.0,
        kd_a_mg_L=0.1,
        kd_b_mg_L=10.0,
        initial_free_a_mg_L=5.0,
        initial_free_b_mg_L=5.0,
    )


class TestReturnType:
    def test_returns_correct_type(self, default_result):
        assert isinstance(default_result, TissueBindingCompetitionResult)

    def test_has_all_fields(self, default_result):
        assert hasattr(default_result, "drug_name_a")
        assert hasattr(default_result, "drug_name_b")
        assert hasattr(default_result, "tissue")
        assert hasattr(default_result, "times_h")
        assert hasattr(default_result, "c_free_a_mg_L")
        assert hasattr(default_result, "c_bound_a_mg_L")
        assert hasattr(default_result, "c_free_b_mg_L")
        assert hasattr(default_result, "c_bound_b_mg_L")
        assert hasattr(default_result, "bound_fraction_a")
        assert hasattr(default_result, "bound_fraction_b")
        assert hasattr(default_result, "competition_index")
        assert hasattr(default_result, "displacement_pct_a")
        assert hasattr(default_result, "displacement_pct_b")
        assert hasattr(default_result, "winner_drug")
        assert hasattr(default_result, "notes")

    def test_drug_names_stored(self, default_result):
        assert default_result.drug_name_a == "DrugA"
        assert default_result.drug_name_b == "DrugB"
        assert default_result.tissue == "liver"


class TestListLengths:
    def test_all_lists_same_length(self, default_result):
        n = len(default_result.times_h)
        assert len(default_result.c_free_a_mg_L) == n
        assert len(default_result.c_bound_a_mg_L) == n
        assert len(default_result.c_free_b_mg_L) == n
        assert len(default_result.c_bound_b_mg_L) == n
        assert len(default_result.bound_fraction_a) == n
        assert len(default_result.bound_fraction_b) == n

    def test_expected_length(self):
        result = simulate_binding_competition(
            drug_name_a="A",
            drug_name_b="B",
            tissue="t",
            bmax_mg_L=10.0,
            kd_a_mg_L=1.0,
            kd_b_mg_L=1.0,
            initial_free_a_mg_L=5.0,
            initial_free_b_mg_L=5.0,
            t_end_h=10.0,
            dt_h=0.1,
        )
        expected = int(10.0 / 0.1) + 1
        assert len(result.times_h) == expected


class TestNonNegative:
    def test_free_concentrations_non_negative(self, default_result):
        assert all(v >= 0 for v in default_result.c_free_a_mg_L)
        assert all(v >= 0 for v in default_result.c_free_b_mg_L)

    def test_bound_concentrations_non_negative(self, default_result):
        assert all(v >= 0 for v in default_result.c_bound_a_mg_L)
        assert all(v >= 0 for v in default_result.c_bound_b_mg_L)

    def test_bound_fractions_non_negative(self, default_result):
        assert all(0 <= v <= 1 for v in default_result.bound_fraction_a)
        assert all(0 <= v <= 1 for v in default_result.bound_fraction_b)


class TestAffinityWins:
    def test_lower_kd_wins(self, a_wins_result):
        """Drug with lower Kd (higher affinity) should win more binding sites."""
        assert a_wins_result.c_bound_a_mg_L[-1] > a_wins_result.c_bound_b_mg_L[-1]
        assert a_wins_result.winner_drug == "A"

    def test_b_wins_when_lower_kd(self):
        result = simulate_binding_competition(
            drug_name_a="DrugA",
            drug_name_b="DrugB",
            tissue="liver",
            bmax_mg_L=10.0,
            kd_a_mg_L=10.0,
            kd_b_mg_L=0.1,
            initial_free_a_mg_L=5.0,
            initial_free_b_mg_L=5.0,
        )
        assert result.winner_drug == "B"
        assert result.c_bound_b_mg_L[-1] > result.c_bound_a_mg_L[-1]


class TestHigherConcentration:
    def test_higher_initial_more_bound(self):
        result_low = simulate_binding_competition(
            drug_name_a="A",
            drug_name_b="B",
            tissue="t",
            bmax_mg_L=10.0,
            kd_a_mg_L=1.0,
            kd_b_mg_L=1.0,
            initial_free_a_mg_L=1.0,
            initial_free_b_mg_L=5.0,
        )
        result_high = simulate_binding_competition(
            drug_name_a="A",
            drug_name_b="B",
            tissue="t",
            bmax_mg_L=10.0,
            kd_a_mg_L=1.0,
            kd_b_mg_L=1.0,
            initial_free_a_mg_L=10.0,
            initial_free_b_mg_L=5.0,
        )
        assert result_high.c_bound_a_mg_L[-1] > result_low.c_bound_a_mg_L[-1]


class TestCompetitionIndex:
    def test_range(self, default_result):
        assert 0 <= default_result.competition_index <= 2

    def test_equal_drugs_near_one(self, default_result):
        assert abs(default_result.competition_index - 1.0) < 0.1

    def test_a_wins_above_one(self, a_wins_result):
        assert a_wins_result.competition_index > 1.0


class TestDisplacement:
    def test_displacement_in_range(self, default_result):
        assert 0 <= default_result.displacement_pct_a <= 100
        assert 0 <= default_result.displacement_pct_b <= 100

    def test_displacement_a_with_competitor(self, a_wins_result):
        # B is displaced by high-affinity A
        assert a_wins_result.displacement_pct_b > 0

    def test_no_displacement_when_alone(self):
        result = simulate_binding_competition(
            drug_name_a="A",
            drug_name_b="B",
            tissue="t",
            bmax_mg_L=10.0,
            kd_a_mg_L=1.0,
            kd_b_mg_L=1.0,
            initial_free_a_mg_L=5.0,
            initial_free_b_mg_L=0.0,
        )
        # No drug B present
        assert result.displacement_pct_b == 0.0


class TestWinnerDrug:
    def test_winner_a(self, a_wins_result):
        assert a_wins_result.winner_drug == "A"

    def test_winner_b(self):
        result = simulate_binding_competition(
            drug_name_a="A",
            drug_name_b="B",
            tissue="t",
            bmax_mg_L=10.0,
            kd_a_mg_L=10.0,
            kd_b_mg_L=0.1,
            initial_free_a_mg_L=5.0,
            initial_free_b_mg_L=5.0,
        )
        assert result.winner_drug == "B"

    def test_tie_with_equal_params(self, default_result):
        assert default_result.winner_drug == "tie"


class TestNoCompetition:
    def test_only_a_present(self):
        result = simulate_binding_competition(
            drug_name_a="A",
            drug_name_b="B",
            tissue="t",
            bmax_mg_L=10.0,
            kd_a_mg_L=1.0,
            kd_b_mg_L=1.0,
            initial_free_a_mg_L=5.0,
            initial_free_b_mg_L=0.0,
        )
        assert result.winner_drug == "A"
        assert all(v == 0 for v in result.c_bound_b_mg_L)
        assert result.displacement_pct_a == 0.0

    def test_only_b_present(self):
        result = simulate_binding_competition(
            drug_name_a="A",
            drug_name_b="B",
            tissue="t",
            bmax_mg_L=10.0,
            kd_a_mg_L=1.0,
            kd_b_mg_L=1.0,
            initial_free_a_mg_L=0.0,
            initial_free_b_mg_L=5.0,
        )
        assert result.winner_drug == "B"
        assert all(v == 0 for v in result.c_bound_a_mg_L)


class TestValidation:
    def test_bmax_zero(self):
        with pytest.raises(ValueError):
            simulate_binding_competition("A", "B", "t", 0.0, 1.0, 1.0, 5.0, 5.0)

    def test_negative_kd_a(self):
        with pytest.raises(ValueError):
            simulate_binding_competition("A", "B", "t", 10.0, -1.0, 1.0, 5.0, 5.0)

    def test_negative_kd_b(self):
        with pytest.raises(ValueError):
            simulate_binding_competition("A", "B", "t", 10.0, 1.0, -1.0, 5.0, 5.0)

    def test_negative_initial_a(self):
        with pytest.raises(ValueError):
            simulate_binding_competition("A", "B", "t", 10.0, 1.0, 1.0, -1.0, 5.0)

    def test_both_zero_initial(self):
        with pytest.raises(ValueError):
            simulate_binding_competition("A", "B", "t", 10.0, 1.0, 1.0, 0.0, 0.0)


class TestDoseProportionality:
    def test_proportional_within_drug_a(self):
        r1 = simulate_binding_competition("A", "B", "t", 100.0, 1.0, 1.0, 2.0, 5.0)
        r2 = simulate_binding_competition("A", "B", "t", 100.0, 1.0, 1.0, 4.0, 5.0)
        # With large Bmax (not saturated), doubling dose ~doubles binding
        ratio = r2.c_bound_a_mg_L[-1] / r1.c_bound_a_mg_L[-1]
        assert 1.5 < ratio < 2.5


class TestNotes:
    def test_notes_contains_drug_names(self, default_result):
        assert "DrugA" in default_result.notes
        assert "DrugB" in default_result.notes
        assert "liver" in default_result.notes
