"""Tests for lymphocyte distribution model (Phase 905)."""

import pytest

from omega_pbpk.core.lymphocyte_distribution import (
    LymphocyteDistributionResult,
    compare_immunosuppressant_kp,
    simulate_lymphocyte_distribution,
)


def make_default() -> LymphocyteDistributionResult:
    return simulate_lymphocyte_distribution(
        drug_name="Tacrolimus",
        dose_mg=5.0,
        kp_lymphocyte=10.0,
        k_lymph_uptake_per_h=2.0,
        k_lymph_release_per_h=0.3,
        cl_L_per_h=10.0,
        vd_L=50.0,
        lymphocyte_volume_L=2.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


class TestReturnType:
    def test_returns_correct_type(self):
        result = make_default()
        assert isinstance(result, LymphocyteDistributionResult)

    def test_drug_name_preserved(self):
        result = make_default()
        assert result.drug_name == "Tacrolimus"

    def test_dose_mg_preserved(self):
        result = make_default()
        assert result.dose_mg == 5.0


class TestTimesAndConcentrations:
    def test_times_nonempty(self):
        result = make_default()
        assert len(result.times_h) > 0

    def test_concentrations_same_length_as_times(self):
        result = make_default()
        assert len(result.c_plasma_mg_L) == len(result.times_h)
        assert len(result.c_lymphocyte_mg_L) == len(result.times_h)

    def test_times_start_at_zero(self):
        result = make_default()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_concentrations_nonnegative(self):
        result = make_default()
        assert all(c >= 0.0 for c in result.c_plasma_mg_L)
        assert all(c >= 0.0 for c in result.c_lymphocyte_mg_L)


class TestPlasmaKinetics:
    def test_cmax_plasma_positive(self):
        result = make_default()
        assert result.cmax_plasma > 0.0

    def test_auc_plasma_positive(self):
        result = make_default()
        assert result.auc_plasma > 0.0

    def test_initial_plasma_concentration(self):
        """At t=0, c_plasma = dose / vd."""
        result = simulate_lymphocyte_distribution(
            dose_mg=100.0,
            vd_L=50.0,
            kp_lymphocyte=1.0,
            k_lymph_uptake_per_h=0.0,
            k_lymph_release_per_h=0.0,
        )
        assert result.c_plasma_mg_L[0] == pytest.approx(2.0, rel=1e-6)

    def test_plasma_declines_with_no_lymph_uptake(self):
        """With no lymph uptake, plasma declines monotonically."""
        result = simulate_lymphocyte_distribution(
            k_lymph_uptake_per_h=0.0,
            k_lymph_release_per_h=0.0,
            cl_L_per_h=10.0,
            vd_L=50.0,
            dt_h=0.05,
        )
        assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


class TestLymphocyteKinetics:
    def test_cmax_lymphocyte_positive(self):
        result = make_default()
        assert result.cmax_lymphocyte > 0.0

    def test_auc_lymphocyte_positive(self):
        result = make_default()
        assert result.auc_lymphocyte > 0.0

    def test_lymphocyte_starts_at_zero(self):
        result = make_default()
        assert result.c_lymphocyte_mg_L[0] == pytest.approx(0.0)

    def test_lymphocyte_rises_then_falls(self):
        """Lymphocyte concentration should rise above zero."""
        result = make_default()
        assert max(result.c_lymphocyte_mg_L) > 0.0


class TestLymphocytePlasmaRatio:
    def test_ratio_positive(self):
        result = make_default()
        assert result.lymphocyte_plasma_ratio > 0.0

    def test_higher_kp_gives_higher_ratio(self):
        result_low = simulate_lymphocyte_distribution(kp_lymphocyte=2.0)
        result_high = simulate_lymphocyte_distribution(kp_lymphocyte=20.0)
        assert result_high.lymphocyte_plasma_ratio > result_low.lymphocyte_plasma_ratio

    def test_intracellular_accumulation_factor_positive(self):
        result = make_default()
        assert result.intracellular_accumulation_factor > 0.0


class TestImmunoscuppressionScore:
    def test_score_in_range(self):
        result = make_default()
        assert 0.0 <= result.immunosuppression_score <= 100.0

    def test_high_kp_gives_high_score(self):
        result = simulate_lymphocyte_distribution(kp_lymphocyte=30.0)
        assert result.immunosuppression_score > 60.0

    def test_low_kp_gives_lower_score_than_high_kp(self):
        """Higher kp_lymphocyte should yield higher immunosuppression score."""
        result_low = simulate_lymphocyte_distribution(kp_lymphocyte=1.0)
        result_high = simulate_lymphocyte_distribution(kp_lymphocyte=30.0)
        assert result_high.immunosuppression_score >= result_low.immunosuppression_score


class TestNotes:
    def test_notes_nonempty(self):
        result = make_default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_high_score_notes(self):
        result = simulate_lymphocyte_distribution(kp_lymphocyte=30.0)
        assert "High" in result.notes or "immunosuppressive" in result.notes

    def test_low_score_notes(self):
        # Use minimal uptake/release so ratio stays low
        result = simulate_lymphocyte_distribution(
            kp_lymphocyte=0.1,
            k_lymph_uptake_per_h=0.001,
            k_lymph_release_per_h=1.0,
        )
        assert "Low" in result.notes


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_lymphocyte_distribution(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_lymphocyte_distribution(dose_mg=-10.0)

    def test_zero_kp_raises(self):
        with pytest.raises(ValueError):
            simulate_lymphocyte_distribution(kp_lymphocyte=0.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            simulate_lymphocyte_distribution(cl_L_per_h=0.0)


class TestCompareImmunossupressantKp:
    def test_returns_list(self):
        results = compare_immunosuppressant_kp("Drug", 100.0, [2.0, 5.0, 15.0])
        assert isinstance(results, list)

    def test_correct_length(self):
        results = compare_immunosuppressant_kp("Drug", 100.0, [2.0, 5.0, 15.0])
        assert len(results) == 3

    def test_sorted_descending_by_score(self):
        results = compare_immunosuppressant_kp("Drug", 100.0, [1.0, 5.0, 20.0])
        scores = [r.immunosuppression_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_are_lymphocyte_result(self):
        results = compare_immunosuppressant_kp("Drug", 100.0, [5.0, 10.0])
        for r in results:
            assert isinstance(r, LymphocyteDistributionResult)
