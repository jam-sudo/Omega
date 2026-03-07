"""Tests for population PK simulation — Phase 835."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.population_pk_simulation import (
    PopulationPKSimResult,
    compare_populations,
    simulate_population_pk,
)

# ---------------------------------------------------------------------------
# Default helper
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> PopulationPKSimResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        n_subjects=50,
        typical_cl_L_per_h=5.0,
        typical_vd_L=50.0,
        omega_cl_cv=0.3,
        omega_vd_cv=0.2,
        ka_per_h=1.0,
        tau_h=12.0,
        n_doses=3,
        target_cmax_mg_L=1.5,
        seed=42,
    )
    defaults.update(kwargs)
    return simulate_population_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_population_pk_sim_result(self):
        result = _default_result()
        assert isinstance(result, PopulationPKSimResult)

    def test_result_is_frozen(self):
        result = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Warfarin")
        assert result.drug_name == "Warfarin"

    def test_n_subjects_preserved(self):
        result = _default_result(n_subjects=30)
        assert result.n_subjects == 30

    def test_dose_preserved(self):
        result = _default_result(dose_mg=200.0)
        assert result.dose_mg == 200.0

    def test_tau_preserved(self):
        result = _default_result(tau_h=24.0)
        assert result.tau_h == 24.0

    def test_n_doses_simulated_preserved(self):
        result = _default_result(n_doses=5)
        assert result.n_doses_simulated == 5

    def test_target_cmax_preserved(self):
        result = _default_result(target_cmax_mg_L=3.0)
        assert result.target_cmax_mg_L == 3.0

    def test_notes_is_string(self):
        result = _default_result()
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Positive-value tests
# ---------------------------------------------------------------------------


class TestPositiveValues:
    def test_mean_cmax_positive(self):
        result = _default_result()
        assert result.mean_cmax_mg_L > 0.0

    def test_mean_auc_positive(self):
        result = _default_result()
        assert result.mean_auc_mg_h_per_L > 0.0

    def test_mean_t_half_positive(self):
        result = _default_result()
        assert result.mean_t_half_h > 0.0

    def test_cv_cmax_positive(self):
        """With BSV, Cmax must vary across subjects."""
        result = _default_result()
        assert result.cv_cmax_pct > 0.0

    def test_cv_auc_positive(self):
        result = _default_result()
        assert result.cv_auc_pct > 0.0


# ---------------------------------------------------------------------------
# Percentile ordering tests
# ---------------------------------------------------------------------------


class TestPercentileOrdering:
    def test_p5_less_than_mean_cmax(self):
        result = _default_result(n_subjects=100)
        assert result.p5_cmax_mg_L <= result.mean_cmax_mg_L

    def test_mean_less_than_p95_cmax(self):
        result = _default_result(n_subjects=100)
        assert result.mean_cmax_mg_L <= result.p95_cmax_mg_L

    def test_p5_less_than_p95_cmax(self):
        result = _default_result(n_subjects=100)
        assert result.p5_cmax_mg_L < result.p95_cmax_mg_L

    def test_p5_less_than_mean_auc(self):
        result = _default_result(n_subjects=100)
        assert result.p5_auc_mg_h_per_L <= result.mean_auc_mg_h_per_L

    def test_mean_less_than_p95_auc(self):
        result = _default_result(n_subjects=100)
        assert result.mean_auc_mg_h_per_L <= result.p95_auc_mg_h_per_L


# ---------------------------------------------------------------------------
# Dose proportionality test
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_higher_dose_higher_mean_cmax(self):
        r_low = _default_result(dose_mg=100.0, seed=99)
        r_high = _default_result(dose_mg=400.0, seed=99)
        assert r_high.mean_cmax_mg_L > r_low.mean_cmax_mg_L

    def test_higher_dose_higher_mean_auc(self):
        r_low = _default_result(dose_mg=100.0, seed=99)
        r_high = _default_result(dose_mg=400.0, seed=99)
        assert r_high.mean_auc_mg_h_per_L > r_low.mean_auc_mg_h_per_L


# ---------------------------------------------------------------------------
# BSV effect tests
# ---------------------------------------------------------------------------


class TestBSVEffect:
    def test_higher_omega_cl_higher_cv_cmax(self):
        r_low = _default_result(omega_cl_cv=0.1, n_subjects=200, seed=7)
        r_high = _default_result(omega_cl_cv=0.5, n_subjects=200, seed=7)
        assert r_high.cv_cmax_pct > r_low.cv_cmax_pct

    def test_zero_omega_gives_low_cv(self):
        """Near-zero BSV should give near-zero CV."""
        result = _default_result(omega_cl_cv=0.01, omega_vd_cv=0.01, n_subjects=50)
        assert result.cv_cmax_pct < 5.0  # should be very small


# ---------------------------------------------------------------------------
# Fraction above target
# ---------------------------------------------------------------------------


class TestFractionAboveTarget:
    def test_fraction_in_valid_range(self):
        result = _default_result()
        assert 0.0 <= result.fraction_above_target_pct <= 100.0

    def test_high_target_gives_low_fraction(self):
        result = _default_result(target_cmax_mg_L=1000.0)
        assert result.fraction_above_target_pct == 0.0

    def test_zero_target_gives_hundred_pct(self):
        result = _default_result(target_cmax_mg_L=0.0)
        assert result.fraction_above_target_pct == 100.0


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self):
        r1 = _default_result(seed=123)
        r2 = _default_result(seed=123)
        assert r1.mean_cmax_mg_L == r2.mean_cmax_mg_L
        assert r1.cv_cmax_pct == r2.cv_cmax_pct

    def test_different_seed_different_result(self):
        r1 = _default_result(seed=1)
        r2 = _default_result(seed=2)
        assert r1.mean_cmax_mg_L != r2.mean_cmax_mg_L


# ---------------------------------------------------------------------------
# compare_populations tests
# ---------------------------------------------------------------------------


class TestComparePopulations:
    def test_returns_list(self):
        pops = [
            {"n_subjects": 20, "dose_mg": 100.0, "seed": 1},
            {"n_subjects": 20, "dose_mg": 200.0, "seed": 1},
        ]
        results = compare_populations("DrugX", dose_mg=100.0, populations=pops)
        assert isinstance(results, list)

    def test_returns_correct_count(self):
        pops = [
            {"n_subjects": 20, "seed": 1},
            {"n_subjects": 20, "seed": 2},
            {"n_subjects": 20, "seed": 3},
        ]
        results = compare_populations("DrugX", dose_mg=100.0, populations=pops)
        assert len(results) == 3

    def test_sorted_by_mean_cmax_descending(self):
        pops = [
            {"n_subjects": 30, "typical_cl_L_per_h": 10.0, "seed": 5},  # low Cmax
            {"n_subjects": 30, "typical_cl_L_per_h": 1.0, "seed": 5},  # high Cmax
        ]
        results = compare_populations("DrugX", dose_mg=100.0, populations=pops)
        # First result should have highest mean_cmax
        assert results[0].mean_cmax_mg_L >= results[1].mean_cmax_mg_L

    def test_all_results_are_population_pk_sim_result(self):
        pops = [{"n_subjects": 20, "seed": i} for i in range(4)]
        results = compare_populations("DrugX", dose_mg=100.0, populations=pops)
        for r in results:
            assert isinstance(r, PopulationPKSimResult)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_n_subjects_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_subjects"):
            simulate_population_pk("Drug", dose_mg=100.0, n_subjects=1)

    def test_n_doses_less_than_1_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            simulate_population_pk("Drug", dose_mg=100.0, n_doses=0)

    def test_typical_cl_zero_raises(self):
        with pytest.raises(ValueError, match="typical_cl"):
            simulate_population_pk("Drug", dose_mg=100.0, typical_cl_L_per_h=0.0)

    def test_typical_cl_negative_raises(self):
        with pytest.raises(ValueError, match="typical_cl"):
            simulate_population_pk("Drug", dose_mg=100.0, typical_cl_L_per_h=-1.0)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_population_pk("Drug", dose_mg=0.0)
