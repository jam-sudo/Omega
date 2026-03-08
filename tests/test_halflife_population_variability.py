"""Tests for drug half-life population variability model (Phase 570)."""

import pytest

from omega_pbpk.clinical.halflife_population_variability import (
    HalflifeVariabilityResult,
    compare_variability_scenarios,
    simulate_halflife_variability,
)

PARAMS = dict(
    drug_name="TestDrug",
    typical_cl_L_per_h=5.0,
    typical_vd_L=50.0,
)


class TestSimulateHalflifeVariability:
    def test_return_type(self):
        result = simulate_halflife_variability(**PARAMS)
        assert isinstance(result, HalflifeVariabilityResult)

    def test_n_subjects_generated(self):
        result = simulate_halflife_variability(**PARAMS, n_subjects=200)
        assert len(result.thalf_values) == 200
        assert result.n_subjects == 200

    def test_mean_positive(self):
        result = simulate_halflife_variability(**PARAMS)
        assert result.mean_thalf_h > 0

    def test_sd_positive(self):
        result = simulate_halflife_variability(**PARAMS)
        assert result.sd_thalf_h > 0

    def test_cv_positive(self):
        result = simulate_halflife_variability(**PARAMS)
        assert result.cv_pct > 0

    def test_percentile_ordering(self):
        result = simulate_halflife_variability(**PARAMS)
        assert result.p5_thalf_h < result.p25_thalf_h
        assert result.p25_thalf_h < result.p50_thalf_h
        assert result.p50_thalf_h < result.p75_thalf_h
        assert result.p75_thalf_h < result.p95_thalf_h

    def test_higher_omega_higher_cv(self):
        r_low = simulate_halflife_variability(**PARAMS, omega_cl=0.1)
        r_high = simulate_halflife_variability(**PARAMS, omega_cl=0.8)
        assert r_high.cv_pct > r_low.cv_pct

    def test_variability_class_low(self):
        result = simulate_halflife_variability(**PARAMS, omega_cl=0.05, omega_vd=0.05)
        assert result.variability_class == "low"

    def test_variability_class_moderate(self):
        result = simulate_halflife_variability(**PARAMS, omega_cl=0.3, omega_vd=0.2)
        assert result.variability_class == "moderate"

    def test_variability_class_high(self):
        result = simulate_halflife_variability(**PARAMS, omega_cl=0.8, omega_vd=0.5)
        assert result.variability_class == "high"

    def test_short_long_counts(self):
        result = simulate_halflife_variability(**PARAMS, n_subjects=100)
        n_middle = result.n_subjects - result.n_short_halflife - result.n_long_halflife
        assert n_middle >= 0
        assert result.n_short_halflife >= 0
        assert result.n_long_halflife >= 0
        assert result.n_short_halflife + result.n_long_halflife + n_middle == result.n_subjects

    def test_seed_reproducibility(self):
        r1 = simulate_halflife_variability(**PARAMS, seed=123)
        r2 = simulate_halflife_variability(**PARAMS, seed=123)
        assert r1.thalf_values == r2.thalf_values
        assert r1.mean_thalf_h == r2.mean_thalf_h

    def test_different_seeds_differ(self):
        r1 = simulate_halflife_variability(**PARAMS, seed=1)
        r2 = simulate_halflife_variability(**PARAMS, seed=2)
        assert r1.thalf_values != r2.thalf_values

    def test_all_thalf_positive(self):
        result = simulate_halflife_variability(**PARAMS, n_subjects=500)
        assert all(v > 0 for v in result.thalf_values)

    def test_drug_name_preserved(self):
        result = simulate_halflife_variability(**PARAMS)
        assert result.drug_name == "TestDrug"

    def test_notes_contains_info(self):
        result = simulate_halflife_variability(**PARAMS, omega_cl=0.3)
        assert "omega_cl=0.3" in result.notes

    def test_typical_halflife_range(self):
        # t_half = 0.693 * 50 / 5 = 6.93 h typically
        result = simulate_halflife_variability(**PARAMS, n_subjects=1000)
        assert 4.0 < result.mean_thalf_h < 12.0


class TestValidation:
    def test_zero_n_subjects(self):
        with pytest.raises(ValueError, match="n_subjects"):
            simulate_halflife_variability(**PARAMS, n_subjects=0)

    def test_negative_cl(self):
        with pytest.raises(ValueError, match="typical_cl"):
            simulate_halflife_variability(drug_name="X", typical_cl_L_per_h=-1, typical_vd_L=50)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="typical_vd"):
            simulate_halflife_variability(drug_name="X", typical_cl_L_per_h=5, typical_vd_L=0)

    def test_negative_omega_cl(self):
        with pytest.raises(ValueError, match="omega_cl"):
            simulate_halflife_variability(**PARAMS, omega_cl=-0.1)


class TestCompareVariabilityScenarios:
    def test_sorted_by_cv_ascending(self):
        results = compare_variability_scenarios("TestDrug", 5.0, 50.0)
        cvs = [r.cv_pct for r in results]
        assert cvs == sorted(cvs)

    def test_returns_list(self):
        results = compare_variability_scenarios("TestDrug", 5.0, 50.0, [0.1, 0.5])
        assert isinstance(results, list)
        assert len(results) == 2
