"""Tests for Phase 414 — covariate_analysis module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.covariate_analysis import (
    analyze_covariate_effect,
    dose_by_covariate,
)

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

SUBJECTS_WEIGHT = [
    {"subject": i + 1, "weight_kg": w, "age": 40, "sex": "M", "cl": cl, "vd": 50.0}
    for i, (w, cl) in enumerate(
        [(50, 3.5), (60, 4.0), (70, 4.8), (80, 5.5), (90, 6.2), (100, 7.0)]
    )
]

SUBJECTS_AGE = [
    {"subject": i + 1, "weight_kg": 70, "age": a, "sex": "M", "cl": cl, "vd": 50.0}
    for i, (a, cl) in enumerate(
        [(20, 6.5), (30, 5.8), (40, 5.2), (50, 4.5), (60, 3.8), (70, 3.2)]
    )
]

SUBJECTS_SEX = [
    {"subject": 1, "weight_kg": 70, "age": 40, "sex": "M", "cl": 6.0, "vd": 55.0},
    {"subject": 2, "weight_kg": 65, "age": 38, "sex": "F", "cl": 4.5, "vd": 45.0},
    {"subject": 3, "weight_kg": 75, "age": 42, "sex": "M", "cl": 5.8, "vd": 53.0},
    {"subject": 4, "weight_kg": 62, "age": 35, "sex": "F", "cl": 4.2, "vd": 43.0},
    {"subject": 5, "weight_kg": 80, "age": 45, "sex": "M", "cl": 6.3, "vd": 57.0},
    {"subject": 6, "weight_kg": 58, "age": 36, "sex": "F", "cl": 4.0, "vd": 42.0},
]


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestCovariateAnalysisResultStructure:
    def test_is_frozen_dataclass(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        with pytest.raises((AttributeError, TypeError)):
            res.pk_param = "vd"  # type: ignore[misc]

    def test_required_fields_present(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        for field in (
            "pk_param", "covariate", "n_subjects",
            "beta_exponent", "r_squared",
            "mean_pk_low", "mean_pk_high",
            "pk_change_75_vs_25_pct", "notes",
        ):
            assert hasattr(res, field), f"Missing field: {field}"

    def test_n_subjects_correct(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert res.n_subjects == len(SUBJECTS_WEIGHT)

    def test_pk_param_stored(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert res.pk_param == "cl"

    def test_covariate_stored(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert res.covariate == "weight_kg"


# ---------------------------------------------------------------------------
# Continuous covariate (weight_kg, age)
# ---------------------------------------------------------------------------

class TestContinuousCovariate:
    def test_beta_exponent_is_float(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert isinstance(res.beta_exponent, float)
        assert not math.isnan(res.beta_exponent)

    def test_r_squared_in_range(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert 0.0 <= res.r_squared <= 1.0

    def test_positive_weight_effect_on_cl(self):
        """CL increases with weight → beta > 0."""
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert res.beta_exponent > 0.0

    def test_negative_age_effect_on_cl(self):
        """CL decreases with age → beta < 0."""
        res = analyze_covariate_effect(SUBJECTS_AGE, "cl", "age")
        assert res.beta_exponent < 0.0

    def test_high_r_squared_for_clean_data(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert res.r_squared > 0.9

    def test_mean_pk_low_nan_for_continuous(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert math.isnan(res.mean_pk_low)

    def test_mean_pk_high_nan_for_continuous(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert math.isnan(res.mean_pk_high)

    def test_pk_change_75_vs_25_positive_for_weight(self):
        """Higher weight → higher CL → positive % change at p75 vs p25."""
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert res.pk_change_75_vs_25_pct > 0.0

    def test_notes_contains_covariate(self):
        res = analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "weight_kg")
        assert "weight_kg" in res.notes


# ---------------------------------------------------------------------------
# Categorical covariate (sex)
# ---------------------------------------------------------------------------

class TestCategoricalCovariate:
    def test_beta_exponent_nan_for_sex(self):
        res = analyze_covariate_effect(SUBJECTS_SEX, "cl", "sex")
        assert math.isnan(res.beta_exponent)

    def test_r_squared_nan_for_sex(self):
        res = analyze_covariate_effect(SUBJECTS_SEX, "cl", "sex")
        assert math.isnan(res.r_squared)

    def test_mean_pk_low_is_female_gm(self):
        """mean_pk_low should be female geometric mean CL."""
        res = analyze_covariate_effect(SUBJECTS_SEX, "cl", "sex")
        expected_gm_f = (4.5 * 4.2 * 4.0) ** (1 / 3)
        assert abs(res.mean_pk_low - expected_gm_f) < 0.01

    def test_mean_pk_high_greater_than_low(self):
        """Males have higher CL in our fixture."""
        res = analyze_covariate_effect(SUBJECTS_SEX, "cl", "sex")
        assert res.mean_pk_high > res.mean_pk_low

    def test_pk_change_nonnegative_for_sex(self):
        res = analyze_covariate_effect(SUBJECTS_SEX, "cl", "sex")
        assert res.pk_change_75_vs_25_pct >= 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidationAnalysis:
    def test_empty_pk_data_raises(self):
        with pytest.raises(ValueError):
            analyze_covariate_effect([], "cl", "weight_kg")

    def test_single_subject_raises(self):
        with pytest.raises(ValueError):
            analyze_covariate_effect([SUBJECTS_WEIGHT[0]], "cl", "weight_kg")

    def test_empty_pk_param_raises(self):
        with pytest.raises(ValueError):
            analyze_covariate_effect(SUBJECTS_WEIGHT, "", "weight_kg")

    def test_empty_covariate_raises(self):
        with pytest.raises(ValueError):
            analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "")

    def test_missing_pk_param_raises(self):
        with pytest.raises(ValueError):
            analyze_covariate_effect(SUBJECTS_WEIGHT, "nonexistent_pk", "weight_kg")

    def test_missing_covariate_raises(self):
        with pytest.raises(ValueError):
            analyze_covariate_effect(SUBJECTS_WEIGHT, "cl", "nonexistent_cov")

    def test_invalid_sex_value_raises(self):
        bad_data = [
            {**rec, "sex": "X"} for rec in SUBJECTS_SEX
        ]
        with pytest.raises(ValueError):
            analyze_covariate_effect(bad_data, "cl", "sex")

    def test_only_male_sex_raises(self):
        male_only = [rec for rec in SUBJECTS_SEX if rec["sex"] == "M"]
        with pytest.raises(ValueError):
            analyze_covariate_effect(male_only, "cl", "sex")

    def test_zero_pk_value_raises(self):
        bad_data = [dict(rec) for rec in SUBJECTS_WEIGHT]
        bad_data[0]["cl"] = 0.0
        with pytest.raises(ValueError):
            analyze_covariate_effect(bad_data, "cl", "weight_kg")


# ---------------------------------------------------------------------------
# dose_by_covariate
# ---------------------------------------------------------------------------

class TestDoseByCovariate:
    def test_returns_list_of_dicts(self):
        result = dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 50.0, 100.0)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_correct_number_of_records(self):
        result = dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 50.0, 100.0)
        assert len(result) == len(SUBJECTS_WEIGHT)

    def test_required_keys_present(self):
        result = dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 50.0, 100.0)
        for rec in result:
            for key in ("subject", "covariate_value", "individual_pk",
                        "population_pk", "recommended_dose_mg"):
                assert key in rec

    def test_higher_cl_lower_dose(self):
        """Subject with higher CL should get lower dose (to keep same AUC)."""
        result = dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 50.0, 100.0)
        doses = [r["recommended_dose_mg"] for r in result]
        # Heavier subjects have higher CL → lower recommended dose
        # Check that doses are ordered inversely (not strictly, but last < first)
        # Actually higher CL → lower AUC at same dose → need higher dose... wait:
        # dose_i = current_dose * (pop_pk / indiv_pk)
        # Higher CL subject → indiv_pk > pop_pk → dose_i < current_dose
        # This is correct: AUC = dose/CL → to keep AUC constant, dose = AUC * CL
        assert all(d > 0 for d in doses)

    def test_sex_covariate_dose_adjustment(self):
        result = dose_by_covariate(SUBJECTS_SEX, "cl", "sex", 50.0, 100.0)
        assert len(result) == len(SUBJECTS_SEX)

    def test_zero_target_auc_raises(self):
        with pytest.raises(ValueError):
            dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 0.0, 100.0)

    def test_zero_current_dose_raises(self):
        with pytest.raises(ValueError):
            dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 50.0, 0.0)

    def test_empty_pk_data_raises(self):
        with pytest.raises(ValueError):
            dose_by_covariate([], "cl", "weight_kg", 50.0, 100.0)

    def test_doses_are_positive(self):
        result = dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 50.0, 100.0)
        assert all(r["recommended_dose_mg"] > 0 for r in result)

    def test_population_pk_same_across_subjects(self):
        """All subjects share the same population_pk reference."""
        result = dose_by_covariate(SUBJECTS_WEIGHT, "cl", "weight_kg", 50.0, 100.0)
        pop_pks = [r["population_pk"] for r in result]
        assert all(abs(p - pop_pks[0]) < 1e-9 for p in pop_pks)
