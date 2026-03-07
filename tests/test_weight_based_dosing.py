"""
Tests for Phase 930: weight-based dosing strategy comparison.
"""

import pytest
from omega_pbpk.clinical.weight_based_dosing import (
    WeightBasedDosingResult,
    compare_dosing_strategies,
    recommend_dosing_strategy,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compare(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        cl_ref_L_per_h=10.0,
        vd_ref_L=50.0,
        flat_dose_mg=100.0,
        dose_per_kg_mg=1.5,
        dose_per_m2_mg=50.0,
        n_patients=100,
        weight_cv=0.20,
        seed=42,
    )
    defaults.update(kwargs)
    return compare_dosing_strategies(**defaults)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_3_results(self):
        results = _compare()
        assert len(results) == 3

    def test_all_results_correct_type(self):
        results = _compare()
        for r in results:
            assert isinstance(r, WeightBasedDosingResult)

    def test_all_strategies_present(self):
        results = _compare()
        strategies = {r.strategy for r in results}
        assert strategies == {"flat", "mg_per_kg", "mg_per_m2"}

    def test_sorted_by_cv_auc_ascending(self):
        results = _compare()
        cvs = [r.cv_auc_pct for r in results]
        assert cvs == sorted(cvs)


# ---------------------------------------------------------------------------
# Statistical expectations
# ---------------------------------------------------------------------------

class TestStatisticalExpectations:
    def test_flat_has_highest_cv_auc(self):
        results = _compare()
        flat_cv = next(r.cv_auc_pct for r in results if r.strategy == "flat")
        for r in results:
            if r.strategy != "flat":
                assert flat_cv >= r.cv_auc_pct

    def test_personalized_lower_cv_than_flat(self):
        results = _compare()
        flat_cv = next(r.cv_auc_pct for r in results if r.strategy == "flat")
        for r in results:
            if r.strategy in ("mg_per_kg", "mg_per_m2"):
                assert r.cv_auc_pct < flat_cv

    def test_mean_auc_positive(self):
        results = _compare()
        for r in results:
            assert r.mean_auc > 0

    def test_cv_auc_nonnegative(self):
        results = _compare()
        for r in results:
            assert r.cv_auc_pct >= 0

    def test_p5_less_than_mean_less_than_p95(self):
        results = _compare()
        for r in results:
            assert r.p5_auc < r.mean_auc < r.p95_auc

    def test_mean_cmax_positive(self):
        results = _compare()
        for r in results:
            assert r.mean_cmax > 0

    def test_cv_cmax_nonnegative(self):
        results = _compare()
        for r in results:
            assert r.cv_cmax_pct >= 0


# ---------------------------------------------------------------------------
# Recommended flag
# ---------------------------------------------------------------------------

class TestRecommendedFlag:
    def test_exactly_one_recommended(self):
        results = _compare()
        recommended = [r for r in results if r.recommended]
        assert len(recommended) == 1

    def test_recommended_has_lowest_cv_auc(self):
        results = _compare()
        recommended = next(r for r in results if r.recommended)
        min_cv = min(r.cv_auc_pct for r in results)
        assert abs(recommended.cv_auc_pct - min_cv) < 1e-9


# ---------------------------------------------------------------------------
# recommend_dosing_strategy
# ---------------------------------------------------------------------------

class TestRecommendDosingStrategy:
    def test_returns_single_result(self):
        result = recommend_dosing_strategy("DrugX")
        assert isinstance(result, WeightBasedDosingResult)

    def test_returns_lowest_cv_auc_strategy(self):
        results = _compare()
        recommended = recommend_dosing_strategy("TestDrug")
        min_cv = min(r.cv_auc_pct for r in results)
        assert abs(recommended.cv_auc_pct - min_cv) < 1e-9

    def test_recommended_flag_true(self):
        result = recommend_dosing_strategy("DrugX")
        assert result.recommended is True


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------

class TestMetadataFields:
    def test_n_patients_preserved(self):
        results = _compare(n_patients=50)
        for r in results:
            assert r.n_patients == 50

    def test_drug_name_preserved(self):
        results = _compare(drug_name="Warfarin")
        for r in results:
            assert r.drug_name == "Warfarin"

    def test_notes_non_empty(self):
        results = _compare()
        for r in results:
            assert r.notes != ""


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_results(self):
        r1 = _compare(seed=42)
        r2 = _compare(seed=42)
        for a, b in zip(r1, r2):
            assert a.mean_auc == b.mean_auc
            assert a.cv_auc_pct == b.cv_auc_pct

    def test_different_seed_runs_without_error(self):
        # Just ensure it doesn't crash — values may or may not differ
        r1 = _compare(seed=42)
        r2 = _compare(seed=1234)
        assert len(r1) == len(r2) == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_cl_ref_zero_raises(self):
        with pytest.raises(ValueError, match="cl_ref_L_per_h"):
            _compare(cl_ref_L_per_h=0.0)

    def test_cl_ref_negative_raises(self):
        with pytest.raises(ValueError, match="cl_ref_L_per_h"):
            _compare(cl_ref_L_per_h=-1.0)

    def test_vd_ref_zero_raises(self):
        with pytest.raises(ValueError, match="vd_ref_L"):
            _compare(vd_ref_L=0.0)

    def test_vd_ref_negative_raises(self):
        with pytest.raises(ValueError, match="vd_ref_L"):
            _compare(vd_ref_L=-5.0)

    def test_flat_dose_zero_raises(self):
        with pytest.raises(ValueError, match="flat_dose_mg"):
            _compare(flat_dose_mg=0.0)

    def test_dose_per_kg_zero_raises(self):
        with pytest.raises(ValueError, match="dose_per_kg_mg"):
            _compare(dose_per_kg_mg=0.0)

    def test_n_patients_below_10_raises(self):
        with pytest.raises(ValueError, match="n_patients"):
            _compare(n_patients=5)

    def test_weight_cv_zero_raises(self):
        with pytest.raises(ValueError, match="weight_cv"):
            _compare(weight_cv=0.0)

    def test_weight_cv_one_raises(self):
        with pytest.raises(ValueError, match="weight_cv"):
            _compare(weight_cv=1.0)

    def test_weight_cv_negative_raises(self):
        with pytest.raises(ValueError, match="weight_cv"):
            _compare(weight_cv=-0.1)
