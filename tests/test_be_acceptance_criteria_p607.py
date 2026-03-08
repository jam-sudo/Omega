"""Tests for Phase 607 — Bioequivalence Acceptance Criteria."""

import math

import pytest

from omega_pbpk.clinical.be_acceptance_criteria_p607 import (
    BEAcceptanceCriteriaResult,
    batch_be_evaluation,
    evaluate_be_criteria,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_result(**kwargs):
    """Build a result with sensible defaults."""
    defaults = dict(
        drug_name="DrugA",
        test_form="TestTab",
        ref_form="RefTab",
        auc_gmr=1.00,
        cmax_gmr=1.00,
        n_subjects=24,
        within_subject_cv_auc=0.20,
        within_subject_cv_cmax=0.25,
        nti_drug=False,
        guideline="FDA",
    )
    defaults.update(kwargs)
    return evaluate_be_criteria(**defaults)


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_be_acceptance_criteria_result(self):
        r = make_result()
        assert isinstance(r, BEAcceptanceCriteriaResult)

    def test_frozen_dataclass_immutable(self):
        r = make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.auc_ratio = 99.0  # type: ignore[misc]

    def test_all_fields_present(self):
        r = make_result()
        for field in (
            "drug_name",
            "test_formulation",
            "reference_formulation",
            "auc_ratio",
            "cmax_ratio",
            "auc_90ci_lower",
            "auc_90ci_upper",
            "cmax_90ci_lower",
            "cmax_90ci_upper",
            "auc_be_pass",
            "cmax_be_pass",
            "overall_be_pass",
            "nti_drug",
            "nti_be_pass",
            "guideline",
            "recommendation",
            "notes",
        ):
            assert hasattr(r, field), f"Missing field: {field}"

    def test_string_fields(self):
        r = make_result(drug_name="Metformin", test_form="Generic", ref_form="Brand")
        assert r.drug_name == "Metformin"
        assert r.test_formulation == "Generic"
        assert r.reference_formulation == "Brand"

    def test_ratio_stored_correctly(self):
        r = make_result(auc_gmr=0.95, cmax_gmr=1.05)
        assert r.auc_ratio == pytest.approx(0.95)
        assert r.cmax_ratio == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# CI calculation
# ---------------------------------------------------------------------------


class TestCICalculation:
    def test_ci_symmetric_around_gmr_in_log_space(self):
        r = make_result(auc_gmr=1.00, n_subjects=24, within_subject_cv_auc=0.20)
        log_lower = math.log(r.auc_90ci_lower)
        log_upper = math.log(r.auc_90ci_upper)
        assert abs(log_lower + log_upper) < 1e-10  # symmetric about 0 when GMR=1

    def test_t_crit_small_n(self):
        r_small = make_result(n_subjects=10, auc_gmr=0.95, within_subject_cv_auc=0.20)
        r_large = make_result(n_subjects=30, auc_gmr=0.95, within_subject_cv_auc=0.20)
        # Smaller n → wider CI
        width_small = r_small.auc_90ci_upper - r_small.auc_90ci_lower
        width_large = r_large.auc_90ci_upper - r_large.auc_90ci_lower
        assert width_small > width_large

    def test_t_crit_medium_n(self):
        # n=15 uses t_crit=1.729; n=25 uses 1.645 → n=15 CI is wider
        r15 = make_result(n_subjects=15, auc_gmr=1.0, within_subject_cv_auc=0.20)
        r25 = make_result(n_subjects=25, auc_gmr=1.0, within_subject_cv_auc=0.20)
        assert r15.auc_90ci_upper > r25.auc_90ci_upper

    def test_ci_lower_less_than_upper(self):
        r = make_result(auc_gmr=0.92, cmax_gmr=1.08)
        assert r.auc_90ci_lower < r.auc_90ci_upper
        assert r.cmax_90ci_lower < r.cmax_90ci_upper

    def test_higher_cv_wider_ci(self):
        r_low = make_result(within_subject_cv_auc=0.10, n_subjects=24)
        r_high = make_result(within_subject_cv_auc=0.40, n_subjects=24)
        assert (
            r_high.auc_90ci_upper - r_high.auc_90ci_lower
            > r_low.auc_90ci_upper - r_low.auc_90ci_lower
        )

    def test_ci_contains_gmr(self):
        r = make_result(auc_gmr=0.93)
        assert r.auc_90ci_lower <= 0.93 <= r.auc_90ci_upper


# ---------------------------------------------------------------------------
# Pass/fail scenarios
# ---------------------------------------------------------------------------


class TestPassFail:
    def test_ideal_gmr_passes(self):
        r = make_result(
            auc_gmr=1.00,
            cmax_gmr=1.00,
            n_subjects=24,
            within_subject_cv_auc=0.15,
            within_subject_cv_cmax=0.20,
        )
        assert r.auc_be_pass is True
        assert r.cmax_be_pass is True
        assert r.overall_be_pass is True

    def test_low_auc_fails(self):
        # GMR=0.75 well below 80% limit
        r = make_result(
            auc_gmr=0.75,
            cmax_gmr=1.00,
            n_subjects=24,
            within_subject_cv_auc=0.10,
            within_subject_cv_cmax=0.10,
        )
        assert r.auc_be_pass is False
        assert r.overall_be_pass is False

    def test_high_auc_fails(self):
        r = make_result(
            auc_gmr=1.35,
            cmax_gmr=1.00,
            n_subjects=24,
            within_subject_cv_auc=0.10,
            within_subject_cv_cmax=0.10,
        )
        assert r.auc_be_pass is False

    def test_cmax_fails_auc_passes(self):
        r = make_result(
            auc_gmr=0.95,
            cmax_gmr=1.40,
            n_subjects=30,
            within_subject_cv_auc=0.10,
            within_subject_cv_cmax=0.10,
        )
        assert r.auc_be_pass is True
        assert r.cmax_be_pass is False
        assert r.overall_be_pass is False

    def test_borderline_pass(self):
        # Tight CI near 80% limit - large n, low CV
        r = make_result(
            auc_gmr=0.82,
            cmax_gmr=0.82,
            n_subjects=100,
            within_subject_cv_auc=0.05,
            within_subject_cv_cmax=0.05,
        )
        assert r.auc_be_pass is True

    def test_wide_ci_fails_despite_central_gmr(self):
        # High CV with small n → CI exceeds 80-125%
        r = make_result(
            auc_gmr=1.00,
            cmax_gmr=1.00,
            n_subjects=4,
            within_subject_cv_auc=0.50,
            within_subject_cv_cmax=0.50,
        )
        # With very high variability and tiny n, CI may be very wide
        assert r.auc_90ci_lower < 0.80 or r.auc_90ci_upper > 1.25


# ---------------------------------------------------------------------------
# NTI criteria
# ---------------------------------------------------------------------------


class TestNTICriteria:
    def test_nti_false_by_default(self):
        r = make_result(nti_drug=False)
        assert r.nti_drug is False
        assert r.nti_be_pass is False

    def test_nti_pass_tight_ci(self):
        r = make_result(
            auc_gmr=0.98,
            cmax_gmr=0.98,
            n_subjects=50,
            within_subject_cv_auc=0.05,
            within_subject_cv_cmax=0.05,
            nti_drug=True,
        )
        assert r.nti_drug is True
        assert r.nti_be_pass is True

    def test_nti_fail_wide_ci(self):
        # Even though GMR=1.0, CV is too high → CI exceeds 90-111.11%
        r = make_result(
            auc_gmr=1.00,
            cmax_gmr=1.00,
            n_subjects=12,
            within_subject_cv_auc=0.25,
            within_subject_cv_cmax=0.25,
            nti_drug=True,
        )
        assert r.nti_drug is True
        # CI would be wide → NTI criteria likely fails
        # Check that the nti result is consistent with the CI
        nti_pass_expected = (
            r.auc_90ci_lower >= 0.90
            and r.auc_90ci_upper <= 1.1111
            and r.cmax_90ci_lower >= 0.90
            and r.cmax_90ci_upper <= 1.1111
        )
        assert r.nti_be_pass is nti_pass_expected

    def test_nti_standard_be_can_differ(self):
        # Standard BE passes (80-125%) but NTI fails (90-111.11%)
        r = make_result(
            auc_gmr=0.87,
            cmax_gmr=0.87,
            n_subjects=100,
            within_subject_cv_auc=0.02,
            within_subject_cv_cmax=0.02,
            nti_drug=True,
        )
        # 87% is within 80-125% but outside 90-111.11%
        assert r.auc_be_pass is True
        assert r.nti_be_pass is False


# ---------------------------------------------------------------------------
# Guideline differences
# ---------------------------------------------------------------------------


class TestGuidelines:
    def test_fda_guideline_stored(self):
        r = make_result(guideline="FDA")
        assert r.guideline == "FDA"

    def test_ema_guideline_stored(self):
        r = make_result(guideline="EMA")
        assert r.guideline == "EMA"

    def test_invalid_guideline_raises(self):
        with pytest.raises(ValueError, match="guideline"):
            make_result(guideline="ICH")


# ---------------------------------------------------------------------------
# Recommendation text
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_recommendation_approve(self):
        r = make_result(
            auc_gmr=1.00,
            cmax_gmr=1.00,
            n_subjects=30,
            within_subject_cv_auc=0.10,
            within_subject_cv_cmax=0.10,
        )
        assert "approve" in r.recommendation.lower()

    def test_recommendation_auc_outside(self):
        r = make_result(
            auc_gmr=0.70,
            cmax_gmr=1.00,
            n_subjects=30,
            within_subject_cv_auc=0.05,
            within_subject_cv_cmax=0.05,
        )
        assert "AUC" in r.recommendation

    def test_recommendation_cmax_outside(self):
        r = make_result(
            auc_gmr=1.00,
            cmax_gmr=1.35,
            n_subjects=30,
            within_subject_cv_auc=0.05,
            within_subject_cv_cmax=0.05,
        )
        assert "Cmax" in r.recommendation

    def test_recommendation_both_outside(self):
        r = make_result(
            auc_gmr=0.70,
            cmax_gmr=1.40,
            n_subjects=30,
            within_subject_cv_auc=0.05,
            within_subject_cv_cmax=0.05,
        )
        assert "both" in r.recommendation.lower()

    def test_notes_contains_guideline(self):
        r = make_result(guideline="EMA")
        assert "EMA" in r.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_auc_gmr_zero(self):
        with pytest.raises(ValueError):
            make_result(auc_gmr=0.0)

    def test_invalid_auc_gmr_too_large(self):
        with pytest.raises(ValueError):
            make_result(auc_gmr=4.0)

    def test_invalid_cmax_gmr_negative(self):
        with pytest.raises(ValueError):
            make_result(cmax_gmr=-0.5)

    def test_invalid_n_subjects(self):
        with pytest.raises(ValueError):
            make_result(n_subjects=1)

    def test_invalid_cv_zero(self):
        with pytest.raises(ValueError):
            make_result(within_subject_cv_auc=0.0)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class TestBatchEvaluation:
    def _make_studies(self):
        return [
            dict(
                drug_name="DrugA",
                test_form="Generic",
                ref_form="Brand",
                auc_gmr=0.70,
                cmax_gmr=0.70,
                n_subjects=24,
                within_subject_cv_auc=0.05,
                within_subject_cv_cmax=0.05,
            ),
            dict(
                drug_name="DrugB",
                test_form="Generic",
                ref_form="Brand",
                auc_gmr=1.00,
                cmax_gmr=1.00,
                n_subjects=24,
                within_subject_cv_auc=0.15,
                within_subject_cv_cmax=0.15,
            ),
            dict(
                drug_name="DrugC",
                test_form="Generic",
                ref_form="Brand",
                auc_gmr=0.95,
                cmax_gmr=0.95,
                n_subjects=24,
                within_subject_cv_auc=0.15,
                within_subject_cv_cmax=0.15,
            ),
        ]

    def test_batch_returns_list(self):
        results = batch_be_evaluation(self._make_studies())
        assert isinstance(results, list)

    def test_batch_length(self):
        studies = self._make_studies()
        results = batch_be_evaluation(studies)
        assert len(results) == len(studies)

    def test_batch_returns_result_objects(self):
        results = batch_be_evaluation(self._make_studies())
        for r in results:
            assert isinstance(r, BEAcceptanceCriteriaResult)

    def test_batch_sorted_pass_first(self):
        results = batch_be_evaluation(self._make_studies())
        # Passing results should come before failing ones
        pass_indices = [i for i, r in enumerate(results) if r.overall_be_pass]
        fail_indices = [i for i, r in enumerate(results) if not r.overall_be_pass]
        if pass_indices and fail_indices:
            assert max(pass_indices) < min(fail_indices)

    def test_batch_empty_list(self):
        results = batch_be_evaluation([])
        assert results == []

    def test_batch_with_nti(self):
        studies = [
            dict(
                drug_name="NTIDrug",
                test_form="T",
                ref_form="R",
                auc_gmr=0.95,
                cmax_gmr=0.95,
                n_subjects=30,
                within_subject_cv_auc=0.05,
                within_subject_cv_cmax=0.05,
                nti_drug=True,
                guideline="FDA",
            )
        ]
        results = batch_be_evaluation(studies)
        assert results[0].nti_drug is True

    def test_batch_sorted_by_auc_proximity_among_passing(self):
        studies = [
            dict(
                drug_name="DrugX",
                test_form="T",
                ref_form="R",
                auc_gmr=0.90,
                cmax_gmr=0.90,
                n_subjects=50,
                within_subject_cv_auc=0.05,
                within_subject_cv_cmax=0.05,
            ),
            dict(
                drug_name="DrugY",
                test_form="T",
                ref_form="R",
                auc_gmr=0.98,
                cmax_gmr=0.98,
                n_subjects=50,
                within_subject_cv_auc=0.05,
                within_subject_cv_cmax=0.05,
            ),
        ]
        results = batch_be_evaluation(studies)
        # Both should pass; DrugY (0.98) closer to 1.0 → comes first
        assert results[0].drug_name == "DrugY"
