"""Tests for omega_pbpk.clinical.bioequivalence module."""

import math

import pytest

from omega_pbpk.clinical.bioequivalence import (
    BioequivalenceResult,
    assess_bioequivalence,
    simulate_be_study,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identical_auc(n=12, value=100.0):
    return [value] * n


# ---------------------------------------------------------------------------
# BioequivalenceResult dataclass
# ---------------------------------------------------------------------------


class TestBioequivalenceResultDataclass:
    def test_is_dataclass_with_required_fields(self):
        result = assess_bioequivalence(
            "DrugA",
            "DrugB",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert isinstance(result, BioequivalenceResult)

    def test_has_drug_test_field(self):
        result = assess_bioequivalence(
            "TestDrug",
            "RefDrug",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert result.drug_test == "TestDrug"

    def test_has_drug_reference_field(self):
        result = assess_bioequivalence(
            "TestDrug",
            "RefDrug",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert result.drug_reference == "RefDrug"

    def test_has_n_subjects_field(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(10),
            auc_reference=_identical_auc(10),
        )
        assert hasattr(result, "n_subjects")

    def test_has_auc_ratio_geometric_mean(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert hasattr(result, "auc_ratio_geometric_mean")

    def test_has_cmax_ratio_geometric_mean(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
            cmax_test=_identical_auc(),
            cmax_reference=_identical_auc(),
        )
        assert hasattr(result, "cmax_ratio_geometric_mean")

    def test_has_ci_90_auc_tuple(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert isinstance(result.ci_90_auc, tuple)
        assert len(result.ci_90_auc) == 2

    def test_has_ci_90_cmax_tuple(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
            cmax_test=_identical_auc(),
            cmax_reference=_identical_auc(),
        )
        assert isinstance(result.ci_90_cmax, tuple)
        assert len(result.ci_90_cmax) == 2

    def test_has_be_auc_bool(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert isinstance(result.be_auc, bool)

    def test_has_be_cmax_bool(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
            cmax_test=_identical_auc(),
            cmax_reference=_identical_auc(),
        )
        assert isinstance(result.be_cmax, bool)

    def test_has_overall_be_bool(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert isinstance(result.overall_be, bool)

    def test_has_study_design_field(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert hasattr(result, "study_design")

    def test_has_intra_subject_cv_auc_pct(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert hasattr(result, "intra_subject_cv_auc_pct")

    def test_has_intra_subject_cv_cmax_pct(self):
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=_identical_auc(),
            auc_reference=_identical_auc(),
        )
        assert hasattr(result, "intra_subject_cv_cmax_pct")


# ---------------------------------------------------------------------------
# assess_bioequivalence — ratio correctness
# ---------------------------------------------------------------------------


class TestAssessBioequivalenceRatio:
    def test_ratio_one_for_identical_data(self):
        data = [80.0, 90.0, 100.0, 110.0, 120.0, 95.0, 85.0, 105.0, 98.0, 102.0, 88.0, 112.0]
        result = assess_bioequivalence("A", "B", auc_test=data, auc_reference=data)
        assert math.isclose(result.auc_ratio_geometric_mean, 1.0, rel_tol=1e-9)

    def test_ratio_two_for_doubled_test(self):
        ref = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0]
        test = [v * 2 for v in ref]
        result = assess_bioequivalence("A", "B", auc_test=test, auc_reference=ref)
        assert math.isclose(result.auc_ratio_geometric_mean, 2.0, rel_tol=1e-6)

    def test_cmax_ratio_one_for_identical_cmax(self):
        data = [50.0] * 12
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=data,
            auc_reference=data,
            cmax_test=data,
            cmax_reference=data,
        )
        assert math.isclose(result.cmax_ratio_geometric_mean, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# assess_bioequivalence — BE decision
# ---------------------------------------------------------------------------


class TestAssessBioequivalenceBEDecision:
    def test_be_auc_true_for_ratio_one_low_cv(self):
        # 24 subjects, ratio=1.0, low variability => must be bioequivalent
        import random

        rng = random.Random(0)
        base = [100.0 * math.exp(rng.gauss(0, 0.05)) for _ in range(24)]
        result = assess_bioequivalence("A", "B", auc_test=base, auc_reference=base)
        assert result.be_auc is True

    def test_be_auc_false_for_ratio_0_7(self):
        import random

        rng = random.Random(1)
        ref = [100.0 * math.exp(rng.gauss(0, 0.05)) for _ in range(24)]
        test = [v * 0.7 for v in ref]
        result = assess_bioequivalence("A", "B", auc_test=test, auc_reference=ref)
        assert result.be_auc is False

    def test_be_auc_false_for_ratio_1_4(self):
        import random

        rng = random.Random(2)
        ref = [100.0 * math.exp(rng.gauss(0, 0.05)) for _ in range(24)]
        test = [v * 1.4 for v in ref]
        result = assess_bioequivalence("A", "B", auc_test=test, auc_reference=ref)
        assert result.be_auc is False

    def test_overall_be_equals_be_auc_when_no_cmax(self):
        data = [100.0] * 12
        result = assess_bioequivalence("A", "B", auc_test=data, auc_reference=data)
        assert result.overall_be == result.be_auc

    def test_overall_be_equals_be_auc_and_be_cmax_when_cmax_provided(self):
        data = [100.0] * 12
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=data,
            auc_reference=data,
            cmax_test=data,
            cmax_reference=data,
        )
        assert result.overall_be == (result.be_auc and result.be_cmax)

    def test_overall_be_false_when_cmax_fails(self):
        import random

        rng = random.Random(3)
        auc_ref = [100.0 * math.exp(rng.gauss(0, 0.05)) for _ in range(24)]
        cmax_ref = [50.0 * math.exp(rng.gauss(0, 0.05)) for _ in range(24)]
        cmax_test = [v * 0.7 for v in cmax_ref]  # fails BE for Cmax
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=auc_ref,
            auc_reference=auc_ref,
            cmax_test=cmax_test,
            cmax_reference=cmax_ref,
        )
        assert result.overall_be is False


# ---------------------------------------------------------------------------
# assess_bioequivalence — CI ordering
# ---------------------------------------------------------------------------


class TestAssessBioequivalenceCIOrdering:
    def test_ci_90_auc_lower_less_than_upper(self):
        import random

        rng = random.Random(4)
        test_data = [100.0 * math.exp(rng.gauss(0, 0.2)) for _ in range(20)]
        ref_data = [100.0 * math.exp(rng.gauss(0, 0.2)) for _ in range(20)]
        result = assess_bioequivalence("A", "B", auc_test=test_data, auc_reference=ref_data)
        assert result.ci_90_auc[0] <= result.ci_90_auc[1]

    def test_ci_90_cmax_lower_less_than_upper(self):
        import random

        rng = random.Random(5)
        test_data = [50.0 * math.exp(rng.gauss(0, 0.2)) for _ in range(20)]
        ref_data = [50.0 * math.exp(rng.gauss(0, 0.2)) for _ in range(20)]
        result = assess_bioequivalence(
            "A",
            "B",
            auc_test=test_data,
            auc_reference=ref_data,
            cmax_test=test_data,
            cmax_reference=ref_data,
        )
        assert result.ci_90_cmax[0] <= result.ci_90_cmax[1]

    def test_ci_90_auc_bounds_are_positive(self):
        data = [100.0] * 12
        result = assess_bioequivalence("A", "B", auc_test=data, auc_reference=data)
        assert result.ci_90_auc[0] > 0
        assert result.ci_90_auc[1] > 0


# ---------------------------------------------------------------------------
# assess_bioequivalence — n_subjects
# ---------------------------------------------------------------------------


class TestAssessBioequivalenceNSubjects:
    def test_n_subjects_preserved_12(self):
        data = _identical_auc(12)
        result = assess_bioequivalence("A", "B", auc_test=data, auc_reference=data)
        assert result.n_subjects == 12

    def test_n_subjects_preserved_24(self):
        data = _identical_auc(24)
        result = assess_bioequivalence("A", "B", auc_test=data, auc_reference=data)
        assert result.n_subjects == 24

    def test_n_subjects_preserved_6(self):
        data = _identical_auc(6)
        result = assess_bioequivalence("A", "B", auc_test=data, auc_reference=data)
        assert result.n_subjects == 6


# ---------------------------------------------------------------------------
# assess_bioequivalence — ValueError conditions
# ---------------------------------------------------------------------------


class TestAssessBioequivalenceValueErrors:
    def test_raises_value_error_for_len_less_than_2_auc_test(self):
        with pytest.raises(ValueError):
            assess_bioequivalence("A", "B", auc_test=[100.0], auc_reference=[100.0])

    def test_raises_value_error_for_empty_auc_test(self):
        with pytest.raises(ValueError):
            assess_bioequivalence("A", "B", auc_test=[], auc_reference=[100.0, 200.0])

    def test_raises_value_error_for_zero_in_auc_test(self):
        with pytest.raises(ValueError):
            assess_bioequivalence(
                "A",
                "B",
                auc_test=[0.0, 100.0, 100.0],
                auc_reference=[100.0, 100.0, 100.0],
            )

    def test_raises_value_error_for_negative_in_auc_reference(self):
        with pytest.raises(ValueError):
            assess_bioequivalence(
                "A",
                "B",
                auc_test=[100.0, 100.0, 100.0],
                auc_reference=[-50.0, 100.0, 100.0],
            )

    def test_raises_value_error_for_zero_in_auc_reference(self):
        with pytest.raises(ValueError):
            assess_bioequivalence(
                "A",
                "B",
                auc_test=[100.0, 100.0, 100.0],
                auc_reference=[0.0, 100.0, 100.0],
            )

    def test_raises_value_error_for_zero_in_cmax_test(self):
        with pytest.raises(ValueError):
            assess_bioequivalence(
                "A",
                "B",
                auc_test=[100.0, 100.0, 100.0],
                auc_reference=[100.0, 100.0, 100.0],
                cmax_test=[0.0, 50.0, 50.0],
                cmax_reference=[50.0, 50.0, 50.0],
            )

    def test_raises_value_error_for_negative_in_cmax_reference(self):
        with pytest.raises(ValueError):
            assess_bioequivalence(
                "A",
                "B",
                auc_test=[100.0, 100.0, 100.0],
                auc_reference=[100.0, 100.0, 100.0],
                cmax_test=[50.0, 50.0, 50.0],
                cmax_reference=[-1.0, 50.0, 50.0],
            )


# ---------------------------------------------------------------------------
# simulate_be_study
# ---------------------------------------------------------------------------


class TestSimulateBeStudy:
    def test_returns_bioequivalence_result(self):
        result = simulate_be_study("DrugA", "DrugB")
        assert isinstance(result, BioequivalenceResult)

    def test_default_n_subjects_24(self):
        result = simulate_be_study("DrugA", "DrugB")
        assert result.n_subjects == 24

    def test_custom_n_subjects_preserved(self):
        result = simulate_be_study("DrugA", "DrugB", n_subjects=36)
        assert result.n_subjects == 36

    def test_true_ratio_one_be_likely_true(self):
        # With true_ratio=1.0 and low CV, should be bioequivalent
        result = simulate_be_study("DrugA", "DrugB", true_ratio=1.0, intra_cv=0.1, seed=42)
        assert result.be_auc is True

    def test_true_ratio_extreme_be_false(self):
        # With true_ratio=0.6, should fail BE
        result = simulate_be_study("DrugA", "DrugB", true_ratio=0.6, intra_cv=0.1, seed=42)
        assert result.be_auc is False

    def test_intra_cv_positive(self):
        result = simulate_be_study("DrugA", "DrugB", seed=42)
        assert result.intra_subject_cv_auc_pct > 0

    def test_study_design_is_2x2_crossover(self):
        result = simulate_be_study("DrugA", "DrugB")
        assert result.study_design == "2x2_crossover"

    def test_drug_names_preserved(self):
        result = simulate_be_study("TestCompound", "ReferenceCompound")
        assert result.drug_test == "TestCompound"
        assert result.drug_reference == "ReferenceCompound"

    def test_seed_reproducibility(self):
        r1 = simulate_be_study("A", "B", seed=123)
        r2 = simulate_be_study("A", "B", seed=123)
        assert math.isclose(r1.auc_ratio_geometric_mean, r2.auc_ratio_geometric_mean)

    def test_different_seeds_give_different_results(self):
        r1 = simulate_be_study("A", "B", seed=1)
        r2 = simulate_be_study("A", "B", seed=999)
        # Extremely unlikely to be equal with different seeds
        assert not math.isclose(
            r1.auc_ratio_geometric_mean, r2.auc_ratio_geometric_mean, rel_tol=1e-6
        )

    def test_auc_ratio_close_to_true_ratio(self):
        # With many subjects, geometric mean should be close to true ratio
        result = simulate_be_study("A", "B", n_subjects=200, true_ratio=1.1, intra_cv=0.2, seed=0)
        assert math.isclose(result.auc_ratio_geometric_mean, 1.1, rel_tol=0.05)

    def test_ci_90_auc_ordered(self):
        result = simulate_be_study("A", "B", seed=42)
        assert result.ci_90_auc[0] < result.ci_90_auc[1]

    def test_overall_be_consistent_with_be_auc_and_be_cmax(self):
        result = simulate_be_study("A", "B", seed=42)
        assert result.overall_be == (result.be_auc and result.be_cmax)
