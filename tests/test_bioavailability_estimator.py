"""Tests for Phase 667 — bioavailability_estimator."""

import pytest

from omega_pbpk.analysis.bioavailability_estimator import (
    BioavailabilityEstimate,
    compare_formulations,
    estimate_absolute_bioavailability,
    estimate_relative_bioavailability,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TIMES_ORAL = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
CONCS_ORAL = [0.0, 5.0, 10.0, 8.0, 5.0, 3.0, 1.5, 0.5]

TIMES_IV = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
CONCS_IV = [20.0, 18.0, 15.0, 11.0, 7.0, 3.0, 1.2, 0.5]

DOSE_ORAL = 100.0  # mg
DOSE_IV = 50.0  # mg

TIMES_REF = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
CONCS_REF = [0.0, 4.0, 8.0, 6.0, 3.0, 0.8]
DOSE_REF = 80.0

TIMES_TEST = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
CONCS_TEST = [0.0, 4.1, 8.2, 6.1, 3.1, 0.82]  # essentially same as ref
DOSE_TEST = 80.0


# ---------------------------------------------------------------------------
# estimate_absolute_bioavailability
# ---------------------------------------------------------------------------


def test_absolute_returns_bioavailability_estimate():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert isinstance(result, BioavailabilityEstimate)


def test_absolute_f_estimated_positive():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert result.f_estimated > 0


def test_absolute_ci_ordering():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert result.f_lower_ci <= result.f_estimated <= result.f_upper_ci


def test_absolute_method_label():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert result.method == "absolute"


def test_absolute_dose_normalized_auc():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert abs(result.dose_normalized_auc_test - result.auc_test / DOSE_ORAL) < 1e-9


def test_absolute_auc_test_positive():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert result.auc_test > 0


def test_absolute_auc_reference_positive():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert result.auc_reference > 0


def test_absolute_formula_consistency():
    """f_estimated should equal (AUC_oral/dose_oral) / (AUC_iv/dose_iv)."""
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    expected_f = (result.auc_test / result.dose_test) / (
        result.auc_reference / result.dose_reference
    )
    assert abs(result.f_estimated - expected_f) < 1e-9


def test_absolute_notes_nonempty():
    result = estimate_absolute_bioavailability(
        "DrugA", TIMES_ORAL, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
    )
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# estimate_relative_bioavailability
# ---------------------------------------------------------------------------


def test_relative_returns_bioavailability_estimate():
    result = estimate_relative_bioavailability(
        "DrugB", TIMES_TEST, CONCS_TEST, DOSE_TEST, TIMES_REF, CONCS_REF, DOSE_REF
    )
    assert isinstance(result, BioavailabilityEstimate)


def test_relative_method_label():
    result = estimate_relative_bioavailability(
        "DrugB", TIMES_TEST, CONCS_TEST, DOSE_TEST, TIMES_REF, CONCS_REF, DOSE_REF
    )
    assert result.method == "relative"


def test_relative_identical_data_f_approx_one():
    """Identical test and ref (same dose) → f ≈ 1.0."""
    result = estimate_relative_bioavailability(
        "DrugC", TIMES_REF, CONCS_REF, DOSE_REF, TIMES_REF, CONCS_REF, DOSE_REF
    )
    assert abs(result.f_estimated - 1.0) < 1e-9


def test_relative_double_auc_f_approx_two():
    """2× AUC for test vs ref (same dose) → f ≈ 2.0."""
    concs_2x = [c * 2.0 for c in CONCS_REF]
    result = estimate_relative_bioavailability(
        "DrugD", TIMES_REF, concs_2x, DOSE_REF, TIMES_REF, CONCS_REF, DOSE_REF
    )
    assert abs(result.f_estimated - 2.0) < 1e-6


def test_relative_higher_auc_test_higher_f():
    concs_high = [c * 1.5 for c in CONCS_REF]
    result_high = estimate_relative_bioavailability(
        "DrugE", TIMES_REF, concs_high, DOSE_REF, TIMES_REF, CONCS_REF, DOSE_REF
    )
    result_low = estimate_relative_bioavailability(
        "DrugE", TIMES_REF, CONCS_REF, DOSE_REF, TIMES_REF, CONCS_REF, DOSE_REF
    )
    assert result_high.f_estimated > result_low.f_estimated


def test_relative_bioequivalent_true_when_similar():
    result = estimate_relative_bioavailability(
        "DrugB", TIMES_TEST, CONCS_TEST, DOSE_TEST, TIMES_REF, CONCS_REF, DOSE_REF
    )
    assert result.bioequivalent is True


def test_relative_bioequivalent_false_when_different():
    concs_very_high = [c * 3.0 for c in CONCS_REF]
    result = estimate_relative_bioavailability(
        "DrugF", TIMES_REF, concs_very_high, DOSE_REF, TIMES_REF, CONCS_REF, DOSE_REF
    )
    assert result.bioequivalent is False


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------

FORMULATIONS = [
    {"name": "IR_tablet", "times_h": TIMES_REF, "concs": CONCS_REF, "dose": DOSE_REF},
    {
        "name": "ER_capsule",
        "times_h": TIMES_REF,
        "concs": [c * 1.2 for c in CONCS_REF],
        "dose": DOSE_REF,
    },
    {
        "name": "Solution",
        "times_h": TIMES_REF,
        "concs": [c * 0.7 for c in CONCS_REF],
        "dose": DOSE_REF,
    },
]


def test_compare_formulations_returns_list():
    results = compare_formulations("DrugG", FORMULATIONS, reference_idx=0)
    assert isinstance(results, list)
    assert len(results) == len(FORMULATIONS)


def test_compare_formulations_sorted_descending():
    results = compare_formulations("DrugG", FORMULATIONS, reference_idx=0)
    fs = [r.f_estimated for r in results]
    assert fs == sorted(fs, reverse=True)


def test_compare_formulations_reference_f_approx_one():
    results = compare_formulations("DrugG", FORMULATIONS, reference_idx=0)
    # reference is formulation 0 (IR_tablet); find it in results by f ≈ 1.0
    ref_results = [r for r in results if abs(r.f_estimated - 1.0) < 1e-6]
    assert len(ref_results) >= 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_empty_times_oral_raises():
    with pytest.raises(ValueError):
        estimate_absolute_bioavailability("X", [], [], DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV)


def test_validation_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        estimate_absolute_bioavailability(
            "X", TIMES_ORAL, CONCS_ORAL[:3], DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
        )


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError):
        estimate_absolute_bioavailability(
            "X", TIMES_ORAL, CONCS_ORAL, 0.0, TIMES_IV, CONCS_IV, DOSE_IV
        )


def test_validation_negative_concentration_raises():
    bad_concs = CONCS_ORAL[:]
    bad_concs[3] = -1.0
    with pytest.raises(ValueError):
        estimate_absolute_bioavailability(
            "X", TIMES_ORAL, bad_concs, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
        )


def test_validation_non_monotone_times_raises():
    bad_times = TIMES_ORAL[:]
    bad_times[3] = bad_times[2]  # duplicate → not strictly increasing
    with pytest.raises(ValueError):
        estimate_absolute_bioavailability(
            "X", bad_times, CONCS_ORAL, DOSE_ORAL, TIMES_IV, CONCS_IV, DOSE_IV
        )
