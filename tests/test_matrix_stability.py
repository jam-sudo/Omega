"""Tests for prediction/matrix_stability.py — Phase 1092.

Drug degradation in biological matrices:
  plasma, whole_blood, urine, liver_s9, brain_homogenate.
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.matrix_stability import (
    MatrixStabilityResult,
    predict_matrix_stability,
    screen_matrices,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def plasma_result(**kwargs) -> MatrixStabilityResult:
    return predict_matrix_stability(drug_name="TestDrug", matrix="plasma", **kwargs)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = plasma_result()
    assert isinstance(r, MatrixStabilityResult)


def test_frozen_dataclass():
    r = plasma_result()
    with pytest.raises((AttributeError, TypeError)):
        r.t_half_h = 999.0  # type: ignore


# ---------------------------------------------------------------------------
# Basic numeric correctness
# ---------------------------------------------------------------------------


def test_t_half_positive():
    r = plasma_result()
    assert r.t_half_h > 0.0


def test_t_half_ln2_over_k():
    r = plasma_result()
    expected = math.log(2.0) / r.k_degradation_per_h
    assert abs(r.t_half_h - expected) < 1e-10


def test_stability_decreasing_over_time():
    r = plasma_result()
    assert r.stability_at_4h_pct <= r.stability_at_1h_pct


def test_stability_24h_le_4h():
    r = plasma_result()
    assert r.stability_at_24h_pct <= r.stability_at_4h_pct


def test_stability_at_1h_in_range():
    r = plasma_result()
    assert 0.0 < r.stability_at_1h_pct <= 100.0


# ---------------------------------------------------------------------------
# Functional groups
# ---------------------------------------------------------------------------


def test_ester_increases_k_in_plasma():
    r_base = plasma_result(has_ester=False)
    r_ester = plasma_result(has_ester=True)
    assert r_ester.k_degradation_per_h > r_base.k_degradation_per_h


def test_whole_blood_k_gt_plasma_for_ester():
    """Esterase activity higher in whole blood than plasma."""
    r_plasma = predict_matrix_stability("Drug", "plasma", has_ester=True)
    r_blood = predict_matrix_stability("Drug", "whole_blood", has_ester=True)
    assert r_blood.k_degradation_per_h > r_plasma.k_degradation_per_h


def test_hydroxamate_increases_k_in_plasma():
    r_base = plasma_result()
    r_hx = plasma_result(has_hydroxamate=True)
    assert r_hx.k_degradation_per_h > r_base.k_degradation_per_h


# ---------------------------------------------------------------------------
# Stability classes
# ---------------------------------------------------------------------------


def test_stable_class_for_low_k():
    """Very low base_k → stable."""
    r = plasma_result(base_k_per_h=1e-6)
    assert r.stability_class == "stable"


def test_labile_class_for_high_k():
    """High k (ester + hydroxamate) → labile."""
    r = predict_matrix_stability(
        "Drug",
        "liver_s9",
        has_ester=True,
        has_hydroxamate=True,
        base_k_per_h=0.5,
    )
    assert r.stability_class == "labile"


def test_moderate_class():
    """Moderate k should give 'moderate' class (t½ between 4-24 h)."""
    # target t½ ~10h: k = ln2/10 ≈ 0.0693
    # k = base * matrix_factor(plasma=1) * temp_factor(37→1) = base
    # so base_k ≈ 0.0693
    r = plasma_result(base_k_per_h=0.069)
    assert r.stability_class == "moderate"


def test_stability_class_valid_values():
    r = plasma_result()
    assert r.stability_class in {"stable", "moderate", "labile"}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_bioanalytical_recommendation_nonempty():
    r = plasma_result()
    assert len(r.bioanalytical_recommendation) > 0


def test_labile_recommendation_mentions_additives():
    r = predict_matrix_stability(
        "Drug",
        "plasma",
        has_ester=True,
        has_hydroxamate=True,
        base_k_per_h=1.0,
    )
    assert r.stability_class == "labile"
    assert (
        "stabilising" in r.bioanalytical_recommendation.lower()
        or "stabiliz" in r.bioanalytical_recommendation.lower()
        or "additive" in r.bioanalytical_recommendation.lower()
    )


# ---------------------------------------------------------------------------
# Temperature effects
# ---------------------------------------------------------------------------


def test_high_temperature_increases_k():
    r_37 = plasma_result(temperature_C=37.0)
    r_50 = plasma_result(temperature_C=50.0)
    assert r_50.k_degradation_per_h > r_37.k_degradation_per_h


def test_low_temperature_decreases_k():
    """4°C storage should give much lower k than 37°C."""
    r_37 = plasma_result(temperature_C=37.0)
    r_4 = plasma_result(temperature_C=4.0)
    assert r_4.k_degradation_per_h < r_37.k_degradation_per_h


# ---------------------------------------------------------------------------
# screen_matrices
# ---------------------------------------------------------------------------


def test_screen_matrices_returns_5():
    results = screen_matrices("TestDrug")
    assert len(results) == 5


def test_screen_matrices_all_result_types():
    results = screen_matrices("TestDrug")
    for r in results:
        assert isinstance(r, MatrixStabilityResult)


def test_screen_matrices_sorted_ascending():
    """Most stable (lowest k) should be first."""
    results = screen_matrices("TestDrug", has_ester=True)
    k_values = [r.k_degradation_per_h for r in results]
    assert k_values == sorted(k_values)


def test_screen_matrices_covers_all_matrices():
    results = screen_matrices("TestDrug")
    matrices = {r.matrix for r in results}
    assert matrices == {"plasma", "whole_blood", "urine", "liver_s9", "brain_homogenate"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_matrix_raises():
    with pytest.raises(ValueError, match="matrix"):
        predict_matrix_stability("Drug", "lymph")


def test_invalid_temperature_high():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_matrix_stability("Drug", "plasma", temperature_C=65.0)


def test_invalid_temperature_negative():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_matrix_stability("Drug", "plasma", temperature_C=-5.0)


def test_invalid_base_k_zero():
    with pytest.raises(ValueError, match="base_k_per_h"):
        predict_matrix_stability("Drug", "plasma", base_k_per_h=0.0)


def test_notes_nonempty():
    r = plasma_result()
    assert len(r.notes) > 0
