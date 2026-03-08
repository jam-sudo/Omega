"""Tests for Phase 327 — Drug Stability in GI Fluid."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.gi_fluid_stability import (
    GIStabilityResult,
    simulate_gi_stability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_KWARGS = dict(
    drug_name="TestDrug",
    pKa=4.5,
    logP=1.5,
    drug_type="acid",
    k_deg_gastric_per_h=0.1,
    k_deg_intestinal_per_h=0.05,
    dose_mg=100.0,
    t_end_min=360.0,
    dt_min=1.0,
)


def _run(**overrides):
    kw = {**_VALID_KWARGS, **overrides}
    return simulate_gi_stability(**kw)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, GIStabilityResult)


# ---------------------------------------------------------------------------
# 2. Array lengths are consistent
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = _run()
    assert len(result.times_min) == len(result.remaining_fraction)


# ---------------------------------------------------------------------------
# 3. Arrays are non-empty
# ---------------------------------------------------------------------------


def test_arrays_non_empty():
    result = _run()
    assert len(result.times_min) > 0


# ---------------------------------------------------------------------------
# 4. Remaining fraction starts at 1.0
# ---------------------------------------------------------------------------


def test_remaining_fraction_starts_at_one():
    result = _run()
    assert result.remaining_fraction[0] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Remaining fraction is monotonically non-increasing
# ---------------------------------------------------------------------------


def test_remaining_fraction_non_increasing():
    result = _run()
    for i in range(1, len(result.remaining_fraction)):
        assert result.remaining_fraction[i] <= result.remaining_fraction[i - 1] + 1e-12


# ---------------------------------------------------------------------------
# 6. Remaining fraction always in [0, 1]
# ---------------------------------------------------------------------------


def test_remaining_fraction_bounds():
    result = _run()
    for frac in result.remaining_fraction:
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# 7. First time point is 0
# ---------------------------------------------------------------------------


def test_first_time_is_zero():
    result = _run()
    assert result.times_min[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Higher k_deg_gastric → lower fraction at absorption site
# ---------------------------------------------------------------------------


def test_higher_gastric_kdeg_lower_fraction():
    r_low = _run(k_deg_gastric_per_h=0.05)
    r_high = _run(k_deg_gastric_per_h=2.0)
    assert r_high.fraction_reaching_absorption_site < r_low.fraction_reaching_absorption_site


# ---------------------------------------------------------------------------
# 9. fraction_reaching_absorption_site in [0, 1]
# ---------------------------------------------------------------------------


def test_fraction_reaching_bounds():
    result = _run()
    assert 0.0 <= result.fraction_reaching_absorption_site <= 1.0


# ---------------------------------------------------------------------------
# 10. stability_class is a valid string
# ---------------------------------------------------------------------------


def test_stability_class_valid():
    valid = {"stable", "labile", "highly_labile"}
    result = _run()
    assert result.stability_class in valid


# ---------------------------------------------------------------------------
# 11. Stable drug produces "stable" classification
# ---------------------------------------------------------------------------


def test_stable_drug_classification():
    # Very small degradation → fraction > 0.8
    result = _run(k_deg_gastric_per_h=0.001, k_deg_intestinal_per_h=0.001)
    assert result.stability_class == "stable"
    assert result.fraction_reaching_absorption_site > 0.8


# ---------------------------------------------------------------------------
# 12. Highly labile drug produces "highly_labile" classification
# ---------------------------------------------------------------------------


def test_highly_labile_drug_classification():
    # Large degradation → fraction < 0.5
    result = _run(k_deg_gastric_per_h=20.0, k_deg_intestinal_per_h=0.1)
    assert result.stability_class == "highly_labile"
    assert result.fraction_reaching_absorption_site < 0.5


# ---------------------------------------------------------------------------
# 13. Labile drug produces "labile" classification
# ---------------------------------------------------------------------------


def test_labile_drug_classification():
    # Choose a rate that gives fraction between 0.5 and 0.8 at 120 min
    # k_gastric_per_min = k/60; fraction = exp(-k_gastric_per_min * 120)
    # We want exp(-k * 2) ≈ 0.6  → k ≈ 0.255 h^-1
    result = _run(k_deg_gastric_per_h=0.255, k_deg_intestinal_per_h=0.05)
    assert result.stability_class == "labile"


# ---------------------------------------------------------------------------
# 14. Zero degradation → fraction_reaching == 1.0
# ---------------------------------------------------------------------------


def test_zero_degradation_full_fraction():
    result = _run(k_deg_gastric_per_h=0.0, k_deg_intestinal_per_h=0.0)
    assert result.fraction_reaching_absorption_site == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 15. gastric_t50_min is finite for non-zero degradation
# ---------------------------------------------------------------------------


def test_gastric_t50_finite():
    result = _run(k_deg_gastric_per_h=1.0)
    assert math.isfinite(result.gastric_t50_min)
    assert result.gastric_t50_min > 0


# ---------------------------------------------------------------------------
# 16. gastric_t50_min is inf for zero gastric degradation
# ---------------------------------------------------------------------------


def test_gastric_t50_infinite():
    result = _run(k_deg_gastric_per_h=0.0)
    assert math.isinf(result.gastric_t50_min)


# ---------------------------------------------------------------------------
# 17. intestinal_t50_min is inf for zero intestinal degradation
# ---------------------------------------------------------------------------


def test_intestinal_t50_infinite():
    result = _run(k_deg_intestinal_per_h=0.0)
    assert math.isinf(result.intestinal_t50_min)


# ---------------------------------------------------------------------------
# 18. notes is a non-empty string
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = _run()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 19. Validation: k_deg_gastric_per_h < 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_negative_kdeg_gastric():
    with pytest.raises(ValueError, match="k_deg_gastric"):
        _run(k_deg_gastric_per_h=-0.1)


# ---------------------------------------------------------------------------
# 20. Validation: k_deg_intestinal_per_h < 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_negative_kdeg_intestinal():
    with pytest.raises(ValueError, match="k_deg_intestinal"):
        _run(k_deg_intestinal_per_h=-0.5)


# ---------------------------------------------------------------------------
# 21. Validation: dose_mg <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


# ---------------------------------------------------------------------------
# 22. Validation: t_end_min <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_zero_t_end():
    with pytest.raises(ValueError, match="t_end_min"):
        _run(t_end_min=0.0)
