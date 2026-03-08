"""Tests for Phase 328 — Drug Degradation in Plasma."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.plasma_drug_degradation import (
    PlasmaDegradationResult,
    simulate_plasma_degradation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    vd_L=50.0,
    cl_L_per_h=5.0,
    k_deg_plasma_per_h=0.2,
    t_end_h=24.0,
    dt_h=0.1,
)


def _run(**overrides):
    kw = {**_VALID_KWARGS, **overrides}
    return simulate_plasma_degradation(**kw)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, PlasmaDegradationResult)


# ---------------------------------------------------------------------------
# 2. Array lengths are consistent
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = _run()
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_plasma_nodeg_mg_L)


# ---------------------------------------------------------------------------
# 3. Arrays are non-empty
# ---------------------------------------------------------------------------


def test_arrays_non_empty():
    result = _run()
    assert len(result.times_h) > 0


# ---------------------------------------------------------------------------
# 4. First time point is 0
# ---------------------------------------------------------------------------


def test_first_time_is_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. AUC with degradation <= AUC without degradation
# ---------------------------------------------------------------------------


def test_auc_with_deg_le_without():
    result = _run()
    assert result.auc_with_deg_mg_h_per_L <= result.auc_without_deg_mg_h_per_L + 1e-9


# ---------------------------------------------------------------------------
# 6. t_half with degradation < t_half without (when k_deg > 0)
# ---------------------------------------------------------------------------


def test_t_half_shorter_with_degradation():
    result = _run(k_deg_plasma_per_h=0.5)
    assert result.t_half_with_deg_h < result.t_half_without_deg_h


# ---------------------------------------------------------------------------
# 7. Zero degradation: auc_ratio ≈ 1.0
# ---------------------------------------------------------------------------


def test_zero_degradation_auc_ratio():
    result = _run(k_deg_plasma_per_h=0.0)
    assert result.auc_ratio == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# 8. Zero degradation: degradation_class == "negligible"
# ---------------------------------------------------------------------------


def test_zero_degradation_class_negligible():
    result = _run(k_deg_plasma_per_h=0.0)
    assert result.degradation_class == "negligible"


# ---------------------------------------------------------------------------
# 9. High degradation: auc_ratio < 1.0
# ---------------------------------------------------------------------------


def test_high_degradation_auc_ratio():
    result = _run(k_deg_plasma_per_h=5.0)
    assert result.auc_ratio < 1.0


# ---------------------------------------------------------------------------
# 10. High degradation: degradation_class == "significant"
# ---------------------------------------------------------------------------


def test_high_degradation_class():
    result = _run(k_deg_plasma_per_h=5.0)
    assert result.degradation_class == "significant"


# ---------------------------------------------------------------------------
# 11. Minor degradation class
# ---------------------------------------------------------------------------


def test_minor_degradation_class():
    # CL/Vd = 5/50 = 0.1 h^-1; add k_deg that contributes 5-20%
    # deg_pct = k_deg / (k_elim + k_deg) * 100
    # For ~10%: k_deg = 0.1 * 0.1111 ≈ 0.011 h^-1
    # k_elim=0.1; k_deg=0.008; pct = 0.008/0.108*100 = 7.4% → minor
    result3 = _run(cl_L_per_h=5.0, vd_L=50.0, k_deg_plasma_per_h=0.008)
    assert result3.degradation_class == "minor"


# ---------------------------------------------------------------------------
# 12. deg_contribution_pct in [0, 100]
# ---------------------------------------------------------------------------


def test_deg_contribution_pct_bounds():
    for k_deg in [0.0, 0.1, 1.0, 10.0]:
        result = _run(k_deg_plasma_per_h=k_deg)
        assert 0.0 <= result.deg_contribution_pct <= 100.0


# ---------------------------------------------------------------------------
# 13. deg_contribution_pct == 0 when k_deg == 0
# ---------------------------------------------------------------------------


def test_deg_contribution_zero():
    result = _run(k_deg_plasma_per_h=0.0)
    assert result.deg_contribution_pct == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 14. degradation_class is a valid string
# ---------------------------------------------------------------------------


def test_degradation_class_valid():
    valid = {"negligible", "minor", "significant"}
    result = _run()
    assert result.degradation_class in valid


# ---------------------------------------------------------------------------
# 15. concentrations are non-negative
# ---------------------------------------------------------------------------


def test_concentrations_non_negative():
    result = _run()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)
    assert all(c >= 0.0 for c in result.c_plasma_nodeg_mg_L)


# ---------------------------------------------------------------------------
# 16. c_plasma_mg_L[0] ≈ dose/Vd (initial condition)
# ---------------------------------------------------------------------------


def test_initial_concentration():
    dose, vd = 200.0, 40.0
    result = _run(dose_mg=dose, vd_L=vd)
    expected_c0 = dose / vd
    assert result.c_plasma_mg_L[0] == pytest.approx(expected_c0, rel=1e-6)


# ---------------------------------------------------------------------------
# 17. notes is a non-empty string
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = _run()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 18. t_half_without_deg_h matches analytical ln(2)/k_elim
# ---------------------------------------------------------------------------


def test_t_half_without_analytical():
    cl, vd = 4.0, 20.0
    k_elim = cl / vd  # 0.2 h^-1
    expected = math.log(2.0) / k_elim
    result = _run(cl_L_per_h=cl, vd_L=vd, k_deg_plasma_per_h=0.0)
    assert result.t_half_without_deg_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 19. Validation: dose_mg <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


# ---------------------------------------------------------------------------
# 20. Validation: vd_L <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_zero_vd():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=0.0)


# ---------------------------------------------------------------------------
# 21. Validation: cl_L_per_h <= 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_zero_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=0.0)


# ---------------------------------------------------------------------------
# 22. Validation: k_deg_plasma_per_h < 0 raises ValueError
# ---------------------------------------------------------------------------


def test_validation_negative_kdeg():
    with pytest.raises(ValueError, match="k_deg_plasma_per_h"):
        _run(k_deg_plasma_per_h=-0.1)
