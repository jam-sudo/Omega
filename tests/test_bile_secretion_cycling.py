"""Tests for Phase 326 — Bile Secretion and Enterohepatic Cycling."""

import pytest

from omega_pbpk.core.bile_secretion_cycling import (
    EHCResult,
    simulate_ehc,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    dose_mg=100.0,
    cl_hepatic_L_per_h=5.0,
    vd_L=20.0,
    f_biliary=0.3,
    k_reabsorption_per_h=0.5,
    k_intestinal_loss_per_h=0.2,
    t_end_h=48.0,
    dt_h=0.05,
)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_ehc(**BASE_KWARGS)
    assert isinstance(result, EHCResult)


# ---------------------------------------------------------------------------
# 2. Array lengths consistent
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = simulate_ehc(**BASE_KWARGS)
    n = len(result.times_h)
    assert n == len(result.c_plasma_mg_L)
    assert n == len(result.c_bile_mg_L)
    assert n == len(result.c_intestine_mg_L)
    assert n >= 2


# ---------------------------------------------------------------------------
# 3. Times start at 0, end close to t_end_h
# ---------------------------------------------------------------------------


def test_times_start_at_zero():
    result = simulate_ehc(**BASE_KWARGS)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    result = simulate_ehc(**BASE_KWARGS)
    # Last recorded time should be close to t_end_h (within one extra step)
    assert result.times_h[-1] >= BASE_KWARGS["t_end_h"] - BASE_KWARGS["dt_h"]


# ---------------------------------------------------------------------------
# 4. Plasma starts at dose / vd
# ---------------------------------------------------------------------------


def test_plasma_initial_concentration():
    result = simulate_ehc(**BASE_KWARGS)
    expected_c0 = 100.0 / 20.0  # = 5.0 mg/L
    assert result.c_plasma_mg_L[0] == pytest.approx(expected_c0, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Plasma generally decreases (last value < initial value)
# ---------------------------------------------------------------------------


def test_plasma_generally_decreases():
    result = simulate_ehc(**BASE_KWARGS)
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# 6. Bile > 0 at some point
# ---------------------------------------------------------------------------


def test_bile_becomes_positive():
    result = simulate_ehc(**BASE_KWARGS)
    assert max(result.c_bile_mg_L) > 0.0


# ---------------------------------------------------------------------------
# 7. Intestine > 0 at some point
# ---------------------------------------------------------------------------


def test_intestine_becomes_positive():
    result = simulate_ehc(**BASE_KWARGS)
    assert max(result.c_intestine_mg_L) > 0.0


# ---------------------------------------------------------------------------
# 8. AUC plasma > 0
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_ehc(**BASE_KWARGS)
    assert result.auc_plasma_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 9. EHC extends effective half-life vs no recycling
# ---------------------------------------------------------------------------


def test_ehc_extends_half_life():
    result_with_ehc = simulate_ehc(**BASE_KWARGS)
    result_no_ehc = simulate_ehc(**{**BASE_KWARGS, "k_reabsorption_per_h": 0.0})
    assert result_with_ehc.t_half_effective_h > result_no_ehc.t_half_effective_h


# ---------------------------------------------------------------------------
# 10. Higher f_biliary → higher recycling_index
# ---------------------------------------------------------------------------


def test_higher_f_biliary_higher_recycling():
    result_low = simulate_ehc(**{**BASE_KWARGS, "f_biliary": 0.1})
    result_high = simulate_ehc(**{**BASE_KWARGS, "f_biliary": 0.8})
    assert result_high.recycling_index > result_low.recycling_index


# ---------------------------------------------------------------------------
# 11. recycling_index in [0, 100]
# ---------------------------------------------------------------------------


def test_recycling_index_range():
    result = simulate_ehc(**BASE_KWARGS)
    assert 0.0 <= result.recycling_index <= 100.0


def test_recycling_index_zero_when_no_reabsorption():
    result = simulate_ehc(**{**BASE_KWARGS, "k_reabsorption_per_h": 0.0})
    assert result.recycling_index == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 12. f_biliary=0 → no bile, no intestinal exposure
# ---------------------------------------------------------------------------


def test_no_biliary_secretion():
    result = simulate_ehc(**{**BASE_KWARGS, "f_biliary": 0.0, "k_reabsorption_per_h": 0.0})
    assert max(result.c_bile_mg_L) == pytest.approx(0.0, abs=1e-9)
    assert max(result.c_intestine_mg_L) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 13. cumulative_reabsorbed_mg >= 0
# ---------------------------------------------------------------------------


def test_cumulative_reabsorbed_non_negative():
    result = simulate_ehc(**BASE_KWARGS)
    assert result.cumulative_reabsorbed_mg >= 0.0


# ---------------------------------------------------------------------------
# 14. cumulative_fecal_mg >= 0
# ---------------------------------------------------------------------------


def test_cumulative_fecal_non_negative():
    result = simulate_ehc(**BASE_KWARGS)
    assert result.cumulative_fecal_mg >= 0.0


# ---------------------------------------------------------------------------
# 15. EHC with recycling has higher AUC than without
# ---------------------------------------------------------------------------


def test_ehc_higher_auc_than_no_recycling():
    result_ehc = simulate_ehc(**BASE_KWARGS)
    result_no = simulate_ehc(**{**BASE_KWARGS, "k_reabsorption_per_h": 0.0})
    assert result_ehc.auc_plasma_mg_h_per_L > result_no.auc_plasma_mg_h_per_L


# ---------------------------------------------------------------------------
# 16. t_half_effective_h > 0
# ---------------------------------------------------------------------------


def test_t_half_positive():
    result = simulate_ehc(**BASE_KWARGS)
    assert result.t_half_effective_h > 0.0


# ---------------------------------------------------------------------------
# 17. notes is a non-empty string
# ---------------------------------------------------------------------------


def test_notes_non_empty_string():
    result = simulate_ehc(**BASE_KWARGS)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 18. Validation: dose <= 0
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ehc(**{**BASE_KWARGS, "dose_mg": 0.0})


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ehc(**{**BASE_KWARGS, "dose_mg": -50.0})


# ---------------------------------------------------------------------------
# 19. Validation: cl <= 0
# ---------------------------------------------------------------------------


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        simulate_ehc(**{**BASE_KWARGS, "cl_hepatic_L_per_h": 0.0})


# ---------------------------------------------------------------------------
# 20. Validation: vd <= 0
# ---------------------------------------------------------------------------


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_ehc(**{**BASE_KWARGS, "vd_L": 0.0})


# ---------------------------------------------------------------------------
# 21. Validation: f_biliary out of range
# ---------------------------------------------------------------------------


def test_validation_f_biliary_negative():
    with pytest.raises(ValueError, match="f_biliary"):
        simulate_ehc(**{**BASE_KWARGS, "f_biliary": -0.1})


def test_validation_f_biliary_greater_than_one():
    with pytest.raises(ValueError, match="f_biliary"):
        simulate_ehc(**{**BASE_KWARGS, "f_biliary": 1.5})


# ---------------------------------------------------------------------------
# 22. f_biliary = 1.0 → maximum biliary routing (recycling_index maximized)
# ---------------------------------------------------------------------------


def test_f_biliary_one_maximum_recycling():
    result_max = simulate_ehc(**{**BASE_KWARGS, "f_biliary": 1.0})
    result_mid = simulate_ehc(**{**BASE_KWARGS, "f_biliary": 0.3})
    assert result_max.recycling_index >= result_mid.recycling_index
