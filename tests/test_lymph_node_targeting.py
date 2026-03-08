"""Tests for Phase 330 — Lymph Node Drug Targeting Model."""

import pytest

from omega_pbpk.core.lymph_node_targeting import (
    LymphNodeTargetingResult,
    simulate_lymph_node_targeting,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    "drug_name": "ImmunoTherapy",
    "dose_mg": 1.0,
    "k_lymph_per_h": 0.5,
    "k_clearance_ln_per_h": 0.1,
    "k_systemic_per_h": 0.2,
    "cl_sys_L_per_h": 1.0,
    "vd_sys_L": 5.0,
    "v_ln_mL": 0.5,
    "t_end_h": 48.0,
    "dt_h": 0.05,
}


def run_base(**overrides) -> LymphNodeTargetingResult:
    params = {**BASE_PARAMS, **overrides}
    return simulate_lymph_node_targeting(**params)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = run_base()
    assert isinstance(result, LymphNodeTargetingResult)


# ---------------------------------------------------------------------------
# 2. Array lengths consistency
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = run_base()
    n = len(result.times_h)
    assert n > 0
    assert len(result.c_injection_site_mg_mL) == n
    assert len(result.c_lymph_node_mg_mL) == n
    assert len(result.c_systemic_mg_L) == n


# ---------------------------------------------------------------------------
# 3. Injection site starts high (dose / 0.5 mL)
# ---------------------------------------------------------------------------


def test_injection_site_starts_high():
    result = run_base(dose_mg=1.0)
    expected = 1.0 / 0.5  # 2 mg/mL
    assert abs(result.c_injection_site_mg_mL[0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# 4. Injection site decreases overall
# ---------------------------------------------------------------------------


def test_injection_site_decreases():
    result = run_base()
    assert result.c_injection_site_mg_mL[-1] < result.c_injection_site_mg_mL[0]


# ---------------------------------------------------------------------------
# 5. Lymph node starts at 0
# ---------------------------------------------------------------------------


def test_lymph_node_starts_at_zero():
    result = run_base()
    assert result.c_lymph_node_mg_mL[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Lymph node rises (achieves nonzero peak)
# ---------------------------------------------------------------------------


def test_lymph_node_rises():
    result = run_base()
    assert result.cmax_lymph_node_mg_mL > 0.0


# ---------------------------------------------------------------------------
# 7. tmax_lymph_node > 0
# ---------------------------------------------------------------------------


def test_tmax_lymph_node_greater_than_zero():
    result = run_base()
    assert result.tmax_lymph_node_h > 0.0


# ---------------------------------------------------------------------------
# 8. cmax_lymph_node > 0
# ---------------------------------------------------------------------------


def test_cmax_lymph_node_positive():
    result = run_base()
    assert result.cmax_lymph_node_mg_mL > 0.0


# ---------------------------------------------------------------------------
# 9. targeting_efficiency_pct in [0, 100]
# ---------------------------------------------------------------------------


def test_targeting_efficiency_in_range():
    result = run_base()
    assert 0.0 <= result.targeting_efficiency_pct <= 100.0


# ---------------------------------------------------------------------------
# 10. AUC lymph node > 0
# ---------------------------------------------------------------------------


def test_auc_lymph_node_positive():
    result = run_base()
    assert result.auc_lymph_node_mg_h_per_mL > 0.0


# ---------------------------------------------------------------------------
# 11. AUC systemic > 0
# ---------------------------------------------------------------------------


def test_auc_systemic_positive():
    result = run_base()
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# 12. Systemic starts at 0
# ---------------------------------------------------------------------------


def test_systemic_starts_at_zero():
    result = run_base()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 13. Higher k_lymph -> faster lymph node uptake (smaller tmax_ln)
# ---------------------------------------------------------------------------


def test_higher_k_lymph_faster_tmax():
    result_low = run_base(k_lymph_per_h=0.1)
    result_high = run_base(k_lymph_per_h=2.0)
    assert result_high.tmax_lymph_node_h <= result_low.tmax_lymph_node_h


# ---------------------------------------------------------------------------
# 14. Higher k_lymph -> higher cmax_ln (more drug delivered faster)
# ---------------------------------------------------------------------------


def test_higher_k_lymph_higher_cmax_ln():
    result_low = run_base(k_lymph_per_h=0.1)
    result_high = run_base(k_lymph_per_h=3.0)
    assert result_high.cmax_lymph_node_mg_mL >= result_low.cmax_lymph_node_mg_mL


# ---------------------------------------------------------------------------
# 15. Dose linearity: 2x dose -> 2x AUC lymph node
# ---------------------------------------------------------------------------


def test_dose_linearity_auc_ln():
    result_1 = run_base(dose_mg=1.0)
    result_2 = run_base(dose_mg=2.0)
    ratio = result_2.auc_lymph_node_mg_h_per_mL / result_1.auc_lymph_node_mg_h_per_mL
    assert abs(ratio - 2.0) < 0.05


# ---------------------------------------------------------------------------
# 16. Dose linearity: 2x dose -> 2x AUC systemic
# ---------------------------------------------------------------------------


def test_dose_linearity_auc_sys():
    result_1 = run_base(dose_mg=1.0)
    result_2 = run_base(dose_mg=2.0)
    ratio = result_2.auc_systemic_mg_h_per_L / result_1.auc_systemic_mg_h_per_L
    assert abs(ratio - 2.0) < 0.05


# ---------------------------------------------------------------------------
# 17. notes is a non-empty string
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = run_base()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 18. times_h starts at 0
# ---------------------------------------------------------------------------


def test_times_starts_at_zero():
    result = run_base()
    assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 19. times_h ends approximately at t_end_h
# ---------------------------------------------------------------------------


def test_times_ends_at_t_end():
    result = run_base(t_end_h=24.0, dt_h=0.05)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.1)


# ---------------------------------------------------------------------------
# 20. Validation: dose <= 0
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=-5.0)


# ---------------------------------------------------------------------------
# 21. Validation: cl <= 0
# ---------------------------------------------------------------------------


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        run_base(cl_sys_L_per_h=0.0)


# ---------------------------------------------------------------------------
# 22. Validation: vd <= 0
# ---------------------------------------------------------------------------


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        run_base(vd_sys_L=0.0)


# ---------------------------------------------------------------------------
# 23. Validation: k_lymph <= 0
# ---------------------------------------------------------------------------


def test_validation_k_lymph_zero():
    with pytest.raises(ValueError, match="k_lymph"):
        run_base(k_lymph_per_h=0.0)


def test_validation_k_lymph_negative():
    with pytest.raises(ValueError, match="k_lymph"):
        run_base(k_lymph_per_h=-1.0)
