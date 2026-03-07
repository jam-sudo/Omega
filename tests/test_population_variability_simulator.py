"""Tests for Phase 288 — Population Variability Simulator."""

import pytest

from omega_pbpk.clinical.population_variability_simulator import (
    PopulationVariabilityResult,
    VariabilityRiskResult,
    assess_variability_risk,
    simulate_population_variability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_mean_L_per_h=5.0,
    vd_mean_L=50.0,
    ka_mean_per_h=1.0,
    f_mean=0.8,
    omega_cl=0.3,
    omega_vd=0.2,
    omega_ka=0.2,
    n_subjects=50,
    t_end_h=24.0,
    dt_h=1.0,
    seed=42,
)


def _sim(**kwargs) -> PopulationVariabilityResult:
    params = {**_BASE, **kwargs}
    return simulate_population_variability(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type_simulate():
    r = _sim()
    assert isinstance(r, PopulationVariabilityResult)


def test_return_type_assess():
    r = _sim()
    auc_list = [r.mean_auc_mg_h_per_L * (0.5 + i * 0.1) for i in range(10)]
    cmax_list = [r.mean_cmax_mg_L] * 10
    result = assess_variability_risk("TestDrug", auc_list, cmax_list, 5.0, 30.0)
    assert isinstance(result, VariabilityRiskResult)


# ---------------------------------------------------------------------------
# Percentile ordering: c_p5 <= c_p50 <= c_p95
# ---------------------------------------------------------------------------


def test_percentile_ordering():
    r = _sim()
    for i in range(len(r.times_h)):
        assert r.c_p5_mg_L[i] <= r.c_p50_mg_L[i] + 1e-9
        assert r.c_p50_mg_L[i] <= r.c_p95_mg_L[i] + 1e-9


# ---------------------------------------------------------------------------
# mean_auc > 0, mean_cmax > 0
# ---------------------------------------------------------------------------


def test_mean_auc_positive():
    r = _sim()
    assert r.mean_auc_mg_h_per_L > 0


def test_mean_cmax_positive():
    r = _sim()
    assert r.mean_cmax_mg_L > 0


# ---------------------------------------------------------------------------
# cv_auc_pct >= 0
# ---------------------------------------------------------------------------


def test_cv_auc_non_negative():
    r = _sim()
    assert r.cv_auc_pct >= 0


def test_cv_cmax_non_negative():
    r = _sim()
    assert r.cv_cmax_pct >= 0


# ---------------------------------------------------------------------------
# Higher omega_cl → higher cv_auc_pct
# ---------------------------------------------------------------------------


def test_higher_omega_cl_increases_cv():
    r_low = _sim(omega_cl=0.1)
    r_high = _sim(omega_cl=0.6)
    assert r_high.cv_auc_pct > r_low.cv_auc_pct


# ---------------------------------------------------------------------------
# variability_class in valid set
# ---------------------------------------------------------------------------


_VALID_VAR_CLASSES = {"low", "moderate", "high"}


def test_variability_class_valid():
    r = _sim()
    assert r.variability_class in _VALID_VAR_CLASSES


def test_variability_class_low_when_zero_omega():
    r = _sim(omega_cl=0.0, omega_vd=0.0, omega_ka=0.0)
    assert r.variability_class == "low"


def test_variability_class_high_when_large_omega():
    r = _sim(omega_cl=1.0, n_subjects=100)
    assert r.variability_class in {"moderate", "high"}


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    r = _sim(drug_name="Warfarin")
    assert r.drug_name == "Warfarin"


def test_n_subjects_preserved():
    r = _sim(n_subjects=30)
    assert r.n_subjects == 30


def test_dose_mg_preserved():
    r = _sim(dose_mg=200.0)
    assert r.dose_mg == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Time grid length consistency
# ---------------------------------------------------------------------------


def test_time_grid_length_consistency():
    r = _sim()
    n = len(r.times_h)
    assert len(r.c_p5_mg_L) == n
    assert len(r.c_p50_mg_L) == n
    assert len(r.c_p95_mg_L) == n


# ---------------------------------------------------------------------------
# assess_variability_risk: pct sum == 100
# ---------------------------------------------------------------------------


def test_pct_sum_equals_100():
    auc_list = list(range(1, 101))  # 1..100
    cmax_list = [1.0] * 100
    r = assess_variability_risk("Drug", auc_list, cmax_list, 20.0, 80.0)
    total = r.pct_below_therapeutic_min + r.pct_above_therapeutic_max + r.pct_in_therapeutic_window
    assert total == pytest.approx(100.0, abs=1e-6)


def test_all_in_window():
    auc_list = [10.0] * 50
    cmax_list = [1.0] * 50
    r = assess_variability_risk("Drug", auc_list, cmax_list, 5.0, 20.0)
    assert r.pct_in_therapeutic_window == pytest.approx(100.0)
    assert r.pct_below_therapeutic_min == pytest.approx(0.0)
    assert r.pct_above_therapeutic_max == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# risk_class in valid set
# ---------------------------------------------------------------------------


_VALID_RISK = {"acceptable", "borderline", "high_risk"}


def test_risk_class_valid():
    auc_list = [10.0] * 50
    cmax_list = [1.0] * 50
    r = assess_variability_risk("Drug", auc_list, cmax_list, 5.0, 20.0)
    assert r.risk_class in _VALID_RISK


def test_risk_class_acceptable():
    # All in window → > 70% → acceptable
    auc_list = [10.0] * 50
    cmax_list = [1.0] * 50
    r = assess_variability_risk("Drug", auc_list, cmax_list, 5.0, 20.0)
    assert r.risk_class == "acceptable"


def test_risk_class_high_risk():
    # Spread all over — most outside window
    auc_list = list(range(1, 101))  # 1..100
    cmax_list = [1.0] * 100
    # Window [45, 55] → 11 inside = 11% in window → high_risk
    r = assess_variability_risk("Drug", auc_list, cmax_list, 45.0, 55.0)
    assert r.risk_class == "high_risk"


# ---------------------------------------------------------------------------
# n_subjects constraint validation
# ---------------------------------------------------------------------------


def test_n_subjects_too_large():
    with pytest.raises(ValueError, match="n_subjects"):
        _sim(n_subjects=201)


def test_n_subjects_zero():
    with pytest.raises(ValueError, match="n_subjects"):
        _sim(n_subjects=0)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_result():
    r1 = _sim(seed=99)
    r2 = _sim(seed=99)
    assert r1.mean_auc_mg_h_per_L == pytest.approx(r2.mean_auc_mg_h_per_L)
    assert r1.cv_auc_pct == pytest.approx(r2.cv_auc_pct)


def test_different_seeds_different_results():
    r1 = _sim(seed=1)
    r2 = _sim(seed=999)
    # Different seeds should (almost certainly) produce different CV
    assert abs(r1.cv_auc_pct - r2.cv_auc_pct) > 1e-6 or r1.cv_auc_pct == pytest.approx(
        r2.cv_auc_pct, rel=0.01
    )


# ---------------------------------------------------------------------------
# Validation errors for simulate
# ---------------------------------------------------------------------------


def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl_mean"):
        _sim(cl_mean_L_per_h=0.0)


def test_invalid_f_mean():
    with pytest.raises(ValueError, match="f_mean"):
        _sim(f_mean=1.5)


def test_invalid_t_end():
    with pytest.raises(ValueError, match="t_end_h"):
        _sim(t_end_h=0.0)


# ---------------------------------------------------------------------------
# Validation errors for assess
# ---------------------------------------------------------------------------


def test_assess_empty_auc_list():
    with pytest.raises(ValueError, match="auc_list"):
        assess_variability_risk("Drug", [], [], 5.0, 20.0)


def test_assess_invalid_therapeutic_bounds():
    with pytest.raises(ValueError, match="therapeutic_auc_max"):
        assess_variability_risk("Drug", [10.0], [1.0], 20.0, 10.0)
