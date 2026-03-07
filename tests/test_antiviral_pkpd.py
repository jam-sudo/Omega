"""Tests for antiviral PK/PD module (Phase 164)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.antiviral_pkpd import (
    AntiviralPKPDResult,
    dose_response_antiviral,
    simulate_antiviral_pkpd,
)

# Common parameters for a drug with reasonable PK and strong antiviral effect
_COMMON = dict(
    drug_name="Remdesivir",
    dose_mg=200.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
)


def _run(**overrides):
    params = {**_COMMON, **overrides}
    return simulate_antiviral_pkpd(**params)


# ---------- 1. Basic simulation ----------


def test_basic_simulation_returns_result():
    result = _run()
    assert isinstance(result, AntiviralPKPDResult)


def test_result_fields_populated():
    result = _run()
    assert result.drug_name == "Remdesivir"
    assert result.dose_mg == 200.0
    assert result.route == "iv"
    assert len(result.times_h) > 0
    assert len(result.c_drug_mg_L) > 0
    assert len(result.viral_load_log10) > 0
    assert result.cmax_mg_L is not None
    assert result.auc_mg_h_L is not None
    assert result.baseline_viral_load_log10 is not None
    assert result.min_viral_load_log10 is not None
    assert result.max_viral_reduction_log10 is not None
    assert result.viral_eradication is not None
    assert result.notes is not None


def test_times_length_matches():
    result = _run()
    assert len(result.times_h) == len(result.c_drug_mg_L)
    assert len(result.times_h) == len(result.viral_load_log10)


# ---------- 4-11. Validation ----------


def test_negative_dose_raises():
    with pytest.raises(ValueError):
        _run(dose_mg=-10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError):
        _run(dose_mg=0.0)


def test_negative_clearance_raises():
    with pytest.raises(ValueError):
        _run(cl_L_per_h=-1.0)


def test_negative_vd_raises():
    with pytest.raises(ValueError):
        _run(vd_L=-10.0)


def test_negative_ec50_raises():
    with pytest.raises(ValueError):
        _run(ec50_mg_L=-0.5)


def test_invalid_route_raises():
    with pytest.raises(ValueError):
        _run(route="im")


def test_invalid_emax_raises():
    with pytest.raises(ValueError):
        _run(emax_antiviral=1.5)
    with pytest.raises(ValueError):
        _run(emax_antiviral=0.0)
    with pytest.raises(ValueError):
        _run(emax_antiviral=-0.1)


def test_invalid_hill_raises():
    with pytest.raises(ValueError):
        _run(hill_n=0.0)
    with pytest.raises(ValueError):
        _run(hill_n=-1.0)


# ---------- 12-13. Viral dynamics ----------


def test_no_drug_viral_load_increases():
    """With negligible dose, viral load should increase when growth > clearance."""
    result = _run(
        dose_mg=0.0001,
        viral_growth_rate_per_h=0.1,
        viral_clearance_per_h_baseline=0.05,
        baseline_viral_load_log10=6.0,
        t_end_h=168.0,
    )
    # Final viral load should be higher than baseline
    final_vl = result.viral_load_log10[-1]
    assert final_vl > result.baseline_viral_load_log10


def test_with_drug_above_ec50_viral_decreases():
    """Adequate dose should reduce viral load below baseline."""
    result = _run(
        dose_mg=500.0,
        ec50_mg_L=0.5,
        emax_antiviral=0.99,
        viral_growth_rate_per_h=0.1,
        viral_clearance_per_h_baseline=0.05,
    )
    assert result.min_viral_load_log10 < result.baseline_viral_load_log10


# ---------- 14-15. Dose comparison / IV vs oral ----------


def test_higher_dose_greater_reduction():
    low = _run(dose_mg=50.0)
    high = _run(dose_mg=500.0)
    assert high.max_viral_reduction_log10 >= low.max_viral_reduction_log10


def test_iv_vs_oral_same_dose():
    """IV route should yield higher Cmax than oral for same dose."""
    iv = _run(dose_mg=200.0, route="iv")
    oral = _run(dose_mg=200.0, route="oral", ka_per_h=1.0, f_oral=0.8)
    assert iv.cmax_mg_L > oral.cmax_mg_L


def test_oral_route_works():
    result = _run(route="oral", ka_per_h=1.0, f_oral=0.8)
    assert result.route == "oral"
    assert len(result.times_h) > 0


# ---------- 17. Dose proportionality ----------


def test_dose_proportionality_auc():
    """Linear PK: doubling dose should approximately double AUC."""
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    ratio = r2.auc_mg_h_L / r1.auc_mg_h_L
    assert 1.8 < ratio < 2.2


# ---------- 18-20. Eradication and time to reduction ----------


def test_eradication_flag_works():
    """Very high dose should achieve viral eradication."""
    result = _run(
        dose_mg=50000.0,
        cl_L_per_h=1.0,
        vd_L=10.0,
        ec50_mg_L=0.01,
        emax_antiviral=0.99,
        viral_growth_rate_per_h=0.05,
        viral_clearance_per_h_baseline=0.1,
        t_end_h=336.0,
    )
    assert result.viral_eradication is True


def test_time_to_1_log_reduction():
    """With adequate dose, time to 1 log reduction should be a positive number."""
    result = _run(
        dose_mg=5000.0,
        cl_L_per_h=1.0,
        vd_L=10.0,
        ec50_mg_L=0.1,
        emax_antiviral=0.99,
        viral_growth_rate_per_h=0.05,
        viral_clearance_per_h_baseline=0.1,
    )
    assert result.time_to_1_log_reduction_h is not None
    assert result.time_to_1_log_reduction_h > 0.0


def test_time_to_1_log_reduction_none():
    """Very low dose should not achieve 1 log reduction."""
    result = _run(
        dose_mg=0.001,
        ec50_mg_L=10.0,
        emax_antiviral=0.5,
        viral_growth_rate_per_h=0.1,
        viral_clearance_per_h_baseline=0.05,
        t_end_h=48.0,
    )
    assert result.time_to_1_log_reduction_h is None


# ---------- 21-23. dose_response_antiviral ----------


def test_dose_response_returns_list():
    results = dose_response_antiviral("Remdesivir", [100.0, 200.0, 400.0], 5.0, 50.0)
    assert len(results) == 3
    assert all(isinstance(r, AntiviralPKPDResult) for r in results)


def test_dose_response_sorted():
    results = dose_response_antiviral("Remdesivir", [400.0, 100.0, 200.0], 5.0, 50.0)
    doses = [r.dose_mg for r in results]
    assert doses == sorted(doses)


def test_dose_response_empty_raises():
    with pytest.raises(ValueError):
        dose_response_antiviral("Remdesivir", [], 5.0, 50.0)


# ---------- 24-27. Additional metrics ----------


def test_cmax_positive():
    result = _run()
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = _run()
    assert result.auc_mg_h_L > 0.0


def test_viral_load_non_negative_log_concept():
    """All viral load log10 values should be finite."""
    import math

    result = _run()
    assert all(math.isfinite(v) for v in result.viral_load_log10)


def test_baseline_viral_load_correct():
    result = _run(baseline_viral_load_log10=5.0)
    assert result.baseline_viral_load_log10 == 5.0
