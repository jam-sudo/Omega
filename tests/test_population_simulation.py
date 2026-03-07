"""Tests for Monte Carlo population PK simulation (Phase 186)."""

import math

import pytest

from omega_pbpk.clinical.population_simulation import (
    PopulationSimResult,
    simulate_population_pk,
    summarize_population_pk,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_population_sim_result():
    result = simulate_population_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_typical_L_per_h=5.0,
        vd_typical_L=50.0,
    )
    assert isinstance(result, PopulationSimResult)


def test_frozen_dataclass():
    result = simulate_population_pk("Drug", 100.0, 5.0, 50.0)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_drug_name_stored():
    result = simulate_population_pk("Ibuprofen", 400.0, 4.0, 40.0)
    assert result.drug_name == "Ibuprofen"


def test_dose_mg_stored():
    result = simulate_population_pk("Drug", 200.0, 5.0, 50.0)
    assert math.isclose(result.dose_mg, 200.0)


# ---------------------------------------------------------------------------
# IV route — zero BSV analytical checks
# ---------------------------------------------------------------------------


def test_iv_zero_bsv_cmax_equals_dose_over_vd():
    result = simulate_population_pk(
        drug_name="Drug",
        dose_mg=100.0,
        cl_typical_L_per_h=5.0,
        vd_typical_L=50.0,
        route="iv",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=10,
    )
    expected_cmax = 100.0 / 50.0  # 2 mg/L
    assert math.isclose(result.cmax_mean, expected_cmax, rel_tol=1e-6)


def test_iv_zero_bsv_auc_equals_dose_over_cl():
    result = simulate_population_pk(
        drug_name="Drug",
        dose_mg=100.0,
        cl_typical_L_per_h=5.0,
        vd_typical_L=50.0,
        route="iv",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=10,
    )
    expected_auc = 100.0 / 5.0  # 20 mg*h/L
    assert math.isclose(result.auc_mean, expected_auc, rel_tol=1e-6)


def test_iv_zero_bsv_cv_near_zero():
    result = simulate_population_pk(
        drug_name="Drug",
        dose_mg=100.0,
        cl_typical_L_per_h=5.0,
        vd_typical_L=50.0,
        route="iv",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=50,
    )
    assert result.cmax_cv_pct < 1e-6


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


def test_oral_route_stored():
    result = simulate_population_pk("Drug", 100.0, 5.0, 50.0, route="oral")
    assert result.route == "oral"


def test_oral_auc_scales_with_bioavailability():
    r1 = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="oral",
        f_oral=1.0,
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=10,
    )
    r2 = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="oral",
        f_oral=0.5,
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=10,
    )
    assert math.isclose(r1.auc_mean / r2.auc_mean, 2.0, rel_tol=1e-6)


def test_oral_cmax_positive():
    result = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="oral",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=10,
    )
    assert result.cmax_mean > 0


# ---------------------------------------------------------------------------
# Linear PK: 2x dose → 2x Cmax mean
# ---------------------------------------------------------------------------


def test_linear_pk_dose_proportionality_iv():
    r1 = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="iv",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=20,
    )
    r2 = simulate_population_pk(
        "Drug",
        200.0,
        5.0,
        50.0,
        route="iv",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=20,
    )
    assert math.isclose(r2.cmax_mean / r1.cmax_mean, 2.0, rel_tol=1e-6)


def test_linear_pk_dose_proportionality_oral_auc():
    r1 = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="oral",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=20,
    )
    r2 = simulate_population_pk(
        "Drug",
        200.0,
        5.0,
        50.0,
        route="oral",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=20,
    )
    assert math.isclose(r2.auc_mean / r1.auc_mean, 2.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# BSV / variability
# ---------------------------------------------------------------------------


def test_bsv_cv_approximately_omega_cl(seed=42):
    """For large n, CL CV% ≈ omega_cl * 100 (log-normal approx for small omega)."""
    omega_cl = 0.3
    result = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="iv",
        omega_cl=omega_cl,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=1000,
        seed=seed,
    )
    # AUC CV is driven by CL BSV for IV; accept wide tolerance
    assert 15 < result.auc_cv_pct < 45  # roughly 30% ± 15


def test_higher_omega_increases_cv():
    r1 = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="iv",
        omega_cl=0.1,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=500,
        seed=0,
    )
    r2 = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="iv",
        omega_cl=0.5,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=500,
        seed=0,
    )
    assert r2.auc_cv_pct > r1.auc_cv_pct


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_result():
    r1 = simulate_population_pk("Drug", 100.0, 5.0, 50.0, seed=123)
    r2 = simulate_population_pk("Drug", 100.0, 5.0, 50.0, seed=123)
    assert math.isclose(r1.cmax_mean, r2.cmax_mean)
    assert math.isclose(r1.auc_mean, r2.auc_mean)


def test_different_seed_different_result():
    r1 = simulate_population_pk("Drug", 100.0, 5.0, 50.0, seed=1)
    r2 = simulate_population_pk("Drug", 100.0, 5.0, 50.0, seed=2)
    assert not math.isclose(r1.cmax_mean, r2.cmax_mean, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# n_subjects
# ---------------------------------------------------------------------------


def test_n_subjects_stored():
    result = simulate_population_pk("Drug", 100.0, 5.0, 50.0, n_subjects=250)
    assert result.n_subjects == 250


def test_single_subject():
    result = simulate_population_pk("Drug", 100.0, 5.0, 50.0, n_subjects=1)
    assert result.n_subjects == 1
    assert result.cmax_mean > 0


# ---------------------------------------------------------------------------
# Percentile ordering
# ---------------------------------------------------------------------------


def test_percentile_ordering_cmax():
    result = simulate_population_pk("Drug", 100.0, 5.0, 50.0, n_subjects=200)
    assert result.cmax_p5 <= result.cmax_mean <= result.cmax_p95


def test_percentile_ordering_auc():
    result = simulate_population_pk("Drug", 100.0, 5.0, 50.0, n_subjects=200)
    assert result.auc_p5 <= result.auc_mean <= result.auc_p95


# ---------------------------------------------------------------------------
# t_half consistency
# ---------------------------------------------------------------------------


def test_t_half_mean_consistent_with_typical():
    """t_half ≈ 0.693 * Vd / CL with zero BSV."""
    result = simulate_population_pk(
        "Drug",
        100.0,
        5.0,
        50.0,
        route="iv",
        omega_cl=0.0,
        omega_vd=0.0,
        omega_ka=0.0,
        n_subjects=10,
    )
    expected = 0.693 * 50.0 / 5.0
    assert math.isclose(result.t_half_mean_h, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# summarize_population_pk
# ---------------------------------------------------------------------------


def test_summarize_basic():
    subjects = [{"cmax": 2.0, "auc": 20.0}] * 10
    result = summarize_population_pk(subjects)
    assert isinstance(result, PopulationSimResult)
    assert math.isclose(result.cmax_mean, 2.0)
    assert math.isclose(result.auc_mean, 20.0)


def test_summarize_n_subjects():
    subjects = [{"cmax": float(i), "auc": float(i * 2)} for i in range(1, 51)]
    result = summarize_population_pk(subjects)
    assert result.n_subjects == 50


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize_population_pk([])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_n_subjects_zero():
    with pytest.raises(ValueError, match="n_subjects"):
        simulate_population_pk("Drug", 100.0, 5.0, 50.0, n_subjects=0)


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_population_pk("Drug", 0.0, 5.0, 50.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_population_pk("Drug", -10.0, 5.0, 50.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_typical"):
        simulate_population_pk("Drug", 100.0, 0.0, 50.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_typical"):
        simulate_population_pk("Drug", 100.0, 5.0, 0.0)


def test_invalid_omega_negative():
    with pytest.raises(ValueError, match="omega_cl"):
        simulate_population_pk("Drug", 100.0, 5.0, 50.0, omega_cl=-0.1)
