"""
Tests for Phase 252: core/implant_pk.py
"""

import pytest

from omega_pbpk.core.implant_pk import (
    VALID_IMPLANT_TYPES,
    ImplantPKResult,
    compare_implant_types,
    simulate_implant_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_result():
    return simulate_implant_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        implant_type="reservoir",
        cl_L_per_h=2.0,
        vd_L=50.0,
        mic_threshold_mg_L=0.01,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type(default_result):
    assert isinstance(default_result, ImplantPKResult)


def test_times_list_non_empty(default_result):
    assert len(default_result.times_days) > 0


def test_c_plasma_list_same_length(default_result):
    assert len(default_result.c_plasma_mg_L) == len(default_result.times_days)


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------


def test_cmax_positive(default_result):
    assert default_result.cmax_plasma_mg_L > 0.0


def test_auc_positive(default_result):
    assert default_result.auc_plasma_mg_day_per_L > 0.0


def test_t50_before_t90(default_result):
    assert default_result.t50_release_days <= default_result.t90_release_days


def test_duration_therapeutic_non_negative(default_result):
    assert default_result.duration_therapeutic_days >= 0.0


def test_mean_residence_time_positive(default_result):
    assert default_result.mean_residence_time_days > 0.0


def test_tmax_within_simulation_time(default_result):
    assert 0.0 <= default_result.tmax_days <= 180.0


# ---------------------------------------------------------------------------
# Slow implant vs fast implant
# ---------------------------------------------------------------------------


def test_slower_implant_longer_duration():
    """reservoir (k=0.01) should sustain levels longer than hydrogel (k=0.05)."""
    slow = simulate_implant_pk(
        drug_name="Drug",
        dose_mg=100.0,
        implant_type="reservoir",
        cl_L_per_h=2.0,
        vd_L=50.0,
        mic_threshold_mg_L=0.005,
        t_end_days=365.0,
        dt_days=1.0,
    )
    fast = simulate_implant_pk(
        drug_name="Drug",
        dose_mg=100.0,
        implant_type="hydrogel",
        cl_L_per_h=2.0,
        vd_L=50.0,
        mic_threshold_mg_L=0.005,
        t_end_days=365.0,
        dt_days=1.0,
    )
    # reservoir releases slower → drug stays above MIC longer
    assert slow.duration_therapeutic_days >= fast.duration_therapeutic_days


def test_osmotic_pump_zero_burst():
    """osmotic_pump has burst_fraction=0.0 so initial plasma should be 0."""
    result = simulate_implant_pk(
        drug_name="Drug",
        dose_mg=100.0,
        implant_type="osmotic_pump",
        cl_L_per_h=2.0,
        vd_L=50.0,
    )
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_hydrogel_high_burst():
    """hydrogel has burst_fraction=0.15 → non-zero initial plasma concentration."""
    result = simulate_implant_pk(
        drug_name="Drug",
        dose_mg=100.0,
        implant_type="hydrogel",
        cl_L_per_h=2.0,
        vd_L=50.0,
    )
    expected_c0 = 100.0 * 0.15 / 50.0
    assert result.c_plasma_mg_L[0] == pytest.approx(expected_c0, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_implant_types
# ---------------------------------------------------------------------------


def test_compare_returns_five():
    results = compare_implant_types("TestDrug", dose_mg=100.0)
    assert len(results) == 5


def test_compare_sorted_descending():
    results = compare_implant_types("TestDrug", dose_mg=100.0)
    durations = [r.duration_therapeutic_days for r in results]
    assert durations == sorted(durations, reverse=True)


def test_compare_covers_all_types():
    results = compare_implant_types("TestDrug", dose_mg=100.0)
    returned_types = {r.implant_type for r in results}
    assert returned_types == set(VALID_IMPLANT_TYPES)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_implant_type_raises():
    with pytest.raises(ValueError, match="implant_type"):
        simulate_implant_pk("Drug", dose_mg=100.0, implant_type="unknown_implant")


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_implant_pk("Drug", dose_mg=0.0, implant_type="reservoir")


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_implant_pk("Drug", dose_mg=-50.0, implant_type="reservoir")


# ---------------------------------------------------------------------------
# Each implant type produces valid results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implant_type", VALID_IMPLANT_TYPES)
def test_each_implant_type_valid(implant_type):
    result = simulate_implant_pk(
        drug_name="GenericDrug",
        dose_mg=200.0,
        implant_type=implant_type,
        cl_L_per_h=3.0,
        vd_L=60.0,
        mic_threshold_mg_L=0.01,
    )
    assert isinstance(result, ImplantPKResult)
    assert result.cmax_plasma_mg_L > 0.0
    assert result.auc_plasma_mg_day_per_L > 0.0
    assert result.t50_release_days <= result.t90_release_days
    assert result.duration_therapeutic_days >= 0.0


# ---------------------------------------------------------------------------
# Higher dose → higher AUC
# ---------------------------------------------------------------------------


def test_higher_dose_higher_auc():
    low_dose = simulate_implant_pk("Drug", dose_mg=50.0, implant_type="matrix_rod")
    high_dose = simulate_implant_pk("Drug", dose_mg=200.0, implant_type="matrix_rod")
    assert high_dose.auc_plasma_mg_day_per_L > low_dose.auc_plasma_mg_day_per_L
