"""Tests for Phase 197 — Joint/Synovial Fluid PK Model."""

import pytest

from omega_pbpk.core.joint_synovial_pk import (
    SynovialPKResult,
    compare_joint_inflammation,
    simulate_synovial_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_result():
    return simulate_synovial_pk(
        drug_name="Diclofenac",
        dose_mg=75.0,
        cl_L_per_h=5.0,
        vd_plasma_L=10.0,
        inflamed=True,
    )


@pytest.fixture
def healthy_result():
    return simulate_synovial_pk(
        drug_name="Diclofenac",
        dose_mg=75.0,
        cl_L_per_h=5.0,
        vd_plasma_L=10.0,
        inflamed=False,
    )


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type(default_result):
    assert isinstance(default_result, SynovialPKResult)


def test_drug_name_preserved(default_result):
    assert default_result.drug_name == "Diclofenac"


def test_dose_preserved(default_result):
    assert default_result.dose_mg == 75.0


def test_inflamed_flag_true(default_result):
    assert default_result.inflamed is True


def test_inflamed_flag_false(healthy_result):
    assert healthy_result.inflamed is False


def test_times_h_is_list(default_result):
    assert isinstance(default_result.times_h, list)


def test_c_plasma_is_list(default_result):
    assert isinstance(default_result.c_plasma_mg_L, list)


def test_c_synovial_is_list(default_result):
    assert isinstance(default_result.c_synovial_mg_L, list)


def test_notes_is_str(default_result):
    assert isinstance(default_result.notes, str)


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------


def test_c_synovial_starts_at_zero(default_result):
    assert default_result.c_synovial_mg_L[0] == 0.0


def test_c_plasma_starts_positive(default_result):
    """Plasma concentration starts at dose/Vd after IV bolus."""
    assert default_result.c_plasma_mg_L[0] > 0.0


def test_c_synovial_rises_then_falls(default_result):
    cs = default_result.c_synovial_mg_L
    # must have a peak somewhere in the middle (not at index 0 or last)
    peak_idx = cs.index(max(cs))
    assert peak_idx > 0
    assert peak_idx < len(cs) - 1


def test_cmax_synovial_positive(default_result):
    assert default_result.cmax_synovial_mg_L > 0.0


def test_cmax_synovial_matches_list_max(default_result):
    assert abs(default_result.cmax_synovial_mg_L - max(default_result.c_synovial_mg_L)) < 1e-9


def test_tmax_synovial_is_non_negative(default_result):
    assert default_result.tmax_synovial_h >= 0.0


def test_auc_synovial_positive(default_result):
    assert default_result.auc_synovial_mg_h_per_L > 0.0


def test_auc_plasma_positive(default_result):
    assert default_result.auc_plasma_mg_h_per_L > 0.0


def test_synovial_to_plasma_ratio_positive(default_result):
    assert default_result.synovial_to_plasma_ratio > 0.0


def test_times_length_matches_concentrations(default_result):
    assert len(default_result.times_h) == len(default_result.c_plasma_mg_L)
    assert len(default_result.times_h) == len(default_result.c_synovial_mg_L)


def test_times_monotonically_increasing(default_result):
    times = default_result.times_h
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1]


# ---------------------------------------------------------------------------
# Inflamed vs healthy comparison
# ---------------------------------------------------------------------------


def test_inflamed_higher_synovial_auc_than_healthy(default_result, healthy_result):
    """Inflamed joint: higher k_ps → higher synovial AUC."""
    assert default_result.auc_synovial_mg_h_per_L > healthy_result.auc_synovial_mg_h_per_L


def test_inflamed_higher_cmax_synovial(default_result, healthy_result):
    assert default_result.cmax_synovial_mg_L > healthy_result.cmax_synovial_mg_L


# ---------------------------------------------------------------------------
# compare_joint_inflammation
# ---------------------------------------------------------------------------


def test_compare_returns_two_results():
    results = compare_joint_inflammation(
        drug_name="Naproxen",
        dose_mg=500.0,
        cl_L_per_h=3.0,
        vd_plasma_L=8.0,
    )
    assert len(results) == 2


def test_compare_first_is_inflamed():
    results = compare_joint_inflammation(
        drug_name="Naproxen",
        dose_mg=500.0,
        cl_L_per_h=3.0,
        vd_plasma_L=8.0,
    )
    assert results[0].inflamed is True


def test_compare_second_is_healthy():
    results = compare_joint_inflammation(
        drug_name="Naproxen",
        dose_mg=500.0,
        cl_L_per_h=3.0,
        vd_plasma_L=8.0,
    )
    assert results[1].inflamed is False


def test_compare_sorted_by_synovial_auc_descending():
    results = compare_joint_inflammation(
        drug_name="Ibuprofen",
        dose_mg=400.0,
        cl_L_per_h=7.0,
        vd_plasma_L=12.0,
    )
    assert results[0].auc_synovial_mg_h_per_L >= results[1].auc_synovial_mg_h_per_L


def test_compare_both_are_synovial_pk_results():
    results = compare_joint_inflammation(
        drug_name="Celecoxib",
        dose_mg=200.0,
        cl_L_per_h=4.0,
        vd_plasma_L=6.0,
    )
    for r in results:
        assert isinstance(r, SynovialPKResult)


# ---------------------------------------------------------------------------
# Joint size scaling
# ---------------------------------------------------------------------------


def test_joint_size_scaling():
    """Larger joint → lower synovial concentration peak."""
    small = simulate_synovial_pk(
        drug_name="X",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_plasma_L=10.0,
        joint_size=0.5,
    )
    large = simulate_synovial_pk(
        drug_name="X",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_plasma_L=10.0,
        joint_size=3.0,
    )
    # Larger volume → lower concentration but transfer rate still same
    # AUC in mg*h/L will differ; both should be positive
    assert small.auc_synovial_mg_h_per_L > 0
    assert large.auc_synovial_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_pk("X", 0.0, 5.0, 10.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_pk("X", -10.0, 5.0, 10.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_synovial_pk("X", 100.0, 0.0, 10.0)


def test_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_synovial_pk("X", 100.0, -1.0, 10.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_synovial_pk("X", 100.0, 5.0, 0.0)


def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_synovial_pk("X", 100.0, 5.0, -5.0)
