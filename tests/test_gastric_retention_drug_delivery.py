"""Tests for Phase 505 — Gastric Retention Drug Delivery."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.gastric_retention_drug_delivery import (
    GastricRetentionResult,
    compare_retention_times,
    simulate_gastric_retention,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_params():
    return dict(
        drug_name="TestDrug",
        dose_mg=200.0,
        formulation_type="floating",
        cl_L_per_h=5.0,
        vd_L=50.0,
    )


@pytest.fixture
def base_result(base_params):
    return simulate_gastric_retention(**base_params)


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_returns_correct_type(base_result):
    assert isinstance(base_result, GastricRetentionResult)


def test_has_required_fields(base_result):
    assert hasattr(base_result, "drug_name")
    assert hasattr(base_result, "dose_mg")
    assert hasattr(base_result, "formulation_type")
    assert hasattr(base_result, "times_h")
    assert hasattr(base_result, "c_systemic_mg_L")
    assert hasattr(base_result, "cmax_mg_L")
    assert hasattr(base_result, "tmax_h")
    assert hasattr(base_result, "auc_mg_h_per_L")
    assert hasattr(base_result, "retention_time_h")
    assert hasattr(base_result, "f_absorbed")
    assert hasattr(base_result, "absorption_phase")
    assert hasattr(base_result, "t_half_h")
    assert hasattr(base_result, "notes")


# ---------------------------------------------------------------------------
# Initial condition: C_sys starts at 0
# ---------------------------------------------------------------------------


def test_c_sys_starts_at_zero(base_result):
    assert base_result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Cmax and AUC positivity
# ---------------------------------------------------------------------------


def test_cmax_positive(base_result):
    assert base_result.cmax_mg_L > 0.0


def test_auc_positive(base_result):
    assert base_result.auc_mg_h_per_L > 0.0


def test_tmax_non_negative(base_result):
    assert base_result.tmax_h >= 0.0


# ---------------------------------------------------------------------------
# Time series sanity
# ---------------------------------------------------------------------------


def test_times_increasing(base_result):
    times = base_result.times_h
    for i in range(1, len(times)):
        assert times[i] > times[i - 1]


def test_times_and_concentration_same_length(base_result):
    assert len(base_result.times_h) == len(base_result.c_systemic_mg_L)


def test_concentrations_non_negative(base_result):
    for c in base_result.c_systemic_mg_L:
        assert c >= 0.0


# ---------------------------------------------------------------------------
# Field value correctness
# ---------------------------------------------------------------------------


def test_drug_name_stored(base_result):
    assert base_result.drug_name == "TestDrug"


def test_dose_stored(base_result):
    assert base_result.dose_mg == pytest.approx(200.0)


def test_formulation_type_stored_floating():
    r = simulate_gastric_retention(
        drug_name="X", dose_mg=100.0, formulation_type="floating", cl_L_per_h=5.0, vd_L=50.0
    )
    assert r.formulation_type == "floating"


def test_formulation_type_stored_swelling():
    r = simulate_gastric_retention(
        drug_name="X", dose_mg=100.0, formulation_type="swelling", cl_L_per_h=5.0, vd_L=50.0
    )
    assert r.formulation_type == "swelling"


def test_retention_time_h_stored(base_result):
    assert base_result.retention_time_h == pytest.approx(4.0)


def test_f_absorbed_between_zero_and_one(base_result):
    assert 0.0 <= base_result.f_absorbed <= 1.0


def test_t_half_h_positive(base_result):
    assert base_result.t_half_h > 0.0


def test_t_half_analytical():
    """t_half should equal ln(2) / k_el = Vd*ln(2)/CL."""
    r = simulate_gastric_retention(
        drug_name="X", dose_mg=100.0, formulation_type="floating", cl_L_per_h=10.0, vd_L=50.0
    )
    expected = math.log(2.0) / (10.0 / 50.0)
    assert r.t_half_h == pytest.approx(expected, rel=1e-6)


def test_absorption_phase_is_string(base_result):
    assert isinstance(base_result.absorption_phase, str)
    assert base_result.absorption_phase in ("gastric", "mixed")


def test_notes_non_empty(base_result):
    assert isinstance(base_result.notes, str)
    assert len(base_result.notes) > 0


# ---------------------------------------------------------------------------
# Retention time effect on tmax
# ---------------------------------------------------------------------------


def test_longer_retention_shifts_tmax():
    """Longer retention time should shift drug release later, changing tmax."""
    short = simulate_gastric_retention(
        drug_name="X",
        dose_mg=200.0,
        formulation_type="floating",
        cl_L_per_h=5.0,
        vd_L=50.0,
        retention_time_h=2.0,
    )
    long = simulate_gastric_retention(
        drug_name="X",
        dose_mg=200.0,
        formulation_type="floating",
        cl_L_per_h=5.0,
        vd_L=50.0,
        retention_time_h=8.0,
    )
    # tmax should be different (longer retention delays peak)
    assert long.tmax_h != short.tmax_h or long.auc_mg_h_per_L != short.auc_mg_h_per_L


def test_longer_retention_tmax_later():
    """With fast intestinal absorption, longer retention should delay tmax."""
    short = simulate_gastric_retention(
        drug_name="X",
        dose_mg=200.0,
        formulation_type="floating",
        cl_L_per_h=5.0,
        vd_L=50.0,
        retention_time_h=1.0,
        ka_gastric=0.1,
        ka_intestine=3.0,
    )
    long = simulate_gastric_retention(
        drug_name="X",
        dose_mg=200.0,
        formulation_type="floating",
        cl_L_per_h=5.0,
        vd_L=50.0,
        retention_time_h=8.0,
        ka_gastric=0.1,
        ka_intestine=3.0,
    )
    assert long.tmax_h >= short.tmax_h


# ---------------------------------------------------------------------------
# Effect of higher intestinal ka
# ---------------------------------------------------------------------------


def test_higher_ka_intestine_faster_peak_after_release():
    """Higher ka_intestine should lead to faster peak after gastric release."""
    slow_int = simulate_gastric_retention(
        drug_name="X",
        dose_mg=200.0,
        formulation_type="floating",
        cl_L_per_h=5.0,
        vd_L=50.0,
        retention_time_h=4.0,
        ka_gastric=0.3,
        ka_intestine=0.5,
    )
    fast_int = simulate_gastric_retention(
        drug_name="X",
        dose_mg=200.0,
        formulation_type="floating",
        cl_L_per_h=5.0,
        vd_L=50.0,
        retention_time_h=4.0,
        ka_gastric=0.3,
        ka_intestine=3.0,
    )
    # Faster intestinal absorption → higher Cmax
    assert fast_int.cmax_mg_L >= slow_int.cmax_mg_L


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_gastric_retention(
            drug_name="X", dose_mg=0.0, formulation_type="floating", cl_L_per_h=5.0, vd_L=50.0
        )


def test_raises_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_gastric_retention(
            drug_name="X", dose_mg=-10.0, formulation_type="floating", cl_L_per_h=5.0, vd_L=50.0
        )


def test_raises_on_zero_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_gastric_retention(
            drug_name="X", dose_mg=100.0, formulation_type="floating", cl_L_per_h=0.0, vd_L=50.0
        )


def test_raises_on_zero_vd():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_gastric_retention(
            drug_name="X", dose_mg=100.0, formulation_type="floating", cl_L_per_h=5.0, vd_L=0.0
        )


def test_raises_on_invalid_formulation_type():
    with pytest.raises(ValueError, match="formulation_type"):
        simulate_gastric_retention(
            drug_name="X", dose_mg=100.0, formulation_type="bioadhesive", cl_L_per_h=5.0, vd_L=50.0
        )


def test_raises_on_zero_retention_time():
    with pytest.raises(ValueError, match="retention_time_h"):
        simulate_gastric_retention(
            drug_name="X",
            dose_mg=100.0,
            formulation_type="floating",
            cl_L_per_h=5.0,
            vd_L=50.0,
            retention_time_h=0.0,
        )


# ---------------------------------------------------------------------------
# compare_retention_times
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    result = compare_retention_times(
        drug_name="X", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, retention_times_h=[2.0, 4.0, 8.0]
    )
    assert isinstance(result, list)
    assert len(result) == 3


def test_compare_returns_gastric_retention_results():
    result = compare_retention_times(
        drug_name="X", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, retention_times_h=[2.0, 6.0]
    )
    for r in result:
        assert isinstance(r, GastricRetentionResult)


def test_compare_sorted_by_auc_descending():
    result = compare_retention_times(
        drug_name="X",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        retention_times_h=[1.0, 3.0, 6.0, 12.0],
    )
    aucs = [r.auc_mg_h_per_L for r in result]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_single_retention_time():
    result = compare_retention_times(
        drug_name="X", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, retention_times_h=[4.0]
    )
    assert len(result) == 1
    assert isinstance(result[0], GastricRetentionResult)
