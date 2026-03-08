"""Tests for core/rectal_absorption_pk_p183.py (Phase 183)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.rectal_absorption_pk_p183 import (
    RectalPKResult,
    compare_suppository_formulations,
    simulate_rectal_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_result() -> RectalPKResult:
    return simulate_rectal_pk(
        drug_name="Aspirin",
        dose_mg=100.0,
        logP=1.2,
        mw_Da=180.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_rect_per_h=1.5,
        t_end_h=12.0,
        dt_h=0.05,
    )


@pytest.fixture()
def high_logp_result() -> RectalPKResult:
    return simulate_rectal_pk(
        drug_name="LipophilicDrug",
        dose_mg=100.0,
        logP=3.0,
        mw_Da=300.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )


@pytest.fixture()
def low_logp_result() -> RectalPKResult:
    return simulate_rectal_pk(
        drug_name="HydrophilicDrug",
        dose_mg=100.0,
        logP=0.0,
        mw_Da=300.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_returns_rectal_pk_result(default_result):
    assert isinstance(default_result, RectalPKResult)


def test_drug_name_preserved(default_result):
    assert default_result.drug_name == "Aspirin"


def test_dose_preserved(default_result):
    assert default_result.dose_mg == 100.0


def test_logp_preserved(default_result):
    assert default_result.logP == 1.2


def test_mw_preserved(default_result):
    assert default_result.mw_Da == 180.0


# ---------------------------------------------------------------------------
# Concentration profile tests
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_zero(default_result):
    """Plasma concentration at t=0 must be 0."""
    assert default_result.c_plasma_mg_L[0] == 0.0


def test_c_plasma_rises_to_cmax(default_result):
    """Concentration must reach a maximum above 0."""
    c = default_result.c_plasma_mg_L
    cmax = default_result.cmax_mg_L
    assert max(c) == cmax
    assert cmax > 0.0


def test_c_plasma_falls_after_cmax(default_result):
    """Concentration must eventually decrease after Tmax."""
    c = default_result.c_plasma_mg_L
    # The last value must be lower than Cmax (terminal elimination)
    assert c[-1] < default_result.cmax_mg_L


def test_times_and_concs_same_length(default_result):
    assert len(default_result.times_h) == len(default_result.c_plasma_mg_L)


def test_times_start_at_zero(default_result):
    assert default_result.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# PK metric tests
# ---------------------------------------------------------------------------


def test_cmax_positive(default_result):
    assert default_result.cmax_mg_L > 0.0


def test_tmax_positive(default_result):
    assert default_result.tmax_h > 0.0


def test_auc_positive(default_result):
    assert default_result.auc_mg_h_per_L > 0.0


def test_t_half_positive(default_result):
    assert default_result.t_half_h > 0.0


def test_f_rect_in_range(default_result):
    assert 0.0 < default_result.f_rect <= 1.0


def test_ka_rect_preserved(default_result):
    assert default_result.ka_rect_per_h == 1.5


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0


# ---------------------------------------------------------------------------
# F_rect physicochemical dependence
# ---------------------------------------------------------------------------


def test_higher_logp_higher_f_rect(high_logp_result, low_logp_result):
    """logP=3 should yield higher F_rect than logP=0."""
    assert high_logp_result.f_rect > low_logp_result.f_rect


def test_higher_mw_lower_f_rect():
    """Higher MW (above 200 Da) penalises F_rect."""
    r_low = simulate_rectal_pk("Drug", 100.0, logP=2.0, mw_Da=200.0)
    r_high = simulate_rectal_pk("Drug", 100.0, logP=2.0, mw_Da=1000.0)
    assert r_low.f_rect >= r_high.f_rect


def test_f_rect_formula_base_case():
    """For logP=0, mw=200: logP_factor=0, mw_factor=0 → F_rect=0.70."""
    r = simulate_rectal_pk("Drug", 100.0, logP=0.0, mw_Da=200.0)
    assert abs(r.f_rect - 0.70) < 1e-9


def test_f_rect_max_capped_at_one():
    """Very high logP should be capped at F_rect=1.0."""
    r = simulate_rectal_pk("Drug", 100.0, logP=100.0, mw_Da=200.0)
    assert r.f_rect <= 1.0


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_double_dose_doubles_auc():
    r1 = simulate_rectal_pk("Drug", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    r2 = simulate_rectal_pk("Drug", dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert abs(ratio - 2.0) < 0.02


# ---------------------------------------------------------------------------
# compare_suppository_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    formulations = [
        {"name": "Slow", "ka_rect_per_h": 0.5},
        {"name": "Fast", "ka_rect_per_h": 3.0},
        {"name": "Medium", "ka_rect_per_h": 1.5},
    ]
    results = compare_suppository_formulations(
        "Aspirin", dose_mg=100.0, logP=1.2, mw_Da=180.0, formulations=formulations
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_results_are_rectal_pk_result():
    formulations = [
        {"name": "A", "ka_rect_per_h": 1.0},
        {"name": "B", "ka_rect_per_h": 2.0},
    ]
    results = compare_suppository_formulations(
        "Drug", dose_mg=50.0, logP=2.0, mw_Da=250.0, formulations=formulations
    )
    for r in results:
        assert isinstance(r, RectalPKResult)


def test_compare_sorted_by_auc_descending():
    """Results must be sorted from highest to lowest AUC."""
    formulations = [
        {"name": "F1", "ka_rect_per_h": 0.3},
        {"name": "F2", "ka_rect_per_h": 1.5},
        {"name": "F3", "ka_rect_per_h": 5.0},
    ]
    results = compare_suppository_formulations(
        "Drug", dose_mg=100.0, logP=2.0, mw_Da=300.0, formulations=formulations
    )
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_single_formulation():
    formulations = [{"name": "Only", "ka_rect_per_h": 1.0}]
    results = compare_suppository_formulations(
        "Drug", dose_mg=100.0, logP=1.5, mw_Da=250.0, formulations=formulations
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_rectal_pk("Drug", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_rectal_pk("Drug", dose_mg=-50.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_rectal_pk("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_rectal_pk("Drug", dose_mg=100.0, vd_L=0.0)


def test_validation_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_rectal_pk("Drug", dose_mg=100.0, vd_L=-10.0)
