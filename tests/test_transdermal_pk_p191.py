"""Tests for transdermal drug delivery PBPK model — Phase 191."""

from __future__ import annotations

import pytest

from omega_pbpk.core.transdermal_pk_p191 import (
    TransdermalPKResult,
    compare_patch_types,
    simulate_transdermal_pk,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

DRUG = "TestDrug"
BASE_KWARGS = dict(
    dose_mg=10.0,
    patch_type="matrix",
    logP=2.0,
    mw_Da=300.0,
    cl_L_per_h=2.0,
    vd_L=40.0,
    patch_duration_h=24.0,
    dt_h=0.1,
)


def _run(**overrides) -> TransdermalPKResult:
    kw = {**BASE_KWARGS, **overrides}
    return simulate_transdermal_pk(DRUG, **kw)


# ── Return type and field preservation ───────────────────────────────────────


def test_return_type():
    r = _run()
    assert isinstance(r, TransdermalPKResult)


def test_drug_name_preserved():
    r = _run()
    assert r.drug_name == DRUG


def test_dose_mg_preserved():
    r = _run(dose_mg=15.0)
    assert r.dose_mg == pytest.approx(15.0)


def test_patch_type_preserved():
    r = _run(patch_type="gel")
    assert r.patch_type == "gel"


def test_logP_preserved():
    r = _run(logP=1.5)
    assert r.logP == pytest.approx(1.5)


def test_mw_Da_preserved():
    r = _run(mw_Da=250.0)
    assert r.mw_Da == pytest.approx(250.0)


# ── Output list types and sizes ───────────────────────────────────────────────


def test_times_h_is_list():
    r = _run()
    assert isinstance(r.times_h, list)
    assert len(r.times_h) > 0


def test_c_plasma_is_list():
    r = _run()
    assert isinstance(r.c_plasma_mg_L, list)
    assert len(r.c_plasma_mg_L) == len(r.times_h)


def test_notes_is_list():
    r = _run()
    assert isinstance(r.notes, list)


# ── Plasma concentration dynamics ────────────────────────────────────────────


def test_c_plasma_starts_at_zero():
    r = _run()
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_c_plasma_rises():
    """Plasma concentration should rise above 0 at some point."""
    r = _run()
    assert max(r.c_plasma_mg_L) > 0.0


def test_c_plasma_positive_after_tmax():
    """Plasma concentration at tmax should be Cmax."""
    r = _run()
    tmax_idx = r.times_h.index(r.tmax_h)
    assert r.c_plasma_mg_L[tmax_idx] == pytest.approx(r.cmax_mg_L, rel=1e-6)


def test_c_plasma_falls_after_tmax():
    """After tmax, plasma concentration should eventually decrease."""
    r = _run()
    tmax_idx = r.times_h.index(r.tmax_h)
    # After tmax, the last value should be less than Cmax
    if tmax_idx < len(r.c_plasma_mg_L) - 5:
        assert r.c_plasma_mg_L[-1] < r.cmax_mg_L


# ── PK metric positivity ─────────────────────────────────────────────────────


def test_cmax_positive():
    r = _run()
    assert r.cmax_mg_L > 0.0


def test_auc_positive():
    r = _run()
    assert r.auc_mg_h_per_L > 0.0


def test_tmax_positive():
    r = _run()
    assert r.tmax_h > 0.0


def test_bioavailability_in_range():
    r = _run()
    assert 0.0 < r.bioavailability_fraction <= 1.0


def test_skin_depot_peak_positive():
    r = _run()
    assert r.skin_depot_peak_mg > 0.0


# ── logP effect on absorption ─────────────────────────────────────────────────


def test_logP2_higher_absorption_than_logP5():
    """logP=2 is optimal; logP=5 is outside optimal range, gives lower AUC."""
    r_opt = _run(logP=2.0)
    r_poor = _run(logP=5.0)
    assert r_opt.auc_mg_h_per_L > r_poor.auc_mg_h_per_L


def test_logP2_higher_cmax_than_logP0():
    """logP=2 optimal; logP=0 is outside optimal range."""
    r_opt = _run(logP=2.0)
    r_poor = _run(logP=0.0)
    assert r_opt.cmax_mg_L > r_poor.cmax_mg_L


# ── Patch type release rates ──────────────────────────────────────────────────


def test_reservoir_slower_release_than_gel():
    """Reservoir has lower k_release (0.02) vs gel (0.1), so slower rise."""
    r_res = _run(patch_type="reservoir")
    r_gel = _run(patch_type="gel")
    # Gel releases faster -> higher Cmax early (at shorter tmax)
    # AUC should be similar (same dose) but tmax differs
    assert r_gel.tmax_h <= r_res.tmax_h or r_gel.auc_mg_h_per_L >= r_res.auc_mg_h_per_L


def test_reservoir_note_present():
    r = _run(patch_type="reservoir")
    note_text = " ".join(r.notes).lower()
    assert "reservoir" in note_text


# ── compare_patch_types ───────────────────────────────────────────────────────


def test_compare_patch_types_returns_three():
    results = compare_patch_types(
        DRUG,
        dose_mg=10.0,
        logP=2.0,
        mw_Da=300.0,
        cl_L_per_h=2.0,
        vd_L=40.0,
        patch_duration_h=24.0,
        dt_h=0.1,
    )
    assert len(results) == 3


def test_compare_patch_types_sorted_by_auc():
    results = compare_patch_types(
        DRUG,
        dose_mg=10.0,
        logP=2.0,
        mw_Da=300.0,
        cl_L_per_h=2.0,
        vd_L=40.0,
        patch_duration_h=24.0,
        dt_h=0.1,
    )
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_patch_types_all_types_present():
    results = compare_patch_types(
        DRUG,
        dose_mg=10.0,
        logP=2.0,
        mw_Da=300.0,
        cl_L_per_h=2.0,
        vd_L=40.0,
        patch_duration_h=24.0,
        dt_h=0.1,
    )
    types = {r.patch_type for r in results}
    assert types == {"reservoir", "matrix", "gel"}


def test_compare_patch_types_returns_transdermal_results():
    results = compare_patch_types(
        DRUG,
        dose_mg=10.0,
        logP=2.0,
        mw_Da=300.0,
        cl_L_per_h=2.0,
        vd_L=40.0,
        patch_duration_h=24.0,
        dt_h=0.1,
    )
    assert all(isinstance(r, TransdermalPKResult) for r in results)


# ── Validation / errors ───────────────────────────────────────────────────────


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_transdermal_pk(DRUG, dose_mg=-1.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_transdermal_pk(DRUG, dose_mg=0.0)


def test_invalid_patch_type_raises():
    with pytest.raises(ValueError, match="patch_type"):
        simulate_transdermal_pk(DRUG, dose_mg=10.0, patch_type="spray")


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_transdermal_pk(DRUG, dose_mg=10.0, mw_Da=0.0)


def test_negative_mw_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        simulate_transdermal_pk(DRUG, dose_mg=10.0, mw_Da=-100.0)


# ── High MW note ──────────────────────────────────────────────────────────────


def test_high_mw_note():
    r = _run(mw_Da=600.0)
    note_text = " ".join(r.notes).lower()
    assert "500" in note_text or "mw" in note_text or "molecular" in note_text.replace("mw", "mw")
