"""Tests for sublingual/buccal mucosal absorption PBPK model."""

import pytest

from omega_pbpk.core.sublingual_pbpk import (
    SublingualPKResult,
    compare_sublingual_oral,
    simulate_sublingual_pk,
)

# --- Validation ---


def test_dose_le_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sublingual_pk("Drug", dose_mg=0)


def test_cl_le_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_sublingual_pk("Drug", dose_mg=10, cl_L_per_h=0)


def test_vd_le_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_sublingual_pk("Drug", dose_mg=10, vd_L=0)


# --- Basic result structure ---


def test_returns_sublingual_pk_result():
    r = simulate_sublingual_pk("TestDrug", dose_mg=5.0)
    assert isinstance(r, SublingualPKResult)


def test_cmax_positive():
    r = simulate_sublingual_pk("TestDrug", dose_mg=5.0)
    assert r.cmax_mg_L > 0


def test_auc_positive():
    r = simulate_sublingual_pk("TestDrug", dose_mg=5.0)
    assert r.auc_0_t > 0


# --- Pharmacokinetic properties ---


def test_sublingual_tmax_earlier_than_oral():
    sl = simulate_sublingual_pk("Drug", dose_mg=10, f_swallowed=0.3)
    oral = simulate_sublingual_pk("Drug", dose_mg=10, f_swallowed=1.0)
    assert sl.tmax_h <= oral.tmax_h


def test_bioavailability_in_range():
    r = simulate_sublingual_pk("Drug", dose_mg=10)
    assert 0 < r.bioavailability <= 1.0


def test_lower_f_swallowed_higher_bioavailability():
    r_low = simulate_sublingual_pk("Drug", dose_mg=10, f_swallowed=0.1)
    r_high = simulate_sublingual_pk("Drug", dose_mg=10, f_swallowed=0.9)
    assert r_low.bioavailability > r_high.bioavailability


def test_mucosal_fraction_absorbed_positive():
    r = simulate_sublingual_pk("Drug", dose_mg=10, f_swallowed=0.3)
    assert r.mucosal_fraction_absorbed > 0


def test_a_mucosa_starts_at_dose_and_declines():
    r = simulate_sublingual_pk("Drug", dose_mg=10)
    assert r.a_mucosa_mg[0] == pytest.approx(10.0)
    assert r.a_mucosa_mg[-1] < r.a_mucosa_mg[0]


def test_first_pass_avoided_positive():
    r = simulate_sublingual_pk("Drug", dose_mg=10, f_swallowed=0.3)
    assert r.first_pass_avoided_mg > 0


# --- Linearity ---


def test_double_dose_doubles_cmax():
    r1 = simulate_sublingual_pk("Drug", dose_mg=5)
    r2 = simulate_sublingual_pk("Drug", dose_mg=10)
    assert r2.cmax_mg_L == pytest.approx(2.0 * r1.cmax_mg_L, rel=0.05)


# --- Contact time ---


def test_longer_contact_more_mucosal_absorption():
    r_short = simulate_sublingual_pk("Drug", dose_mg=10, t_contact_h=0.1)
    r_long = simulate_sublingual_pk("Drug", dose_mg=10, t_contact_h=1.0)
    assert r_long.mucosal_fraction_absorbed > r_short.mucosal_fraction_absorbed


# --- Notes and route ---


def test_notes_contain_drug_name_and_route():
    r = simulate_sublingual_pk("Nitroglycerin", dose_mg=0.4, route="sublingual")
    assert "Nitroglycerin" in r.notes
    assert "sublingual" in r.notes


def test_buccal_route_works():
    r = simulate_sublingual_pk("Drug", dose_mg=5, route="buccal")
    assert r.route == "buccal"
    assert r.cmax_mg_L > 0


# --- Comparison function ---


def test_compare_sublingual_oral_returns_two_results():
    sl, oral = compare_sublingual_oral("Drug", dose_mg=10, cl_L_per_h=1.0, vd_L=30.0)
    assert isinstance(sl, SublingualPKResult)
    assert isinstance(oral, SublingualPKResult)


def test_sublingual_auc_greater_than_oral():
    sl, oral = compare_sublingual_oral(
        "Drug",
        dose_mg=10,
        cl_L_per_h=1.0,
        vd_L=30.0,
        f_oral_hepatic=0.5,
    )
    assert sl.auc_0_t > oral.auc_0_t
