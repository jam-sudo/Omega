"""Tests for Phase 268: Mucoadhesive drug delivery PK model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.mucoadhesive_pk import MucoadhesivePKResult, simulate_mucoadhesive_pk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def buccal_tablet() -> MucoadhesivePKResult:
    return simulate_mucoadhesive_pk(
        drug_name="fentanyl",
        dose_mg=0.2,
        cl_L_per_h=50.0,
        vd_L=400.0,
        adhesion_site="buccal",
        formulation_type="tablet",
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mucoadhesive_pk("drug", dose_mg=0.0, cl_L_per_h=5.0, vd_L=50.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mucoadhesive_pk("drug", dose_mg=-1.0, cl_L_per_h=5.0, vd_L=50.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_mucoadhesive_pk("drug", dose_mg=1.0, cl_L_per_h=0.0, vd_L=50.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_mucoadhesive_pk("drug", dose_mg=1.0, cl_L_per_h=5.0, vd_L=0.0)


def test_invalid_adhesion_site():
    with pytest.raises(ValueError, match="adhesion_site"):
        simulate_mucoadhesive_pk(
            "drug", dose_mg=1.0, cl_L_per_h=5.0, vd_L=50.0, adhesion_site="rectal"
        )


def test_invalid_formulation_type():
    with pytest.raises(ValueError, match="formulation_type"):
        simulate_mucoadhesive_pk(
            "drug", dose_mg=1.0, cl_L_per_h=5.0, vd_L=50.0, formulation_type="patch"
        )


# ---------------------------------------------------------------------------
# Basic result checks
# ---------------------------------------------------------------------------

def test_cmax_positive(buccal_tablet: MucoadhesivePKResult):
    assert buccal_tablet.cmax > 0.0


def test_auc_positive(buccal_tablet: MucoadhesivePKResult):
    assert buccal_tablet.auc > 0.0


def test_tmax_nonnegative(buccal_tablet: MucoadhesivePKResult):
    assert buccal_tablet.tmax_h >= 0.0


def test_t_half_adhesion_positive(buccal_tablet: MucoadhesivePKResult):
    assert buccal_tablet.t_half_adhesion_h > 0.0


def test_f_mucosal_absorption_range(buccal_tablet: MucoadhesivePKResult):
    assert 0.0 <= buccal_tablet.f_mucosal_absorption <= 1.0


# ---------------------------------------------------------------------------
# Array length consistency
# ---------------------------------------------------------------------------

def test_array_lengths_consistent(buccal_tablet: MucoadhesivePKResult):
    n = len(buccal_tablet.times_h)
    assert len(buccal_tablet.c_plasma_mg_L) == n
    assert len(buccal_tablet.m_mucosal_mg) == n


# ---------------------------------------------------------------------------
# All adhesion sites work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("site", ["buccal", "nasal", "gastric", "vaginal"])
def test_all_adhesion_sites(site: str):
    r = simulate_mucoadhesive_pk(
        "drug", dose_mg=1.0, cl_L_per_h=5.0, vd_L=50.0, adhesion_site=site
    )
    assert r.cmax > 0.0
    assert r.adhesion_site == site


# ---------------------------------------------------------------------------
# All formulation types work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["tablet", "gel", "microsphere"])
def test_all_formulation_types(form: str):
    r = simulate_mucoadhesive_pk(
        "drug", dose_mg=1.0, cl_L_per_h=5.0, vd_L=50.0, formulation_type=form
    )
    assert r.cmax > 0.0
    assert r.formulation_type == form


# ---------------------------------------------------------------------------
# Microsphere slower release → lower Cmax than tablet
# ---------------------------------------------------------------------------

def test_microsphere_lower_cmax_than_tablet():
    common = dict(drug_name="drug", dose_mg=1.0, cl_L_per_h=5.0, vd_L=50.0)
    r_tab = simulate_mucoadhesive_pk(**common, formulation_type="tablet")
    r_ms = simulate_mucoadhesive_pk(**common, formulation_type="microsphere")
    assert r_ms.cmax < r_tab.cmax


# ---------------------------------------------------------------------------
# Drug name stored
# ---------------------------------------------------------------------------

def test_drug_name_stored():
    r = simulate_mucoadhesive_pk("buprenorphine", dose_mg=0.8, cl_L_per_h=50.0, vd_L=300.0)
    assert r.drug_name == "buprenorphine"
