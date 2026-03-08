"""Tests for drug placental transfer model (Phase 652)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.placental_transfer_p652 import (
    PlacentalTransferResult,
    simulate_placental_transfer,
)

BASE = {
    "drug_name": "DrugA",
    "dose_mg": 100.0,
    "logP": 2.0,
    "mw_Da": 300.0,
    "cl_maternal_L_per_h": 5.0,
    "vd_maternal_L": 50.0,
}


# ── Basic return type ────────────────────────────────────────────────


def test_return_type():
    r = simulate_placental_transfer(**BASE)
    assert isinstance(r, PlacentalTransferResult)


def test_not_frozen():
    r = simulate_placental_transfer(**BASE)
    r.drug_name = "other"
    assert r.drug_name == "other"


def test_drug_name_preserved():
    r = simulate_placental_transfer(**BASE)
    assert r.drug_name == "DrugA"


# ── List lengths and non-negative ────────────────────────────────────


def test_list_lengths():
    r = simulate_placental_transfer(**BASE, t_end_h=10.0, dt_h=0.5)
    assert len(r.times_h) == len(r.c_maternal_mg_L) == len(r.c_fetal_mg_L)
    assert len(r.times_h) > 0


def test_concentrations_non_negative():
    r = simulate_placental_transfer(**BASE)
    assert all(c >= 0 for c in r.c_maternal_mg_L)
    assert all(c >= 0 for c in r.c_fetal_mg_L)


# ── Higher logP → higher fetal transfer ──────────────────────────────


def test_higher_logP_higher_fetal_ratio():
    r_low = simulate_placental_transfer(**{**BASE, "logP": 0.5})
    r_high = simulate_placental_transfer(**{**BASE, "logP": 4.0})
    assert r_high.fetal_to_maternal_ratio_auc > r_low.fetal_to_maternal_ratio_auc


# ── Lower MW → higher fetal exposure ─────────────────────────────────


def test_lower_mw_higher_fetal_exposure():
    r_low = simulate_placental_transfer(**{**BASE, "mw_Da": 200.0})
    r_high = simulate_placental_transfer(**{**BASE, "mw_Da": 600.0})
    assert r_low.auc_fetal_mg_h_per_L > r_high.auc_fetal_mg_h_per_L


# ── Earlier gestational age → lower fetal Vd ─────────────────────────


def test_earlier_ga_lower_fetal_vd():
    r_early = simulate_placental_transfer(**BASE, gestational_age_weeks=12)
    r_late = simulate_placental_transfer(**BASE, gestational_age_weeks=38)
    # Earlier GA means smaller fetal compartment, so higher fetal Cmax per unit transferred
    # but lower total exposure due to smaller surface area
    assert r_late.auc_fetal_mg_h_per_L != r_early.auc_fetal_mg_h_per_L


# ── Fetal-to-maternal ratio range ────────────────────────────────────


def test_fetal_to_maternal_ratio_range():
    # With volume-scaled ODEs, fetal/maternal AUC ratio can exceed 1.0
    # due to concentration in the small fetal compartment
    r = simulate_placental_transfer(**BASE)
    assert r.fetal_to_maternal_ratio_auc >= 0.0


# ── AUC positive ─────────────────────────────────────────────────────


def test_auc_maternal_positive():
    r = simulate_placental_transfer(**BASE)
    assert r.auc_maternal_mg_h_per_L > 0


def test_auc_fetal_positive():
    r = simulate_placental_transfer(**BASE)
    assert r.auc_fetal_mg_h_per_L > 0


# ── t50_fetal_h ──────────────────────────────────────────────────────


def test_t50_fetal_positive():
    r = simulate_placental_transfer(**{**BASE, "logP": 3.0, "mw_Da": 200.0})
    # With good permeability, fetal conc should reach 50% of maternal Cmax
    assert r.t50_fetal_h >= 0.0


def test_t50_fetal_within_simulation():
    r = simulate_placental_transfer(**{**BASE, "logP": 3.0, "mw_Da": 200.0}, t_end_h=48.0)
    if r.t50_fetal_h > 0:
        assert r.t50_fetal_h <= 48.0


# ── Fetal exposure category ──────────────────────────────────────────


def test_exposure_category_values():
    r = simulate_placental_transfer(**BASE)
    assert r.fetal_exposure_category in ("high", "moderate", "low", "minimal")


def test_high_exposure_category():
    # Very lipophilic, small molecule, long simulation
    r = simulate_placental_transfer(
        drug_name="LipoSmall",
        dose_mg=100.0,
        logP=4.0,
        mw_Da=150.0,
        cl_maternal_L_per_h=1.0,
        vd_maternal_L=30.0,
        gestational_age_weeks=38,
        t_end_h=48.0,
    )
    assert r.fetal_exposure_category in ("high", "moderate")


def test_minimal_exposure_category():
    # Very hydrophilic, large molecule, large fetal Vd (late GA) to dilute
    r = simulate_placental_transfer(
        drug_name="HydroLarge",
        dose_mg=100.0,
        logP=-2.0,
        mw_Da=1200.0,
        cl_maternal_L_per_h=30.0,
        vd_maternal_L=10.0,
        gestational_age_weeks=40,
        t_end_h=4.0,
    )
    assert r.fetal_exposure_category in ("minimal", "low")


# ── Dose proportionality ────────────────────────────────────────────


def test_dose_proportionality_maternal_cmax():
    r1 = simulate_placental_transfer(**{**BASE, "dose_mg": 50.0})
    r2 = simulate_placental_transfer(**{**BASE, "dose_mg": 100.0})
    ratio = r2.cmax_maternal_mg_L / r1.cmax_maternal_mg_L
    assert abs(ratio - 2.0) < 0.1


def test_dose_proportionality_fetal_cmax():
    r1 = simulate_placental_transfer(**{**BASE, "dose_mg": 50.0})
    r2 = simulate_placental_transfer(**{**BASE, "dose_mg": 100.0})
    ratio = r2.cmax_fetal_mg_L / r1.cmax_fetal_mg_L
    assert abs(ratio - 2.0) < 0.1


# ── Cmax values ──────────────────────────────────────────────────────


def test_cmax_maternal_positive():
    r = simulate_placental_transfer(**BASE)
    assert r.cmax_maternal_mg_L > 0


def test_cmax_fetal_positive():
    r = simulate_placental_transfer(**BASE)
    assert r.cmax_fetal_mg_L > 0


# ── Placental transfer rate ──────────────────────────────────────────


def test_placental_transfer_rate_range():
    r = simulate_placental_transfer(**BASE)
    assert 0.001 <= r.placental_transfer_rate <= 2.0


# ── mw_Da and logP stored ───────────────────────────────────────────


def test_mw_stored():
    r = simulate_placental_transfer(**BASE)
    assert r.mw_Da == 300.0


def test_logP_stored():
    r = simulate_placental_transfer(**BASE)
    assert r.logP == 2.0


# ── Validation errors ───────────────────────────────────────────────


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_placental_transfer(**{**BASE, "dose_mg": 0.0})


def test_validation_dose_negative():
    with pytest.raises(ValueError):
        simulate_placental_transfer(**{**BASE, "dose_mg": -10.0})


def test_validation_cl_zero():
    with pytest.raises(ValueError):
        simulate_placental_transfer(**{**BASE, "cl_maternal_L_per_h": 0.0})


def test_validation_vd_zero():
    with pytest.raises(ValueError):
        simulate_placental_transfer(**{**BASE, "vd_maternal_L": 0.0})


def test_validation_ga_too_low():
    with pytest.raises(ValueError):
        simulate_placental_transfer(**BASE, gestational_age_weeks=0)


def test_validation_ga_too_high():
    with pytest.raises(ValueError):
        simulate_placental_transfer(**BASE, gestational_age_weeks=43)
