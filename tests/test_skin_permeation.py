"""Tests for core/skin_permeation.py — Phase 187."""

import pytest

from omega_pbpk.core.skin_permeation import (
    VEHICLE_EF,
    SkinPermeationResult,
    calculate_skin_permeation,
    compare_vehicles,
)


# ---------------------------------------------------------------------------
# Basic return type and fields
# ---------------------------------------------------------------------------

def test_returns_dataclass():
    result = calculate_skin_permeation("TestDrug", logP=2.0, mw=300.0)
    assert isinstance(result, SkinPermeationResult)


def test_fields_populated():
    r = calculate_skin_permeation("TestDrug", logP=2.0, mw=300.0)
    assert r.drug_name == "TestDrug"
    assert r.logP == 2.0
    assert r.mw == 300.0
    assert r.vehicle == "cream"


# ---------------------------------------------------------------------------
# Potts-Guy physics: higher logP → higher Kp
# ---------------------------------------------------------------------------

def test_high_logP_higher_kp():
    r_low = calculate_skin_permeation("Drug", logP=0.0, mw=300.0)
    r_high = calculate_skin_permeation("Drug", logP=4.0, mw=300.0)
    assert r_high.kp_cm_per_h > r_low.kp_cm_per_h


# ---------------------------------------------------------------------------
# Higher MW → lower Kp
# ---------------------------------------------------------------------------

def test_high_mw_lower_kp():
    r_low = calculate_skin_permeation("Drug", logP=2.0, mw=200.0)
    r_high = calculate_skin_permeation("Drug", logP=2.0, mw=600.0)
    assert r_high.kp_cm_per_h < r_low.kp_cm_per_h


# ---------------------------------------------------------------------------
# Vehicle ordering
# ---------------------------------------------------------------------------

def test_ointment_greater_than_cream():
    r_ointment = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, vehicle="ointment")
    r_cream = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, vehicle="cream")
    # EF ointment=1.5 > cream=1.0
    assert r_ointment.kp_cm_per_h > r_cream.kp_cm_per_h


def test_patch_greater_than_gel():
    r_patch = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, vehicle="patch")
    r_gel = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, vehicle="gel")
    # EF patch=1.2 > gel=0.8
    assert r_patch.kp_cm_per_h > r_gel.kp_cm_per_h


def test_vehicle_enhancement_factor_stored():
    r = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, vehicle="ointment")
    assert r.vehicle_enhancement_factor == VEHICLE_EF["ointment"]


# ---------------------------------------------------------------------------
# Skin condition
# ---------------------------------------------------------------------------

def test_damaged_skin_higher_penetration_than_normal():
    r_normal = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, skin_condition="normal")
    r_damaged = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, skin_condition="damaged")
    assert r_damaged.kp_cm_per_h > r_normal.kp_cm_per_h


def test_diseased_skin_higher_penetration_than_normal():
    r_normal = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, skin_condition="normal")
    r_diseased = calculate_skin_permeation("Drug", logP=2.0, mw=300.0, skin_condition="diseased")
    assert r_diseased.kp_cm_per_h > r_normal.kp_cm_per_h


# ---------------------------------------------------------------------------
# Lag time
# ---------------------------------------------------------------------------

def test_lag_time_positive():
    r = calculate_skin_permeation("Drug", logP=2.0, mw=300.0)
    assert r.lag_time_h > 0.0


def test_lag_time_capped_at_12h():
    # Very low Kp → very small D → large lag time → capped at 12h
    r = calculate_skin_permeation("Drug", logP=-5.0, mw=800.0)
    assert r.lag_time_h <= 12.0


def test_lag_time_min_01():
    # Even very high Kp should not produce lag < 0.1h
    r = calculate_skin_permeation("Drug", logP=6.0, mw=100.0)
    assert r.lag_time_h >= 0.1


# ---------------------------------------------------------------------------
# Flux and absorbed dose
# ---------------------------------------------------------------------------

def test_steady_state_flux_positive():
    r = calculate_skin_permeation("Drug", logP=2.0, mw=300.0)
    assert r.steady_state_flux_ug_cm2_h > 0.0


def test_absorbed_dose_pct_between_0_and_100():
    r = calculate_skin_permeation("Drug", logP=2.0, mw=300.0)
    assert 0.0 <= r.absorbed_dose_pct <= 100.0


def test_higher_kp_higher_flux():
    r_low = calculate_skin_permeation("Drug", logP=0.0, mw=300.0)
    r_high = calculate_skin_permeation("Drug", logP=4.0, mw=300.0)
    assert r_high.steady_state_flux_ug_cm2_h > r_low.steady_state_flux_ug_cm2_h


# ---------------------------------------------------------------------------
# Penetration class
# ---------------------------------------------------------------------------

def test_penetration_class_low():
    # Very hydrophilic + high MW → low permeability
    r = calculate_skin_permeation("Drug", logP=-3.0, mw=700.0)
    assert r.penetration_class == "low"


def test_penetration_class_high():
    # Very lipophilic + low MW → high permeability
    r = calculate_skin_permeation("Drug", logP=5.0, mw=150.0)
    assert r.penetration_class == "high"


# ---------------------------------------------------------------------------
# compare_vehicles
# ---------------------------------------------------------------------------

def test_compare_vehicles_returns_5_results():
    results = compare_vehicles("Drug", logP=2.0, mw=300.0)
    assert len(results) == 5


def test_compare_vehicles_sorted_descending():
    results = compare_vehicles("Drug", logP=2.0, mw=300.0)
    kps = [r.kp_cm_per_h for r in results]
    assert kps == sorted(kps, reverse=True)


def test_compare_vehicles_ointment_first():
    results = compare_vehicles("Drug", logP=2.0, mw=300.0)
    # Ointment has highest EF (1.5) so should be first
    assert results[0].vehicle == "ointment"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_vehicle_raises():
    with pytest.raises(ValueError, match="Invalid vehicle"):
        calculate_skin_permeation("Drug", logP=2.0, mw=300.0, vehicle="lotion")


def test_invalid_skin_condition_raises():
    with pytest.raises(ValueError, match="Invalid skin_condition"):
        calculate_skin_permeation("Drug", logP=2.0, mw=300.0, skin_condition="burned")


def test_zero_mw_raises():
    with pytest.raises(ValueError, match="mw must be"):
        calculate_skin_permeation("Drug", logP=2.0, mw=0.0)


def test_negative_concentration_raises():
    with pytest.raises(ValueError, match="concentration_mg_mL must be"):
        calculate_skin_permeation("Drug", logP=2.0, mw=300.0, concentration_mg_mL=-1.0)
