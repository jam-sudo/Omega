"""Tests for Phase 643 — drug_accumulation_risk.py"""

import pytest

from omega_pbpk.risk.drug_accumulation_risk import (
    AccumulationRiskResult,
    assess_accumulation_risk,
    screen_accumulation_risk,
)

VALID_KWARGS = dict(
    compound_name="TestDrug",
    logP=2.0,
    mw=350.0,
    psa=60.0,
    vd_L_per_kg=1.0,
    cl_L_per_h_per_kg=0.1,
    fup=0.1,
)


def make_result(**overrides):
    kwargs = dict(VALID_KWARGS)
    kwargs.update(overrides)
    return assess_accumulation_risk(**kwargs)


# --- Basic return type ---
def test_returns_accumulation_risk_result():
    r = make_result()
    assert isinstance(r, AccumulationRiskResult)


def test_accumulation_index_in_range():
    r = make_result()
    assert 0.0 <= r.accumulation_index <= 100.0


def test_risk_level_valid():
    r = make_result()
    assert r.risk_level in {"low", "moderate", "high", "very_high"}


def test_tissue_of_concern_valid():
    r = make_result()
    assert r.tissue_of_concern in {"adipose", "muscle", "liver", "brain"}


def test_wash_out_t_half_h_positive():
    r = make_result()
    assert r.wash_out_t_half_h > 0.0


def test_steady_state_ratio_positive():
    r = make_result()
    assert r.steady_state_ratio > 0.0


def test_notes_nonempty():
    r = make_result()
    assert len(r.notes) > 0


# --- Physicochemical relationships ---
def test_high_logP_high_accumulation_index():
    r = make_result(logP=5.0)
    assert r.accumulation_index > 50.0


def test_low_logP_low_accumulation_index():
    r = make_result(logP=-1.0)
    assert r.accumulation_index < 30.0


def test_basic_drug_higher_muscle_kp():
    basic = make_result(logP=2.0, pka_base=9.0)
    neutral = make_result(logP=2.0, pka_base=None)
    assert basic.muscle_kp > neutral.muscle_kp


def test_basic_drug_higher_liver_kp():
    basic = make_result(logP=2.0, pka_base=9.0)
    neutral = make_result(logP=2.0, pka_base=None)
    assert basic.liver_kp > neutral.liver_kp


def test_high_logP_low_psa_positive_brain_kp():
    r = make_result(logP=4.0, psa=30.0)
    assert r.brain_kp > 0.1


def test_high_psa_low_brain_kp():
    r = make_result(logP=4.0, psa=120.0)
    assert r.brain_kp <= 0.1


# --- Risk level thresholds ---
def test_high_logP_high_risk_level():
    r = make_result(logP=5.0, pka_base=9.0)
    assert r.risk_level in {"high", "very_high"}


def test_low_logP_low_vd_low_risk():
    r = make_result(logP=-1.0, vd_L_per_kg=0.2, pka_base=None)
    assert r.risk_level == "low"


# --- Screening ---
def test_screen_returns_list_sorted_descending():
    compounds = [
        dict(compound_name="LowLogP", logP=0.5, mw=200.0, psa=50.0),
        dict(compound_name="HighLogP", logP=5.0, mw=400.0, psa=40.0),
        dict(compound_name="MidLogP", logP=2.5, mw=300.0, psa=60.0),
    ]
    results = screen_accumulation_risk(compounds)
    assert len(results) == 3
    indices = [r.accumulation_index for r in results]
    assert indices == sorted(indices, reverse=True)


def test_screen_empty_input_returns_empty():
    assert screen_accumulation_risk([]) == []


# --- Validation errors ---
def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        make_result(mw=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L_per_kg"):
        make_result(vd_L_per_kg=0.0)


def test_validation_fup_zero_raises():
    with pytest.raises(ValueError, match="fup"):
        make_result(fup=0.0)


def test_validation_fup_above_one_raises():
    with pytest.raises(ValueError, match="fup"):
        make_result(fup=1.2)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h_per_kg"):
        make_result(cl_L_per_h_per_kg=0.0)


# --- T-half estimation ---
def test_t_half_estimated_from_vd_cl_when_not_provided():
    r = make_result(vd_L_per_kg=2.0, cl_L_per_h_per_kg=0.5)
    # wash_out_t_half should be > 0 and related to t_half
    assert r.wash_out_t_half_h > 0.0


def test_explicit_t_half_used():
    r = assess_accumulation_risk(
        compound_name="Test", logP=2.0, mw=300.0, psa=50.0, t_half_h=10.0, fup=0.1
    )
    assert r.wash_out_t_half_h > 0.0
