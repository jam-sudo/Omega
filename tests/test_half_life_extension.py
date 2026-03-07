"""Tests for Phase 1045 — half-life extension strategies."""

import math
import pytest

from omega_pbpk.prediction.half_life_extension import (
    HalfLifeExtensionResult,
    predict_half_life_extension,
)

_VALID_MECHANISMS = {"depot", "recycling", "reduced_clearance", "prodrug_conversion", "PEGylation"}

# Baseline parameters used across tests
_CL = 5.0   # L/h
_VD = 50.0  # L


def _baseline_t_half() -> float:
    return math.log(2) * _VD / _CL


# --- Return type ---

def test_returns_dataclass():
    result = predict_half_life_extension("DrugA", _CL, _VD)
    assert isinstance(result, HalfLifeExtensionResult)


def test_frozen():
    result = predict_half_life_extension("DrugA", _CL, _VD)
    with pytest.raises((AttributeError, TypeError)):
        result.extended_t_half_h = 99.0  # type: ignore[misc]


# --- All strategies extend half-life ---

@pytest.mark.parametrize("strategy,kwargs", [
    ("PEGylation", {"peg_mw_kDa": 20.0}),
    ("albumin_binding", {"albumin_affinity_uM": 1.0}),
    ("FcRn_recycling", {"mw_Da": 150000.0}),
    ("depot_formulation", {"depot_release_t50_h": 48.0}),
    ("prodrug", {"prodrug_conversion_t_half_h": 24.0}),
    ("nanoparticle", {}),
])
def test_extended_gt_baseline(strategy, kwargs):
    result = predict_half_life_extension("Drug", _CL, _VD, strategy=strategy, **kwargs)
    assert result.extended_t_half_h > result.baseline_t_half_h


@pytest.mark.parametrize("strategy,kwargs", [
    ("PEGylation", {"peg_mw_kDa": 20.0}),
    ("albumin_binding", {"albumin_affinity_uM": 1.0}),
    ("FcRn_recycling", {"mw_Da": 150000.0}),
    ("depot_formulation", {"depot_release_t50_h": 48.0}),
    ("prodrug", {"prodrug_conversion_t_half_h": 24.0}),
    ("nanoparticle", {}),
])
def test_extension_ratio_gt_1(strategy, kwargs):
    result = predict_half_life_extension("Drug", _CL, _VD, strategy=strategy, **kwargs)
    assert result.extension_ratio > 1.0


# --- AUC ratio ---

@pytest.mark.parametrize("strategy,kwargs", [
    ("PEGylation", {"peg_mw_kDa": 20.0}),
    ("albumin_binding", {"albumin_affinity_uM": 1.0}),
    ("FcRn_recycling", {"mw_Da": 150000.0}),
    ("nanoparticle", {}),
])
def test_auc_ratio_gte_1(strategy, kwargs):
    result = predict_half_life_extension("Drug", _CL, _VD, strategy=strategy, **kwargs)
    assert result.auc_ratio >= 1.0


# --- Mechanism field ---

def test_peg_mechanism():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="PEGylation", peg_mw_kDa=10)
    assert result.mechanism == "PEGylation"


def test_albumin_mechanism():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="albumin_binding")
    assert result.mechanism == "reduced_clearance"


def test_fcrn_mechanism():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="FcRn_recycling", mw_Da=150000)
    assert result.mechanism == "recycling"


def test_depot_mechanism():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="depot_formulation", depot_release_t50_h=24)
    assert result.mechanism == "depot"


def test_prodrug_mechanism():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="prodrug", prodrug_conversion_t_half_h=12)
    assert result.mechanism == "prodrug_conversion"


def test_nanoparticle_mechanism():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="nanoparticle")
    assert result.mechanism == "depot"


def test_all_mechanisms_valid():
    for strategy, kwargs in [
        ("PEGylation", {"peg_mw_kDa": 10}),
        ("albumin_binding", {}),
        ("FcRn_recycling", {"mw_Da": 150000}),
        ("depot_formulation", {"depot_release_t50_h": 24}),
        ("prodrug", {"prodrug_conversion_t_half_h": 12}),
        ("nanoparticle", {}),
    ]:
        result = predict_half_life_extension("Drug", _CL, _VD, strategy=strategy, **kwargs)
        assert result.mechanism in _VALID_MECHANISMS


# --- PEGylation high kDa ---

def test_peg_high_kDa_extension_gt2():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="PEGylation", peg_mw_kDa=40.0)
    assert result.extension_ratio > 2.0


# --- Albumin binding tight ---

def test_albumin_tight_Kd_strong():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="albumin_binding", albumin_affinity_uM=0.1)
    assert result.extension_ratio > 2.0


# --- FcRn for large mAb ---

def test_fcrn_large_mab():
    result = predict_half_life_extension("mAb", _CL, _VD, strategy="FcRn_recycling", mw_Da=150000.0)
    assert result.extension_ratio > 1.0
    assert result.mechanism == "recycling"


# --- dosing_frequency_reduction == extension_ratio ---

def test_dosing_freq_equals_extension_ratio():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="PEGylation", peg_mw_kDa=20)
    assert abs(result.dosing_frequency_reduction - result.extension_ratio) < 1e-9


# --- recommended flag ---

def test_recommended_when_ratio_ge_2():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="PEGylation", peg_mw_kDa=40)
    assert result.extension_ratio >= 2.0
    assert result.recommended is True


def test_not_recommended_when_ratio_lt_2():
    # Very small PEG → minimal extension
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="PEGylation", peg_mw_kDa=0.0)
    # With 0 kDa PEG: cl_reduction=1.0, vd_increase=1.0 → ratio=1.0
    assert result.extension_ratio < 2.0
    assert result.recommended is False


# --- notes nonempty ---

def test_notes_nonempty():
    result = predict_half_life_extension("Drug", _CL, _VD)
    assert len(result.notes) > 0


# --- Validation ---

def test_invalid_cl_raises():
    with pytest.raises(ValueError):
        predict_half_life_extension("Drug", 0.0, _VD)


def test_invalid_vd_raises():
    with pytest.raises(ValueError):
        predict_half_life_extension("Drug", _CL, 0.0)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        predict_half_life_extension("Drug", _CL, _VD, strategy="magic_pill")


# --- Baseline half-life computed correctly ---

def test_baseline_t_half_correct():
    result = predict_half_life_extension("Drug", _CL, _VD, strategy="nanoparticle")
    expected = math.log(2) * _VD / _CL
    assert abs(result.baseline_t_half_h - expected) < 1e-9
