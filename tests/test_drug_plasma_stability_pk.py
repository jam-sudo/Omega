"""Tests for Phase 946 — Drug Plasma Stability PK."""

import pytest

from omega_pbpk.prediction.drug_plasma_stability_pk import (
    PlasmaStabilityPKResult,
    predict_plasma_stability_pk,
    screen_plasma_stability,
)


def make_default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        logp=2.5,
        mw=350.0,
        cl_hepatic_L_per_h=10.0,
        vd_L=50.0,
        cl_renal_L_per_h=0.0,
        has_ester=False,
        has_carbonate=False,
        has_lactone=False,
        has_amide=False,
        experimental_t_half_min=None,
    )
    defaults.update(kwargs)
    return predict_plasma_stability_pk(**defaults)


# --- Return type ---


def test_return_type():
    result = make_default()
    assert isinstance(result, PlasmaStabilityPKResult)


def test_frozen_dataclass():
    result = make_default()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore


# --- Basic positive metrics ---


def test_t_half_plasma_stability_positive():
    result = make_default()
    assert result.t_half_plasma_stability_min > 0


def test_stability_category_valid():
    result = make_default()
    assert result.stability_category in {"unstable", "labile", "stable"}


def test_k_plasma_deg_positive():
    result = make_default()
    assert result.k_plasma_deg_per_h > 0


def test_cl_plasma_deg_positive():
    result = make_default()
    assert result.cl_plasma_deg_L_per_h > 0


def test_fraction_in_range():
    result = make_default()
    assert 0.0 <= result.fraction_degraded_vs_metabolized <= 1.0


def test_predicted_cl_positive():
    result = make_default()
    assert result.predicted_in_vivo_cl_L_per_h > 0


def test_t_half_in_vivo_positive():
    result = make_default()
    assert result.t_half_in_vivo_h > 0


def test_notes_non_empty():
    result = make_default()
    assert len(result.notes) > 0


# --- Structural alert categories ---


def test_ester_labile_or_unstable():
    result = make_default(has_ester=True)
    # t_half=60 min → boundary: < 60 unstable, >=60 & <240 labile
    # exactly 60 min → "labile" (60 is not < 60)
    assert result.stability_category in {"unstable", "labile"}


def test_carbonate_unstable():
    result = make_default(has_carbonate=True)
    # t_half = 30 min < 60 → unstable
    assert result.stability_category == "unstable"


def test_no_alerts_stable():
    result = make_default()
    # t_half = 10000 min → stable
    assert result.stability_category == "stable"


# --- Experimental t_half override ---


def test_experimental_t_half_used():
    result = make_default(experimental_t_half_min=30.0)
    assert result.t_half_plasma_stability_min == 30.0


def test_experimental_overrides_structural():
    # structural would be stable (no alerts), but experimental says 30 min
    result = make_default(experimental_t_half_min=30.0)
    assert result.stability_category == "unstable"


# --- Ester faster than amide ---


def test_ester_faster_deg_than_amide():
    ester = make_default(has_ester=True)
    amide = make_default(has_amide=True)
    assert ester.k_plasma_deg_per_h > amide.k_plasma_deg_per_h


# --- Fraction degraded comparison ---


def test_ester_higher_fraction_than_stable():
    ester = make_default(has_ester=True)
    stable = make_default()
    assert ester.fraction_degraded_vs_metabolized > stable.fraction_degraded_vs_metabolized


# --- screen_plasma_stability ---


def test_screen_correct_length():
    compounds = [
        {"drug_name": "DrugA", "logp": 1.0, "mw": 200.0, "has_ester": True},
        {"drug_name": "DrugB", "logp": 2.0, "mw": 300.0, "has_carbonate": True},
        {"drug_name": "DrugC", "logp": 3.0, "mw": 400.0},
    ]
    results = screen_plasma_stability(compounds)
    assert len(results) == 3


def test_screen_sorted_ascending():
    compounds = [
        {"drug_name": "DrugA", "logp": 1.0, "mw": 200.0},  # stable, t=10000
        {"drug_name": "DrugB", "logp": 2.0, "mw": 300.0, "has_ester": True},  # t=60
        {"drug_name": "DrugC", "logp": 3.0, "mw": 400.0, "has_carbonate": True},  # t=30
    ]
    results = screen_plasma_stability(compounds)
    t_halfs = [r.t_half_plasma_stability_min for r in results]
    assert t_halfs == sorted(t_halfs)


def test_screen_drug_name_preserved():
    compounds = [
        {"drug_name": "SpecialDrug", "logp": 2.0, "mw": 250.0},
    ]
    results = screen_plasma_stability(compounds)
    assert results[0].drug_name == "SpecialDrug"


# --- Validation errors ---


def test_validation_cl_hepatic_negative():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        make_default(cl_hepatic_L_per_h=-1.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        make_default(vd_L=0.0)


def test_validation_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        make_default(vd_L=-10.0)


def test_validation_cl_renal_negative():
    with pytest.raises(ValueError, match="cl_renal_L_per_h"):
        make_default(cl_renal_L_per_h=-0.5)


def test_validation_experimental_t_half_zero():
    with pytest.raises(ValueError, match="experimental_t_half_min"):
        make_default(experimental_t_half_min=0.0)


def test_validation_experimental_t_half_negative():
    with pytest.raises(ValueError, match="experimental_t_half_min"):
        make_default(experimental_t_half_min=-10.0)
