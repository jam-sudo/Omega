"""Tests for Phase 1053 — Drug Tumor Penetration Index."""

import pytest

from omega_pbpk.prediction.tumor_penetration_index import (
    TumorPenetrationResult,
    predict_tumor_penetration,
)


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        mw_Da=400.0,
        logp=2.5,
        pka=7.0,
        ionization_type="neutral",
        tumor_type="solid",
        tumor_vascularity="moderate",
        has_efflux_pump=False,
        protein_binding_pct=90.0,
    )
    params.update(kwargs)
    return predict_tumor_penetration(**params)


# --- Return type ---


def test_returns_tumor_penetration_result():
    result = _default()
    assert isinstance(result, TumorPenetrationResult)


def test_result_is_frozen():
    result = _default()
    with pytest.raises((AttributeError, TypeError)):
        result.penetration_index = 0.5  # type: ignore[misc]


# --- Range checks ---


def test_penetration_index_in_range():
    result = _default()
    assert 0.01 <= result.penetration_index <= 0.95


def test_diffusion_coefficient_positive():
    result = _default()
    assert result.diffusion_coefficient_cm2_h > 0


def test_effective_diffusion_depth_positive():
    result = _default()
    assert result.effective_diffusion_depth_um > 0


def test_dose_required_fold_increase_ge_1():
    result = _default()
    assert result.dose_required_fold_increase >= 1.0


def test_penetration_class_valid():
    valid = {"excellent", "good", "moderate", "poor"}
    for tumor_type in ("solid", "hematologic", "pancreatic", "brain"):
        result = _default(tumor_type=tumor_type)
        assert result.penetration_class in valid


def test_notes_nonempty():
    result = _default()
    assert len(result.notes) > 0


# --- Comparative / directional ---


def test_pancreatic_lower_than_hematologic():
    panc = _default(tumor_type="pancreatic", protein_binding_pct=0.0)
    hema = _default(tumor_type="hematologic", protein_binding_pct=0.0)
    assert panc.penetration_index < hema.penetration_index


def test_efflux_pump_reduces_penetration():
    no_efflux = _default(protein_binding_pct=50.0)
    with_efflux = _default(protein_binding_pct=50.0, has_efflux_pump=True)
    assert with_efflux.penetration_index < no_efflux.penetration_index


def test_high_protein_binding_reduces_penetration():
    low_bind = _default(protein_binding_pct=20.0)
    high_bind = _default(protein_binding_pct=95.0)
    assert high_bind.penetration_index < low_bind.penetration_index


def test_low_vascularity_reduces_vascular_access_score():
    low_vasc = _default(tumor_vascularity="low")
    high_vasc = _default(tumor_vascularity="high")
    assert low_vasc.vascular_access_score < high_vasc.vascular_access_score


def test_ifp_factor_matches_tumor_type():
    result_panc = _default(tumor_type="pancreatic")
    result_hema = _default(tumor_type="hematologic")
    assert (
        result_panc.interstitial_fluid_pressure_factor
        < result_hema.interstitial_fluid_pressure_factor
    )


def test_binding_factor_zero_binding():
    result = _default(protein_binding_pct=0.0)
    assert abs(result.binding_factor - 1.0) < 1e-9


def test_binding_factor_full_binding():
    result = _default(protein_binding_pct=100.0)
    assert abs(result.binding_factor - 0.0) < 1e-9


def test_penetration_class_consistency():
    result = _default(protein_binding_pct=0.0, tumor_type="hematologic", tumor_vascularity="high")
    if result.penetration_index > 0.7:
        assert result.penetration_class == "excellent"
    elif result.penetration_index >= 0.5:
        assert result.penetration_class == "good"
    elif result.penetration_index >= 0.3:
        assert result.penetration_class == "moderate"
    else:
        assert result.penetration_class == "poor"


def test_dose_fold_increase_inverse_of_pi():
    result = _default(protein_binding_pct=50.0)
    expected = max(1.0, 1.0 / result.penetration_index)
    assert abs(result.dose_required_fold_increase - expected) < 1e-9


def test_large_mw_reduces_diffusion_coefficient():
    small_mw = _default(mw_Da=200.0)
    large_mw = _default(mw_Da=1000.0)
    assert large_mw.diffusion_coefficient_cm2_h <= small_mw.diffusion_coefficient_cm2_h


# --- Validation ---


def test_raises_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _default(mw_Da=0.0)


def test_raises_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        _default(mw_Da=-1.0)


def test_raises_invalid_tumor_type():
    with pytest.raises(ValueError, match="tumor_type"):
        _default(tumor_type="unknown_tumor")


def test_raises_invalid_vascularity():
    with pytest.raises(ValueError, match="tumor_vascularity"):
        _default(tumor_vascularity="very_high")


def test_raises_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        _default(ionization_type="zwitterion")


def test_raises_protein_binding_out_of_range():
    with pytest.raises(ValueError, match="protein_binding_pct"):
        _default(protein_binding_pct=110.0)
