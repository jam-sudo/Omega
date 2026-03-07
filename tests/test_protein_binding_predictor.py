"""Tests for omega_pbpk.prediction.protein_binding_predictor (Phase 440)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.protein_binding_predictor import (
    ProteinBindingPredResult,
    _binding_class,
    disease_effect_on_binding,
    predict_protein_binding,
)

# ---------------------------------------------------------------------------
# Default valid inputs
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    mw=350.0,
    logP=2.5,
    psa=60.0,
    pka=7.4,
    molecule_type="neutral",
    logD_pH74=1.8,
)


def pred(**kwargs):
    params = {**BASE_KWARGS, **kwargs}
    return predict_protein_binding(**params)


# ---------------------------------------------------------------------------
# Input validation — predict_protein_binding
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        pred(mw=0)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        pred(mw=-100)


def test_invalid_psa_negative():
    with pytest.raises(ValueError, match="psa"):
        pred(psa=-1)


def test_invalid_molecule_type():
    with pytest.raises(ValueError, match="molecule_type"):
        pred(molecule_type="amphoteric")


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_frozen_dataclass():
    result = pred()
    assert isinstance(result, ProteinBindingPredResult)


def test_frozen_dataclass_immutable():
    result = pred()
    with pytest.raises((AttributeError, TypeError)):
        result.fu_plasma = 0.5  # type: ignore[misc]


def test_fu_plasma_in_range():
    result = pred()
    assert 0.001 <= result.fu_plasma <= 1.0


def test_protein_binding_pct_matches_fu():
    result = pred()
    expected_pct = (1.0 - result.fu_plasma) * 100.0
    assert result.protein_binding_pct == pytest.approx(expected_pct, rel=1e-6)


def test_notes_nonempty():
    result = pred()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Primary protein assignment
# ---------------------------------------------------------------------------


def test_base_binds_agp():
    result = pred(molecule_type="base")
    assert result.primary_protein == "AGP"


def test_acid_binds_albumin():
    result = pred(molecule_type="acid")
    assert result.primary_protein == "Albumin"


def test_neutral_binds_albumin():
    result = pred(molecule_type="neutral")
    assert result.primary_protein == "Albumin"


def test_zwitterion_binds_albumin():
    result = pred(molecule_type="zwitterion")
    assert result.primary_protein == "Albumin"


def test_base_fu_lower_than_neutral_same_props():
    """Bases have AGP correction (×0.8) → lower fu."""
    r_base = pred(molecule_type="base")
    r_neutral = pred(molecule_type="neutral")
    assert r_base.fu_plasma < r_neutral.fu_plasma


# ---------------------------------------------------------------------------
# Lipophilicity effects
# ---------------------------------------------------------------------------


def test_high_logP_reduces_fu():
    """Higher logP → more hydrophobic → more bound → lower fu."""
    r_low = pred(logP=0.0)
    r_high = pred(logP=5.0)
    assert r_high.fu_plasma < r_low.fu_plasma


def test_very_low_logP_fu_close_to_one():
    """Very negative logP → essentially no hydrophobic binding."""
    result = pred(logP=-3.0, psa=0.0, logD_pH74=2.0)
    assert result.fu_plasma > 0.5


# ---------------------------------------------------------------------------
# Binding class classification
# ---------------------------------------------------------------------------


def test_binding_class_low():
    assert _binding_class(0.8) == "low"      # 20% bound


def test_binding_class_moderate():
    assert _binding_class(0.3) == "moderate"  # 70% bound


def test_binding_class_high():
    assert _binding_class(0.05) == "high"    # 95% bound


def test_binding_class_boundary_50():
    # exactly 50% bound → fu=0.5 → "moderate" (50-90 range, inclusive at 50)
    assert _binding_class(0.5) == "moderate"


def test_binding_class_boundary_10():
    # exactly 10% unbound → 90% bound → "moderate" (spec: >90% = high, so 90% = moderate)
    assert _binding_class(0.1) == "moderate"


def test_result_binding_class_matches_fu():
    result = pred(logP=4.0, psa=20.0, logD_pH74=3.5)
    expected = _binding_class(result.fu_plasma)
    assert result.binding_class == expected


# ---------------------------------------------------------------------------
# Disease effect on binding
# ---------------------------------------------------------------------------


def test_disease_hypoalbuminemia_increases_fu():
    d = disease_effect_on_binding(**BASE_KWARGS, disease="hypoalbuminemia")
    assert d["disease_fu"] > d["normal_fu"]


def test_disease_renal_failure_increases_fu():
    d = disease_effect_on_binding(**BASE_KWARGS, disease="renal_failure")
    assert d["disease_fu"] > d["normal_fu"]


def test_disease_hepatic_failure_increases_fu():
    d = disease_effect_on_binding(**BASE_KWARGS, disease="hepatic_failure")
    assert d["disease_fu"] > d["normal_fu"]


def test_disease_fu_ratio_hypoalbuminemia():
    d = disease_effect_on_binding(**BASE_KWARGS, disease="hypoalbuminemia")
    expected_ratio = d["disease_fu"] / d["normal_fu"]
    assert d["fu_ratio"] == pytest.approx(expected_ratio, rel=1e-6)


def test_disease_fu_clamped_at_one():
    """Even with high multiplier, fu cannot exceed 1.0."""
    d = disease_effect_on_binding(
        mw=100.0, logP=-5.0, psa=200.0, pka=7.0,
        molecule_type="neutral", logD_pH74=-4.0,
        disease="hypoalbuminemia",
    )
    assert d["disease_fu"] <= 1.0


def test_disease_invalid_name():
    with pytest.raises(ValueError, match="disease"):
        disease_effect_on_binding(**BASE_KWARGS, disease="cirrhosis")


def test_disease_invalid_mw():
    with pytest.raises(ValueError, match="mw"):
        disease_effect_on_binding(
            mw=-1, logP=2.0, psa=50.0, pka=7.0,
            molecule_type="neutral", logD_pH74=1.0,
            disease="renal_failure",
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_output():
    r1 = pred()
    r2 = pred()
    assert r1.fu_plasma == r2.fu_plasma
    assert r1.binding_class == r2.binding_class


def test_fu_clamped_minimum():
    """Very high logP → minimum fu = 0.001."""
    result = pred(logP=20.0, psa=0.0, logD_pH74=20.0)
    assert result.fu_plasma >= 0.001


def test_fu_clamped_maximum():
    """Very low logP → fu capped at 1.0."""
    result = pred(logP=-10.0, psa=0.0, logD_pH74=-10.0)
    assert result.fu_plasma <= 1.0
