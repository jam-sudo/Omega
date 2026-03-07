"""Tests for prediction/oral_bioavailability_model.py — Phase 434."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.oral_bioavailability_model import (
    OralBioavailabilityResult,
    estimate_oral_bioavailability,
    food_effect_on_f,
)

# ---------------------------------------------------------------------------
# Common test parameters
# ---------------------------------------------------------------------------

_DRUG = "test_drug"
_CLINT_LOW = 0.5  # L/h/kg — low CL → high fh
_CLINT_HIGH = 20.0  # L/h/kg — high CL → low fh
_FU = 0.1
_QH = 1.3  # L/h/kg
_MW = 350.0
_LOGP_LOW = 0.5
_LOGP_HIGH = 4.5
_PSA = 60.0

# ---------------------------------------------------------------------------
# estimate_oral_bioavailability — explicit fa/fg/fh provided
# ---------------------------------------------------------------------------


def test_explicit_f_components_used_directly():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.8,
        fg=0.9,
        fh=0.7,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    assert result.fa == pytest.approx(0.8)
    assert result.fg == pytest.approx(0.9)
    assert result.fh == pytest.approx(0.7)


def test_f_absolute_equals_product_of_components():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.8,
        fg=0.9,
        fh=0.7,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    expected = 0.8 * 0.9 * 0.7
    assert result.f_absolute == pytest.approx(expected, rel=1e-4)


def test_returns_result_dataclass():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=None,
        fg=None,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
        bcs_class=1,
    )
    assert isinstance(result, OralBioavailabilityResult)


def test_result_is_frozen():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.5,
        fg=0.8,
        fh=0.9,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.fa = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Estimated components from physicochemical properties
# ---------------------------------------------------------------------------


def test_bcs1_high_f_estimated():
    """BCS I drug with low CLint should have high estimated F."""
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=None,
        fg=None,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=0.1,
        fu_plasma=0.5,
        logP=2.0,
        psa=50.0,
        bcs_class=1,
    )
    assert result.f_absolute > 0.4


def test_bcs4_low_f_estimated():
    """BCS IV drug should have lower estimated F than BCS I."""
    bcs1 = estimate_oral_bioavailability(
        _DRUG,
        fa=None,
        fg=None,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=0.1,
        fu_plasma=0.5,
        logP=1.0,
        psa=120.0,
        bcs_class=1,
    )
    bcs4 = estimate_oral_bioavailability(
        _DRUG,
        fa=None,
        fg=None,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=0.1,
        fu_plasma=0.5,
        logP=1.0,
        psa=120.0,
        bcs_class=4,
    )
    assert bcs4.f_absolute < bcs1.f_absolute


def test_high_clint_reduces_fh():
    """Higher CLint should yield lower fh (more first-pass)."""
    low_cl = estimate_oral_bioavailability(
        _DRUG,
        fa=0.9,
        fg=0.9,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    high_cl = estimate_oral_bioavailability(
        _DRUG,
        fa=0.9,
        fg=0.9,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=_CLINT_HIGH,
        fu_plasma=_FU,
    )
    assert high_cl.fh < low_cl.fh


def test_eh_in_0_1_range():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.8,
        fg=0.8,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=5.0,
        fu_plasma=0.2,
    )
    assert 0.0 <= result.eh_hepatic <= 1.0


def test_fh_plus_eh_equals_1():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.8,
        fg=0.8,
        fh=None,
        cl_intrinsic_L_per_h_per_kg=3.0,
        fu_plasma=0.3,
    )
    assert result.fh + result.eh_hepatic == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# f_classification
# ---------------------------------------------------------------------------


def test_high_f_classification():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.95,
        fg=0.95,
        fh=0.9,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    assert result.f_classification == "high"


def test_moderate_f_classification():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.8,
        fg=0.8,
        fh=0.7,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    # 0.8*0.8*0.7 = 0.448 → moderate
    assert result.f_classification == "moderate"


def test_low_f_classification():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.1,
        fg=0.5,
        fh=0.4,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    # 0.1*0.5*0.4 = 0.02 → low
    assert result.f_classification == "low"


# ---------------------------------------------------------------------------
# limiting_factor
# ---------------------------------------------------------------------------


def test_limiting_factor_absorption():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.1,
        fg=0.95,
        fh=0.95,
        cl_intrinsic_L_per_h_per_kg=0.01,
        fu_plasma=0.5,
    )
    assert result.limiting_factor == "absorption"


def test_limiting_factor_hepatic():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.95,
        fg=0.95,
        fh=0.1,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    assert result.limiting_factor == "hepatic"


def test_limiting_factor_gut_wall():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.9,
        fg=0.2,
        fh=0.9,
        cl_intrinsic_L_per_h_per_kg=0.01,
        fu_plasma=0.5,
    )
    assert result.limiting_factor == "gut_wall"


# ---------------------------------------------------------------------------
# f_absolute clamping
# ---------------------------------------------------------------------------


def test_f_absolute_clamped_min_001():
    """Even with very high CLint and low fa, F should not go below 0.01."""
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.001,
        fg=0.001,
        fh=0.001,
        cl_intrinsic_L_per_h_per_kg=_CLINT_HIGH,
        fu_plasma=_FU,
    )
    assert result.f_absolute >= 0.01


def test_f_absolute_clamped_max_1():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=1.0,
        fg=1.0,
        fh=1.0,
        cl_intrinsic_L_per_h_per_kg=0.0,
        fu_plasma=_FU,
    )
    assert result.f_absolute <= 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_empty_drug_name_raises():
    with pytest.raises(ValueError, match="drug_name"):
        estimate_oral_bioavailability(
            "",
            fa=0.5,
            fg=0.5,
            fh=0.5,
            cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
            fu_plasma=_FU,
        )


def test_fa_out_of_range_raises():
    with pytest.raises(ValueError, match="fa"):
        estimate_oral_bioavailability(
            _DRUG,
            fa=1.5,
            fg=None,
            fh=None,
            cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
            fu_plasma=_FU,
        )


def test_invalid_bcs_class_raises():
    with pytest.raises(ValueError, match="bcs_class"):
        estimate_oral_bioavailability(
            _DRUG,
            fa=None,
            fg=None,
            fh=None,
            cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
            fu_plasma=_FU,
            bcs_class=5,
        )


def test_invalid_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        estimate_oral_bioavailability(
            _DRUG,
            fa=0.5,
            fg=0.5,
            fh=0.5,
            cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
            fu_plasma=0.0,
        )


def test_invalid_qh_zero_raises():
    with pytest.raises(ValueError, match="qh_L_per_h_per_kg"):
        estimate_oral_bioavailability(
            _DRUG,
            fa=0.5,
            fg=0.5,
            fh=0.5,
            cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
            fu_plasma=_FU,
            qh_L_per_h_per_kg=0.0,
        )


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_non_empty_string():
    result = estimate_oral_bioavailability(
        _DRUG,
        fa=0.7,
        fg=0.8,
        fh=0.9,
        cl_intrinsic_L_per_h_per_kg=_CLINT_LOW,
        fu_plasma=_FU,
    )
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# food_effect_on_f
# ---------------------------------------------------------------------------


def test_food_effect_returns_dict():
    result = food_effect_on_f("drug_a", bcs_class=2, logP=4.0, dissolution_rate_mg_min=0.5)
    assert isinstance(result, dict)


def test_food_effect_keys_present():
    result = food_effect_on_f("drug_a", bcs_class=1, logP=2.0, dissolution_rate_mg_min=5.0)
    for key in ("fasted_f", "fed_f", "food_effect_ratio", "recommendation"):
        assert key in result


def test_bcs2_lipophilic_food_increases_f():
    result = food_effect_on_f("drug_b", bcs_class=2, logP=5.0, dissolution_rate_mg_min=0.3)
    assert result["fed_f"] > result["fasted_f"]


def test_bcs2_food_effect_ratio_gt_1():
    result = food_effect_on_f("drug_b", bcs_class=2, logP=4.0, dissolution_rate_mg_min=1.0)
    assert result["food_effect_ratio"] > 1.0


def test_bcs1_minimal_food_effect():
    bcs1 = food_effect_on_f("drug_c", bcs_class=1, logP=1.5, dissolution_rate_mg_min=10.0)
    bcs2 = food_effect_on_f("drug_c", bcs_class=2, logP=4.5, dissolution_rate_mg_min=0.5)
    assert bcs2["food_effect_ratio"] > bcs1["food_effect_ratio"]


def test_fed_f_clamped_to_1():
    result = food_effect_on_f("drug_d", bcs_class=1, logP=3.0, dissolution_rate_mg_min=20.0)
    assert result["fed_f"] <= 1.0


def test_recommendation_is_string():
    result = food_effect_on_f("drug_e", bcs_class=2, logP=3.0, dissolution_rate_mg_min=2.0)
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 0


def test_food_effect_invalid_bcs_raises():
    with pytest.raises(ValueError, match="bcs_class"):
        food_effect_on_f("drug_f", bcs_class=0, logP=2.0, dissolution_rate_mg_min=5.0)


def test_food_effect_empty_drug_name_raises():
    with pytest.raises(ValueError, match="drug_name"):
        food_effect_on_f("", bcs_class=1, logP=2.0, dissolution_rate_mg_min=5.0)


def test_food_effect_negative_dissolution_raises():
    with pytest.raises(ValueError, match="dissolution_rate_mg_min"):
        food_effect_on_f("drug_g", bcs_class=1, logP=2.0, dissolution_rate_mg_min=-1.0)


def test_drug_name_stored_in_result():
    result = food_effect_on_f("warfarin", bcs_class=2, logP=2.7, dissolution_rate_mg_min=1.0)
    assert result["drug_name"] == "warfarin"


def test_bcs4_food_effect_positive():
    """BCS IV drug with lipophilicity should still benefit from food."""
    result = food_effect_on_f("drug_h", bcs_class=4, logP=4.0, dissolution_rate_mg_min=0.5)
    assert result["food_effect_ratio"] >= 1.0
