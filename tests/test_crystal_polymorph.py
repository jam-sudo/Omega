"""Tests for crystal polymorph impact assessment (Phase 185)."""

import math

import pytest

from omega_pbpk.biopharmaceutics.crystal_polymorph import (
    CrystalForm,
    PolymorphResult,
    assess_polymorph_impact,
    compare_polymorphs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FORM_I = CrystalForm(
    form_name="Form I",
    relative_solubility=1.0,
    melting_point_C=180.0,
    is_amorphous=False,
)

FORM_II = CrystalForm(
    form_name="Form II",
    relative_solubility=1.8,
    melting_point_C=175.0,
    is_amorphous=False,
)

FORM_HIGH = CrystalForm(
    form_name="Form III",
    relative_solubility=3.5,
    melting_point_C=160.0,
    is_amorphous=False,
)

AMORPHOUS = CrystalForm(
    form_name="Amorphous",
    relative_solubility=10.0,
    melting_point_C=0.0,  # no defined melting point
    is_amorphous=True,
)

FORM_POOR = CrystalForm(
    form_name="Form IV",
    relative_solubility=0.5,
    melting_point_C=185.0,
    is_amorphous=False,
)


# ---------------------------------------------------------------------------
# Basic construction / dataclass
# ---------------------------------------------------------------------------


def test_crystal_form_dataclass():
    assert FORM_I.form_name == "Form I"
    assert FORM_I.relative_solubility == 1.0
    assert not FORM_I.is_amorphous


def test_polymorph_result_frozen():
    result = assess_polymorph_impact("Drug A", 0.1, FORM_I)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Solubility computation
# ---------------------------------------------------------------------------


def test_form_i_solubility_equals_reference():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_I)
    assert math.isclose(result.solubility_mg_mL, 0.5, rel_tol=1e-9)


def test_form_ii_solubility_scales():
    result = assess_polymorph_impact("Drug A", 0.2, FORM_II)
    expected = 0.2 * 1.8
    assert math.isclose(result.solubility_mg_mL, expected, rel_tol=1e-9)


def test_amorphous_high_solubility():
    result = assess_polymorph_impact("Drug A", 0.1, AMORPHOUS)
    assert result.solubility_mg_mL > 0.1


# ---------------------------------------------------------------------------
# Dissolution rate
# ---------------------------------------------------------------------------


def test_dissolution_rate_positive():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_I)
    assert result.dissolution_rate_mg_per_min > 0


def test_dissolution_rate_scales_with_surface_area():
    r1 = assess_polymorph_impact("Drug A", 0.5, FORM_I, surface_area_cm2=1.0)
    r2 = assess_polymorph_impact("Drug A", 0.5, FORM_I, surface_area_cm2=2.0)
    assert math.isclose(
        r2.dissolution_rate_mg_per_min, r1.dissolution_rate_mg_per_min * 2, rel_tol=1e-9
    )


def test_dissolution_rate_scales_with_solubility():
    r1 = assess_polymorph_impact("Drug A", 0.2, FORM_I)
    r2 = assess_polymorph_impact("Drug A", 0.2, FORM_II)
    assert r2.dissolution_rate_mg_per_min > r1.dissolution_rate_mg_per_min


# ---------------------------------------------------------------------------
# Bioavailability impact
# ---------------------------------------------------------------------------


def test_form_i_ba_impact_zero_bcs_ii():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_I, bcs_class="II")
    assert math.isclose(result.bioavailability_impact_pct, 0.0, abs_tol=1e-9)


def test_bcs_i_ba_impact_always_zero():
    for form in [FORM_I, FORM_II, AMORPHOUS]:
        result = assess_polymorph_impact("Drug A", 0.5, form, bcs_class="I")
        assert math.isclose(result.bioavailability_impact_pct, 0.0, abs_tol=1e-9)


def test_bcs_iii_ba_impact_zero():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_II, bcs_class="III")
    assert math.isclose(result.bioavailability_impact_pct, 0.0, abs_tol=1e-9)


def test_bcs_ii_higher_solubility_positive_ba_impact():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_II, bcs_class="II")
    assert result.bioavailability_impact_pct > 0.0


def test_bcs_iv_higher_solubility_positive_ba_impact():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_II, bcs_class="IV")
    assert result.bioavailability_impact_pct > 0.0


def test_bcs_ii_ba_impact_formula():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_II, bcs_class="II")
    expected = (FORM_II.relative_solubility**0.5 - 1.0) * 100.0
    assert math.isclose(result.bioavailability_impact_pct, expected, rel_tol=1e-9)


def test_bcs_ii_lower_solubility_negative_ba_impact():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_POOR, bcs_class="II")
    assert result.bioavailability_impact_pct < 0.0


# ---------------------------------------------------------------------------
# Stability risk
# ---------------------------------------------------------------------------


def test_amorphous_stability_high():
    result = assess_polymorph_impact("Drug A", 0.5, AMORPHOUS)
    assert result.stability_risk == "high"


def test_high_relative_solubility_moderate_risk():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_HIGH)
    assert result.stability_risk == "moderate"


def test_normal_form_low_risk():
    result = assess_polymorph_impact("Drug A", 0.5, FORM_I)
    assert result.stability_risk == "low"


def test_form_ii_below_2x_low_risk():
    # relative_solubility=1.8 < 2.0 and not amorphous → low
    result = assess_polymorph_impact("Drug A", 0.5, FORM_II)
    assert result.stability_risk == "low"


# ---------------------------------------------------------------------------
# compare_polymorphs
# ---------------------------------------------------------------------------


def test_compare_polymorphs_returns_sorted():
    forms = [FORM_I, FORM_II, AMORPHOUS, FORM_POOR]
    results = compare_polymorphs("Drug A", 0.2, forms, bcs_class="II")
    ba_impacts = [r.bioavailability_impact_pct for r in results]
    assert ba_impacts == sorted(ba_impacts, reverse=True)


def test_compare_polymorphs_length():
    forms = [FORM_I, FORM_II, AMORPHOUS]
    results = compare_polymorphs("Drug A", 0.2, forms)
    assert len(results) == 3


def test_compare_polymorphs_types():
    results = compare_polymorphs("Drug A", 0.2, [FORM_I, FORM_II])
    for r in results:
        assert isinstance(r, PolymorphResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_reference_solubility_zero():
    with pytest.raises(ValueError, match="reference_solubility_mg_mL"):
        assess_polymorph_impact("Drug A", 0.0, FORM_I)


def test_invalid_reference_solubility_negative():
    with pytest.raises(ValueError, match="reference_solubility_mg_mL"):
        assess_polymorph_impact("Drug A", -1.0, FORM_I)


def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        assess_polymorph_impact("Drug A", 0.5, FORM_I, dose_mg=0.0)


def test_invalid_bcs_class():
    with pytest.raises(ValueError, match="bcs_class"):
        assess_polymorph_impact("Drug A", 0.5, FORM_I, bcs_class="V")
