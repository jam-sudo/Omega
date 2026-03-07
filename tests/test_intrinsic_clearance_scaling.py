"""Tests for Phase 181 — intrinsic_clearance_scaling."""

import pytest

from omega_pbpk.prediction.intrinsic_clearance_scaling import (
    ClintScalingResult,
    compare_species,
    scale_clint,
)

# ---------------------------------------------------------------------------
# Basic return-type and field-preservation tests
# ---------------------------------------------------------------------------


def test_scale_clint_returns_dataclass():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    assert isinstance(result, ClintScalingResult)


def test_scale_clint_drug_name_preserved():
    result = scale_clint("Warfarin", clint_uL_per_min_per_mg=5.0, fu_mic=0.1)
    assert result.drug_name == "Warfarin"


def test_scale_clint_species_default_human():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    assert result.species == "human"


def test_scale_clint_species_custom():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5, species="rat")
    assert result.species == "rat"


def test_scale_clint_fu_mic_preserved():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.3)
    assert result.fu_mic == pytest.approx(0.3)


def test_scale_clint_hpgl_preserved():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5, hpgl=50.0)
    assert result.hpgl == pytest.approx(50.0)


def test_scale_clint_invitro_preserved():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=15.0, fu_mic=0.5)
    assert result.clint_invitro_uL_min_mg == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Numerical scaling tests
# ---------------------------------------------------------------------------


def test_clint_invivo_proportional_to_invitro():
    """Doubling CLint_invitro should double CLint_invivo."""
    r1 = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0)
    r2 = scale_clint("D", clint_uL_per_min_per_mg=20.0, fu_mic=1.0)
    assert r2.clint_invivo_mL_per_min == pytest.approx(2.0 * r1.clint_invivo_mL_per_min, rel=1e-6)


def test_clint_invivo_proportional_to_hpgl():
    """Doubling HPGL should double CLint_invivo."""
    r1 = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0, hpgl=45.0)
    r2 = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0, hpgl=90.0)
    assert r2.clint_invivo_mL_per_min == pytest.approx(2.0 * r1.clint_invivo_mL_per_min, rel=1e-6)


def test_clint_invivo_proportional_to_liver_weight():
    """Doubling liver weight should double CLint_invivo."""
    r1 = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0, liver_weight_g=1000.0)
    r2 = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0, liver_weight_g=2000.0)
    assert r2.clint_invivo_mL_per_min == pytest.approx(2.0 * r1.clint_invivo_mL_per_min, rel=1e-6)


def test_extraction_ratio_in_unit_interval():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    assert 0.0 < result.extraction_ratio < 1.0


def test_extraction_ratio_bounded_above_by_one():
    """Even with very high CLint, ER must stay below 1."""
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=1e6, fu_mic=1.0)
    assert result.extraction_ratio < 1.0


def test_high_clint_gives_high_extraction_ratio():
    result = scale_clint("HighCL", clint_uL_per_min_per_mg=500.0, fu_mic=1.0)
    assert result.extraction_ratio > 0.7


def test_low_clint_gives_low_extraction_ratio():
    result = scale_clint("LowCL", clint_uL_per_min_per_mg=0.1, fu_mic=1.0)
    assert result.extraction_ratio < 0.3


def test_cl_hepatic_positive():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    assert result.cl_hepatic_L_per_h > 0.0


def test_cl_hepatic_less_than_qh():
    """CL_h must never exceed hepatic blood flow (90 L/h by default)."""
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=1e6, fu_mic=1.0)
    assert result.cl_hepatic_L_per_h < 90.0


def test_notes_not_empty():
    result = scale_clint("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    assert len(result.notes) > 0


def test_default_liver_weight_from_bw():
    """Default liver weight = 21.4 g/kg * 70 kg = 1498 g."""
    r_default = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0)
    r_explicit = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0, liver_weight_g=1498.0)
    assert r_default.clint_invivo_mL_per_min == pytest.approx(
        r_explicit.clint_invivo_mL_per_min, rel=1e-4
    )


# ---------------------------------------------------------------------------
# compare_species tests
# ---------------------------------------------------------------------------


def test_compare_species_returns_four():
    results = compare_species("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    assert len(results) == 4


def test_compare_species_all_results_are_dataclass():
    results = compare_species("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    for r in results:
        assert isinstance(r, ClintScalingResult)


def test_compare_species_species_names():
    results = compare_species("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    species_names = {r.species for r in results}
    assert species_names == {"mouse", "rat", "dog", "human"}


def test_compare_species_extraction_ratios_in_bounds():
    results = compare_species("DrugA", clint_uL_per_min_per_mg=10.0, fu_mic=0.5)
    for r in results:
        assert 0.0 < r.extraction_ratio < 1.0


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


def test_validate_clint_positive():
    with pytest.raises(ValueError, match="clint"):
        scale_clint("D", clint_uL_per_min_per_mg=0.0, fu_mic=0.5)


def test_validate_clint_negative():
    with pytest.raises(ValueError, match="clint"):
        scale_clint("D", clint_uL_per_min_per_mg=-5.0, fu_mic=0.5)


def test_validate_fu_mic_zero():
    with pytest.raises(ValueError, match="fu_mic"):
        scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=0.0)


def test_validate_fu_mic_above_one():
    with pytest.raises(ValueError, match="fu_mic"):
        scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.1)


def test_validate_body_weight_positive():
    with pytest.raises(ValueError, match="body_weight_kg"):
        scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=0.5, body_weight_kg=0.0)


def test_fu_mic_one_allowed():
    """fu_mic = 1.0 (fully unbound) should be valid."""
    result = scale_clint("D", clint_uL_per_min_per_mg=10.0, fu_mic=1.0)
    assert isinstance(result, ClintScalingResult)
