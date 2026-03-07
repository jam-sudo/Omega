"""Tests for clearance scaling (IVIVE and allometric) (Phase 676)."""

import pytest

from omega_pbpk.prediction.clearance_scaling import (
    CLScalingResult,
    allometric_cl_scaling,
    multi_species_cl_scaling,
    scale_clint_to_clh,
)

# ---------------------------------------------------------------------------
# scale_clint_to_clh — return type and keys
# ---------------------------------------------------------------------------


def test_scale_clint_returns_dict():
    result = scale_clint_to_clh(clint_uL_per_min_per_mg=10.0)
    assert isinstance(result, dict)


def test_scale_clint_has_expected_keys():
    result = scale_clint_to_clh(clint_uL_per_min_per_mg=10.0)
    expected_keys = {
        "clint_scaled_L_per_h",
        "fu_mic",
        "clint_u_L_per_h",
        "clh_L_per_h",
        "er",
        "notes",
    }
    assert expected_keys.issubset(result.keys())


def test_scale_clint_clh_positive():
    result = scale_clint_to_clh(clint_uL_per_min_per_mg=10.0)
    assert result["clh_L_per_h"] > 0


def test_scale_clint_er_in_zero_one():
    result = scale_clint_to_clh(clint_uL_per_min_per_mg=10.0)
    assert 0 < result["er"] < 1


def test_scale_clint_er_increases_with_higher_clint():
    r_low = scale_clint_to_clh(clint_uL_per_min_per_mg=1.0)
    r_high = scale_clint_to_clh(clint_uL_per_min_per_mg=100.0)
    assert r_high["er"] > r_low["er"]


def test_scale_clint_lower_fup_lower_clh():
    r_high_fup = scale_clint_to_clh(clint_uL_per_min_per_mg=10.0, fup=1.0)
    r_low_fup = scale_clint_to_clh(clint_uL_per_min_per_mg=10.0, fup=0.1)
    assert r_low_fup["clh_L_per_h"] < r_high_fup["clh_L_per_h"]


def test_scale_clint_fu_mic_formula():
    protein_conc = 1.0
    result = scale_clint_to_clh(clint_uL_per_min_per_mg=10.0, protein_conc_mg_per_mL=protein_conc)
    expected_fu_mic = 1.0 / (1.0 + 0.072 * protein_conc)
    assert result["fu_mic"] == pytest.approx(expected_fu_mic, rel=1e-6)


def test_scale_clint_er_cannot_exceed_one():
    # Very high CLint -> ER approaches 1 but not exceed
    result = scale_clint_to_clh(clint_uL_per_min_per_mg=1e6)
    assert result["er"] < 1.0


# ---------------------------------------------------------------------------
# scale_clint_to_clh — validation errors
# ---------------------------------------------------------------------------


def test_scale_clint_zero_clint_raises():
    with pytest.raises(ValueError, match="clint"):
        scale_clint_to_clh(clint_uL_per_min_per_mg=0.0)


def test_scale_clint_zero_liver_weight_raises():
    with pytest.raises(ValueError, match="liver_weight_g"):
        scale_clint_to_clh(clint_uL_per_min_per_mg=10.0, liver_weight_g=0.0)


def test_scale_clint_invalid_fup_raises():
    with pytest.raises(ValueError, match="fup"):
        scale_clint_to_clh(clint_uL_per_min_per_mg=10.0, fup=0.0)


def test_scale_clint_fup_gt_one_raises():
    with pytest.raises(ValueError, match="fup"):
        scale_clint_to_clh(clint_uL_per_min_per_mg=10.0, fup=1.5)


# ---------------------------------------------------------------------------
# allometric_cl_scaling
# ---------------------------------------------------------------------------


def test_allometric_returns_dict():
    result = allometric_cl_scaling(cl_animal_L_per_h=1.0, bw_animal_kg=0.25)
    assert isinstance(result, dict)


def test_allometric_has_expected_keys():
    result = allometric_cl_scaling(cl_animal_L_per_h=1.0, bw_animal_kg=0.25)
    assert {"cl_human_L_per_h", "scaling_factor", "exponent", "notes"}.issubset(result.keys())


def test_allometric_rat_to_human():
    # Rat 0.25 kg, human 70 kg, exponent 0.75
    cl_rat = 2.0
    bw_rat = 0.25
    result = allometric_cl_scaling(cl_animal_L_per_h=cl_rat, bw_animal_kg=bw_rat)
    expected = cl_rat * (70.0 / bw_rat) ** 0.75
    assert result["cl_human_L_per_h"] == pytest.approx(expected, rel=1e-6)


def test_allometric_custom_exponent():
    result = allometric_cl_scaling(
        cl_animal_L_per_h=1.0, bw_animal_kg=5.0, bw_human_kg=70.0, exponent=0.8
    )
    expected = 1.0 * (70.0 / 5.0) ** 0.8
    assert result["cl_human_L_per_h"] == pytest.approx(expected, rel=1e-6)


def test_allometric_same_species_gives_same_cl():
    # If animal BW == human BW, CL should be unchanged
    result = allometric_cl_scaling(cl_animal_L_per_h=5.0, bw_animal_kg=70.0, bw_human_kg=70.0)
    assert result["cl_human_L_per_h"] == pytest.approx(5.0, rel=1e-6)


def test_allometric_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_animal"):
        allometric_cl_scaling(cl_animal_L_per_h=0.0, bw_animal_kg=0.25)


def test_allometric_zero_bw_animal_raises():
    with pytest.raises(ValueError, match="bw_animal_kg"):
        allometric_cl_scaling(cl_animal_L_per_h=1.0, bw_animal_kg=0.0)


# ---------------------------------------------------------------------------
# multi_species_cl_scaling
# ---------------------------------------------------------------------------

_MULTI_SPECIES = [
    {"species": "mouse", "cl_L_per_h": 0.5, "bw_kg": 0.025},
    {"species": "rat", "cl_L_per_h": 2.0, "bw_kg": 0.25},
    {"species": "dog", "cl_L_per_h": 15.0, "bw_kg": 10.0},
]


def test_multi_species_returns_dict():
    result = multi_species_cl_scaling(_MULTI_SPECIES)
    assert isinstance(result, dict)


def test_multi_species_has_expected_keys():
    result = multi_species_cl_scaling(_MULTI_SPECIES)
    assert {
        "predicted_cl_human_L_per_h",
        "fitted_exponent",
        "r_squared",
        "species_used",
        "notes",
    }.issubset(result.keys())


def test_multi_species_predicted_cl_positive():
    result = multi_species_cl_scaling(_MULTI_SPECIES)
    assert result["predicted_cl_human_L_per_h"] > 0


def test_multi_species_r_squared_in_range():
    result = multi_species_cl_scaling(_MULTI_SPECIES)
    assert 0.0 <= result["r_squared"] <= 1.0


def test_multi_species_fitted_exponent_reasonable():
    # Standard allometric exponent for CL should be near 0.75
    result = multi_species_cl_scaling(_MULTI_SPECIES)
    assert 0.5 <= result["fitted_exponent"] <= 1.1


def test_multi_species_returns_correct_species_used():
    result = multi_species_cl_scaling(_MULTI_SPECIES)
    assert "mouse" in result["species_used"]
    assert "rat" in result["species_used"]
    assert "dog" in result["species_used"]


def test_multi_species_fewer_than_2_raises():
    with pytest.raises(ValueError, match="At least 2 species"):
        multi_species_cl_scaling([{"species": "rat", "cl_L_per_h": 2.0, "bw_kg": 0.25}])


def test_multi_species_empty_raises():
    with pytest.raises(ValueError, match="At least 2 species"):
        multi_species_cl_scaling([])


# ---------------------------------------------------------------------------
# CLScalingResult dataclass
# ---------------------------------------------------------------------------


def test_cl_scaling_result_created_correctly():
    r = CLScalingResult(
        compound_name="TestCompound",
        clint_in_vitro=10.0,
        clh_predicted=25.0,
        er=0.28,
        method="well-stirred",
        notes="Test note.",
    )
    assert r.compound_name == "TestCompound"
    assert r.clint_in_vitro == pytest.approx(10.0)
    assert r.clh_predicted == pytest.approx(25.0)
    assert r.er == pytest.approx(0.28)
    assert r.method == "well-stirred"
    assert "Test note" in r.notes
