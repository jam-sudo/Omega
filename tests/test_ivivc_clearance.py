"""Tests for Phase 386: In Vitro-In Vivo Correlation for Clearance."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.ivivc_clearance import (
    IVIVCClearanceResult,
    compare_species,
    predict_clearance_ivivc,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_ivivc_clearance_result():
    result = predict_clearance_ivivc("DrugA", cl_int_uL_per_min_per_mg_mic=10.0, fu_plasma=0.1)
    assert isinstance(result, IVIVCClearanceResult)


def test_species_stored_correctly():
    result = predict_clearance_ivivc("DrugA", 10.0, 0.1, species="rat")
    assert result.species == "rat"


def test_drug_name_stored():
    result = predict_clearance_ivivc("Ibuprofen", 5.0, 0.05)
    assert result.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# Scaling factor correctness for human
# ---------------------------------------------------------------------------


def test_human_scaling_factor():
    """Human: MPPGL=45, liver_weight=25.7 g/kg -> scaling_factor=45*25.7=1156.5."""
    result = predict_clearance_ivivc("DrugB", 10.0, 1.0)
    expected_sf = 45.0 * 25.7
    assert abs(result.scaling_factor_used - expected_sf) < 1e-6


def test_rat_scaling_factor():
    """Rat: MPPGL=60, liver_weight=40.0 g/kg -> scaling_factor=2400."""
    result = predict_clearance_ivivc("DrugB", 10.0, 1.0, species="rat")
    expected_sf = 60.0 * 40.0
    assert abs(result.scaling_factor_used - expected_sf) < 1e-6


# ---------------------------------------------------------------------------
# Higher CLint -> higher CLh
# ---------------------------------------------------------------------------


def test_higher_clint_gives_higher_clh():
    r_low = predict_clearance_ivivc("DrugC", 1.0, 0.5)
    r_high = predict_clearance_ivivc("DrugC", 100.0, 0.5)
    assert (
        r_high.cl_hepatic_well_stirred_L_per_h_per_kg > r_low.cl_hepatic_well_stirred_L_per_h_per_kg
    )


# ---------------------------------------------------------------------------
# Well-stirred <= parallel tube (WS < PT always except at very low CLint)
# ---------------------------------------------------------------------------


def test_well_stirred_le_parallel_tube():
    """Well-stirred model always gives CLh <= parallel-tube model."""
    for cl_int in [1.0, 10.0, 100.0, 500.0]:
        r = predict_clearance_ivivc("DrugD", cl_int, 0.5)
        assert r.cl_hepatic_well_stirred_L_per_h_per_kg <= (
            r.cl_hepatic_parallel_tube_L_per_h_per_kg + 1e-9
        ), f"WS > PT at cl_int={cl_int}"


def test_dispersion_between_ws_and_pt():
    """Dispersion model is average of WS and PT."""
    r = predict_clearance_ivivc("DrugD", 50.0, 0.5)
    expected = 0.5 * (
        r.cl_hepatic_well_stirred_L_per_h_per_kg + r.cl_hepatic_parallel_tube_L_per_h_per_kg
    )
    assert abs(r.cl_hepatic_dispersion_L_per_h_per_kg - expected) < 1e-9


# ---------------------------------------------------------------------------
# Extraction ratio bounds
# ---------------------------------------------------------------------------


def test_extraction_ratio_between_0_and_1():
    for cl_int in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        r = predict_clearance_ivivc("DrugE", cl_int, 0.5)
        assert 0.0 <= r.extraction_ratio_well_stirred <= 1.0


def test_high_clint_high_extraction():
    r = predict_clearance_ivivc("DrugE", 10000.0, 1.0)
    assert r.extraction_ratio_well_stirred > 0.9


def test_low_clint_low_extraction():
    r = predict_clearance_ivivc("DrugE", 0.01, 1.0)
    assert r.extraction_ratio_well_stirred < 0.1


# ---------------------------------------------------------------------------
# Hepatic bioavailability
# ---------------------------------------------------------------------------


def test_high_clh_low_f_hepatic():
    r = predict_clearance_ivivc("DrugF", 5000.0, 1.0)
    assert r.predicted_f_hepatic < 0.2


def test_f_hepatic_equals_one_minus_er():
    r = predict_clearance_ivivc("DrugF", 20.0, 0.3)
    assert abs(r.predicted_f_hepatic - (1.0 - r.extraction_ratio_well_stirred)) < 1e-9


# ---------------------------------------------------------------------------
# fu_plasma / fu_mic correction
# ---------------------------------------------------------------------------


def test_fu_correction_effect():
    """Lower fu_mic means less free drug in microsomes -> lower effective CLint."""
    r_no_correction = predict_clearance_ivivc("DrugG", 10.0, 1.0, fu_mic=1.0)
    r_with_correction = predict_clearance_ivivc("DrugG", 10.0, 1.0, fu_mic=0.1)
    # fu_plasma/fu_mic: 1/1=1 vs 1/0.1=10 -> with correction has higher scaled CLint
    assert (
        r_with_correction.cl_int_scaled_mL_per_min_per_kg
        > r_no_correction.cl_int_scaled_mL_per_min_per_kg
    )


def test_fu_mic_none_uses_default():
    r = predict_clearance_ivivc("DrugG", 10.0, 0.5, fu_mic=None)
    assert r.fu_mic == 1.0  # default for all species


# ---------------------------------------------------------------------------
# compare_species
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_species("DrugH", 20.0, 0.5)
    assert len(results) == 4


def test_compare_sorted_descending():
    results = compare_species("DrugH", 20.0, 0.5)
    clhs = [r.cl_hepatic_well_stirred_L_per_h_per_kg for r in results]
    assert clhs == sorted(clhs, reverse=True)


def test_compare_all_species_represented():
    results = compare_species("DrugH", 20.0, 0.5)
    species_set = {r.species for r in results}
    assert species_set == {"human", "rat", "dog", "monkey"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_zero_clint():
    with pytest.raises(ValueError, match="cl_int"):
        predict_clearance_ivivc("DrugX", 0.0, 0.5)


def test_validation_negative_clint():
    with pytest.raises(ValueError, match="cl_int"):
        predict_clearance_ivivc("DrugX", -1.0, 0.5)


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_clearance_ivivc("DrugX", 10.0, 0.0)


def test_validation_fu_plasma_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_clearance_ivivc("DrugX", 10.0, 1.1)


def test_validation_unknown_species():
    with pytest.raises(ValueError, match="species"):
        predict_clearance_ivivc("DrugX", 10.0, 0.5, species="mouse")


def test_validation_negative_microsomal_protein():
    with pytest.raises(ValueError, match="microsomal_protein"):
        predict_clearance_ivivc("DrugX", 10.0, 0.5, microsomal_protein_mg_per_mL=-1.0)
