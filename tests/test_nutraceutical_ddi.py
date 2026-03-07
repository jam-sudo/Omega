"""Tests for nutraceutical/food-drug interactions — Phase 316."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.nutraceutical_ddi import (
    ChelationResult,
    NutraDDIResult,
    classify_nutra_interaction_risk,
    grapefruit_juice_interaction,
    mineral_chelation,
    st_johns_wort_interaction,
)

# ---------------------------------------------------------------------------
# grapefruit_juice_interaction
# ---------------------------------------------------------------------------


def test_gfj_returns_result():
    r = grapefruit_juice_interaction("Simvastatin", 10.0, 200.0, 0.9, 40.0)
    assert isinstance(r, NutraDDIResult)


def test_gfj_nutraceutical_name():
    r = grapefruit_juice_interaction("Midazolam", 5.0, 50.0, 0.8, 5.0)
    assert r.nutraceutical == "grapefruit_juice"


def test_gfj_aucr_greater_than_one_for_cyp3a4_substrate():
    # High fm_cyp3a4 => GFJ increases AUC
    r = grapefruit_juice_interaction("Drug", 10.0, 100.0, fm_cyp3a4=0.9, dose_mg=10.0)
    assert r.aucr > 1.0


def test_gfj_aucr_increases_with_fm():
    low = grapefruit_juice_interaction("D", 10.0, 100.0, fm_cyp3a4=0.1, dose_mg=10.0)
    high = grapefruit_juice_interaction("D", 10.0, 100.0, fm_cyp3a4=0.9, dose_mg=10.0)
    assert high.aucr > low.aucr


def test_gfj_cmax_ratio_positive():
    r = grapefruit_juice_interaction("Drug", 10.0, 100.0, 0.7, 20.0)
    assert r.cmax_ratio > 0


def test_gfj_profiles_have_same_length():
    r = grapefruit_juice_interaction("Drug", 10.0, 100.0, 0.7, 20.0)
    assert len(r.times_h_normal) == len(r.c_plasma_normal)
    assert len(r.times_h_affected) == len(r.c_plasma_affected)


def test_gfj_concentration_nonnegative():
    r = grapefruit_juice_interaction("Drug", 10.0, 100.0, 0.7, 20.0)
    assert all(c >= 0 for c in r.c_plasma_normal)
    assert all(c >= 0 for c in r.c_plasma_affected)


def test_gfj_zero_cl_raises():
    with pytest.raises(ValueError, match="cl"):
        grapefruit_juice_interaction("D", 0.0, 100.0, 0.7, 20.0)


def test_gfj_zero_vd_raises():
    with pytest.raises(ValueError, match="vd"):
        grapefruit_juice_interaction("D", 10.0, 0.0, 0.7, 20.0)


def test_gfj_zero_dose_raises():
    with pytest.raises(ValueError, match="dose"):
        grapefruit_juice_interaction("D", 10.0, 100.0, 0.7, 0.0)


def test_gfj_fm_out_of_range_raises():
    with pytest.raises(ValueError, match="fm_cyp3a4"):
        grapefruit_juice_interaction("D", 10.0, 100.0, 1.5, 20.0)


def test_gfj_interaction_class_is_string():
    r = grapefruit_juice_interaction("Drug", 10.0, 100.0, 0.9, 20.0)
    assert isinstance(r.interaction_class, str)


def test_gfj_no_interaction_near_zero_fm():
    # fm_cyp3a4 ~ 0 => very little GFJ effect => AUCR ~ 1.0
    r = grapefruit_juice_interaction("D", 10.0, 100.0, fm_cyp3a4=0.0, dose_mg=10.0)
    assert abs(r.aucr - 1.0) < 0.01


# ---------------------------------------------------------------------------
# st_johns_wort_interaction
# ---------------------------------------------------------------------------


def test_sjw_returns_result():
    r = st_johns_wort_interaction("Cyclosporine", 30.0, 500.0, 0.7, 0.6, 100.0)
    assert isinstance(r, NutraDDIResult)


def test_sjw_nutraceutical_name():
    r = st_johns_wort_interaction("Drug", 10.0, 100.0, 0.7, 0.5, 10.0)
    assert r.nutraceutical == "st_johns_wort"


def test_sjw_aucr_less_than_one():
    # SJW induces metabolism => AUC decreases
    r = st_johns_wort_interaction("Drug", 10.0, 100.0, fm_cyp3a4=0.8, fm_pgp=0.5, dose_mg=10.0)
    assert r.aucr < 1.0


def test_sjw_aucr_decreases_with_fm():
    low = st_johns_wort_interaction("D", 10.0, 100.0, fm_cyp3a4=0.1, fm_pgp=0.1, dose_mg=10.0)
    high = st_johns_wort_interaction("D", 10.0, 100.0, fm_cyp3a4=0.9, fm_pgp=0.9, dose_mg=10.0)
    assert high.aucr < low.aucr


def test_sjw_zero_cl_raises():
    with pytest.raises(ValueError, match="cl"):
        st_johns_wort_interaction("D", 0.0, 100.0, 0.7, 0.5, 10.0)


def test_sjw_fm_cyp3a4_out_of_range_raises():
    with pytest.raises(ValueError, match="fm_cyp3a4"):
        st_johns_wort_interaction("D", 10.0, 100.0, 1.2, 0.5, 10.0)


def test_sjw_fm_pgp_out_of_range_raises():
    with pytest.raises(ValueError, match="fm_pgp"):
        st_johns_wort_interaction("D", 10.0, 100.0, 0.7, -0.1, 10.0)


def test_sjw_short_duration_less_effect():
    short = st_johns_wort_interaction("D", 10.0, 100.0, 0.8, 0.6, 10.0, induction_days=3)
    long_ = st_johns_wort_interaction("D", 10.0, 100.0, 0.8, 0.6, 10.0, induction_days=14)
    # More induction days => lower AUCR
    assert short.aucr > long_.aucr


def test_sjw_profiles_nonnegative():
    r = st_johns_wort_interaction("Drug", 10.0, 100.0, 0.7, 0.5, 10.0)
    assert all(c >= 0 for c in r.c_plasma_normal)
    assert all(c >= 0 for c in r.c_plasma_affected)


# ---------------------------------------------------------------------------
# mineral_chelation
# ---------------------------------------------------------------------------


def test_chelation_returns_result():
    r = mineral_chelation("Ciprofloxacin", 500.0, "calcium", 1000.0, 6.0, "fluoroquinolone")
    assert isinstance(r, ChelationResult)


def test_chelation_fraction_in_bounds():
    r = mineral_chelation("Drug", 500.0, "iron", 200.0, 4.0)
    assert 0.0 <= r.fraction_chelated <= 1.0


def test_chelation_effective_dose_less_than_drug_dose():
    r = mineral_chelation("Tetracycline", 250.0, "calcium", 500.0, 3.5, "tetracycline")
    assert r.effective_dose_mg < 250.0


def test_chelation_effective_dose_nonnegative():
    r = mineral_chelation("Drug", 100.0, "zinc", 500.0, 5.0)
    assert r.effective_dose_mg >= 0.0


def test_chelation_tetracycline_higher_than_other():
    tet = mineral_chelation("Tet", 250.0, "calcium", 1000.0, 3.5, "tetracycline")
    other = mineral_chelation("Drug", 250.0, "calcium", 1000.0, 3.5, "other")
    assert tet.fraction_chelated > other.fraction_chelated


def test_chelation_more_mineral_more_chelation():
    low = mineral_chelation("Drug", 250.0, "iron", 100.0, 4.0, "fluoroquinolone")
    high = mineral_chelation("Drug", 250.0, "iron", 1000.0, 4.0, "fluoroquinolone")
    assert high.fraction_chelated > low.fraction_chelated


def test_chelation_invalid_mineral_raises():
    with pytest.raises(ValueError, match="mineral"):
        mineral_chelation("Drug", 500.0, "lead", 200.0, 4.0)


def test_chelation_zero_drug_dose_raises():
    with pytest.raises(ValueError, match="drug_dose_mg"):
        mineral_chelation("Drug", 0.0, "calcium", 500.0, 4.0)


def test_chelation_zero_mineral_dose_raises():
    with pytest.raises(ValueError, match="mineral_dose_mg"):
        mineral_chelation("Drug", 500.0, "calcium", 0.0, 4.0)


def test_chelation_all_minerals():
    for mineral in ("calcium", "magnesium", "iron", "aluminum", "zinc"):
        r = mineral_chelation("Drug", 500.0, mineral, 1000.0, 4.0)
        assert r.mineral == mineral


def test_chelation_effective_dose_mass_balance():
    r = mineral_chelation("Drug", 200.0, "magnesium", 400.0, 5.0)
    expected = 200.0 * (1.0 - r.fraction_chelated)
    assert abs(r.effective_dose_mg - expected) < 1e-9


# ---------------------------------------------------------------------------
# classify_nutra_interaction_risk
# ---------------------------------------------------------------------------


def test_classify_significant_increase():
    assert classify_nutra_interaction_risk(2.5, "increase") == "significant_increase"


def test_classify_moderate_increase():
    assert classify_nutra_interaction_risk(1.5, "increase") == "moderate"


def test_classify_negligible_increase():
    assert classify_nutra_interaction_risk(1.1, "increase") == "negligible"


def test_classify_significant_decrease():
    assert classify_nutra_interaction_risk(0.3, "decrease") == "significant_decrease"


def test_classify_moderate_decrease():
    assert classify_nutra_interaction_risk(0.65, "decrease") == "moderate"


def test_classify_negligible_decrease():
    assert classify_nutra_interaction_risk(0.95, "decrease") == "negligible"


def test_classify_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        classify_nutra_interaction_risk(1.5, "sideways")


def test_classify_boundary_increase():
    # Exactly 2.0 => significant_increase
    assert classify_nutra_interaction_risk(2.0, "increase") == "significant_increase"


def test_classify_boundary_decrease():
    # Exactly 0.5 => significant_decrease
    assert classify_nutra_interaction_risk(0.5, "decrease") == "significant_decrease"
