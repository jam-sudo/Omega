"""Tests for Phase 462 — Mucoadhesive Drug Delivery."""

import pytest

from omega_pbpk.prediction.mucoadhesive_drug_delivery import (
    MucoadhesiveResult,
    predict_mucoadhesive_delivery,
    screen_polymers,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> MucoadhesiveResult:
    params = dict(
        drug_name="TestDrug",
        polymer="chitosan",
        route="oral",
        mw_Da=300.0,
        logp=1.5,
        charge=0.0,
        dose_mg=10.0,
    )
    params.update(kwargs)
    return predict_mucoadhesive_delivery(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default_result()
    assert isinstance(result, MucoadhesiveResult)


# ---------------------------------------------------------------------------
# Field presence and types
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = _default_result(drug_name="Clotrimazole")
    assert result.drug_name == "Clotrimazole"


def test_polymer_preserved():
    result = _default_result(polymer="carbopol")
    assert result.polymer == "carbopol"


def test_route_preserved():
    result = _default_result(route="nasal")
    assert result.route == "nasal"


def test_recommended_is_bool():
    result = _default_result()
    assert isinstance(result.recommended, bool)


def test_notes_is_string():
    result = _default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Carbopol has higher residence time than PVA
# ---------------------------------------------------------------------------


def test_carbopol_higher_rt_than_pva():
    r_carbopol = _default_result(polymer="carbopol")
    r_pva = _default_result(polymer="PVA")
    assert r_carbopol.residence_time_h > r_pva.residence_time_h


# ---------------------------------------------------------------------------
# Charged drug has higher WOA than neutral
# ---------------------------------------------------------------------------


def test_charged_higher_woa_than_neutral():
    r_charged = _default_result(charge=2.0)
    r_neutral = _default_result(charge=0.0)
    assert r_charged.work_of_adhesion_mJ_per_m2 > r_neutral.work_of_adhesion_mJ_per_m2


def test_negative_charge_same_woa_as_positive():
    r_neg = _default_result(charge=-2.0)
    r_pos = _default_result(charge=2.0)
    assert r_neg.work_of_adhesion_mJ_per_m2 == pytest.approx(r_pos.work_of_adhesion_mJ_per_m2)


# ---------------------------------------------------------------------------
# Nasal route has smaller mucus thickness than oral
# ---------------------------------------------------------------------------


def test_nasal_smaller_thickness_than_oral():
    r_nasal = _default_result(route="nasal")
    r_oral = _default_result(route="oral")
    assert r_nasal.mucus_thickness_mm < r_oral.mucus_thickness_mm


def test_ophthalmic_smallest_thickness():
    r_ophthalmic = _default_result(route="ophthalmic")
    r_vaginal = _default_result(route="vaginal")
    assert r_ophthalmic.mucus_thickness_mm < r_vaginal.mucus_thickness_mm


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def test_bioavailability_enhancement_factor_positive():
    result = _default_result()
    assert result.bioavailability_enhancement_factor > 0.0


def test_bef_max_5():
    # carbopol has res_time=8h: bef = min(5, 8/2) = 4 (without charge/logP factor)
    # with neutral charge and logP<3: bef = min(5, 8*1*1/2) = 4.0
    result = _default_result(polymer="carbopol", charge=0.0, logp=1.0)
    assert result.bioavailability_enhancement_factor <= 5.0


def test_local_auc_positive():
    result = _default_result()
    assert result.local_auc_mg_h_per_mL > 0.0


def test_permeation_rate_positive():
    result = _default_result()
    assert result.drug_permeation_rate_mg_per_cm2_per_h > 0.0


def test_permeation_rate_clamped_to_range():
    # Very large logP might push raw permeation rate up
    result = _default_result(logp=10.0, mw_Da=100.0)
    assert result.drug_permeation_rate_mg_per_cm2_per_h <= 1.0
    assert result.drug_permeation_rate_mg_per_cm2_per_h >= 0.001


# ---------------------------------------------------------------------------
# recommended field logic
# ---------------------------------------------------------------------------


def test_recommended_true_for_carbopol():
    # carbopol has res_time=8h > 3h, permeation > 0 → True
    result = _default_result(polymer="carbopol", logp=1.0)
    assert result.recommended is True


def test_recommended_false_for_low_rt():
    # PVA base res_time=2h < 3h → recommended False (neutral, logP<3)
    result = predict_mucoadhesive_delivery("Drug", "PVA", "oral", 300.0, 1.0, charge=0.0)
    assert result.recommended is False


# ---------------------------------------------------------------------------
# screen_polymers
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_polymers("Drug", "oral", 300.0, 1.5)
    assert isinstance(results, list)
    assert len(results) == 5  # all 5 polymers


def test_screen_sorted_by_local_auc_descending():
    results = screen_polymers("Drug", "nasal", 300.0, 1.5)
    aucs = [r.local_auc_mg_h_per_mL for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_screen_all_are_mucoadhesive_results():
    results = screen_polymers("Drug", "oral", 300.0, 1.5)
    for r in results:
        assert isinstance(r, MucoadhesiveResult)


def test_screen_subset_of_polymers():
    results = screen_polymers("Drug", "oral", 300.0, 1.5, polymers=["HPMC", "carbopol"])
    assert len(results) == 2
    polymer_names = {r.polymer for r in results}
    assert polymer_names == {"HPMC", "carbopol"}


def test_screen_none_uses_all_polymers():
    results = screen_polymers("Drug", "vaginal", 300.0, 2.0, polymers=None)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_mucoadhesive_delivery("D", "chitosan", "oral", 0.0, 1.5)


def test_invalid_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_mucoadhesive_delivery("D", "chitosan", "oral", -100.0, 1.5)


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        predict_mucoadhesive_delivery("D", "chitosan", "buccal_wrong", 300.0, 1.5)


def test_invalid_polymer():
    with pytest.raises(ValueError, match="polymer"):
        predict_mucoadhesive_delivery("D", "unknown_polymer", "oral", 300.0, 1.5)


def test_all_routes_valid():
    for route in ["oral", "nasal", "vaginal", "rectal", "ophthalmic"]:
        result = predict_mucoadhesive_delivery("Drug", "HPMC", route, 300.0, 1.0)
        assert isinstance(result, MucoadhesiveResult)


def test_all_polymers_valid():
    for polymer in ["HPMC", "carbopol", "chitosan", "PVA", "alginate"]:
        result = predict_mucoadhesive_delivery("Drug", polymer, "oral", 300.0, 1.0)
        assert isinstance(result, MucoadhesiveResult)
