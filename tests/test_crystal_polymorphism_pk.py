"""Tests for Phase 466 — Drug Crystal Polymorphism PK Impact."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.crystal_polymorphism_pk import (
    PolymorphismPKResult,
    compare_polymorphs,
    predict_polymorphism_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _predict(
    drug_name: str = "DrugX",
    polymorph_form: str = "Form_I",
    base_solubility_mg_mL: float = 0.1,
    base_fa: float = 0.3,
) -> PolymorphismPKResult:
    return predict_polymorphism_pk(
        drug_name=drug_name,
        polymorph_form=polymorph_form,
        base_solubility_mg_mL=base_solubility_mg_mL,
        base_fa=base_fa,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type_form_i():
    result = _predict(polymorph_form="Form_I")
    assert isinstance(result, PolymorphismPKResult)


def test_return_type_amorphous():
    result = _predict(polymorph_form="Amorphous")
    assert isinstance(result, PolymorphismPKResult)


# ---------------------------------------------------------------------------
# Form_I baseline tests
# ---------------------------------------------------------------------------


def test_form_i_solubility_ratio():
    result = _predict(polymorph_form="Form_I")
    assert result.solubility_ratio == pytest.approx(1.0, rel=1e-6)


def test_form_i_dissolution_rate_ratio():
    result = _predict(polymorph_form="Form_I")
    assert result.dissolution_rate_ratio == pytest.approx(1.0, rel=1e-6)


def test_form_i_conversion_risk_none():
    result = _predict(polymorph_form="Form_I")
    assert result.conversion_risk == "none"


def test_form_i_no_bioavailability_advantage():
    result = _predict(polymorph_form="Form_I")
    assert result.bioavailability_advantage is False


def test_form_i_not_recommended():
    # Form_I has no bioavailability advantage, so recommended_for_development = False
    result = _predict(polymorph_form="Form_I")
    assert result.recommended_for_development is False


def test_form_i_stability_risk_score_zero():
    result = _predict(polymorph_form="Form_I")
    assert result.stability_risk_score == pytest.approx(0.0, rel=1e-6)


def test_form_i_predicted_fa_equals_base_fa():
    result = _predict(polymorph_form="Form_I", base_fa=0.3)
    assert result.predicted_fa == pytest.approx(0.3, rel=1e-6)


def test_form_i_predicted_cmax_ratio_is_one():
    result = _predict(polymorph_form="Form_I")
    assert result.predicted_cmax_ratio == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Amorphous tests
# ---------------------------------------------------------------------------


def test_amorphous_has_highest_predicted_fa():
    forms = ["Form_I", "Form_II", "Form_III", "Amorphous"]
    results = {f: _predict(polymorph_form=f, base_fa=0.05) for f in forms}
    amorphous_fa = results["Amorphous"].predicted_fa
    for f in ["Form_I", "Form_II", "Form_III"]:
        assert amorphous_fa >= results[f].predicted_fa


def test_amorphous_highest_stability_risk_score():
    forms = ["Form_I", "Form_II", "Form_III", "Amorphous"]
    results = {f: _predict(polymorph_form=f) for f in forms}
    amorphous_score = results["Amorphous"].stability_risk_score
    for f in ["Form_I", "Form_II", "Form_III"]:
        assert amorphous_score >= results[f].stability_risk_score


def test_amorphous_stability_risk_score_value():
    result = _predict(polymorph_form="Amorphous")
    assert result.stability_risk_score == pytest.approx(95.0, rel=1e-6)


def test_amorphous_conversion_risk_very_high():
    result = _predict(polymorph_form="Amorphous")
    assert result.conversion_risk == "very_high"


def test_amorphous_predicted_cmax_ratio_capped():
    # dissolution_rate_ratio = 15.0 but capped at 5.0
    result = _predict(polymorph_form="Amorphous")
    assert result.predicted_cmax_ratio == pytest.approx(5.0, rel=1e-6)


def test_amorphous_not_recommended_due_to_stability():
    # stability_risk_score = 95 >= 50, so not recommended
    result = _predict(polymorph_form="Amorphous")
    assert result.recommended_for_development is False


# ---------------------------------------------------------------------------
# Form_II tests
# ---------------------------------------------------------------------------


def test_form_ii_may_be_recommended():
    # stability_risk_score = 25 < 50, and if base_fa is low enough, has bioavailability advantage
    result = _predict(polymorph_form="Form_II", base_fa=0.3)
    # predicted_fa = min(1.0, 1.5 * 0.3) = 0.45 > 0.3 → advantage
    assert result.bioavailability_advantage is True
    assert result.recommended_for_development is True


def test_form_ii_predicted_fa_value():
    result = _predict(polymorph_form="Form_II", base_fa=0.3)
    assert result.predicted_fa == pytest.approx(min(1.0, 1.5 * 0.3), rel=1e-6)


def test_form_ii_cmax_ratio_capped_at_five():
    # dissolution_rate_ratio = 2.0 < 5.0 → not capped
    result = _predict(polymorph_form="Form_II")
    assert result.predicted_cmax_ratio == pytest.approx(2.0, rel=1e-6)


def test_form_iii_predicted_fa():
    result = _predict(polymorph_form="Form_III", base_fa=0.3)
    assert result.predicted_fa == pytest.approx(min(1.0, 3.0 * 0.3), rel=1e-6)


def test_form_iii_stability_risk_score():
    result = _predict(polymorph_form="Form_III")
    assert result.stability_risk_score == pytest.approx(75.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Intrinsic solubility tests
# ---------------------------------------------------------------------------


def test_intrinsic_solubility_form_i():
    result = _predict(polymorph_form="Form_I", base_solubility_mg_mL=0.5)
    assert result.intrinsic_solubility_mg_mL == pytest.approx(0.5, rel=1e-6)


def test_intrinsic_solubility_amorphous():
    result = _predict(polymorph_form="Amorphous", base_solubility_mg_mL=0.1)
    assert result.intrinsic_solubility_mg_mL == pytest.approx(1.0, rel=1e-6)  # 10x


# ---------------------------------------------------------------------------
# compare_polymorphs tests
# ---------------------------------------------------------------------------


def test_compare_returns_four_items():
    results = compare_polymorphs("DrugX")
    assert len(results) == 4


def test_compare_sorted_by_fa_descending():
    results = compare_polymorphs("DrugX", base_fa=0.05)
    fas = [r.predicted_fa for r in results]
    assert fas == sorted(fas, reverse=True)


def test_compare_all_forms_present():
    results = compare_polymorphs("DrugX")
    forms = {r.polymorph_form for r in results}
    assert forms == {"Form_I", "Form_II", "Form_III", "Amorphous"}


def test_compare_high_base_fa_all_capped_at_one():
    # With base_fa=0.7, Form_II=min(1,1.5*0.7)=1.0, Form_III=min(1,3*0.7)=1.0, Amorphous=1.0
    results = compare_polymorphs("DrugX", base_fa=0.7)
    for r in results:
        assert r.predicted_fa <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_invalid_polymorph_form_raises():
    with pytest.raises(ValueError, match="polymorph_form"):
        predict_polymorphism_pk("DrugX", "Form_IV")


def test_invalid_base_solubility_raises():
    with pytest.raises(ValueError, match="base_solubility_mg_mL"):
        predict_polymorphism_pk("DrugX", "Form_I", base_solubility_mg_mL=0.0)


def test_invalid_base_solubility_negative_raises():
    with pytest.raises(ValueError, match="base_solubility_mg_mL"):
        predict_polymorphism_pk("DrugX", "Form_I", base_solubility_mg_mL=-1.0)


def test_invalid_base_fa_zero_raises():
    with pytest.raises(ValueError, match="base_fa"):
        predict_polymorphism_pk("DrugX", "Form_I", base_fa=0.0)


def test_invalid_base_fa_above_one_raises():
    with pytest.raises(ValueError, match="base_fa"):
        predict_polymorphism_pk("DrugX", "Form_I", base_fa=1.5)
