"""Tests for Phase 229: pediatric dose scaling (prediction module)."""

import pytest

from omega_pbpk.prediction.pediatric_dose_scaling import (
    PediatricDoseResult,
    compare_scaling_methods,
    scale_pediatric_dose,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ADULT_DOSE = 500.0
ADULT_WT = 70.0
CHILD_WT = 20.0
CHILD_AGE = 36.0  # 3 yr toddler


def _default(**kwargs) -> PediatricDoseResult:
    defaults = dict(
        drug_name="TestDrug",
        adult_dose_mg=ADULT_DOSE,
        adult_weight_kg=ADULT_WT,
        child_weight_kg=CHILD_WT,
        child_age_months=CHILD_AGE,
        dose_metric="allometric",
    )
    defaults.update(kwargs)
    return scale_pediatric_dose(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    res = _default()
    assert isinstance(res, PediatricDoseResult)


def test_field_drug_name():
    res = _default(drug_name="Morphine")
    assert res.drug_name == "Morphine"


def test_field_adult_dose():
    res = _default()
    assert res.adult_dose_mg == ADULT_DOSE


def test_field_child_weight():
    res = _default()
    assert res.child_weight_kg == CHILD_WT


def test_field_child_age():
    res = _default()
    assert res.child_age_months == CHILD_AGE


# ---------------------------------------------------------------------------
# Allometric scaling
# ---------------------------------------------------------------------------


def test_allometric_dose_less_than_adult():
    res = _default(dose_metric="allometric")
    assert res.scaled_dose_mg < ADULT_DOSE


def test_allometric_dose_positive():
    res = _default(dose_metric="allometric")
    assert res.scaled_dose_mg > 0


def test_allometric_formula():
    res = _default(dose_metric="allometric")
    expected = ADULT_DOSE * (CHILD_WT / ADULT_WT) ** 0.75
    assert abs(res.scaled_dose_mg - expected) < 0.01


def test_allometric_dose_ratio():
    res = _default(dose_metric="allometric")
    assert abs(res.dose_ratio - res.scaled_dose_mg / ADULT_DOSE) < 1e-4


# ---------------------------------------------------------------------------
# mg/kg scaling
# ---------------------------------------------------------------------------


def test_mg_per_kg_proportional():
    res = _default(dose_metric="mg_per_kg")
    expected = ADULT_DOSE / ADULT_WT * CHILD_WT
    assert abs(res.scaled_dose_mg - expected) < 0.01


def test_mg_per_kg_dose_per_kg_constant():
    res = _default(dose_metric="mg_per_kg")
    adult_per_kg = ADULT_DOSE / ADULT_WT
    assert abs(res.dose_per_kg - adult_per_kg) < 0.01


# ---------------------------------------------------------------------------
# BSA scaling
# ---------------------------------------------------------------------------


def test_bsa_dose_positive():
    res = _default(dose_metric="bsa")
    assert res.scaled_dose_mg > 0


def test_bsa_dose_less_than_adult():
    res = _default(dose_metric="bsa")
    assert res.scaled_dose_mg < ADULT_DOSE


# ---------------------------------------------------------------------------
# Maturation function
# ---------------------------------------------------------------------------


def test_maturation_factor_range():
    res = _default(dose_metric="maturation_function")
    assert 0.0 <= res.maturation_factor <= 1.0


def test_neonate_lower_maturation_than_child():
    neonate = scale_pediatric_dose("Drug", 100.0, 70.0, 3.5, 1.0, dose_metric="maturation_function")
    child = scale_pediatric_dose("Drug", 100.0, 70.0, 20.0, 60.0, dose_metric="maturation_function")
    assert neonate.maturation_factor < child.maturation_factor


def test_maturation_dose_lower_than_mg_per_kg_for_neonate():
    mat = scale_pediatric_dose("Drug", 100.0, 70.0, 3.5, 1.0, dose_metric="maturation_function")
    mpk = scale_pediatric_dose("Drug", 100.0, 70.0, 3.5, 1.0, dose_metric="mg_per_kg")
    # Neonate maturation is very low → maturation dose should be much lower
    assert mat.scaled_dose_mg < mpk.scaled_dose_mg


# ---------------------------------------------------------------------------
# Max dose cap
# ---------------------------------------------------------------------------


def test_scaled_dose_never_exceeds_adult():
    # Child heavier than adult — dose must be capped
    res = scale_pediatric_dose("Drug", 100.0, 30.0, 80.0, 200.0, dose_metric="mg_per_kg")
    assert res.scaled_dose_mg <= res.adult_dose_mg


def test_cap_reflected_in_dose_ratio():
    res = scale_pediatric_dose("Drug", 100.0, 30.0, 80.0, 200.0, dose_metric="mg_per_kg")
    assert res.dose_ratio <= 1.0


# ---------------------------------------------------------------------------
# Age groups
# ---------------------------------------------------------------------------


def test_age_group_neonate():
    res = scale_pediatric_dose("Drug", 100.0, 70.0, 2.0, 0.5)
    assert res.age_group == "neonates"


def test_age_group_infant():
    res = scale_pediatric_dose("Drug", 100.0, 70.0, 5.0, 6.0)
    assert res.age_group == "infants"


def test_age_group_toddler():
    res = scale_pediatric_dose("Drug", 100.0, 70.0, 15.0, 36.0)
    assert res.age_group == "toddlers"


def test_age_group_child():
    res = scale_pediatric_dose("Drug", 100.0, 70.0, 25.0, 96.0)
    assert res.age_group == "children"


def test_age_group_adolescent():
    res = scale_pediatric_dose("Drug", 100.0, 70.0, 55.0, 168.0)
    assert res.age_group == "adolescents"


# ---------------------------------------------------------------------------
# compare_scaling_methods
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_scaling_methods("Drug", 500.0, 70.0, 20.0, 36.0)
    assert len(results) == 4


def test_compare_sorted_descending():
    results = compare_scaling_methods("Drug", 500.0, 70.0, 20.0, 36.0)
    doses = [r.scaled_dose_mg for r in results]
    assert doses == sorted(doses, reverse=True)


def test_compare_all_methods_present():
    results = compare_scaling_methods("Drug", 500.0, 70.0, 20.0, 36.0)
    metrics = {r.dose_metric for r in results}
    assert metrics == {"allometric", "bsa", "mg_per_kg", "maturation_function"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_adult_dose():
    with pytest.raises(ValueError, match="adult_dose_mg"):
        scale_pediatric_dose("Drug", 0.0, 70.0, 20.0, 36.0)


def test_invalid_child_weight():
    with pytest.raises(ValueError, match="child_weight_kg"):
        scale_pediatric_dose("Drug", 100.0, 70.0, 0.0, 36.0)


def test_invalid_adult_weight():
    with pytest.raises(ValueError, match="adult_weight_kg"):
        scale_pediatric_dose("Drug", 100.0, 0.0, 20.0, 36.0)
