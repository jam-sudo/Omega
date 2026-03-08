"""Tests for Phase 287 — Crystal Form Bioavailability Predictor."""

import pytest

from omega_pbpk.prediction.crystal_form_bioavailability import (
    CrystalFormBAResult,
    compare_crystal_forms,
    predict_crystal_form_ba,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_COMMON = dict(
    drug_name="TestDrug",
    logp=2.5,
    mw=350.0,
    dose_mg=100.0,
)


def _make_result(
    crystal_form="polymorph_A",
    idr=0.1,
    sol_ratio=1.0,
    bcs="II",
    **kwargs,
) -> CrystalFormBAResult:
    params = {**_COMMON, **kwargs}
    return predict_crystal_form_ba(
        crystal_form=crystal_form,
        intrinsic_dissolution_rate_mg_per_cm2_per_min=idr,
        solubility_ratio_vs_reference=sol_ratio,
        bcs_class_reference=bcs,
        **params,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = _make_result()
    assert isinstance(r, CrystalFormBAResult)


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


def test_field_drug_name():
    r = _make_result()
    assert r.drug_name == "TestDrug"


def test_field_crystal_form():
    r = _make_result(crystal_form="hydrate")
    assert r.crystal_form == "hydrate"


def test_field_solubility_ratio():
    r = _make_result(sol_ratio=1.5)
    assert r.solubility_ratio_vs_reference == pytest.approx(1.5)


def test_field_idr():
    r = _make_result(idr=0.2)
    assert r.intrinsic_dissolution_rate_mg_per_cm2_per_min == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# ba_ratio > 0
# ---------------------------------------------------------------------------


def test_ba_ratio_positive():
    r = _make_result()
    assert r.ba_ratio > 0


def test_ba_ratio_positive_bcs4():
    r = _make_result(bcs="IV")
    assert r.ba_ratio > 0


# ---------------------------------------------------------------------------
# Higher sol_ratio → higher ba_ratio (BCS II)
# ---------------------------------------------------------------------------


def test_higher_sol_ratio_increases_ba_ratio_bcs2():
    r1 = _make_result(sol_ratio=1.0, bcs="II")
    r2 = _make_result(sol_ratio=2.0, bcs="II")
    assert r2.ba_ratio >= r1.ba_ratio


def test_higher_sol_ratio_increases_ba_ratio_bcs4():
    r1 = _make_result(sol_ratio=1.0, bcs="IV")
    r2 = _make_result(sol_ratio=3.0, bcs="IV")
    assert r2.ba_ratio >= r1.ba_ratio


# ---------------------------------------------------------------------------
# BCS I and III: ba_ratio close to 1.0
# ---------------------------------------------------------------------------


def test_bcs1_ba_ratio_is_1():
    r = _make_result(bcs="I")
    assert r.ba_ratio == pytest.approx(1.0)


def test_bcs3_ba_ratio_is_1():
    r = _make_result(bcs="III")
    assert r.ba_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# predicted_f_abs in [0, 1]
# ---------------------------------------------------------------------------


def test_f_abs_range():
    for bcs in ("I", "II", "III", "IV"):
        r = _make_result(bcs=bcs)
        assert 0.0 <= r.predicted_f_abs <= 1.0, f"f_abs out of range for BCS {bcs}"


# ---------------------------------------------------------------------------
# manufacturability_score in [0, 100]
# ---------------------------------------------------------------------------


def test_manufacturability_score_range():
    for sol_ratio in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0):
        r = _make_result(sol_ratio=sol_ratio)
        assert 0.0 <= r.manufacturability_score <= 100.0


# ---------------------------------------------------------------------------
# recommendation in valid set
# ---------------------------------------------------------------------------


_VALID_RECS = {"preferred", "acceptable", "avoid"}


def test_recommendation_valid_set():
    r = _make_result()
    assert r.recommendation in _VALID_RECS


def test_recommendation_preferred_for_high_ba():
    # High sol_ratio with BCS II should push ba_ratio > 1.3
    r = _make_result(sol_ratio=4.0, idr=0.5, bcs="II")
    assert r.recommendation == "preferred"


def test_recommendation_avoid_for_low_ba():
    # sol_ratio < 1 (less soluble) and low IDR → ba_ratio < 0.8
    r = _make_result(sol_ratio=0.1, idr=0.01, bcs="II")
    assert r.recommendation == "avoid"


# ---------------------------------------------------------------------------
# stability_risk
# ---------------------------------------------------------------------------


def test_stability_risk_high_when_ratio_gt_2():
    r = _make_result(sol_ratio=2.5)
    assert r.stability_risk == "high"


def test_stability_risk_low_when_ratio_le_2():
    r = _make_result(sol_ratio=1.5)
    assert r.stability_risk == "low"


# ---------------------------------------------------------------------------
# compare_crystal_forms sorted descending by ba_ratio
# ---------------------------------------------------------------------------


def test_compare_crystal_forms_sorted():
    forms_data = [
        dict(
            crystal_form="polymorph_A",
            intrinsic_dissolution_rate_mg_per_cm2_per_min=0.05,
            solubility_ratio_vs_reference=0.5,
            logp=2.5,
            mw=350.0,
            dose_mg=100.0,
            bcs_class_reference="II",
        ),
        dict(
            crystal_form="cocrystal",
            intrinsic_dissolution_rate_mg_per_cm2_per_min=0.3,
            solubility_ratio_vs_reference=3.0,
            logp=2.5,
            mw=350.0,
            dose_mg=100.0,
            bcs_class_reference="II",
        ),
        dict(
            crystal_form="hydrate",
            intrinsic_dissolution_rate_mg_per_cm2_per_min=0.1,
            solubility_ratio_vs_reference=1.0,
            logp=2.5,
            mw=350.0,
            dose_mg=100.0,
            bcs_class_reference="II",
        ),
    ]
    results = compare_crystal_forms("TestDrug", forms_data)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].ba_ratio >= results[i + 1].ba_ratio


def test_compare_crystal_forms_return_type():
    forms_data = [
        dict(
            crystal_form="polymorph_A",
            intrinsic_dissolution_rate_mg_per_cm2_per_min=0.1,
            solubility_ratio_vs_reference=1.0,
            logp=2.5,
            mw=350.0,
            dose_mg=100.0,
            bcs_class_reference="II",
        ),
    ]
    results = compare_crystal_forms("TestDrug", forms_data)
    assert isinstance(results, list)
    assert isinstance(results[0], CrystalFormBAResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_idr_zero():
    with pytest.raises(ValueError, match="intrinsic_dissolution_rate"):
        _make_result(idr=0.0)


def test_invalid_idr_negative():
    with pytest.raises(ValueError, match="intrinsic_dissolution_rate"):
        _make_result(idr=-0.1)


def test_invalid_sol_ratio_zero():
    with pytest.raises(ValueError, match="solubility_ratio"):
        _make_result(sol_ratio=0.0)


def test_invalid_bcs_class():
    with pytest.raises(ValueError, match="bcs_class_reference"):
        _make_result(bcs="V")


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_crystal_form_ba(
            drug_name="X",
            crystal_form="A",
            intrinsic_dissolution_rate_mg_per_cm2_per_min=0.1,
            solubility_ratio_vs_reference=1.0,
            logp=2.0,
            mw=300.0,
            dose_mg=0.0,
            bcs_class_reference="II",
        )
