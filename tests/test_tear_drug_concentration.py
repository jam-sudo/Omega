"""
Tests for Phase 508 — Drug Concentration in Tears
"""

import pytest

from omega_pbpk.prediction.tear_drug_concentration import (
    TearDrugConcentrationResult,
    compare_tear_monitoring,
    predict_tear_concentration,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def make_neutral(drug_name="DrugN", plasma=1.0, fu=0.5, pKa=7.4):
    return predict_tear_concentration(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma,
        fu_plasma=fu,
        pKa=pKa,
        drug_type="neutral",
    )


def make_base(drug_name="DrugB", plasma=1.0, fu=0.5, pKa=9.0):
    return predict_tear_concentration(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma,
        fu_plasma=fu,
        pKa=pKa,
        drug_type="base",
    )


def make_acid(drug_name="DrugA", plasma=1.0, fu=0.5, pKa=4.0):
    return predict_tear_concentration(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma,
        fu_plasma=fu,
        pKa=pKa,
        drug_type="acid",
    )


# ── Return type and fields ───────────────────────────────────────────────────


def test_return_type_neutral():
    r = make_neutral()
    assert isinstance(r, TearDrugConcentrationResult)


def test_return_type_base():
    r = make_base()
    assert isinstance(r, TearDrugConcentrationResult)


def test_return_type_acid():
    r = make_acid()
    assert isinstance(r, TearDrugConcentrationResult)


def test_result_has_required_fields():
    r = make_neutral()
    for attr in [
        "drug_name",
        "plasma_conc_mg_L",
        "fu_plasma",
        "pKa",
        "drug_type",
        "tear_plasma_ratio",
        "tear_conc_mg_L",
        "monitoring_application",
        "notes",
    ]:
        assert hasattr(r, attr), f"Missing field: {attr}"


def test_drug_name_preserved():
    r = make_neutral(drug_name="Lidocaine")
    assert r.drug_name == "Lidocaine"


def test_drug_type_preserved():
    r = make_base()
    assert r.drug_type == "base"


def test_frozen_dataclass():
    """Result should be immutable (frozen dataclass)."""
    r = make_neutral()
    with pytest.raises((AttributeError, TypeError)):
        r.tear_conc_mg_L = 999.0  # type: ignore[misc]


# ── Neutral drug ratio = fu * 0.6 ────────────────────────────────────────────


def test_neutral_ratio_equals_fu_times_dilution():
    """For neutral drug with pH_tear = pH_plasma, ratio = fu * 0.6."""
    fu = 0.5
    r = make_neutral(fu=fu, pKa=7.0)
    expected = fu * 0.6
    assert r.tear_plasma_ratio == pytest.approx(expected, rel=1e-6)


def test_neutral_ratio_different_fu():
    fu = 0.8
    r = make_neutral(fu=fu)
    expected = fu * 0.6
    assert r.tear_plasma_ratio == pytest.approx(expected, rel=1e-6)


def test_neutral_ratio_low_fu_clamped_to_min():
    """Very low fu * 0.6 may hit the 0.01 floor."""
    fu = 0.01
    r = make_neutral(fu=fu)
    # fu * 0.6 = 0.006 < 0.01, so should be clamped to 0.01
    assert r.tear_plasma_ratio == pytest.approx(0.01, rel=1e-6)


# ── Tear concentration calculation ────────────────────────────────────────────


def test_tear_conc_equals_plasma_times_ratio():
    plasma = 2.5
    r = make_neutral(plasma=plasma, fu=0.5)
    expected = plasma * r.tear_plasma_ratio
    assert r.tear_conc_mg_L == pytest.approx(expected, rel=1e-6)


def test_tear_conc_positive():
    r = make_neutral()
    assert r.tear_conc_mg_L > 0


def test_tear_conc_scales_with_plasma():
    r1 = make_neutral(plasma=1.0)
    r2 = make_neutral(plasma=2.0)
    assert r2.tear_conc_mg_L == pytest.approx(2.0 * r1.tear_conc_mg_L, rel=1e-6)


# ── Ratio clamping ────────────────────────────────────────────────────────────


def test_ratio_within_bounds():
    """Ratio must always be in [0.01, 5.0]."""
    for drug_type, pKa in [("base", 12.0), ("acid", 1.0), ("neutral", 7.0)]:
        r = predict_tear_concentration(
            drug_name="T",
            plasma_conc_mg_L=1.0,
            fu_plasma=1.0,
            pKa=pKa,
            drug_type=drug_type,
        )
        assert 0.01 <= r.tear_plasma_ratio <= 5.0, (
            f"Ratio {r.tear_plasma_ratio} out of [0.01, 5.0] for {drug_type} pKa={pKa}"
        )


def test_ratio_max_clamp():
    """High fu, base with very high pKa can exceed 5.0 before clamping."""
    r = predict_tear_concentration(
        drug_name="T",
        plasma_conc_mg_L=1.0,
        fu_plasma=1.0,
        pKa=14.0,
        drug_type="base",
    )
    assert r.tear_plasma_ratio <= 5.0


# ── Monitoring application ────────────────────────────────────────────────────


def test_tdm_feasible_when_ratio_in_range():
    """ratio in [0.5, 2.0] → TDM_feasible"""
    # neutral with fu=1.0: ratio = 1.0 * 0.6 = 0.6, within range
    r = make_neutral(fu=1.0)
    assert r.monitoring_application == "TDM_feasible"


def test_research_only_when_ratio_below_range():
    """ratio < 0.5 → research_only"""
    # neutral with fu=0.01: ratio = 0.01 * 0.6 = 0.006 → clamped to 0.01 → research_only
    r = make_neutral(fu=0.01)
    assert r.monitoring_application == "research_only"


def test_monitoring_application_is_string():
    r = make_neutral()
    assert isinstance(r.monitoring_application, str)


def test_monitoring_application_valid_values():
    r = make_neutral()
    assert r.monitoring_application in {"TDM_feasible", "research_only"}


# ── Validation errors ─────────────────────────────────────────────────────────


def test_negative_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc"):
        predict_tear_concentration(
            drug_name="D",
            plasma_conc_mg_L=-1.0,
            fu_plasma=0.5,
            pKa=7.4,
            drug_type="neutral",
        )


def test_zero_plasma_conc_raises():
    with pytest.raises(ValueError, match="plasma_conc"):
        predict_tear_concentration(
            drug_name="D",
            plasma_conc_mg_L=0.0,
            fu_plasma=0.5,
            pKa=7.4,
            drug_type="neutral",
        )


def test_zero_fu_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_tear_concentration(
            drug_name="D",
            plasma_conc_mg_L=1.0,
            fu_plasma=0.0,
            pKa=7.4,
            drug_type="neutral",
        )


def test_fu_above_one_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_tear_concentration(
            drug_name="D",
            plasma_conc_mg_L=1.0,
            fu_plasma=1.5,
            pKa=7.4,
            drug_type="neutral",
        )


def test_invalid_drug_type_raises():
    with pytest.raises(ValueError, match="drug_type"):
        predict_tear_concentration(
            drug_name="D",
            plasma_conc_mg_L=1.0,
            fu_plasma=0.5,
            pKa=7.4,
            drug_type="zwitterion",
        )


# ── compare_tear_monitoring ───────────────────────────────────────────────────


def test_compare_returns_list():
    compounds = [
        {
            "drug_name": "A",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.8,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
        {
            "drug_name": "B",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.3,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
    ]
    results = compare_tear_monitoring(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_sorted_by_ratio_descending():
    compounds = [
        {
            "drug_name": "A",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.3,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
        {
            "drug_name": "B",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.8,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
        {
            "drug_name": "C",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.6,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
    ]
    results = compare_tear_monitoring(compounds)
    ratios = [r.tear_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_all_results_are_correct_type():
    compounds = [
        {
            "drug_name": "X",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.5,
            "pKa": 7.4,
            "drug_type": "neutral",
        },
    ]
    results = compare_tear_monitoring(compounds)
    assert all(isinstance(r, TearDrugConcentrationResult) for r in results)
