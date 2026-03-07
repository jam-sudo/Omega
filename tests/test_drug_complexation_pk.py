"""Tests for Phase 1018: drug-cyclodextrin complexation PK prediction."""

import pytest

from omega_pbpk.prediction.drug_complexation_pk import (
    CyclodextrinComplexResult,
    predict_cyclodextrin_complexation,
    screen_cyclodextrins,
)


# ---------------------------------------------------------------------------
# Basic return-type and field checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    res = predict_cyclodextrin_complexation("TestDrug", "HP-beta-CD", 2.0)
    assert isinstance(res, CyclodextrinComplexResult)


def test_complex_stability_constant_positive():
    res = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 2.0)
    assert res.complex_stability_constant_Ka > 0.0


def test_complexed_solubility_exceeds_intrinsic():
    res = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 2.0)
    assert res.complexed_solubility_mg_mL > res.drug_intrinsic_solubility_mg_mL


def test_solubility_enhancement_greater_than_one():
    res = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 2.0)
    assert res.solubility_enhancement > 1.0


def test_solubilization_efficiency_in_range():
    res = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 2.0)
    assert 0.0 < res.solubilization_efficiency_pct <= 100.0


def test_phase_solubility_slope_positive():
    res = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 2.0)
    assert res.phase_solubility_slope > 0.0


def test_bioavailability_benefit_valid():
    res = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 2.0)
    assert res.bioavailability_benefit in {"high", "moderate", "low", "none"}


# ---------------------------------------------------------------------------
# logP effects
# ---------------------------------------------------------------------------


def test_high_logp_lower_intrinsic_solubility():
    """Higher logP → lower intrinsic solubility (S0 = 10^(-0.7*logP + 0.44))."""
    low = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 1.0)
    high = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 4.0)
    assert high.drug_intrinsic_solubility_mg_mL < low.drug_intrinsic_solubility_mg_mL


def test_high_logp_higher_ka():
    """Higher logP → higher Ka (better CD binding affinity)."""
    low = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 0.0)
    high = predict_cyclodextrin_complexation("Drug", "HP-beta-CD", 3.0)
    assert high.complex_stability_constant_Ka > low.complex_stability_constant_Ka


# ---------------------------------------------------------------------------
# Cyclodextrin comparison
# ---------------------------------------------------------------------------


def test_sbe_beta_cd_highest_ka_factor():
    """SBE-beta-CD has highest Ka_factor → highest Ka at same logP."""
    sbe = predict_cyclodextrin_complexation("Drug", "SBE-beta-CD", 2.0)
    beta = predict_cyclodextrin_complexation("Drug", "beta-CD", 2.0)
    alpha = predict_cyclodextrin_complexation("Drug", "alpha-CD", 2.0)
    assert sbe.complex_stability_constant_Ka > beta.complex_stability_constant_Ka
    assert sbe.complex_stability_constant_Ka > alpha.complex_stability_constant_Ka


def test_higher_cd_concentration_higher_enhancement():
    """More CD → greater solubility enhancement."""
    low_conc = predict_cyclodextrin_complexation(
        "Drug", "HP-beta-CD", 2.0, cd_concentration_mM=5.0
    )
    high_conc = predict_cyclodextrin_complexation(
        "Drug", "HP-beta-CD", 2.0, cd_concentration_mM=50.0
    )
    assert high_conc.solubility_enhancement > low_conc.solubility_enhancement


# ---------------------------------------------------------------------------
# screen_cyclodextrins
# ---------------------------------------------------------------------------


def test_screen_returns_five_results():
    results = screen_cyclodextrins("Drug", 2.0, 300.0)
    assert len(results) == 5


def test_screen_sorted_by_enhancement_descending():
    results = screen_cyclodextrins("Drug", 2.0, 300.0)
    for i in range(1, len(results)):
        assert results[i - 1].solubility_enhancement >= results[i].solubility_enhancement


def test_screen_all_valid_types():
    results = screen_cyclodextrins("Drug", 1.5, 350.0)
    names = {r.cyclodextrin_name for r in results}
    assert names == {"beta-CD", "HP-beta-CD", "SBE-beta-CD", "gamma-CD", "alpha-CD"}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_cyclodextrin_raises():
    with pytest.raises(ValueError, match="cyclodextrin_name"):
        predict_cyclodextrin_complexation("Drug", "unknown-CD", 2.0)


def test_invalid_cd_concentration_raises():
    with pytest.raises(ValueError, match="cd_concentration_mM"):
        predict_cyclodextrin_complexation(
            "Drug", "HP-beta-CD", 2.0, cd_concentration_mM=0.0
        )


def test_negative_cd_concentration_raises():
    with pytest.raises(ValueError, match="cd_concentration_mM"):
        predict_cyclodextrin_complexation(
            "Drug", "HP-beta-CD", 2.0, cd_concentration_mM=-5.0
        )


def test_invalid_drug_mw_raises():
    with pytest.raises(ValueError, match="drug_mw"):
        predict_cyclodextrin_complexation(
            "Drug", "HP-beta-CD", 2.0, drug_mw=0.0
        )
