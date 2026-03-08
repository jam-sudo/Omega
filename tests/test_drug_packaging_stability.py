"""Tests for Phase 1042 — Drug Packaging Stability."""

import pytest

from omega_pbpk.prediction.drug_packaging_stability import (
    PackagingStabilityResult,
    compare_packaging_options,
    predict_packaging_stability,
)

VALID_PATHWAYS = {"hydrolysis", "oxidation", "photodegradation", "combined"}


@pytest.fixture
def default_result():
    return predict_packaging_stability(
        drug_name="TestDrug",
        mw_Da=350.0,
        logp=2.0,
        hydrolysis_susceptibility="low",
        oxidation_susceptibility="low",
        packaging="HDPE",
        storage_condition="25C_60RH",
        initial_purity_pct=99.5,
    )


def test_return_type(default_result):
    assert isinstance(default_result, PackagingStabilityResult)


def test_frozen_dataclass(default_result):
    with pytest.raises((AttributeError, TypeError)):
        default_result.shelf_life_months = 999.0  # type: ignore[misc]


def test_predicted_purity_12m_range(default_result):
    assert 0 < default_result.predicted_purity_at_12m <= 100


def test_purity_decreasing(default_result):
    assert default_result.predicted_purity_at_24m <= default_result.predicted_purity_at_12m
    assert default_result.predicted_purity_at_36m <= default_result.predicted_purity_at_24m


def test_shelf_life_range(default_result):
    assert 0 <= default_result.shelf_life_months <= 120


def test_degradation_rate_positive(default_result):
    assert default_result.degradation_rate_per_month > 0


def test_degradation_pathway_valid(default_result):
    assert default_result.degradation_pathway in VALID_PATHWAYS


def test_ict_compliant_matches_shelf_life(default_result):
    assert default_result.ict_compliant == (default_result.shelf_life_months >= 24)


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0


def test_high_hydrolysis_lower_shelf_life():
    low = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
        hydrolysis_susceptibility="low",
        packaging="HDPE",
        storage_condition="25C_60RH",
    )
    high = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
        hydrolysis_susceptibility="high",
        packaging="HDPE",
        storage_condition="25C_60RH",
    )
    assert high.shelf_life_months < low.shelf_life_months


def test_higher_temperature_lower_shelf_life():
    cold = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
        storage_condition="25C_60RH",
        packaging="glass",
    )
    hot = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
        storage_condition="40C_75RH",
        packaging="glass",
    )
    assert hot.shelf_life_months < cold.shelf_life_months


def test_aluminum_foil_better_than_blister_pvc():
    alu = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
        hydrolysis_susceptibility="moderate",
        oxidation_susceptibility="moderate",
        packaging="aluminum_foil",
        storage_condition="25C_60RH",
    )
    pvc = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
        hydrolysis_susceptibility="moderate",
        oxidation_susceptibility="moderate",
        packaging="blister_PVC",
        storage_condition="25C_60RH",
    )
    assert alu.shelf_life_months > pvc.shelf_life_months


def test_compare_packaging_returns_5():
    results = compare_packaging_options(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
    )
    assert len(results) == 5


def test_compare_packaging_sorted_desc():
    results = compare_packaging_options(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
    )
    shelf_lives = [r.shelf_life_months for r in results]
    assert shelf_lives == sorted(shelf_lives, reverse=True)


def test_compare_packaging_all_types():
    results = compare_packaging_options(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
    )
    pkgs = {r.packaging for r in results}
    assert pkgs == {"HDPE", "glass", "blister_PVC", "blister_PVDC", "aluminum_foil"}


def test_recommended_packaging_flag():
    result = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=2.0,
        packaging="aluminum_foil",
        storage_condition="25C_60RH",
        hydrolysis_susceptibility="low",
        oxidation_susceptibility="low",
    )
    assert result.recommended_packaging == (result.shelf_life_months >= 36)


# Validation tests
def test_validation_purity_zero():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        predict_packaging_stability(
            drug_name="D",
            mw_Da=300.0,
            logp=2.0,
            initial_purity_pct=0.0,
        )


def test_validation_invalid_packaging():
    with pytest.raises(ValueError, match="packaging"):
        predict_packaging_stability(
            drug_name="D",
            mw_Da=300.0,
            logp=2.0,
            packaging="cardboard",
        )


def test_validation_invalid_storage():
    with pytest.raises(ValueError, match="storage_condition"):
        predict_packaging_stability(
            drug_name="D",
            mw_Da=300.0,
            logp=2.0,
            storage_condition="50C_80RH",
        )


def test_validation_invalid_susceptibility():
    with pytest.raises(ValueError, match="hydrolysis_susceptibility"):
        predict_packaging_stability(
            drug_name="D",
            mw_Da=300.0,
            logp=2.0,
            hydrolysis_susceptibility="extreme",
        )


def test_photodegradation_effect():
    """High logP with blister_PVC should trigger photodegradation (k_total*1.2)."""
    high_logp = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=4.0,  # > 3 triggers photodegradation
        packaging="blister_PVC",  # O2_factor=0.9 > 0.5
        storage_condition="25C_60RH",
    )
    low_logp = predict_packaging_stability(
        drug_name="D",
        mw_Da=300.0,
        logp=1.0,  # no photodegradation
        packaging="blister_PVC",
        storage_condition="25C_60RH",
    )
    # High logP should degrade faster (lower shelf life)
    assert high_logp.shelf_life_months < low_logp.shelf_life_months
