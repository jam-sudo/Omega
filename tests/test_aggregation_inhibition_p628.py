"""Tests for Phase 628 — Drug Protein Aggregation Inhibition."""

import pytest

from omega_pbpk.prediction.aggregation_inhibition_p628 import (
    AggregationInhibitionResult,
    predict_aggregation_inhibition,
    screen_aggregation_inhibitors,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        target_protein="IgG1",
        drug_mw_Da=300.0,
        drug_logP=1.0,
        drug_charge="neutral",
    )
    params.update(kwargs)
    return predict_aggregation_inhibition(**params)


# ── return type and frozen dataclass ─────────────────────────────────────────


def test_returns_aggregation_inhibition_result():
    result = default_result()
    assert isinstance(result, AggregationInhibitionResult)


def test_frozen_dataclass_immutable():
    result = default_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Modified"  # type: ignore[misc]


def test_all_fields_present():
    result = default_result()
    assert hasattr(result, "drug_name")
    assert hasattr(result, "target_protein")
    assert hasattr(result, "drug_mw_Da")
    assert hasattr(result, "drug_logP")
    assert hasattr(result, "inhibition_mechanism")
    assert hasattr(result, "ki_uM")
    assert hasattr(result, "inhibition_pct_at_1uM")
    assert hasattr(result, "inhibition_pct_at_10uM")
    assert hasattr(result, "aggregation_rate_reduction")
    assert hasattr(result, "preferred_excipient")
    assert hasattr(result, "formulation_ph_optimum")
    assert hasattr(result, "stability_improvement_pct")
    assert hasattr(result, "notes")


# ── mechanism classification ──────────────────────────────────────────────────


def test_mechanism_none_for_negative_neutral():
    # drug_charge=negative → no electrostatic, logP<2 → no hydrophobic, not (large+neutral) → none
    result = default_result(drug_charge="negative", drug_logP=0.5, drug_mw_Da=200.0)
    assert result.inhibition_mechanism == "none"


def test_mechanism_electrostatic_for_positive_low_logP():
    result = default_result(drug_charge="positive", drug_logP=1.0, drug_mw_Da=200.0)
    assert result.inhibition_mechanism == "electrostatic"


def test_mechanism_hydrophobic_for_neutral_high_logP():
    result = default_result(drug_charge="neutral", drug_logP=3.0, drug_mw_Da=200.0)
    assert result.inhibition_mechanism == "hydrophobic"


def test_mechanism_steric_for_large_neutral():
    result = default_result(drug_charge="neutral", drug_logP=1.0, drug_mw_Da=600.0)
    assert result.inhibition_mechanism == "steric"


def test_mechanism_mixed_positive_high_logP():
    # positive + logP>2 → 2 mechanisms → mixed
    result = default_result(drug_charge="positive", drug_logP=3.0, drug_mw_Da=200.0)
    assert result.inhibition_mechanism == "mixed"


def test_mechanism_mixed_positive_large_mw():
    # positive charge doesn't count toward steric (steric requires neutral)
    # positive=1, hydrophobic: logP<2=0, steric: mw>500 but not neutral=0 → electrostatic
    result = default_result(drug_charge="positive", drug_logP=1.0, drug_mw_Da=700.0)
    assert result.inhibition_mechanism == "electrostatic"


def test_mechanism_mixed_neutral_high_logP_large_mw():
    # logP>2 + mw>500 neutral → 2 mechanisms → mixed
    result = default_result(drug_charge="neutral", drug_logP=3.0, drug_mw_Da=600.0)
    assert result.inhibition_mechanism == "mixed"


# ── ki estimation ─────────────────────────────────────────────────────────────


def test_ki_none_mechanism():
    result = default_result(drug_charge="negative", drug_logP=0.5, drug_mw_Da=200.0)
    assert result.ki_uM == pytest.approx(50.0)


def test_ki_electrostatic():
    result = default_result(drug_charge="positive", drug_logP=1.0, drug_mw_Da=200.0)
    assert result.ki_uM == pytest.approx(50.0 / 3.0, rel=1e-6)


def test_ki_hydrophobic_logP3():
    result = default_result(drug_charge="neutral", drug_logP=3.0, drug_mw_Da=200.0)
    # hydrophobic: ki = 50 / (1 + 3) = 12.5
    assert result.ki_uM == pytest.approx(50.0 / 4.0, rel=1e-6)


def test_ki_steric():
    result = default_result(drug_charge="neutral", drug_logP=1.0, drug_mw_Da=600.0)
    assert result.ki_uM == pytest.approx(50.0 / 1.5, rel=1e-6)


def test_ki_mixed():
    result = default_result(drug_charge="positive", drug_logP=3.0, drug_mw_Da=200.0)
    assert result.ki_uM == pytest.approx(50.0 / 5.0, rel=1e-6)


def test_ki_clamp_lower():
    # Very high logP → ki could go below 0.1
    result = predict_aggregation_inhibition(
        "X", "P", drug_mw_Da=200.0, drug_logP=1000.0, drug_charge="neutral"
    )
    assert result.ki_uM >= 0.1


def test_ki_clamp_upper():
    result = default_result(drug_charge="negative", drug_logP=0.5, drug_mw_Da=200.0)
    assert result.ki_uM <= 500.0


# ── inhibition percentages ────────────────────────────────────────────────────


def test_inhibition_at_1uM_less_than_at_10uM():
    result = default_result(drug_charge="positive", drug_logP=1.0, drug_mw_Da=200.0)
    assert result.inhibition_pct_at_1uM < result.inhibition_pct_at_10uM


def test_inhibition_pct_range():
    result = default_result()
    assert 0.0 <= result.inhibition_pct_at_1uM <= 100.0
    assert 0.0 <= result.inhibition_pct_at_10uM <= 100.0


def test_inhibition_pct_formula():
    result = default_result(drug_charge="positive", drug_logP=1.0, drug_mw_Da=200.0)
    ki = result.ki_uM
    expected_1 = 1.0 / (1.0 + ki) * 100.0
    expected_10 = 10.0 / (10.0 + ki) * 100.0
    assert result.inhibition_pct_at_1uM == pytest.approx(expected_1, rel=1e-6)
    assert result.inhibition_pct_at_10uM == pytest.approx(expected_10, rel=1e-6)


# ── aggregation rate reduction ────────────────────────────────────────────────


def test_rate_reduction_at_least_1():
    result = default_result()
    assert result.aggregation_rate_reduction >= 1.0


def test_rate_reduction_at_most_100():
    result = default_result()
    assert result.aggregation_rate_reduction <= 100.0


def test_rate_reduction_higher_for_better_inhibitor():
    good = default_result(drug_charge="positive", drug_logP=3.0, drug_mw_Da=200.0)  # mixed
    poor = default_result(drug_charge="negative", drug_logP=0.5, drug_mw_Da=200.0)  # none
    assert good.aggregation_rate_reduction >= poor.aggregation_rate_reduction


# ── excipient recommendation ──────────────────────────────────────────────────


def test_excipient_electrostatic():
    result = default_result(drug_charge="positive", drug_logP=1.0, drug_mw_Da=200.0)
    assert result.preferred_excipient == "arginine"


def test_excipient_mixed():
    result = default_result(drug_charge="positive", drug_logP=3.0, drug_mw_Da=200.0)
    assert result.preferred_excipient == "arginine"


def test_excipient_hydrophobic():
    result = default_result(drug_charge="neutral", drug_logP=3.0, drug_mw_Da=200.0)
    assert result.preferred_excipient == "polysorbate 80"


def test_excipient_steric():
    result = default_result(drug_charge="neutral", drug_logP=1.0, drug_mw_Da=600.0)
    assert result.preferred_excipient == "PEG 4000"


def test_excipient_none():
    result = default_result(drug_charge="negative", drug_logP=0.5, drug_mw_Da=200.0)
    assert result.preferred_excipient == "sucrose"


# ── pH optimum ────────────────────────────────────────────────────────────────


def test_ph_positive_charge():
    result = default_result(drug_charge="positive")
    assert result.formulation_ph_optimum == pytest.approx(5.0)


def test_ph_negative_charge():
    result = default_result(drug_charge="negative")
    assert result.formulation_ph_optimum == pytest.approx(7.4)


def test_ph_neutral():
    result = default_result(drug_charge="neutral")
    assert result.formulation_ph_optimum == pytest.approx(6.5)


def test_ph_zwitterionic():
    result = default_result(drug_charge="zwitterionic")
    assert result.formulation_ph_optimum == pytest.approx(6.5)


# ── stability improvement ──────────────────────────────────────────────────────


def test_stability_improvement_range():
    result = default_result()
    assert 0.0 <= result.stability_improvement_pct <= 100.0


def test_stability_improvement_formula():
    result = default_result(drug_charge="positive", drug_logP=1.0, drug_mw_Da=200.0)
    expected = result.inhibition_pct_at_10uM * 0.7
    assert result.stability_improvement_pct == pytest.approx(expected, rel=1e-6)


# ── screening ─────────────────────────────────────────────────────────────────


def test_screen_returns_list():
    compounds = [
        {
            "drug_name": "A",
            "target_protein": "P",
            "drug_mw_Da": 300.0,
            "drug_logP": 3.0,
            "drug_charge": "positive",
        },
        {
            "drug_name": "B",
            "target_protein": "P",
            "drug_mw_Da": 200.0,
            "drug_logP": 0.5,
            "drug_charge": "negative",
        },
    ]
    results = screen_aggregation_inhibitors(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_descending():
    compounds = [
        {
            "drug_name": "Poor",
            "target_protein": "P",
            "drug_mw_Da": 200.0,
            "drug_logP": 0.5,
            "drug_charge": "negative",
        },
        {
            "drug_name": "Good",
            "target_protein": "P",
            "drug_mw_Da": 300.0,
            "drug_logP": 3.0,
            "drug_charge": "positive",
        },
    ]
    results = screen_aggregation_inhibitors(compounds)
    assert results[0].stability_improvement_pct >= results[1].stability_improvement_pct


def test_screen_empty_list():
    results = screen_aggregation_inhibitors([])
    assert results == []


def test_screen_with_optional_params():
    compounds = [
        {
            "drug_name": "X",
            "target_protein": "mAb",
            "drug_mw_Da": 500.0,
            "drug_logP": 2.5,
            "drug_charge": "positive",
            "drug_pka": 8.0,
            "concentration_uM": 5.0,
        }
    ]
    results = screen_aggregation_inhibitors(compounds)
    assert len(results) == 1
    assert isinstance(results[0], AggregationInhibitionResult)


# ── validation ────────────────────────────────────────────────────────────────


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="drug_mw_Da"):
        predict_aggregation_inhibition("X", "P", 0.0, 1.0, "neutral")


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="drug_mw_Da"):
        predict_aggregation_inhibition("X", "P", -100.0, 1.0, "neutral")


def test_validation_invalid_charge():
    with pytest.raises(ValueError, match="drug_charge"):
        predict_aggregation_inhibition("X", "P", 300.0, 1.0, "cationic")


def test_validation_concentration_zero():
    with pytest.raises(ValueError, match="concentration_uM"):
        predict_aggregation_inhibition("X", "P", 300.0, 1.0, "neutral", concentration_uM=0.0)


# ── notes field ───────────────────────────────────────────────────────────────


def test_notes_contains_drug_name():
    result = predict_aggregation_inhibition("Aspirin", "BSA", 180.0, 1.0, "neutral")
    assert "Aspirin" in result.notes


def test_notes_contains_protein():
    result = predict_aggregation_inhibition("Aspirin", "BSA", 180.0, 1.0, "neutral")
    assert "BSA" in result.notes
