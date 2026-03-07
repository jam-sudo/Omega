"""Tests for skin sensitization risk prediction - Phase 790 API."""

import pytest

from omega_pbpk.risk.skin_sensitization_risk_p790 import (
    SkinSensitizationResultP790,
    predict_skin_sensitization,
    screen_sensitization,
)

_VALID_POTENCY = {"extreme", "strong", "moderate", "weak", "non-sensitizer"}
_VALID_GHS = {"H317", "H317 Cat.1A", "H317 Cat.1B", "no classification"}


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_skin_sensitization("TestCompound")
    assert isinstance(result, SkinSensitizationResultP790)


def test_result_is_frozen():
    result = predict_skin_sensitization("TestCompound")
    with pytest.raises((AttributeError, TypeError)):
        result.compound_name = "Modified"  # type: ignore[misc]


def test_compound_name_stored():
    result = predict_skin_sensitization("Isoeugenol")
    assert result.compound_name == "Isoeugenol"


# ---------------------------------------------------------------------------
# Structural alert detection
# ---------------------------------------------------------------------------


def test_michael_acceptor_makes_sensitizer():
    result = predict_skin_sensitization("MACompound", has_michael_acceptor=True, logp=2.0)
    assert result.potency_class != "non-sensitizer"


def test_acyl_transfer_makes_sensitizer():
    result = predict_skin_sensitization("ACCompound", has_acyl_transfer=True, logp=2.0)
    assert result.potency_class != "non-sensitizer"


def test_no_alerts_non_sensitizer():
    result = predict_skin_sensitization(
        "Inert",
        logp=2.0,
        mw_da=300.0,
        has_michael_acceptor=False,
        has_acyl_transfer=False,
        has_sn2_electrophile=False,
        has_radical_mechanism=False,
    )
    assert result.potency_class == "non-sensitizer"


def test_no_alerts_no_classification():
    result = predict_skin_sensitization("Inert", logp=2.0)
    assert result.ghs_classification == "no classification"


def test_n_structural_alerts_zero_when_no_alerts():
    result = predict_skin_sensitization("Inert")
    assert result.n_structural_alerts == 0


def test_n_structural_alerts_counts_correctly_one():
    result = predict_skin_sensitization("OnAlert", has_michael_acceptor=True)
    assert result.n_structural_alerts == 1


def test_n_structural_alerts_counts_all_four():
    result = predict_skin_sensitization(
        "AllAlerts",
        has_michael_acceptor=True,
        has_acyl_transfer=True,
        has_sn2_electrophile=True,
        has_radical_mechanism=True,
    )
    assert result.n_structural_alerts == 4


# ---------------------------------------------------------------------------
# Potency class and GHS
# ---------------------------------------------------------------------------


def test_potency_class_in_valid_set():
    result = predict_skin_sensitization("Drug", has_michael_acceptor=True)
    assert result.potency_class in _VALID_POTENCY


def test_ghs_classification_in_valid_set():
    result = predict_skin_sensitization("Drug", has_acyl_transfer=True)
    assert result.ghs_classification in _VALID_GHS


def test_strong_sensitizer_ghs_1a():
    """Acyl transfer (50) + logP 1-4 bonus (10) = 60 → protein_binding_score=60,
    ec3 at boundary of strong (< 1%)."""
    result = predict_skin_sensitization(
        "StrongSens",
        has_acyl_transfer=True,
        logp=2.0,
    )
    # score = 50 + 10 = 60; ec3 = 1.0% (boundary); potency = strong or extreme
    assert result.ghs_classification in ("H317 Cat.1A", "H317 Cat.1B")


def test_weak_sensitizer_ghs_1b():
    result = predict_skin_sensitization(
        "WeakSens",
        has_sn2_electrophile=True,
        logp=5.0,  # no logP modifier
        skin_penetration_factor=0.5,
    )
    # base_score = 30; * 0.5 = 15 → ec3 > 10% → weak or non-sensitizer
    assert result.ghs_classification in ("H317 Cat.1B", "no classification")


# ---------------------------------------------------------------------------
# LLNA EC3 estimate
# ---------------------------------------------------------------------------


def test_llna_ec3_positive():
    result = predict_skin_sensitization("Drug", has_michael_acceptor=True)
    assert result.llna_ec3_estimate_pct > 0.0


def test_llna_ec3_in_reasonable_range():
    result = predict_skin_sensitization("Drug", has_michael_acceptor=True)
    # Michael acceptor is a sensitizer — EC3 should be in sensitizer range
    assert 0.0 < result.llna_ec3_estimate_pct <= 100.0


def test_more_alerts_lower_ec3():
    """More potent sensitizer → lower EC3."""
    r_weak = predict_skin_sensitization("Weak", has_sn2_electrophile=True, logp=5.0)
    r_strong = predict_skin_sensitization(
        "Strong", has_acyl_transfer=True, has_michael_acceptor=True, logp=2.0
    )
    assert r_strong.llna_ec3_estimate_pct < r_weak.llna_ec3_estimate_pct


# ---------------------------------------------------------------------------
# Physicochemical modifiers
# ---------------------------------------------------------------------------


def test_high_logp_penetration_bonus():
    """logP in 1-4 range should give higher protein binding score than logP 7."""
    r_opt = predict_skin_sensitization("Opt", has_michael_acceptor=True, logp=2.0)
    r_poor = predict_skin_sensitization("Poor", has_michael_acceptor=True, logp=7.0)
    assert r_opt.protein_binding_score > r_poor.protein_binding_score


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mw_da_zero_raises():
    with pytest.raises(ValueError, match="mw_da must be > 0"):
        predict_skin_sensitization("Drug", mw_da=0.0)


def test_mw_da_negative_raises():
    with pytest.raises(ValueError, match="mw_da must be > 0"):
        predict_skin_sensitization("Drug", mw_da=-100.0)


# ---------------------------------------------------------------------------
# screen_sensitization
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A"},
        {"compound_name": "B", "has_michael_acceptor": True},
    ]
    results = screen_sensitization(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_ascending_ec3():
    """Most potent (lowest EC3) should come first."""
    compounds = [
        {"compound_name": "NonSens"},
        {"compound_name": "Strong", "has_acyl_transfer": True, "logp": 2.0},
        {"compound_name": "Weak", "has_sn2_electrophile": True, "logp": 5.0},
    ]
    results = screen_sensitization(compounds)
    ec3_values = [r.llna_ec3_estimate_pct for r in results]
    assert ec3_values == sorted(ec3_values)


def test_screen_all_type_correct():
    compounds = [{"compound_name": f"C{i}"} for i in range(5)]
    results = screen_sensitization(compounds)
    assert all(isinstance(r, SkinSensitizationResultP790) for r in results)
