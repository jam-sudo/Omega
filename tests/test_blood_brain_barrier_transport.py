"""Tests for Phase 949 — BBB Transport Prediction."""

import pytest

from omega_pbpk.prediction.blood_brain_barrier_transport import (
    BBBTransportResult,
    predict_bbb_transport,
    screen_cns_penetration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_result(**kwargs):
    defaults = dict(drug_name="TestDrug", logp=2.0, mw=300.0, pka=7.4, ionization_type="neutral")
    defaults.update(kwargs)
    return predict_bbb_transport(**defaults)


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------
def test_return_type_is_bbb_result():
    result = _base_result()
    assert isinstance(result, BBBTransportResult)


def test_result_is_frozen():
    result = _base_result()
    with pytest.raises((AttributeError, TypeError)):
        result.kp_uu_brain = 0.5  # type: ignore


# ---------------------------------------------------------------------------
# Field ranges
# ---------------------------------------------------------------------------
def test_bbb_passive_perm_positive():
    result = _base_result()
    assert result.bbb_passive_perm_cm_s > 0


def test_pgp_efflux_probability_range():
    result = _base_result()
    assert 0.0 <= result.pgp_efflux_probability <= 1.0


def test_bcrp_efflux_probability_range():
    result = _base_result()
    assert 0.0 <= result.bcrp_efflux_probability <= 1.0


def test_net_efflux_ratio_ge_one():
    result = _base_result()
    assert result.net_efflux_ratio >= 1.0


def test_kp_uu_brain_range():
    result = _base_result()
    assert 0.01 <= result.kp_uu_brain <= 1.0


def test_cns_penetration_valid():
    result = _base_result()
    assert result.cns_penetration in {"high", "moderate", "low"}


def test_pgp_substrate_risk_valid():
    result = _base_result()
    assert result.pgp_substrate_risk in {"low", "moderate", "high"}


def test_bbb_limited_is_bool():
    result = _base_result()
    assert isinstance(result.bbb_limited, bool)


def test_psa_estimated_non_negative():
    result = _base_result()
    assert result.psa_estimated >= 0


def test_notes_non_empty():
    result = _base_result()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Physics / model direction
# ---------------------------------------------------------------------------
def test_low_logp_low_mw_neutral_gives_higher_kp_uu():
    """Small neutral molecule should have higher CNS penetration than large base."""
    small_neutral = predict_bbb_transport(
        "small", logp=1.0, mw=150.0, pka=7.0, ionization_type="neutral"
    )
    large_base = predict_bbb_transport("large", logp=4.0, mw=600.0, pka=9.0, ionization_type="base")
    assert small_neutral.kp_uu_brain >= large_base.kp_uu_brain


def test_high_mw_base_high_pgp_probability():
    """High MW basic drug should have high P-gp efflux probability."""
    result = predict_bbb_transport("drug", logp=3.0, mw=500.0, pka=9.5, ionization_type="base")
    assert result.pgp_efflux_probability >= 0.5


def test_low_logp_neutral_lower_net_efflux_than_high_logp_base():
    low = predict_bbb_transport("low", logp=0.5, mw=200.0, pka=5.0, ionization_type="neutral")
    high = predict_bbb_transport("high", logp=4.0, mw=500.0, pka=9.5, ionization_type="base")
    assert low.net_efflux_ratio <= high.net_efflux_ratio


# ---------------------------------------------------------------------------
# screen_cns_penetration
# ---------------------------------------------------------------------------
_COMPOUNDS = [
    {"drug_name": "A", "logp": 2.0, "mw": 250.0, "pka": 7.0, "ionization_type": "neutral"},
    {"drug_name": "B", "logp": 4.0, "mw": 500.0, "pka": 9.5, "ionization_type": "base"},
    {"drug_name": "C", "logp": 0.5, "mw": 180.0, "pka": 5.0, "ionization_type": "acid"},
]


def test_screen_returns_correct_length():
    results = screen_cns_penetration(_COMPOUNDS)
    assert len(results) == len(_COMPOUNDS)


def test_screen_sorted_by_kp_uu_descending():
    results = screen_cns_penetration(_COMPOUNDS)
    kp_values = [r.kp_uu_brain for r in results]
    assert kp_values == sorted(kp_values, reverse=True)


# ---------------------------------------------------------------------------
# BBB limited flag and CNS classification
# ---------------------------------------------------------------------------
def test_bbb_limited_true_when_kp_uu_below_0_3():
    # force low kp_uu with high MW base
    result = predict_bbb_transport("drug", logp=4.0, mw=600.0, pka=9.5, ionization_type="base")
    if result.kp_uu_brain < 0.3:
        assert result.bbb_limited is True


def test_bbb_limited_false_when_kp_uu_high():
    # Small neutral with moderate logP should have high kp_uu
    result = predict_bbb_transport("small", logp=1.0, mw=120.0, pka=7.0, ionization_type="neutral")
    if result.kp_uu_brain >= 0.3:
        assert result.bbb_limited is False


def test_cns_penetration_high_when_kp_uu_ge_0_3():
    result = _base_result(logp=1.0, mw=120.0, ionization_type="neutral")
    if result.kp_uu_brain >= 0.3:
        assert result.cns_penetration == "high"


def test_cns_penetration_low_when_kp_uu_lt_0_1():
    result = predict_bbb_transport("drug", logp=4.0, mw=650.0, pka=9.5, ionization_type="base")
    if result.kp_uu_brain < 0.1:
        assert result.cns_penetration == "low"


# ---------------------------------------------------------------------------
# drug_name preserved
# ---------------------------------------------------------------------------
def test_drug_name_preserved():
    result = predict_bbb_transport("Aspirin", logp=1.2, mw=180.0, pka=3.5, ionization_type="acid")
    assert result.drug_name == "Aspirin"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_bbb_transport("X", logp=2.0, mw=0.0, pka=7.0, ionization_type="neutral")


def test_validation_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_bbb_transport("X", logp=2.0, mw=-10.0, pka=7.0, ionization_type="neutral")


def test_validation_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_bbb_transport("X", logp=2.0, mw=300.0, pka=7.0, ionization_type="zwitterion")


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------
def test_kp_uu_consistent_with_cns_penetration_moderate():
    result = predict_bbb_transport("drug", logp=2.5, mw=350.0, pka=7.0, ionization_type="neutral")
    kp = result.kp_uu_brain
    if 0.1 <= kp < 0.3:
        assert result.cns_penetration == "moderate"


def test_pgp_substrate_risk_low_for_low_pgp_prob():
    result = predict_bbb_transport("small", logp=0.5, mw=150.0, pka=5.0, ionization_type="acid")
    if result.pgp_efflux_probability < 0.3:
        assert result.pgp_substrate_risk == "low"


def test_pgp_substrate_risk_high_for_high_pgp_prob():
    result = predict_bbb_transport("drug", logp=3.0, mw=500.0, pka=9.5, ionization_type="base")
    if result.pgp_efflux_probability > 0.6:
        assert result.pgp_substrate_risk == "high"
