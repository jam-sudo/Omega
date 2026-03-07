"""Tests for Phase 794 CSF penetration predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.csf_penetration_predictor import (
    CSFPenetrationResult,
    predict_csf_penetration,
    screen_csf_penetration,
)

# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = predict_csf_penetration("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert isinstance(result, CSFPenetrationResult)


def test_frozen_dataclass_immutable():
    result = predict_csf_penetration("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    with pytest.raises((AttributeError, TypeError)):
        result.predicted_kp_uu_csf = 0.5  # type: ignore[misc]


def test_compound_name_stored():
    result = predict_csf_penetration("Aspirin", logp=1.2, mw_da=180.0, psa_A2=63.6, fup=0.05)
    assert result.compound_name == "Aspirin"


def test_logp_stored():
    result = predict_csf_penetration("DrugB", logp=3.5, mw_da=350.0, psa_A2=50.0, fup=0.2)
    assert result.logp == 3.5


def test_fup_stored():
    result = predict_csf_penetration("DrugC", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.15)
    assert result.fup == 0.15


def test_pka_basic_stored_none():
    result = predict_csf_penetration("DrugD", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert result.pka_basic is None


def test_pka_basic_stored_value():
    result = predict_csf_penetration(
        "DrugE", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1, pka_basic=8.5
    )
    assert result.pka_basic == 8.5


# ---------------------------------------------------------------------------
# BBB permeability classification
# ---------------------------------------------------------------------------


def test_high_logp_better_bbb():
    """High logP (lipophilic) should give better BBB permeability than low logP."""
    r_high = predict_csf_penetration("HighLogP", logp=5.0, mw_da=300.0, psa_A2=20.0, fup=0.3)
    r_low = predict_csf_penetration("LowLogP", logp=-2.0, mw_da=300.0, psa_A2=20.0, fup=0.3)
    assert r_high.predicted_kp_uu_csf >= r_low.predicted_kp_uu_csf


def test_high_psa_lower_penetration():
    """High PSA reduces BBB penetration (P-gp held constant to isolate PSA effect)."""
    r_low_psa = predict_csf_penetration(
        "LowPSA", logp=3.0, mw_da=300.0, psa_A2=20.0, fup=0.2, is_pgp_substrate=False
    )
    r_high_psa = predict_csf_penetration(
        "HighPSA", logp=3.0, mw_da=300.0, psa_A2=150.0, fup=0.2, is_pgp_substrate=False
    )
    assert r_low_psa.predicted_kp_uu_csf >= r_high_psa.predicted_kp_uu_csf


def test_bbb_permeability_valid_values():
    result = predict_csf_penetration("DrugF", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert result.bbb_permeability in {"high", "moderate", "low"}


# ---------------------------------------------------------------------------
# Kp_uu and Kp_total
# ---------------------------------------------------------------------------


def test_kp_uu_positive():
    result = predict_csf_penetration("DrugG", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert result.predicted_kp_uu_csf > 0.0


def test_kp_total_equals_kp_uu_times_fup():
    """Kp_total should equal Kp_uu * fup (no P-gp correction case)."""
    fup = 0.2
    result = predict_csf_penetration(
        "DrugH", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=fup, is_pgp_substrate=False
    )
    expected = result.predicted_kp_uu_csf * fup
    assert abs(result.predicted_kp_total_csf - expected) < 1e-10


def test_pgp_substrate_reduces_kp_uu():
    """P-gp substrate should have lower Kp_uu than non-substrate."""
    r_sub = predict_csf_penetration(
        "PgpSub", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.2, is_pgp_substrate=True
    )
    r_nonsub = predict_csf_penetration(
        "PgpNon", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.2, is_pgp_substrate=False
    )
    assert r_sub.predicted_kp_uu_csf < r_nonsub.predicted_kp_uu_csf


# ---------------------------------------------------------------------------
# CSF exposure class
# ---------------------------------------------------------------------------


def test_csf_exposure_class_valid():
    result = predict_csf_penetration("DrugI", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert result.csf_exposure_class in {"excellent", "good", "limited", "poor"}


def test_excellent_exposure_for_high_kp_uu():
    """A drug with very high logP and low PSA should have excellent exposure."""
    result = predict_csf_penetration(
        "CNSDrug", logp=6.0, mw_da=250.0, psa_A2=10.0, fup=0.5, is_pgp_substrate=False
    )
    assert result.predicted_kp_uu_csf >= 0.3
    assert result.csf_exposure_class == "excellent"


# ---------------------------------------------------------------------------
# Efflux and MDR
# ---------------------------------------------------------------------------


def test_efflux_risk_valid():
    result = predict_csf_penetration("DrugJ", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert result.efflux_risk in {"high", "low"}


def test_pgp_substrate_high_efflux_risk():
    result = predict_csf_penetration(
        "PgpDrug", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1, is_pgp_substrate=True
    )
    assert result.efflux_risk == "high"


def test_mdr_score_range():
    result = predict_csf_penetration("DrugK", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert 0.0 <= result.mdr_mdr_score <= 100.0


def test_pgp_probability_range():
    result = predict_csf_penetration("DrugL", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.1)
    assert 0.0 <= result.pgp_substrate_probability <= 1.0


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "logp": 2.0, "mw_da": 300.0, "psa_A2": 60.0, "fup": 0.2},
        {"compound_name": "B", "logp": 4.0, "mw_da": 280.0, "psa_A2": 30.0, "fup": 0.3},
    ]
    results = screen_csf_penetration(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_descending():
    compounds = [
        {"compound_name": "Poor", "logp": -2.0, "mw_da": 600.0, "psa_A2": 200.0, "fup": 0.05},
        {"compound_name": "Good", "logp": 4.5, "mw_da": 250.0, "psa_A2": 20.0, "fup": 0.5},
    ]
    results = screen_csf_penetration(compounds)
    assert results[0].predicted_kp_uu_csf >= results[-1].predicted_kp_uu_csf


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_fup_zero():
    with pytest.raises(ValueError):
        predict_csf_penetration("DrugM", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=0.0)


def test_raises_on_fup_negative():
    with pytest.raises(ValueError):
        predict_csf_penetration("DrugN", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=-0.1)


def test_raises_on_fup_greater_than_one():
    with pytest.raises(ValueError):
        predict_csf_penetration("DrugO", logp=2.0, mw_da=300.0, psa_A2=60.0, fup=1.5)


def test_raises_on_zero_mw():
    with pytest.raises(ValueError):
        predict_csf_penetration("DrugP", logp=2.0, mw_da=0.0, psa_A2=60.0, fup=0.1)


def test_raises_on_negative_mw():
    with pytest.raises(ValueError):
        predict_csf_penetration("DrugQ", logp=2.0, mw_da=-100.0, psa_A2=60.0, fup=0.1)
