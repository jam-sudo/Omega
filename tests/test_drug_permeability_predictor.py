"""Tests for drug_permeability_predictor module (Phase 777)."""

import pytest

from omega_pbpk.prediction.drug_permeability_predictor import (
    PermeabilityPredictionResult,
    predict_permeability,
    screen_permeability,
)

# ── Return type and field types ───────────────────────────────────────────────


def test_returns_correct_type():
    result = predict_permeability("DrugA", logp=2.0, mw_da=350.0, psa_A2=60.0)
    assert isinstance(result, PermeabilityPredictionResult)


def test_compound_name_stored():
    result = predict_permeability("Aspirin", logp=1.2, mw_da=180.0, psa_A2=63.6)
    assert result.compound_name == "Aspirin"


def test_logp_stored():
    result = predict_permeability("DrugA", logp=2.5, mw_da=300.0, psa_A2=50.0)
    assert result.logp == 2.5


def test_mw_da_stored():
    result = predict_permeability("DrugA", logp=2.0, mw_da=400.0, psa_A2=70.0)
    assert result.mw_da == 400.0


def test_psa_stored():
    result = predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=80.0)
    assert result.psa_A2 == 80.0


# ── All Papp values > 0 ───────────────────────────────────────────────────────


def test_papp_caco2_positive():
    result = predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0)
    assert result.papp_caco2_cm_s > 0.0


def test_papp_mdck_positive():
    result = predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0)
    assert result.papp_mdck_cm_s > 0.0


def test_papp_pampa_positive():
    result = predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0)
    assert result.papp_pampa_cm_s > 0.0


def test_bbe_positive():
    result = predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0)
    assert result.bbe_cm_s > 0.0


# ── Physical ranges ───────────────────────────────────────────────────────────


def test_papp_caco2_in_range():
    result = predict_permeability("DrugA", logp=3.0, mw_da=350.0, psa_A2=50.0)
    assert 1e-8 <= result.papp_caco2_cm_s <= 1e-4


def test_papp_mdck_is_80pct_caco2():
    # Use high logP / low PSA so Caco-2 is well above the lower clamp
    result = predict_permeability("DrugB", logp=4.0, mw_da=200.0, psa_A2=20.0)
    # MDCK = 0.8 * Caco-2, but both are individually clamped; verify ratio only when
    # Caco-2 is above the lower clamp so MDCK isn't independently clamped
    assert result.papp_mdck_cm_s == pytest.approx(0.8 * result.papp_caco2_cm_s, rel=1e-9)


# ── Physicochemical trends ────────────────────────────────────────────────────


def test_high_logp_higher_caco2_than_low_logp():
    low = predict_permeability("Low", logp=-1.0, mw_da=300.0, psa_A2=60.0)
    high = predict_permeability("High", logp=4.0, mw_da=300.0, psa_A2=60.0)
    assert high.papp_caco2_cm_s > low.papp_caco2_cm_s


def test_high_psa_lower_caco2_than_low_psa():
    low_psa = predict_permeability("LowPSA", logp=2.0, mw_da=300.0, psa_A2=20.0)
    high_psa = predict_permeability("HighPSA", logp=2.0, mw_da=300.0, psa_A2=160.0)
    assert low_psa.papp_caco2_cm_s > high_psa.papp_caco2_cm_s


def test_high_hbd_reduces_caco2():
    # Use higher logP so both compounds are above the lower clamp and the difference is visible
    no_hbd = predict_permeability("NoHBD", logp=3.5, mw_da=250.0, psa_A2=40.0, hbd_count=0)
    high_hbd = predict_permeability("HiHBD", logp=3.5, mw_da=250.0, psa_A2=40.0, hbd_count=6)
    assert no_hbd.papp_caco2_cm_s > high_hbd.papp_caco2_cm_s


# ── BCS classification ────────────────────────────────────────────────────────


def test_bcs_high_for_lipophilic_low_psa():
    result = predict_permeability("Lipophilic", logp=5.0, mw_da=250.0, psa_A2=20.0)
    assert result.bcs_permeability_class == "high"
    assert result.papp_caco2_cm_s >= 1e-6


def test_bcs_low_for_hydrophilic_high_psa():
    result = predict_permeability("Hydrophilic", logp=-2.0, mw_da=500.0, psa_A2=200.0)
    assert result.bcs_permeability_class == "low"


def test_bcs_class_is_high_or_low():
    result = predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0)
    assert result.bcs_permeability_class in {"high", "low"}


# ── BBB label ─────────────────────────────────────────────────────────────────


def test_bbb_label_in_expected_set():
    result = predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0)
    assert result.bbb_permeability in {"permeable", "limited", "impermeable"}


def test_bbb_permeable_for_small_lipophilic():
    # Small, lipophilic, low PSA — should penetrate BBB
    result = predict_permeability("CNSDrug", logp=3.5, mw_da=200.0, psa_A2=10.0)
    assert result.bbb_permeability in {"permeable", "limited"}


# ── Validation errors ─────────────────────────────────────────────────────────


def test_mw_zero_raises_value_error():
    with pytest.raises(ValueError):
        predict_permeability("DrugA", logp=2.0, mw_da=0.0, psa_A2=60.0)


def test_mw_negative_raises_value_error():
    with pytest.raises(ValueError):
        predict_permeability("DrugA", logp=2.0, mw_da=-100.0, psa_A2=60.0)


def test_hbd_negative_raises_value_error():
    with pytest.raises(ValueError):
        predict_permeability("DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0, hbd_count=-1)


# ── screen_permeability ───────────────────────────────────────────────────────


def test_screen_returns_list():
    compounds = [
        {"compound_name": "A", "logp": 1.0, "mw_da": 300.0, "psa_A2": 80.0},
        {"compound_name": "B", "logp": 3.0, "mw_da": 280.0, "psa_A2": 40.0},
    ]
    results = screen_permeability(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_caco2_descending():
    compounds = [
        {"compound_name": "A", "logp": -1.0, "mw_da": 400.0, "psa_A2": 150.0},
        {"compound_name": "B", "logp": 4.0, "mw_da": 250.0, "psa_A2": 20.0},
        {"compound_name": "C", "logp": 2.0, "mw_da": 320.0, "psa_A2": 60.0},
    ]
    results = screen_permeability(compounds)
    papp_values = [r.papp_caco2_cm_s for r in results]
    assert papp_values == sorted(papp_values, reverse=True)


def test_screen_empty_list():
    results = screen_permeability([])
    assert results == []


# ── Ionization / logD correction ─────────────────────────────────────────────


def test_ionized_anion_lower_permeability_than_neutral():
    # Use high logP so neutral is well above lower clamp; ionized will be lower
    neutral = predict_permeability("N", logp=4.0, mw_da=250.0, psa_A2=30.0, charge="neutral")
    anion = predict_permeability(
        "A", logp=4.0, mw_da=250.0, psa_A2=30.0, charge="anion", pka=5.0, ph=7.4
    )
    # At pH 7.4, acid with pKa=5.0 is mostly ionized → lower effective logP → lower Papp
    assert neutral.papp_caco2_cm_s > anion.papp_caco2_cm_s
