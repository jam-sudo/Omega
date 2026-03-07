"""Tests for Phase 623 permeability predictor — predict_permeability_623 / screen_permeability_623."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.permeability_predictor import (
    PermeabilityResult623,
    predict_permeability_623,
    screen_permeability_623,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> PermeabilityResult623:
    defaults = dict(drug_name="TestDrug", logP=2.0, mw=300.0, psa=70.0, n_hbd=2)
    defaults.update(kwargs)
    return predict_permeability_623(**defaults)


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_result():
    r = _default()
    assert isinstance(r, PermeabilityResult623)


def test_papp_ab_positive():
    r = _default()
    assert r.papp_ab_cm_s > 0.0


def test_papp_ba_ge_papp_ab():
    r = _default()
    assert r.papp_ba_cm_s >= r.papp_ab_cm_s


def test_efflux_ratio_ge_1():
    r = _default()
    assert r.efflux_ratio >= 1.0


def test_peff_positive():
    r = _default()
    assert r.peff_cm_s > 0.0


def test_log_papp_is_correct():
    r = _default()
    # papp_ab = 10^log_papp (within clamping)
    assert abs(math.log10(r.papp_ab_cm_s) - r.log_papp) < 1e-9


# ---------------------------------------------------------------------------
# QSAR directionality
# ---------------------------------------------------------------------------


def test_high_logP_higher_papp():
    low = _default(logP=0.0)
    high = _default(logP=5.0)
    assert high.papp_ab_cm_s >= low.papp_ab_cm_s


def test_high_psa_lower_papp():
    low_psa = _default(psa=20.0)
    high_psa = _default(psa=150.0)
    assert low_psa.papp_ab_cm_s >= high_psa.papp_ab_cm_s


# ---------------------------------------------------------------------------
# Permeability class
# ---------------------------------------------------------------------------


def test_permeability_class_valid():
    r = _default()
    assert r.permeability_class in {"high", "moderate", "low", "very_low"}


def test_high_papp_high_class():
    # Very favourable physicochemistry → should give high class
    r = predict_permeability_623(drug_name="A", logP=5.0, mw=200.0, psa=10.0, n_hbd=0)
    assert r.permeability_class == "high"


# ---------------------------------------------------------------------------
# P-gp substrate risk
# ---------------------------------------------------------------------------


def test_pgp_risk_with_affinity():
    r = predict_permeability_623(drug_name="Sub", logP=2.0, mw=300.0, psa=70.0, pgp_affinity=1.0)
    assert r.pgp_substrate_risk is True


def test_pgp_risk_false_no_affinity():
    r = predict_permeability_623(drug_name="NonSub", logP=2.0, mw=300.0, psa=70.0, pgp_affinity=0.0)
    assert r.pgp_substrate_risk is False


# ---------------------------------------------------------------------------
# BBB penetration
# ---------------------------------------------------------------------------


def test_bbb_penetration_for_small_lipophilic():
    r = predict_permeability_623(drug_name="CNS", logP=2.0, mw=300.0, psa=40.0)
    assert r.bbb_penetration is True


def test_bbb_no_penetration_high_psa():
    r = predict_permeability_623(drug_name="NoCNS", logP=2.0, mw=300.0, psa=100.0)
    assert r.bbb_penetration is False


# ---------------------------------------------------------------------------
# Fraction absorbed
# ---------------------------------------------------------------------------


def test_fraction_absorbed_in_0_1():
    r = _default()
    assert 0.0 <= r.fraction_absorbed <= 1.0


def test_high_papp_high_fa():
    # logP=5, mw=200, psa=10, n_hbd=0 → log_papp well above -6 → fa close to 1
    r = predict_permeability_623(drug_name="A", logP=5.0, mw=200.0, psa=10.0, n_hbd=0)
    assert r.fraction_absorbed > 0.8


# ---------------------------------------------------------------------------
# Assay type corrections
# ---------------------------------------------------------------------------


def test_mdck_assay_higher():
    r_caco2 = _default(assay_type="caco2")
    r_mdck = _default(assay_type="mdck")
    assert r_mdck.papp_ab_cm_s >= r_caco2.papp_ab_cm_s


def test_pampa_no_efflux():
    r = _default(assay_type="pampa", pgp_affinity=1.0)
    assert r.efflux_ratio == 1.0


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"name": "A", "logP": 2.0, "mw": 300.0, "psa": 70.0},
        {"name": "B", "logP": 4.0, "mw": 200.0, "psa": 30.0},
    ]
    results = screen_permeability_623(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_descending():
    compounds = [
        {"name": "Low", "logP": -2.0, "mw": 600.0, "psa": 200.0},
        {"name": "High", "logP": 5.0, "mw": 200.0, "psa": 10.0},
        {"name": "Mid", "logP": 2.0, "mw": 300.0, "psa": 70.0},
    ]
    results = screen_permeability_623(compounds)
    papp_vals = [r.papp_ab_cm_s for r in results]
    assert papp_vals == sorted(papp_vals, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_permeability_623(drug_name="X", logP=2.0, mw=0.0, psa=70.0)


def test_invalid_assay_raises():
    with pytest.raises(ValueError, match="assay_type"):
        predict_permeability_623(drug_name="X", logP=2.0, mw=300.0, psa=70.0, assay_type="other")
