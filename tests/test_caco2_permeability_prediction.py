"""Tests for prediction/caco2_permeability_prediction.py (Phase 184)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.caco2_permeability_prediction import (
    Caco2Result,
    predict_caco2,
    screen_permeability,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_result() -> Caco2Result:
    return predict_caco2(
        drug_name="Metoprolol",
        mw_Da=267.4,
        logP=1.88,
        psa_A2=50.7,
        hbd=3,
        hba=4,
    )


@pytest.fixture()
def high_logp_result() -> Caco2Result:
    return predict_caco2(
        drug_name="LipophilicDrug",
        mw_Da=300.0,
        logP=5.0,
        psa_A2=20.0,
        hbd=0,
        hba=2,
    )


@pytest.fixture()
def low_logp_result() -> Caco2Result:
    return predict_caco2(
        drug_name="HydrophilicDrug",
        mw_Da=300.0,
        logP=0.0,
        psa_A2=20.0,
        hbd=0,
        hba=2,
    )


@pytest.fixture()
def high_psa_result() -> Caco2Result:
    return predict_caco2(
        drug_name="HighPSADrug",
        mw_Da=300.0,
        logP=2.0,
        psa_A2=150.0,
        hbd=2,
        hba=5,
    )


@pytest.fixture()
def low_psa_result() -> Caco2Result:
    return predict_caco2(
        drug_name="LowPSADrug",
        mw_Da=300.0,
        logP=2.0,
        psa_A2=20.0,
        hbd=2,
        hba=5,
    )


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


def test_returns_caco2_result(default_result):
    assert isinstance(default_result, Caco2Result)


def test_drug_name_preserved(default_result):
    assert default_result.drug_name == "Metoprolol"


def test_mw_preserved(default_result):
    assert default_result.mw_Da == 267.4


def test_logp_preserved(default_result):
    assert default_result.logP == 1.88


def test_psa_preserved(default_result):
    assert default_result.psa_A2 == 50.7


def test_hbd_preserved(default_result):
    assert default_result.hbd == 3


def test_hba_preserved(default_result):
    assert default_result.hba == 4


def test_result_is_frozen(default_result):
    """Caco2Result must be immutable (frozen dataclass)."""
    with pytest.raises((AttributeError, TypeError)):
        default_result.papp_nm_s = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Papp range and positivity
# ---------------------------------------------------------------------------


def test_papp_positive(default_result):
    assert default_result.papp_nm_s > 0.0


def test_papp_within_clamped_range(default_result):
    assert 0.01 <= default_result.papp_nm_s <= 1000.0


def test_log_papp_finite(default_result):
    import math

    assert math.isfinite(default_result.log_papp)


# ---------------------------------------------------------------------------
# Physicochemical effects on Papp
# ---------------------------------------------------------------------------


def test_higher_logp_higher_papp(high_logp_result, low_logp_result):
    """logP increase should raise Papp."""
    assert high_logp_result.papp_nm_s > low_logp_result.papp_nm_s


def test_higher_psa_lower_papp(high_psa_result, low_psa_result):
    """Higher PSA should lower Papp."""
    assert high_psa_result.papp_nm_s < low_psa_result.papp_nm_s


def test_higher_hbd_lower_papp():
    """More HBDs should lower Papp."""
    r_low = predict_caco2("Drug", mw_Da=300.0, logP=2.0, psa_A2=50.0, hbd=0)
    r_high = predict_caco2("Drug", mw_Da=300.0, logP=2.0, psa_A2=50.0, hbd=4)
    assert r_high.papp_nm_s < r_low.papp_nm_s


# ---------------------------------------------------------------------------
# Permeability class
# ---------------------------------------------------------------------------


def test_high_papp_gives_high_class():
    """Papp > 100 nm/s → 'high'."""
    # logP=5, PSA=5, HBD=0, MW=200 → very high Papp
    r = predict_caco2("Drug", mw_Da=200.0, logP=5.0, psa_A2=5.0, hbd=0)
    assert r.permeability_class == "high"


def test_low_papp_gives_low_class():
    """Very low Papp → 'low'."""
    # logP=-2, PSA=200, HBD=10 → very low Papp
    r = predict_caco2("Drug", mw_Da=600.0, logP=-2.0, psa_A2=200.0, hbd=10)
    assert r.permeability_class == "low"


def test_permeability_class_options(default_result):
    assert default_result.permeability_class in {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# fa_fraction
# ---------------------------------------------------------------------------


def test_fa_fraction_in_range(default_result):
    assert 0.0 <= default_result.fa_fraction <= 1.0


def test_fa_fraction_range_all_drugs(high_logp_result, low_logp_result, high_psa_result):
    for r in [high_logp_result, low_logp_result, high_psa_result]:
        assert 0.0 <= r.fa_fraction <= 1.0


def test_high_papp_high_fa():
    """High Papp drug should have high fa."""
    r = predict_caco2("Drug", mw_Da=200.0, logP=5.0, psa_A2=5.0, hbd=0)
    assert r.fa_fraction > 0.5


# ---------------------------------------------------------------------------
# Absorption prediction categories
# ---------------------------------------------------------------------------


def test_absorption_prediction_options(default_result):
    assert default_result.absorption_prediction in {
        "complete",
        "high",
        "moderate",
        "poor",
    }


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0


# ---------------------------------------------------------------------------
# screen_permeability
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    drugs = [
        {"drug_name": "A", "mw_Da": 200.0, "logP": 2.0, "psa_A2": 40.0, "hbd": 1, "hba": 2},
        {"drug_name": "B", "mw_Da": 350.0, "logP": 4.0, "psa_A2": 60.0, "hbd": 0, "hba": 3},
    ]
    results = screen_permeability(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_papp_descending():
    drugs = [
        {"drug_name": "Low", "mw_Da": 500.0, "logP": 0.0, "psa_A2": 150.0, "hbd": 5, "hba": 8},
        {"drug_name": "High", "mw_Da": 200.0, "logP": 5.0, "psa_A2": 10.0, "hbd": 0, "hba": 1},
        {"drug_name": "Mid", "mw_Da": 300.0, "logP": 2.0, "psa_A2": 50.0, "hbd": 2, "hba": 4},
    ]
    results = screen_permeability(drugs)
    papps = [r.papp_nm_s for r in results]
    assert papps == sorted(papps, reverse=True)


def test_screen_returns_caco2_result_objects():
    drugs = [
        {"drug_name": "X", "mw_Da": 250.0, "logP": 1.5, "psa_A2": 45.0},
    ]
    results = screen_permeability(drugs)
    assert isinstance(results[0], Caco2Result)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_caco2("Drug", mw_Da=0.0, logP=2.0, psa_A2=50.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_caco2("Drug", mw_Da=-100.0, logP=2.0, psa_A2=50.0)


def test_validation_psa_negative():
    with pytest.raises(ValueError, match="psa_A2"):
        predict_caco2("Drug", mw_Da=300.0, logP=2.0, psa_A2=-5.0)


def test_validation_hbd_negative():
    with pytest.raises(ValueError, match="hbd"):
        predict_caco2("Drug", mw_Da=300.0, logP=2.0, psa_A2=50.0, hbd=-1)
