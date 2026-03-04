"""Tests for Phase 21 dynamic DDI simulation — classify_ddi, simulate_ddi, API endpoint.

Covers:
  - classify_ddi: all FDA 2020 AUCR severity classes and boundary values
  - simulate_ddi: no-inhibition baseline, potent inhibition, profile shapes
  - POST /ddi/simulate: endpoint contract and profile consistency
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.ddi_simulation import (
    DDISimulationResult,
    PerpetratorSpec,
    classify_ddi,
    simulate_ddi,
)
from omega_pbpk.drugs.drug import Drug

client = TestClient(app)

# ---------------------------------------------------------------------------
# Test victim drug — midazolam-like (CYP3A4 primary metabolism)
# ---------------------------------------------------------------------------

TEST_DRUG = Drug(
    name="test_victim",
    mw=325.8,
    logP=3.89,
    pka=[6.15],
    drug_type="monoprotic_base",
    fup=0.032,
    rbp=0.66,
    clint_hepatic_L_per_h=750.0,
    peff=5.37,
    fm={"CYP3A4": 0.93, "other": 0.07},
    kp={
        "lung": 0.6,
        "brain": 6.0,
        "heart": 5.0,
        "kidney": 5.0,
        "liver": 5.8,
        "spleen": 3.8,
        "gut_wall": 4.2,
        "rest": 2.5,
    },
)

# Corresponding DrugRequest payload for API tests
TEST_DRUG_PAYLOAD = {
    "name": "test_victim",
    "mw": 325.8,
    "logP": 3.89,
    "pka": [6.15],
    "drug_type": "monoprotic_base",
    "fup": 0.032,
    "rbp": 0.66,
    "clint_hepatic_L_per_h": 750.0,
    "peff": 5.37,
    "fm": {"CYP3A4": 0.93, "other": 0.07},
}


# ===========================================================================
# classify_ddi — unit tests
# ===========================================================================


@pytest.mark.integration
def test_classify_ddi_no_interaction():
    """AUCR = 1.1 → no_interaction."""
    assert classify_ddi(1.1) == "no_interaction"


@pytest.mark.integration
def test_classify_ddi_weak():
    """AUCR = 1.5 → weak."""
    assert classify_ddi(1.5) == "weak"


@pytest.mark.integration
def test_classify_ddi_moderate():
    """AUCR = 3.0 → moderate."""
    assert classify_ddi(3.0) == "moderate"


@pytest.mark.integration
def test_classify_ddi_strong():
    """AUCR = 6.0 → strong."""
    assert classify_ddi(6.0) == "strong"


@pytest.mark.integration
def test_classify_ddi_weak_induction():
    """AUCR = 0.7 → weak_induction."""
    assert classify_ddi(0.7) == "weak_induction"


@pytest.mark.integration
def test_classify_ddi_boundary_1_25():
    """AUCR = 1.25 (lower boundary of weak) → weak."""
    assert classify_ddi(1.25) == "weak"


@pytest.mark.integration
def test_classify_ddi_boundary_5():
    """AUCR = 5.0 (lower boundary of strong) → strong."""
    assert classify_ddi(5.0) == "strong"


# ===========================================================================
# simulate_ddi — integration tests
# ===========================================================================


@pytest.mark.integration
def test_no_inhibition_aucr_near_one():
    """Perpetrator with huge Ki (1e6 µM) should produce AUCR ≈ 1.0 (no meaningful inhibition)."""
    perp = PerpetratorSpec(name="non_inhibitor", ki_uM=1_000_000.0, cmax_uM=1.0)
    result = simulate_ddi(TEST_DRUG, perp, dose_mg=1.0, t_end_h=12.0)
    assert isinstance(result, DDISimulationResult)
    assert abs(result.auc_ratio - 1.0) < 0.1, f"AUCR={result.auc_ratio:.4f} expected ≈ 1.0"


@pytest.mark.integration
def test_competitive_inhibition_increases_auc():
    """Potent inhibitor (Ki=0.01 µM, Cmax=1 µM) should yield AUCR > 1.5."""
    perp = PerpetratorSpec(name="potent_inhibitor", ki_uM=0.01, cmax_uM=1.0)
    result = simulate_ddi(TEST_DRUG, perp, dose_mg=1.0, t_end_h=12.0)
    assert result.auc_ratio > 1.5, f"AUCR={result.auc_ratio:.4f} expected > 1.5"


@pytest.mark.integration
def test_result_has_profiles():
    """time_h, cp_alone_mg_L, cp_ddi_mg_L must be non-empty tuples of equal length."""
    perp = PerpetratorSpec(name="test_perp", ki_uM=1.0, cmax_uM=1.0)
    result = simulate_ddi(TEST_DRUG, perp, dose_mg=1.0, t_end_h=12.0)
    assert len(result.time_h) > 0
    assert len(result.cp_alone_mg_L) == len(result.time_h)
    assert len(result.cp_ddi_mg_L) == len(result.time_h)


@pytest.mark.integration
def test_static_r1_positive():
    """Static R1 = 1 + Cmax/Ki must be > 1 for any positive Cmax and Ki."""
    perp = PerpetratorSpec(name="r1_test", ki_uM=1.0, cmax_uM=2.0)
    result = simulate_ddi(TEST_DRUG, perp, dose_mg=1.0, t_end_h=12.0)
    assert result.static_r1 > 1.0, f"static_r1={result.static_r1:.4f} expected > 1.0"


# ===========================================================================
# POST /ddi/simulate — API endpoint tests
# ===========================================================================


def _ddi_simulate_payload(**overrides) -> dict:
    payload = {
        "victim_drug": TEST_DRUG_PAYLOAD,
        "perpetrator_name": "ketoconazole",
        "perpetrator_ki_uM": 0.018,
        "perpetrator_cmax_uM": 1.0,
        "dose_mg": 1.0,
        "route": "oral",
        "duration_h": 12.0,
    }
    payload.update(overrides)
    return payload


@pytest.mark.integration
def test_api_ddi_simulate_endpoint_200():
    """POST /ddi/simulate returns 200 and has expected top-level keys."""
    r = client.post("/ddi/simulate", json=_ddi_simulate_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    expected_keys = {
        "victim_name",
        "perpetrator_name",
        "auc_ratio",
        "cmax_ratio",
        "classification",
        "static_r1",
        "time_h",
        "cp_alone_mg_L",
        "cp_ddi_mg_L",
    }
    assert expected_keys.issubset(body.keys()), f"Missing keys: {expected_keys - body.keys()}"


@pytest.mark.integration
def test_api_ddi_simulate_returns_profiles():
    """Response time_h, cp_alone_mg_L, cp_ddi_mg_L must be non-empty lists of equal length."""
    r = client.post("/ddi/simulate", json=_ddi_simulate_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    time_h = body["time_h"]
    assert len(time_h) > 0
    assert len(body["cp_alone_mg_L"]) == len(time_h)
    assert len(body["cp_ddi_mg_L"]) == len(time_h)
