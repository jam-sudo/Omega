"""Tests for the Omega PBPK FastAPI server.

Tests exercise all major endpoints using FastAPI's TestClient (synchronous).
Run with: python3 -m pytest tests/test_api.py -v --tb=short
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Reference Drug payload (caffeine-like parameters)
# ---------------------------------------------------------------------------

CAFFEINE_DRUG = {
    "name": "caffeine",
    "mw": 194.19,
    "logP": -0.07,
    "pka": [14.0],
    "drug_type": "neutral",
    "fup": 0.65,
    "rbp": 1.0,
    "clint_hepatic_L_per_h": 2.3,
    "clr_L_per_h": 0.0,
    "peff": 2.0,
    "vdss_L_per_kg": 0.6,
    "route_of_elimination": "hepatic",
}

CAFFEINE_SIMULATE_PAYLOAD = {
    "drug": CAFFEINE_DRUG,
    "dose_mg": 200.0,
    "route": "oral",
    "duration_h": 12.0,
    "body_weight": 70.0,
}

CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

def test_health_ok():
    """GET /health should return 200 with status='ok' and version='0.9.0'."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.9.0"


# ---------------------------------------------------------------------------
# 2. Root redirect
# ---------------------------------------------------------------------------

def test_root_redirects_to_docs():
    """GET / should redirect to /docs."""
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert "/docs" in r.headers.get("location", "")


# ---------------------------------------------------------------------------
# 3. Docs available
# ---------------------------------------------------------------------------

def test_docs_available():
    """GET /docs should return 200 (Swagger UI)."""
    r = client.get("/docs")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 4. Simulate oral (caffeine)
# ---------------------------------------------------------------------------

def test_simulate_oral_caffeine():
    """POST /simulate with caffeine oral dose should return positive Cmax."""
    r = client.post("/simulate", json=CAFFEINE_SIMULATE_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cmax_mg_L"] > 0, f"Expected Cmax > 0, got {body['cmax_mg_L']}"
    assert body["auc0t_mg_h_L"] > 0
    assert isinstance(body["time_h"], list)
    assert isinstance(body["cp_mg_L"], list)
    assert len(body["time_h"]) > 1
    assert len(body["cp_mg_L"]) == len(body["time_h"])


# ---------------------------------------------------------------------------
# 5. Simulate IV
# ---------------------------------------------------------------------------

def test_simulate_iv():
    """POST /simulate with IV route should return positive Cmax."""
    payload = {**CAFFEINE_SIMULATE_PAYLOAD, "route": "iv"}
    r = client.post("/simulate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cmax_mg_L"] > 0
    assert body["auc0t_mg_h_L"] > 0


# ---------------------------------------------------------------------------
# 6. Simulate invalid route returns 422
# ---------------------------------------------------------------------------

def test_simulate_invalid_route_returns_422():
    """POST /simulate with bad route should return 422 Unprocessable Entity."""
    payload = {**CAFFEINE_SIMULATE_PAYLOAD, "route": "suppository"}
    r = client.post("/simulate", json=payload)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# ---------------------------------------------------------------------------
# 7. Predict from SMILES
# ---------------------------------------------------------------------------

def test_predict_smiles():
    """POST /predict with caffeine SMILES should return a valid PK response."""
    r = client.post("/predict", json={
        "smiles": CAFFEINE_SMILES,
        "dose_mg": 200.0,
        "route": "oral",
        "duration_h": 12.0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cmax_mg_L"] > 0
    assert body["auc0t_mg_h_L"] > 0
    assert "confidence" in body
    assert isinstance(body["warnings"], list)
    assert isinstance(body["time_h"], list)
    assert isinstance(body["cp_mg_L"], list)


# ---------------------------------------------------------------------------
# 8. NCA endpoint
# ---------------------------------------------------------------------------

def test_nca_endpoint():
    """POST /nca with a simple biexponential profile should return NCA params."""
    import numpy as np
    t = list(np.linspace(0, 12, 50))
    # Simple oral-like profile
    ka, ke, vd, dose = 2.0, 0.3, 50.0, 200.0
    c = [(dose / vd) * (ka / (ka - ke)) * (np.exp(-ke * ti) - np.exp(-ka * ti)) for ti in t]

    r = client.post("/nca", json={
        "time_h": t,
        "conc_mg_L": c,
        "dose_mg": dose,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cmax_mg_L" in body
    assert "auc0t_mg_h_L" in body
    assert body["cmax_mg_L"] > 0


# ---------------------------------------------------------------------------
# 9. NCA returns Cmax and AUC
# ---------------------------------------------------------------------------

def test_nca_returns_cmax_and_auc():
    """POST /nca should include all expected NCA output keys."""
    import numpy as np
    t = list(np.linspace(0, 24, 100))
    c = [2.0 * np.exp(-0.15 * ti) for ti in t]

    r = client.post("/nca", json={
        "time_h": t,
        "conc_mg_L": c,
        "dose_mg": 100.0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    expected_keys = {
        "cmax_mg_L", "tmax_h", "auc0t_mg_h_L", "auc0inf_mg_h_L",
        "t_half_h", "kel_per_h", "vz_L", "cl_L_per_h", "mrt_h", "r_squared",
    }
    for key in expected_keys:
        assert key in body, f"Missing key '{key}' in NCA response"


# ---------------------------------------------------------------------------
# 10. DDI endpoint
# ---------------------------------------------------------------------------

def test_ddi_endpoint():
    """POST /ddi with an inhibitor should return DDI risk report."""
    r = client.post("/ddi", json={
        "inhibitors": [
            {
                "name": "ketoconazole",
                "cmax_uM": 3.7,
                "ki_3a4_uM": 0.0037,
                "dose_mg": 200.0,
                "mw": 531.43,
                "fa": 0.85,
                "fg": 0.6,
                "ka_per_h": 1.0,
                "victim_fm_3a4": 0.5,
            }
        ]
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1
    report = body[0]
    assert "R1_3a4" in report
    assert report["R1_3a4"] > 1.0  # ketoconazole is a strong CYP3A4 inhibitor
    assert "flags" in report
    assert "recommendation" in report


# ---------------------------------------------------------------------------
# 11. Pipeline health check
# ---------------------------------------------------------------------------

def test_pipeline_health():
    """GET /pipeline/health should confirm the caffeine PBPK simulation succeeds."""
    r = client.get("/pipeline/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["pipeline"] == "operational"
    assert body["cmax_mg_L"] > 0


# ---------------------------------------------------------------------------
# 12. Population endpoint
# ---------------------------------------------------------------------------

def test_population_endpoint():
    """POST /population should simulate N subjects and return summary stats."""
    r = client.post("/population", json={
        "drug": CAFFEINE_DRUG,
        "dose_mg": 200.0,
        "route": "oral",
        "n_subjects": 10,
        "duration_h": 12.0,
        "seed": 42,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "n_subjects" in body
    assert body["n_subjects"] > 0
    assert "cmax_median_mg_L" in body
    assert body["cmax_median_mg_L"] > 0
    assert "auc_median_mg_h_L" in body
