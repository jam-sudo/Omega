"""Tests for M3-2 FastAPI integration — new endpoints and validators."""
from __future__ import annotations

import pytest

try:
    from omega_pbpk.api.app import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")

if HAS_FASTAPI:
    from fastapi.testclient import TestClient
    client = TestClient(app)
else:
    client = None  # type: ignore[assignment]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_version_header():
    r = client.get("/health")
    assert "X-Omega-Version" in r.headers


def test_simulate_invalid_dose():
    r = client.post(
        "/simulate",
        json={"compound_path": "compounds/caffeine.yaml", "dose_mg": -1, "route": "oral"},
    )
    assert r.status_code == 422


def test_train_surrogate_endpoint():
    r = client.post("/train/surrogate", json={"n_samples": 20, "epochs": 5})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"


def test_validate_endpoint():
    r = client.post("/validate", json={"mode": "sanity"})
    assert r.status_code == 200, r.text
    assert r.json()["passed"] is True
