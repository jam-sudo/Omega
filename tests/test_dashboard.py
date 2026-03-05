"""Phase 45 — Dashboard frontend tests."""
from __future__ import annotations
import pathlib
import pytest
from fastapi.testclient import TestClient
from omega_pbpk.api.app import app

client = TestClient(app)
_STATIC_DIR = pathlib.Path(__file__).parent.parent / "src" / "omega_pbpk" / "static"


@pytest.mark.unit
def test_dashboard_root_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


@pytest.mark.unit
def test_dashboard_html_contains_tabs():
    r = client.get("/")
    body = r.text
    for label in ["Compound Explorer", "PK Simulator", "Dose Optimizer", "DDI Checker"]:
        assert label in body


@pytest.mark.unit
def test_static_app_js_served():
    if not (_STATIC_DIR / "app.js").exists():
        pytest.skip("Static files not present")
    r = client.get("/static/app.js")
    assert r.status_code == 200


@pytest.mark.unit
def test_static_style_css_served():
    if not (_STATIC_DIR / "style.css").exists():
        pytest.skip("Static files not present")
    r = client.get("/static/style.css")
    assert r.status_code == 200


@pytest.mark.unit
def test_simulate_endpoint_dashboard_payload():
    payload = {
        "drug": {
            "name": "test", "mw": 300.0, "logP": 2.0, "fup": 0.1,
            "rbp": 1.0, "clint_hepatic_L_per_h": 5.0, "clr_L_per_h": 0.0,
            "peff": 1.0, "vdss_L_per_kg": 0.6, "drug_type": "neutral",
        },
        "dose_mg": 100.0, "route": "oral", "duration_h": 24.0, "body_weight": 70.0,
    }
    r = client.post("/simulate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["cmax_mg_L"] > 0
    assert isinstance(body["time_h"], list)


@pytest.mark.unit
def test_ddi_simulate_endpoint_dashboard_payload():
    payload = {
        "victim_drug": {
            "name": "victim", "mw": 300.0, "logP": 2.0, "fup": 0.1,
            "rbp": 1.0, "clint_hepatic_L_per_h": 5.0, "clr_L_per_h": 0.0,
            "peff": 1.0, "vdss_L_per_kg": 0.6, "drug_type": "neutral",
        },
        "perpetrator_name": "ketoconazole",
        "perpetrator_ki_uM": 0.1,
        "perpetrator_target_enzyme": "CYP3A4",
        "perpetrator_mechanism": "competitive",
        "perpetrator_cmax_uM": 5.0,
        "perpetrator_kinact_per_h": 0.0,
        "perpetrator_kdeg_per_h": 0.02,
        "perpetrator_fold_induction": 1.0,
        "dose_mg": 100.0, "route": "oral", "body_weight": 70.0, "duration_h": 24.0,
    }
    r = client.post("/ddi/simulate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "auc_ratio" in body
    assert isinstance(body["time_h"], list)
