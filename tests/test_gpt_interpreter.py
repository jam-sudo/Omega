"""Unit tests for omega_pbpk.interpretation.gpt_interpreter (Phase 41).

14 tests — all @pytest.mark.unit, no real API calls.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from omega_pbpk.interpretation.gpt_interpreter import (
    GPTInterpreterConfig,
    InterpretationResult,
    _build_user_prompt,
    _result_to_dict,
    _rule_based_interpret,
    interpret_pk_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE = {
    "drug_name": "TestDrug",
    "cmax_mg_L": 2.5,
    "tmax_h": 1.5,
    "auc0t_mg_h_L": 20.0,
    "t_half_h": 6.0,
    "overall_risk_level": "low",
    "risk_flags": {},
    "confidence": "medium",
    "transporter_ddi_flags": [],
}


def _mock_openai_module(json_payload: dict) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(json_payload)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_defaults():
    cfg = GPTInterpreterConfig()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.max_tokens == 500
    assert cfg.temperature == 0.3
    assert cfg.oauth_token == ""


@pytest.mark.unit
def test_resolve_token_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN", "sk-env-token")
    cfg = GPTInterpreterConfig()
    assert cfg.resolve_token() == "sk-env-token"


@pytest.mark.unit
def test_resolve_token_prefers_config(monkeypatch):
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN", "sk-env-token")
    cfg = GPTInterpreterConfig(oauth_token="sk-explicit")
    assert cfg.resolve_token() == "sk-explicit"


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_interpretation_result_frozen():
    result = InterpretationResult(
        summary="ok",
        key_findings=(),
        safety_flags=(),
        recommendations=(),
        confidence="high",
        model_used="rule-based",
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        result.summary = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestPromptBuilding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_user_prompt_contains_metrics():
    data = {
        "drug_name": "Caffeine",
        "cmax_mg_L": 3.0,
        "auc0t_mg_h_L": 15.0,
        "t_half_h": 5.0,
        "unrelated": "ignored",
    }
    prompt = _build_user_prompt(data)
    parsed = json.loads(prompt)
    assert parsed["drug_name"] == "Caffeine"
    assert parsed["cmax_mg_L"] == 3.0
    assert parsed["auc0t_mg_h_L"] == 15.0
    assert "unrelated" not in parsed


@pytest.mark.unit
def test_result_to_dict_passthrough():
    d = {"drug_name": "X", "cmax_mg_L": 1.0}
    assert _result_to_dict(d) is d


@pytest.mark.unit
def test_result_to_dict_dataclass():
    @dataclass(frozen=True)
    class FakePK:
        drug_name: str
        flags: tuple

    r = FakePK(drug_name="Aspirin", flags=("a", "b"))
    d = _result_to_dict(r)
    assert d["drug_name"] == "Aspirin"
    assert d["flags"] == ["a", "b"]  # tuple → list


# ---------------------------------------------------------------------------
# TestRuleBased
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rule_based_low_risk():
    d = {
        "drug_name": "SafeDrug",
        "cmax_mg_L": 1.0,
        "auc0t_mg_h_L": 10.0,
        "t_half_h": 6.0,
        "overall_risk_level": "low",
        "risk_flags": {},
        "transporter_ddi_flags": [],
        "confidence": "high",
    }
    result = _rule_based_interpret(d)
    assert result.model_used == "rule-based"
    assert not result.safety_flags
    assert any("Standard monitoring" in r for r in result.recommendations)


@pytest.mark.unit
def test_rule_based_high_risk():
    d = {
        "drug_name": "DangerDrug",
        "cmax_mg_L": 10.0,
        "auc0t_mg_h_L": 100.0,
        "t_half_h": 30.0,
        "overall_risk_level": "high",
        "risk_flags": {"hepatotoxicity": True},
        "transporter_ddi_flags": [],
        "confidence": "medium",
    }
    result = _rule_based_interpret(d)
    assert "high" in result.summary
    assert any("Hepatotoxicity" in f for f in result.safety_flags)
    assert any("dose reduction" in r.lower() for r in result.recommendations)


@pytest.mark.unit
def test_rule_based_zero_exposure():
    d = {
        "drug_name": "ZeroDrug",
        "cmax_mg_L": 0.0,
        "auc0t_mg_h_L": 0.0,
        "t_half_h": 0.0,
        "overall_risk_level": "unknown",
        "risk_flags": {},
        "transporter_ddi_flags": [],
        "confidence": "low",
    }
    result = _rule_based_interpret(d)
    assert result.model_used == "rule-based"
    assert "0.00" in result.summary
    assert not result.key_findings  # no t_half/auc findings for zero values


@pytest.mark.unit
def test_rule_based_ddi_flags():
    d = {
        "drug_name": "DDIDrug",
        "cmax_mg_L": 2.0,
        "auc0t_mg_h_L": 20.0,
        "t_half_h": 8.0,
        "overall_risk_level": "moderate",
        "risk_flags": {},
        "transporter_ddi_flags": ["CYP3A4", "P-gp"],
        "confidence": "medium",
    }
    result = _rule_based_interpret(d)
    ddi_safety = [f for f in result.safety_flags if "DDI:" in f]
    assert len(ddi_safety) == 2
    assert any("CYP3A4" in f for f in ddi_safety)
    assert any("P-gp" in f for f in ddi_safety)


# ---------------------------------------------------------------------------
# TestGPTPath
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_interpret_pk_result_gpt_success():
    payload = {
        "summary": "Good PK profile.",
        "key_findings": ["Moderate half-life"],
        "safety_flags": [],
        "recommendations": ["Proceed to Phase I"],
    }
    mock_openai = _mock_openai_module(payload)
    cfg = GPTInterpreterConfig(oauth_token="sk-test", model="gpt-4o-mini")
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = interpret_pk_result(_SAMPLE, cfg)
    assert result.model_used == "gpt-4o-mini"
    assert result.summary == "Good PK profile."
    assert "Moderate half-life" in result.key_findings
    assert "Proceed to Phase I" in result.recommendations


@pytest.mark.unit
def test_interpret_pk_result_gpt_fallback():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_OAUTH_TOKEN"}
    with patch.dict(os.environ, env, clear=True):
        result = interpret_pk_result(_SAMPLE)
    assert result.model_used == "rule-based"
    assert isinstance(result, InterpretationResult)


# ---------------------------------------------------------------------------
# TestAPI
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_interpret_api_endpoint():
    from fastapi.testclient import TestClient

    from omega_pbpk.api.app import app

    client = TestClient(app)
    payload = {
        "drug_name": "TestDrug",
        "cmax_mg_L": 2.5,
        "auc0t_mg_h_L": 20.0,
        "t_half_h": 6.0,
        "overall_risk_level": "low",
        "confidence": "medium",
    }
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_OAUTH_TOKEN"}
    with patch.dict(os.environ, env, clear=True):
        resp = client.post("/interpret", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_used"] == "rule-based"
    assert "summary" in data
    assert isinstance(data["key_findings"], list)
