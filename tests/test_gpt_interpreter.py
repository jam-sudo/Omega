"""Unit tests for omega_pbpk.interpretation.gpt_interpreter (Phase 41)."""

from __future__ import annotations

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
    _call_openai,
    _result_to_dict,
    _rule_based_interpret,
    interpret_pk_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RESULT_DICT = {
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


def _make_mock_client(json_payload: dict) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(json_payload)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _make_openai_module(mock_client: MagicMock) -> MagicMock:
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai


# ---------------------------------------------------------------------------
# GPTInterpreterConfig tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_defaults():
    cfg = GPTInterpreterConfig()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.max_tokens == 500
    assert cfg.temperature == 0.3
    assert cfg.oauth_token == ""


@pytest.mark.unit
def test_config_resolve_token_explicit():
    cfg = GPTInterpreterConfig(oauth_token="sk-abc123")
    assert cfg.resolve_token() == "sk-abc123"


@pytest.mark.unit
def test_config_resolve_token_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN", "sk-from-env")
    cfg = GPTInterpreterConfig()
    assert cfg.resolve_token() == "sk-from-env"


# ---------------------------------------------------------------------------
# _result_to_dict tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_result_to_dict_passthrough():
    d = {"key": "value", "num": 42}
    assert _result_to_dict(d) is d


@pytest.mark.unit
def test_result_to_dict_from_dataclass():
    @dataclass(frozen=True)
    class FakeResult:
        drug_name: str
        cmax: float
        flags: tuple

    r = FakeResult(drug_name="Aspirin", cmax=1.5, flags=("a", "b"))
    d = _result_to_dict(r)
    assert d["drug_name"] == "Aspirin"
    assert d["cmax"] == 1.5
    assert d["flags"] == ["a", "b"]  # tuples converted to list


# ---------------------------------------------------------------------------
# _build_user_prompt tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_user_prompt_includes_known_keys():
    data = {"drug_name": "Caffeine", "cmax_mg_L": 3.0, "extra_field": "ignored"}
    prompt = _build_user_prompt(data)
    parsed = json.loads(prompt)
    assert parsed["drug_name"] == "Caffeine"
    assert parsed["cmax_mg_L"] == 3.0
    assert "extra_field" not in parsed


@pytest.mark.unit
def test_build_user_prompt_missing_keys_skipped():
    data = {"drug_name": "X"}
    prompt = _build_user_prompt(data)
    parsed = json.loads(prompt)
    assert list(parsed.keys()) == ["drug_name"]


# ---------------------------------------------------------------------------
# _call_openai tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_call_openai_no_token_raises():
    cfg = GPTInterpreterConfig(oauth_token="")
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_OAUTH_TOKEN"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="No OpenAI OAuth token"):
            _call_openai("prompt", cfg)


@pytest.mark.unit
def test_call_openai_strips_markdown_fences():
    payload = {"summary": "OK", "key_findings": [], "safety_flags": [], "recommendations": []}
    fenced = f"```json\n{json.dumps(payload)}\n```"
    mock_client = _make_mock_client({})
    mock_client.chat.completions.create.return_value.choices[0].message.content = fenced
    mock_openai = _make_openai_module(mock_client)
    cfg = GPTInterpreterConfig(oauth_token="sk-test")
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = _call_openai("prompt", cfg)
    assert result["summary"] == "OK"


@pytest.mark.unit
def test_call_openai_raises_on_non_json():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "not-json"
    mock_openai = _make_openai_module(mock_client)
    cfg = GPTInterpreterConfig(oauth_token="sk-test")
    with patch.dict(sys.modules, {"openai": mock_openai}):
        with pytest.raises(RuntimeError, match="non-JSON"):
            _call_openai("prompt", cfg)


# ---------------------------------------------------------------------------
# _rule_based_interpret tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rule_based_interpret_high_risk():
    d = {
        "drug_name": "DangerDrug",
        "cmax_mg_L": 10.0,
        "auc0t_mg_h_L": 100.0,
        "t_half_h": 30.0,
        "overall_risk_level": "high",
        "risk_flags": {"hepatotoxicity": True},
        "transporter_ddi_flags": ["CYP3A4"],
        "confidence": "medium",
    }
    result = _rule_based_interpret(d)
    assert result.model_used == "rule-based"
    assert "high" in result.summary
    assert any("Hepatotoxicity" in f for f in result.safety_flags)
    assert any("CYP3A4" in f for f in result.safety_flags)
    assert any("dose reduction" in r.lower() for r in result.recommendations)


@pytest.mark.unit
def test_rule_based_interpret_low_risk():
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


# ---------------------------------------------------------------------------
# interpret_pk_result integration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_interpret_pk_result_uses_gpt_when_token_available():
    payload = {
        "summary": "Good PK profile.",
        "key_findings": ["Moderate half-life"],
        "safety_flags": [],
        "recommendations": ["Proceed to Phase I"],
    }
    mock_client = _make_mock_client(payload)
    mock_openai = _make_openai_module(mock_client)
    cfg = GPTInterpreterConfig(oauth_token="sk-test", model="gpt-4o-mini")
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = interpret_pk_result(_SAMPLE_RESULT_DICT, cfg)
    assert result.model_used == "gpt-4o-mini"
    assert result.summary == "Good PK profile."
    assert "Moderate half-life" in result.key_findings
    assert "Proceed to Phase I" in result.recommendations


@pytest.mark.unit
def test_interpret_pk_result_fallback_no_token():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_OAUTH_TOKEN"}
    with patch.dict(os.environ, env, clear=True):
        result = interpret_pk_result(_SAMPLE_RESULT_DICT)
    assert result.model_used == "rule-based"
    assert isinstance(result, InterpretationResult)
