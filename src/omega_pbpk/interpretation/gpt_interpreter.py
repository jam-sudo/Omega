"""OpenAI GPT interpretation layer for PBPK simulation results.

Converts structured PK assessment data into natural-language clinical
summaries with key findings, safety flags, and dosing recommendations.
Falls back to rule-based interpretation when OpenAI is unavailable.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------

_KEYS_FOR_PROMPT = (
    "drug_name",
    "cmax_mg_L",
    "tmax_h",
    "auc0t_mg_h_L",
    "t_half_h",
    "cmax_p5",
    "cmax_p95",
    "auc_p5",
    "auc_p95",
    "overall_risk_level",
    "risk_flags",
    "risk_count",
    "transporter_ddi_flags",
    "adme",
    "confidence",
    "warnings",
)

_SYSTEM_PROMPT = (
    "You are a clinical pharmacology expert. "
    "Analyze the pharmacokinetic assessment data and respond ONLY with valid JSON "
    "containing exactly these keys: "
    '"summary" (string), '
    '"key_findings" (list of strings), '
    '"safety_flags" (list of strings), '
    '"recommendations" (list of strings). '
    "Do not include any text outside the JSON object."
)


@dataclass(frozen=True)
class GPTInterpreterConfig:
    """Configuration for the GPT interpretation layer."""

    model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.3
    oauth_token: str = ""

    def resolve_token(self) -> str:
        """Return explicit token or fall back to env variable."""
        return self.oauth_token or os.environ.get("OPENAI_OAUTH_TOKEN", "")


@dataclass(frozen=True)
class InterpretationResult:
    """Natural-language interpretation of a PK assessment."""

    summary: str
    key_findings: tuple[str, ...]
    safety_flags: tuple[str, ...]
    recommendations: tuple[str, ...]
    confidence: str
    model_used: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert result to plain dict; handles dicts and frozen dataclasses."""
    if isinstance(result, dict):
        return result
    # frozen dataclass → dict
    d: dict[str, Any] = {}
    for f in dataclasses.fields(result):  # type: ignore[arg-type]
        val = getattr(result, f.name)
        if isinstance(val, tuple):
            val = list(val)
        d[f.name] = val
    return d


def _build_user_prompt(result_dict: dict[str, Any]) -> str:
    """Extract relevant PK fields and return as JSON string."""
    subset: dict[str, Any] = {}
    for key in _KEYS_FOR_PROMPT:
        if key in result_dict:
            val = result_dict[key]
            # Convert tuples to lists for JSON serialisation
            if isinstance(val, tuple):
                val = list(val)
            subset[key] = val
    return json.dumps(subset, default=str)


def _call_openai(user_prompt: str, config: GPTInterpreterConfig) -> dict[str, Any]:
    """Call the OpenAI chat completions API and parse the JSON response."""
    token = config.resolve_token()
    if not token:
        raise RuntimeError("No OpenAI OAuth token")

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("openai not installed") from exc

    client = openai.OpenAI(api_key=token)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    content: str = response.choices[0].message.content or ""
    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # Remove first and last fence lines
        content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GPT returned non-JSON response: {content[:200]}") from exc


def _rule_based_interpret(result_dict: dict[str, Any]) -> InterpretationResult:
    """Generate a deterministic interpretation without any external API call."""
    drug = result_dict.get("drug_name", "compound")
    cmax = result_dict.get("cmax_mg_L", 0.0)
    auc = result_dict.get("auc0t_mg_h_L", 0.0)
    t_half = result_dict.get("t_half_h", 0.0)
    risk_level = result_dict.get("overall_risk_level", "unknown")
    confidence = str(result_dict.get("confidence", "low"))

    # --- summary ---
    summary = (
        f"{drug} shows {risk_level} overall risk. "
        f"Cmax {cmax:.2f} mg/L, AUC {auc:.1f} mg·h/L, t½ {t_half:.1f} h."
    )

    # --- key findings ---
    findings: list[str] = []
    if t_half > 0:
        if t_half < 4:
            findings.append(f"Short half-life ({t_half:.1f} h) may require frequent dosing.")
        elif t_half > 24:
            findings.append(
                f"Long half-life ({t_half:.1f} h); accumulation risk with repeat dosing."
            )
        else:
            findings.append(f"Moderate half-life ({t_half:.1f} h) suitable for once/twice daily.")
    if auc > 0:
        findings.append(f"Total exposure AUC {auc:.1f} mg·h/L.")
    cmax_p5 = result_dict.get("cmax_p5", 0.0)
    cmax_p95 = result_dict.get("cmax_p95", 0.0)
    if cmax_p95 > 0 and cmax_p5 >= 0:
        fold = cmax_p95 / cmax_p5 if cmax_p5 > 1e-9 else float("inf")
        if fold > 3:
            findings.append(f"High Cmax variability (p5={cmax_p5:.2f}, p95={cmax_p95:.2f} mg/L).")

    # --- safety flags ---
    safety: list[str] = []
    risk_flags: dict[str, Any] = result_dict.get("risk_flags", {})
    for flag, active in risk_flags.items():
        if active:
            safety.append(flag.replace("_", " ").capitalize())
    ddi_flags = result_dict.get("transporter_ddi_flags", [])
    if isinstance(ddi_flags, (list, tuple)):
        for f in ddi_flags:
            safety.append(f"DDI: {f}")

    # --- recommendations ---
    recommendations: list[str] = []
    if risk_level == "high":
        recommendations.append("Consider dose reduction or enhanced monitoring.")
        recommendations.append("Clinical DDI studies strongly recommended.")
    elif risk_level == "moderate":
        recommendations.append("Monitor plasma levels during initial dosing.")
    else:
        recommendations.append("Standard monitoring is appropriate.")
    if confidence == "low":
        recommendations.append("Confirm predictions with in vivo data; low model confidence.")

    return InterpretationResult(
        summary=summary,
        key_findings=tuple(findings),
        safety_flags=tuple(safety),
        recommendations=tuple(recommendations),
        confidence=confidence,
        model_used="rule-based",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interpret_pk_result(
    result: Any,
    config: GPTInterpreterConfig | None = None,
) -> InterpretationResult:
    """Interpret a PK assessment result with GPT or rule-based fallback.

    Args:
        result: FullPredictionResult dataclass or equivalent dict.
        config: GPTInterpreterConfig; defaults to GPTInterpreterConfig().

    Returns:
        InterpretationResult with natural-language summary and flags.
    """
    if config is None:
        config = GPTInterpreterConfig()

    result_dict = _result_to_dict(result)
    user_prompt = _build_user_prompt(result_dict)

    try:
        raw = _call_openai(user_prompt, config)
        confidence = str(result_dict.get("confidence", "low"))
        return InterpretationResult(
            summary=str(raw.get("summary", "")),
            key_findings=tuple(raw.get("key_findings", [])),
            safety_flags=tuple(raw.get("safety_flags", [])),
            recommendations=tuple(raw.get("recommendations", [])),
            confidence=confidence,
            model_used=config.model,
        )
    except RuntimeError as exc:
        logger.warning("GPT interpretation unavailable (%s); using rule-based fallback.", exc)
        return _rule_based_interpret(result_dict)


__all__ = [
    "GPTInterpreterConfig",
    "InterpretationResult",
    "interpret_pk_result",
]
