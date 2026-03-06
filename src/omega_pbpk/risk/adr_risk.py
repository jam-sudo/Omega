"""Adverse drug reaction (ADR) risk scoring — physicochemical + exposure-based."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ADRRiskResult", "score_adr_risk", "batch_adr_screen"]

ADR_CATEGORIES = [
    "hepatotoxicity",
    "nephrotoxicity",
    "cardiotoxicity",
    "neurotoxicity",
    "gi_toxicity",
]


@dataclass(frozen=True)
class ADRRiskResult:
    drug_name: str
    composite_risk_score: float  # 0-100 normalized
    risk_tier: str  # "low" (<25), "moderate" (25-50), "high" (50-75), "very_high" (>75)
    category_scores: dict  # {category: score 0-10}
    risk_flags: list[str]  # specific flags
    safety_margin: float  # Cmax / predicted_tox_threshold
    bsa_mg_per_m2: float  # dose normalized by body surface area


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _hepatotoxicity_score(logP: float, mw: float, reactive_metabolite: bool = False) -> float:
    lipophilicity_risk = _clip(logP * 1.5, 0, 4)
    mw_risk = 1.0 if mw > 300 else 0.0
    rm_risk = 4.0 if reactive_metabolite else 0.0
    return _clip(lipophilicity_risk + mw_risk + rm_risk, 0, 10)


def _nephrotoxicity_score(fup: float, fraction_renal: float, gfr_mL_min: float = 120.0) -> float:
    renal_load = fraction_renal * 5.0
    low_gfr_risk = _clip((120 - gfr_mL_min) / 12, 0, 3)
    free_drug_risk = _clip((1 - fup) * 2, 0, 2)
    return _clip(renal_load + low_gfr_risk + free_drug_risk, 0, 10)


def _cardiotoxicity_score(delta_qtc_ms: float = 0.0, is_pgp_inhibitor: bool = False) -> float:
    qtc_risk = _clip(delta_qtc_ms / 10.0, 0, 6)
    pgp_risk = 2.0 if is_pgp_inhibitor else 0.0
    return _clip(qtc_risk + pgp_risk, 0, 10)


def _neurotoxicity_score(kp_uu_brain: float = 0.0, cns_active: bool = False) -> float:
    bbb_risk = _clip(kp_uu_brain * 8, 0, 6)
    cns_risk = 3.0 if cns_active else 0.0
    return _clip(bbb_risk + cns_risk, 0, 10)


def _gi_toxicity_score(logP: float, sol_mg_mL: float = 1.0) -> float:
    insol_risk = _clip(-math.log10(max(sol_mg_mL, 1e-4)), 0, 4)
    lipophilicity_risk = _clip(max(logP - 2, 0) * 1.5, 0, 4)
    return _clip(insol_risk + lipophilicity_risk, 0, 10)


def score_adr_risk(
    drug_name: str,
    logP: float,
    MW: float,
    fup: float,
    cmax_mg_L: float,
    dose_mg: float,
    fraction_renal: float = 0.5,
    kp_uu_brain: float = 0.1,
    delta_qtc_ms: float = 0.0,
    sol_mg_mL: float = 1.0,
    reactive_metabolite: bool = False,
    is_pgp_inhibitor: bool = False,
    cns_active: bool = False,
    gfr_mL_min: float = 120.0,
    body_surface_area_m2: float = 1.73,
) -> ADRRiskResult:
    h = _hepatotoxicity_score(logP, MW, reactive_metabolite)
    n = _nephrotoxicity_score(fup, fraction_renal, gfr_mL_min)
    c = _cardiotoxicity_score(delta_qtc_ms, is_pgp_inhibitor)
    neu = _neurotoxicity_score(kp_uu_brain, cns_active)
    gi = _gi_toxicity_score(logP, sol_mg_mL)

    category_scores = {
        "hepatotoxicity": h,
        "nephrotoxicity": n,
        "cardiotoxicity": c,
        "neurotoxicity": neu,
        "gi_toxicity": gi,
    }

    composite = (h + n + c + neu + gi) / 5.0 * 10.0

    if composite < 25:
        risk_tier = "low"
    elif composite < 50:
        risk_tier = "moderate"
    elif composite < 75:
        risk_tier = "high"
    else:
        risk_tier = "very_high"

    safety_margin = 100.0 / max(composite, 0.1)
    bsa_mg_per_m2 = dose_mg / body_surface_area_m2

    risk_flags: list[str] = []
    if reactive_metabolite:
        risk_flags.append("Reactive metabolite risk")
    if h > 6:
        risk_flags.append("High hepatotoxicity potential")
    if n > 6:
        risk_flags.append("Renal accumulation risk")
    if delta_qtc_ms > 20:
        risk_flags.append("QT prolongation risk")
    if kp_uu_brain > 0.3:
        risk_flags.append("CNS penetration")
    if sol_mg_mL < 0.1:
        risk_flags.append("Low GI solubility")

    return ADRRiskResult(
        drug_name=drug_name,
        composite_risk_score=composite,
        risk_tier=risk_tier,
        category_scores=category_scores,
        risk_flags=risk_flags,
        safety_margin=safety_margin,
        bsa_mg_per_m2=bsa_mg_per_m2,
    )


def batch_adr_screen(compounds: list[dict]) -> list[ADRRiskResult]:
    results = [score_adr_risk(**compound) for compound in compounds]
    results.sort(key=lambda r: r.composite_risk_score, reverse=True)
    return results
