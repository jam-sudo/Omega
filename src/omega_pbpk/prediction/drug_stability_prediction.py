"""
Phase 972 — Drug chemical stability prediction under stress conditions.

Predicts hydrolysis, oxidation, photodegradation rates using Arrhenius
kinetics and structural feature-based rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DrugStabilityResult", "predict_drug_stability", "screen_stability"]

_R_kJ = 8.314e-3  # kJ / (mol * K)
_Ea_HYD_kJ = 80.0  # kJ/mol typical hydrolysis activation energy


@dataclass(frozen=True)
class DrugStabilityResult:
    drug_name: str
    logp: float
    mw: float
    has_ester: bool
    has_lactam: bool
    has_amide_primary: bool
    has_catechol: bool
    has_thiol: bool
    k_hyd_25_per_h: float
    k_hyd_40_per_h: float
    shelf_life_months_25c: float
    shelf_life_months_40c: float
    oxidation_index: float
    light_sensitivity: str
    stability_category: str
    notes: str


def _shelf_life_months(k_hyd_per_h: float) -> float:
    """Months to 5% degradation (t95%)."""
    # t = ln(0.95) / (-k) = -ln(0.95) / k
    t_h = -math.log(0.95) / k_hyd_per_h
    t_months = t_h / (30.0 * 24.0)
    return max(0.1, min(120.0, t_months))


def predict_drug_stability(
    drug_name: str,
    logp: float,
    mw: float,
    has_ester: bool = False,
    has_lactam: bool = False,
    has_amide_primary: bool = False,
    has_catechol: bool = False,
    has_thiol: bool = False,
    has_aldehyde: bool = False,
    ionization_type: str = "neutral",
) -> DrugStabilityResult:
    """Predict drug chemical stability from structural features."""
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError("ionization_type must be 'acid', 'base', or 'neutral'")

    # Base hydrolysis rate at 25°C from structural lability
    k_hyd_25 = 1e-6  # /h background (no labile group) — slow degradation, ~71m shelf life
    if has_ester:
        k_hyd_25 = max(k_hyd_25, 5e-3)
    if has_lactam:
        k_hyd_25 = max(k_hyd_25, 1e-3)
    if has_amide_primary:
        k_hyd_25 = max(k_hyd_25, 5e-4)

    # Temperature scaling using Q10 rule (rate doubles per 10°C)
    # k_40 = k_25 * 2^((40-25)/10) = k_25 * 2^1.5
    q10_factor = 2.0 ** ((40.0 - 25.0) / 10.0)
    k_hyd_40 = k_hyd_25 * q10_factor

    # Shelf life calculations
    shelf_25 = _shelf_life_months(k_hyd_25)
    shelf_40 = _shelf_life_months(k_hyd_40)

    # Oxidation susceptibility index (0-100)
    ox_index = 5.0  # baseline
    if has_catechol:
        ox_index += 40.0
    if has_thiol:
        ox_index += 30.0
    if has_aldehyde:
        ox_index += 30.0
    if logp > 2 and ionization_type == "base":
        ox_index += 20.0
    ox_index = min(100.0, ox_index)

    # Light sensitivity
    if has_catechol:
        light_sensitivity = "high"
    elif mw > 200 and logp > 1:
        light_sensitivity = "moderate"
    else:
        light_sensitivity = "low"

    # Stability category based on shelf life at 25°C
    if shelf_25 > 24.0:
        stability_category = "stable"
    elif shelf_25 > 12.0:
        stability_category = "moderate"
    else:
        stability_category = "unstable"

    labile_groups = []
    if has_ester:
        labile_groups.append("ester")
    if has_lactam:
        labile_groups.append("lactam")
    if has_amide_primary:
        labile_groups.append("primary amide")
    if has_catechol:
        labile_groups.append("catechol")
    if has_thiol:
        labile_groups.append("thiol")
    if has_aldehyde:
        labile_groups.append("aldehyde")

    notes_parts = [
        f"Labile groups: {', '.join(labile_groups) if labile_groups else 'none'}",
        f"k_hyd_25={k_hyd_25:.2e}/h, k_hyd_40={k_hyd_40:.2e}/h",
        f"Shelf life: {shelf_25:.1f}m @25C, {shelf_40:.1f}m @40C",
        f"OxIndex={ox_index:.0f}, Light={light_sensitivity}",
    ]
    notes = "; ".join(notes_parts)

    return DrugStabilityResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        has_ester=has_ester,
        has_lactam=has_lactam,
        has_amide_primary=has_amide_primary,
        has_catechol=has_catechol,
        has_thiol=has_thiol,
        k_hyd_25_per_h=k_hyd_25,
        k_hyd_40_per_h=k_hyd_40,
        shelf_life_months_25c=shelf_25,
        shelf_life_months_40c=shelf_40,
        oxidation_index=ox_index,
        light_sensitivity=light_sensitivity,
        stability_category=stability_category,
        notes=notes,
    )


def screen_stability(compounds: list) -> list:
    """Screen a list of compound dicts and rank by shelf life at 25°C (descending)."""
    results = []
    for c in compounds:
        result = predict_drug_stability(**c)
        results.append(result)
    results.sort(key=lambda r: r.shelf_life_months_25c, reverse=True)
    return results
