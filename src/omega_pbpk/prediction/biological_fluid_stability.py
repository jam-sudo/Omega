"""Phase 736 — Drug Stability in Biological Fluids.

Predicts first-order degradation rate constants and half-lives in plasma,
gastric fluid (pH 1.2), and intestinal fluid (pH 6.8) based on structural
features (ester, amide, beta-lactam, acid-labile groups).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_LN2 = math.log(2.0)


@dataclass(frozen=True)
class DrugStabilityResult:
    """Stability prediction for a drug in biological fluids."""

    compound_name: str
    k_deg_plasma_per_h: float
    k_deg_gastric_per_h: float
    k_deg_intestinal_per_h: float
    t_half_plasma_h: float
    t_half_gastric_h: float
    t_half_intestinal_h: float
    fraction_intact_plasma_1h: float
    fraction_intact_gastric_30min: float
    fraction_intact_intestinal_1h: float
    plasma_stability: str
    gastric_stability: str
    overall_stability: str
    notes: str


def _classify_plasma(t_half_h: float) -> str:
    if t_half_h > 6.0:
        return "stable"
    elif t_half_h >= 1.0:
        return "moderate"
    else:
        return "labile"


def _classify_gastric(t_half_h: float) -> str:
    if t_half_h > 2.0:
        return "stable"
    elif t_half_h >= 0.5:
        return "moderate"
    else:
        return "labile"


def _overall_stability(plasma_stab: str, gastric_stab: str) -> str:
    if plasma_stab == "labile" or gastric_stab == "labile":
        return "poor"
    elif plasma_stab == "stable" and gastric_stab == "stable":
        return "good"
    else:
        return "moderate"


def predict_drug_stability(
    compound_name: str,
    has_ester: bool = False,
    has_amide: bool = False,
    has_lactam: bool = False,
    has_acid_labile_group: bool = False,
    logp: float = 2.0,
    mw_da: float = 300.0,
) -> DrugStabilityResult:
    """Predict drug stability in plasma, gastric fluid, and intestinal fluid.

    Parameters
    ----------
    compound_name:
        Name of the compound. Empty strings are handled gracefully.
    has_ester:
        Compound contains an ester bond (susceptible to plasma esterases and
        acid hydrolysis).
    has_amide:
        Compound contains an amide bond (mild susceptibility to hydrolysis).
    has_lactam:
        Compound contains a beta-lactam ring (acid-labile; also susceptible
        to plasma hydrolysis).
    has_acid_labile_group:
        Compound has another acid-labile functional group (e.g. acid-sensitive
        prodrugs, acid-labile protecting groups).
    logp:
        Octanol-water partition coefficient (stored for extensibility).
    mw_da:
        Molecular weight in Da (stored for extensibility).

    Returns
    -------
    DrugStabilityResult
    """
    # --- plasma degradation (L/h) ---
    k_plasma = 0.001  # baseline
    if has_ester:
        k_plasma += 0.15
    if has_amide:
        k_plasma += 0.02
    if has_lactam:
        k_plasma += 0.05

    # --- gastric degradation at pH 1.2 ---
    k_gastric = 0.0001  # baseline
    if has_acid_labile_group or has_lactam:
        k_gastric += 2.0
    if has_ester:
        k_gastric += 0.5

    # --- intestinal degradation at pH 6.8 (milder conditions) ---
    k_intestinal = k_gastric * 0.01 + k_plasma * 0.5

    # --- half-lives ---
    t_half_plasma = _LN2 / k_plasma
    t_half_gastric = _LN2 / k_gastric
    t_half_intestinal = _LN2 / k_intestinal

    # --- fraction intact ---
    fi_plasma_1h = math.exp(-k_plasma * 1.0)
    fi_gastric_30min = math.exp(-k_gastric * 0.5)
    fi_intestinal_1h = math.exp(-k_intestinal * 1.0)

    # --- stability classifications ---
    plasma_stab = _classify_plasma(t_half_plasma)
    gastric_stab = _classify_gastric(t_half_gastric)
    overall_stab = _overall_stability(plasma_stab, gastric_stab)

    # --- notes ---
    flags = []
    if has_ester:
        flags.append("ester")
    if has_amide:
        flags.append("amide")
    if has_lactam:
        flags.append("lactam")
    if has_acid_labile_group:
        flags.append("acid-labile")
    flag_str = ", ".join(flags) if flags else "none"
    notes = (
        f"Structural flags: {flag_str}; "
        f"plasma t\u00bd={t_half_plasma:.2f}h ({plasma_stab}); "
        f"gastric t\u00bd={t_half_gastric:.3f}h ({gastric_stab}); "
        f"overall={overall_stab}"
    )

    return DrugStabilityResult(
        compound_name=compound_name,
        k_deg_plasma_per_h=k_plasma,
        k_deg_gastric_per_h=k_gastric,
        k_deg_intestinal_per_h=k_intestinal,
        t_half_plasma_h=t_half_plasma,
        t_half_gastric_h=t_half_gastric,
        t_half_intestinal_h=t_half_intestinal,
        fraction_intact_plasma_1h=fi_plasma_1h,
        fraction_intact_gastric_30min=fi_gastric_30min,
        fraction_intact_intestinal_1h=fi_intestinal_1h,
        plasma_stability=plasma_stab,
        gastric_stability=gastric_stab,
        overall_stability=overall_stab,
        notes=notes,
    )


def screen_stability(compounds: list[dict]) -> list[DrugStabilityResult]:
    """Screen a list of compound parameter dicts and return sorted results.

    Compounds with ``overall_stability="good"`` appear first, then
    ``"moderate"``, then ``"poor"``.

    Parameters
    ----------
    compounds:
        Each dict must contain ``compound_name`` and may contain any other
        keyword arguments accepted by :func:`predict_drug_stability`.

    Returns
    -------
    List[DrugStabilityResult]
        Sorted by overall stability (good first).
    """
    _order = {"good": 0, "moderate": 1, "poor": 2}
    results = [predict_drug_stability(**c) for c in compounds]
    results.sort(key=lambda r: _order.get(r.overall_stability, 3))
    return results
