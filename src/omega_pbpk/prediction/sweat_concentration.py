"""Phase 478 — Drug Concentration Prediction in Sweat.

Models drug secretion into sweat for drug monitoring/doping detection.
Uses Henderson-Hasselbalch partitioning and Stinchcomb model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}
_LOD_MG_L = 0.001  # limit of detection in sweat (mg/L)
_SWEAT_PH_DEFAULT = 6.3
_PLASMA_PH = 7.4
_RATIO_MIN = 0.01
_RATIO_MAX = 10.0


@dataclass(frozen=True)
class SweatConcentrationResult:
    """Result of sweat drug concentration prediction."""

    drug_name: str
    logP: float
    pKa: float
    ionization_type: str
    plasma_conc_mg_L: float
    sweat_ph: float
    sweat_plasma_ratio: float
    sweat_conc_mg_L: float
    sweat_excretion_rate_mg_per_h: float
    detection_window_h: float
    detection_class: str
    notes: str


def _ratio_neutral(logP: float) -> float:
    """Sweat/plasma ratio for neutral drug (Stinchcomb simplified model)."""
    # Lipophilic drugs bind plasma proteins more → lower ratio
    r = 1.0 / (1.0 + 0.3 * logP)
    return max(_RATIO_MIN, min(_RATIO_MAX, r))


def _fraction_ionized_acid(ph: float, pka: float) -> float:
    """Fraction ionized for weak acid at given pH."""
    # HA <-> H+ + A-  ; fi = 1 / (1 + 10^(pKa - pH))
    exponent = pka - ph
    # clamp to avoid overflow
    if exponent > 300:
        return 0.0
    if exponent < -300:
        return 1.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


def _fraction_ionized_base(ph: float, pka: float) -> float:
    """Fraction ionized for weak base at given pH."""
    # BH+ <-> B + H+  ; fi = 1 / (1 + 10^(pH - pKa))
    exponent = ph - pka
    if exponent > 300:
        return 0.0
    if exponent < -300:
        return 1.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


def _compute_ratio(
    logP: float,
    pKa: float,
    ionization_type: str,
    sweat_ph: float,
) -> float:
    """Compute sweat/plasma concentration ratio."""
    r_neutral = _ratio_neutral(logP)

    if ionization_type == "neutral":
        return max(_RATIO_MIN, min(_RATIO_MAX, r_neutral))

    if ionization_type == "acid":
        fi_plasma = _fraction_ionized_acid(_PLASMA_PH, pKa)
        fi_sweat = _fraction_ionized_acid(sweat_ph, pKa)
        fu_plasma = 1.0 - fi_plasma  # unionized fraction in plasma
        fu_sweat = 1.0 - fi_sweat  # unionized fraction in sweat
        if fu_plasma <= 0.0:
            ratio = _RATIO_MIN
        else:
            ratio = r_neutral * fu_sweat / fu_plasma
        return max(_RATIO_MIN, min(_RATIO_MAX, ratio))

    if ionization_type == "base":
        fi_plasma = _fraction_ionized_base(_PLASMA_PH, pKa)
        fi_sweat = _fraction_ionized_base(sweat_ph, pKa)
        fu_plasma = 1.0 - fi_plasma
        fu_sweat = 1.0 - fi_sweat
        if fu_plasma <= 0.0:
            ratio = _RATIO_MIN
        else:
            ratio = r_neutral * fu_sweat / fu_plasma
        return max(_RATIO_MIN, min(_RATIO_MAX, ratio))

    return r_neutral


def _detection_class(ratio: float) -> str:
    """Classify sweat/plasma ratio into detection class."""
    if ratio > 1.0:
        return "high"
    if ratio >= 0.1:
        return "moderate"
    return "low"


def predict_sweat_concentration(
    drug_name: str,
    logP: float,
    pKa: float,
    ionization_type: str,
    plasma_conc_mg_L: float,
    sweat_flow_mL_per_h: float = 0.5,
    sweat_ph: float = _SWEAT_PH_DEFAULT,
) -> SweatConcentrationResult:
    """Predict drug concentration in sweat using Henderson-Hasselbalch partitioning.

    Args:
        drug_name: Name of the drug.
        logP: Octanol-water partition coefficient (log scale).
        pKa: Acid dissociation constant.
        ionization_type: "acid", "base", or "neutral".
        plasma_conc_mg_L: Plasma concentration (mg/L).
        sweat_flow_mL_per_h: Sweat production rate (mL/h).
        sweat_ph: pH of sweat (default 6.3).

    Returns:
        SweatConcentrationResult with ratio, concentration, and detection info.
    """
    # --- validation ---
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )
    if plasma_conc_mg_L <= 0:
        raise ValueError("plasma_conc_mg_L must be > 0")
    if sweat_flow_mL_per_h <= 0:
        raise ValueError("sweat_flow_mL_per_h must be > 0")
    if not (2.0 <= sweat_ph <= 8.0):
        raise ValueError("sweat_ph must be in [2, 8]")

    ratio = _compute_ratio(logP, pKa, ionization_type, sweat_ph)
    sweat_conc = plasma_conc_mg_L * ratio
    # rate: mg/h = sweat_conc (mg/L) * flow (mL/h) / 1000
    excretion_rate = sweat_conc * sweat_flow_mL_per_h / 1000.0

    detection_window = 24.0 if sweat_conc > _LOD_MG_L else 0.0
    det_class = _detection_class(ratio)

    notes_parts = [
        f"Sweat/plasma ratio: {ratio:.3f}",
        f"Detection class: {det_class}",
        f"Sweat pH: {sweat_ph}",
    ]
    if ionization_type != "neutral":
        notes_parts.append(f"Ionization type: {ionization_type} (pKa={pKa})")
    notes = "; ".join(notes_parts)

    return SweatConcentrationResult(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        ionization_type=ionization_type,
        plasma_conc_mg_L=plasma_conc_mg_L,
        sweat_ph=sweat_ph,
        sweat_plasma_ratio=ratio,
        sweat_conc_mg_L=sweat_conc,
        sweat_excretion_rate_mg_per_h=excretion_rate,
        detection_window_h=detection_window,
        detection_class=det_class,
        notes=notes,
    )


def screen_sweat_detectability(
    compounds: list,
) -> list:
    """Screen multiple compounds for sweat detectability.

    Each compound dict must contain:
        drug_name, logP, pKa, ionization_type, plasma_conc_mg_L
    Optional keys: sweat_flow_mL_per_h, sweat_ph

    Returns list of SweatConcentrationResult sorted by sweat_plasma_ratio descending.
    """
    results = []
    for compound in compounds:
        result = predict_sweat_concentration(
            drug_name=compound["drug_name"],
            logP=compound["logP"],
            pKa=compound["pKa"],
            ionization_type=compound["ionization_type"],
            plasma_conc_mg_L=compound["plasma_conc_mg_L"],
            sweat_flow_mL_per_h=compound.get("sweat_flow_mL_per_h", 0.5),
            sweat_ph=compound.get("sweat_ph", _SWEAT_PH_DEFAULT),
        )
        results.append(result)

    results.sort(key=lambda r: r.sweat_plasma_ratio, reverse=True)
    return results
