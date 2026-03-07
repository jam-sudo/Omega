"""Phase 670 — LogD Predictor: predict logD from logP and pKa at physiological pH."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _logD_at_pH(
    logP: float,
    pka_acid: float | None,
    pka_base: float | None,
    pH: float,
) -> float:
    """Compute logD at a given pH for a compound with given logP and pKa values."""
    if pka_acid is not None and pka_base is None:
        # Pure acid
        delta = pH - pka_acid
        logD = logP - math.log10(1.0 + 10.0**delta)
    elif pka_base is not None and pka_acid is None:
        # Pure base
        delta = pka_base - pH
        logD = logP - math.log10(1.0 + 10.0**delta)
    elif pka_acid is not None and pka_base is not None:
        # Zwitterion: weighted blend
        logD_acid = logP - math.log10(1.0 + 10.0 ** (pH - pka_acid))
        logD_base = logP - math.log10(1.0 + 10.0 ** (pka_base - pH))
        logD = (logD_acid + logD_base) / 2.0
    else:
        # Neutral compound
        logD = logP
    return logD


def _ionization_at_pH74(
    pka_acid: float | None,
    pka_base: float | None,
) -> tuple[float, float]:
    """Return (ionized_fraction, net_charge) at pH 7.4."""
    pH = 7.4

    if pka_acid is not None and pka_base is None:
        # Acid
        ionized = 1.0 / (1.0 + 10.0 ** (pka_acid - pH))
        charge = -ionized  # negative for deprotonated acid
    elif pka_base is not None and pka_acid is None:
        # Base
        ionized = 1.0 / (1.0 + 10.0 ** (pH - pka_base))
        charge = ionized  # positive for protonated base
    elif pka_acid is not None and pka_base is not None:
        # Zwitterion: both ionizations contribute
        ionized_acid = 1.0 / (1.0 + 10.0 ** (pka_acid - pH))
        ionized_base = 1.0 / (1.0 + 10.0 ** (pH - pka_base))
        ionized = (ionized_acid + ionized_base) / 2.0
        charge = ionized_base - ionized_acid  # net
    else:
        ionized = 0.0
        charge = 0.0

    return ionized, charge


def _membrane_permeability_class(logD74: float) -> str:
    if logD74 > 1.0:
        return "high"
    elif logD74 >= 0.0:
        return "moderate"
    else:
        return "low"


def _cns_penetration_likelihood(logD74: float) -> str:
    if 1.0 <= logD74 <= 3.0:
        return "high"
    elif (0.0 <= logD74 < 1.0) or (3.0 < logD74 <= 5.0):
        return "moderate"
    else:
        return "low"


def _gi_absorption_prediction(logD74: float) -> str:
    if logD74 > -1.0:
        return "excellent"
    elif logD74 >= -1.0:
        return "good"
    else:
        return "poor"


def _build_logd_notes(
    logP: float,
    pka_acid: float | None,
    pka_base: float | None,
    logD74: float,
    ionized: float,
) -> str:
    compound_type = "neutral"
    if pka_acid is not None and pka_base is not None:
        compound_type = "zwitterion"
    elif pka_acid is not None:
        compound_type = f"acid (pKa={pka_acid:.1f})"
    elif pka_base is not None:
        compound_type = f"base (pKa={pka_base:.1f})"

    return (
        f"Compound type: {compound_type}. "
        f"logP={logP:.2f}, logD(7.4)={logD74:.2f}. "
        f"Ionized fraction at pH 7.4: {ionized:.2%}. "
        "LogD calculated using Henderson-Hasselbalch equation."
    )


@dataclass(frozen=True)
class LogDPrediction:
    """LogD prediction result at various physiologically relevant pH values."""

    compound_name: str
    logP: float
    pka_acid: float | None
    pka_base: float | None
    logD_at_pH74: float
    logD_at_pH_gastric: float
    logD_at_pH_intestinal: float
    ionization_at_pH74: float
    charge_at_pH74: float
    membrane_permeability_class: str
    cns_penetration_likelihood: str
    gi_absorption_prediction: str
    notes: str


def predict_logD(
    compound_name: str,
    logP: float,
    pka_acid: float | None = None,
    pka_base: float | None = None,
) -> LogDPrediction:
    """Predict logD at physiological and GI pH values.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient (log scale). May be any float.
    pka_acid:
        Acid dissociation constant. Must be in [0, 14] if provided.
    pka_base:
        Base dissociation constant. Must be in [0, 14] if provided.

    Returns
    -------
    LogDPrediction
    """
    # --- Validation ---
    if pka_acid is not None and not (0.0 <= pka_acid <= 14.0):
        raise ValueError(f"pka_acid must be in [0, 14], got {pka_acid}")
    if pka_base is not None and not (0.0 <= pka_base <= 14.0):
        raise ValueError(f"pka_base must be in [0, 14], got {pka_base}")

    # --- Compute logD at key pH values ---
    logD74 = _logD_at_pH(logP, pka_acid, pka_base, 7.4)
    logD_gastric = _logD_at_pH(logP, pka_acid, pka_base, 1.5)
    logD_intestinal = _logD_at_pH(logP, pka_acid, pka_base, 6.5)

    # --- Ionization at pH 7.4 ---
    ionized, charge = _ionization_at_pH74(pka_acid, pka_base)

    # --- Classification ---
    perm_class = _membrane_permeability_class(logD74)
    cns_class = _cns_penetration_likelihood(logD74)
    gi_class = _gi_absorption_prediction(logD74)

    notes = _build_logd_notes(logP, pka_acid, pka_base, logD74, ionized)

    return LogDPrediction(
        compound_name=compound_name,
        logP=logP,
        pka_acid=pka_acid,
        pka_base=pka_base,
        logD_at_pH74=logD74,
        logD_at_pH_gastric=logD_gastric,
        logD_at_pH_intestinal=logD_intestinal,
        ionization_at_pH74=ionized,
        charge_at_pH74=charge,
        membrane_permeability_class=perm_class,
        cns_penetration_likelihood=cns_class,
        gi_absorption_prediction=gi_class,
        notes=notes,
    )


def logD_pH_profile(
    compound_name: str,
    logP: float,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    pH_range: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Return list of (pH, logD) tuples for pH profile visualization.

    Parameters
    ----------
    compound_name:
        Name of the compound (used for context, not computation).
    logP:
        Octanol-water partition coefficient.
    pka_acid:
        Acid pKa. Must be in [0, 14] if provided.
    pka_base:
        Base pKa. Must be in [0, 14] if provided.
    pH_range:
        List of pH values. Defaults to [1, 2, ..., 10].

    Returns
    -------
    list of (pH, logD) tuples
    """
    # Validation
    if pka_acid is not None and not (0.0 <= pka_acid <= 14.0):
        raise ValueError(f"pka_acid must be in [0, 14], got {pka_acid}")
    if pka_base is not None and not (0.0 <= pka_base <= 14.0):
        raise ValueError(f"pka_base must be in [0, 14], got {pka_base}")

    if pH_range is None:
        pH_range = [float(i) for i in range(1, 11)]

    return [(pH, _logD_at_pH(logP, pka_acid, pka_base, pH)) for pH in pH_range]


def screen_logD(compounds: list[dict]) -> list[LogDPrediction]:
    """Screen multiple compounds and return sorted by logD_at_pH74 descending.

    Each dict should have keys: compound_name, logP, and optionally pka_acid, pka_base.

    Parameters
    ----------
    compounds:
        List of compound parameter dicts.

    Returns
    -------
    list of LogDPrediction sorted by logD_at_pH74 descending
    """
    if not compounds:
        return []

    results: list[LogDPrediction] = []
    for cmpd in compounds:
        result = predict_logD(
            compound_name=cmpd.get("compound_name", "unknown"),
            logP=cmpd["logP"],
            pka_acid=cmpd.get("pka_acid"),
            pka_base=cmpd.get("pka_base"),
        )
        results.append(result)

    results.sort(key=lambda r: r.logD_at_pH74, reverse=True)
    return results
