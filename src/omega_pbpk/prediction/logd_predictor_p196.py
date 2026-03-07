"""Phase 196 — LogD Predictor.

Predict logD (distribution coefficient) at physiological pH values
from logP and pKa using the Henderson-Hasselbalch equation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain @dataclass — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class LogDResult:
    """LogD prediction result at multiple pH values."""

    drug_name: str
    logP: float
    pka: float
    ionization_type: str  # "acid", "base", "neutral"
    ph_values: list  # list of float
    logd_values: list  # list of float corresponding to ph_values
    logd_at_ph74: float
    logd_at_ph68: float
    logd_at_ph12: float
    optimal_absorption_ph_range: object  # tuple (ph_lo, ph_hi) or None
    lipophilicity_class_at_74: str  # "hydrophilic", "optimal", "lipophilic"
    notes: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}
_LOGP_MIN = -5.0
_LOGP_MAX = 10.0
_PKA_MIN = 0.0
_PKA_MAX = 14.0


def _validate(logP: float, pka: float, ionization_type: str) -> None:
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )
    if not (_LOGP_MIN <= logP <= _LOGP_MAX):
        raise ValueError(f"logP must be in [{_LOGP_MIN}, {_LOGP_MAX}], got {logP}")
    if not (_PKA_MIN <= pka <= _PKA_MAX):
        raise ValueError(f"pKa must be in [{_PKA_MIN}, {_PKA_MAX}], got {pka}")


# ---------------------------------------------------------------------------
# Core Henderson-Hasselbalch corrections
# ---------------------------------------------------------------------------


def _logd_at_ph(logP: float, pka: float, ionization_type: str, ph: float) -> float:
    """Compute logD at a single pH using Henderson-Hasselbalch.

    - Acid:    logD(pH) = logP - log10(1 + 10^(pH - pKa))
    - Base:    logD(pH) = logP - log10(1 + 10^(pKa - pH))
    - Neutral: logD(pH) = logP (constant)
    """
    if ionization_type == "acid":
        exponent = ph - pka
        correction = math.log10(1.0 + 10.0**exponent)
        return logP - correction
    elif ionization_type == "base":
        exponent = pka - ph
        correction = math.log10(1.0 + 10.0**exponent)
        return logP - correction
    else:  # neutral
        return logP


# ---------------------------------------------------------------------------
# Lipophilicity classification
# ---------------------------------------------------------------------------


def _lipophilicity_class(logd_74: float) -> str:
    """Classify lipophilicity at pH 7.4."""
    if logd_74 < 0.0:
        return "hydrophilic"
    elif logd_74 <= 3.0:
        return "optimal"
    else:
        return "lipophilic"


# ---------------------------------------------------------------------------
# Optimal absorption window
# ---------------------------------------------------------------------------


def _optimal_absorption_range(
    ph_values: list[float],
    logd_values: list[float],
) -> object:
    """Find pH range where logD is in [0, 3] (optimal absorption window).

    Returns (ph_lo, ph_hi) tuple or None if no such range exists.
    """
    in_window = [ph for ph, ld in zip(ph_values, logd_values) if 0.0 <= ld <= 3.0]
    if not in_window:
        return None
    return (min(in_window), max(in_window))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_logd(
    drug_name: str,
    logP: float,
    pka: float,
    ionization_type: str,
    ph_values: list[float] | None = None,
) -> LogDResult:
    """Predict logD at multiple pH values using Henderson-Hasselbalch.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient; must be in [-5, 10].
    pka:
        Acid/base dissociation constant; must be in [0, 14].
    ionization_type:
        One of "acid", "base", "neutral".
    ph_values:
        List of pH values at which to compute logD.
        Defaults to [1.0, 2.0, ..., 14.0].

    Returns
    -------
    LogDResult
    """
    _validate(logP, pka, ionization_type)

    if ph_values is None:
        ph_values = [float(i) for i in range(1, 15)]

    # Compute logD profile
    logd_values = [_logd_at_ph(logP, pka, ionization_type, ph) for ph in ph_values]

    # Compute special pH values
    logd_74 = _logd_at_ph(logP, pka, ionization_type, 7.4)
    logd_68 = _logd_at_ph(logP, pka, ionization_type, 6.8)
    logd_12 = _logd_at_ph(logP, pka, ionization_type, 1.2)

    # Optimal absorption range
    opt_range = _optimal_absorption_range(ph_values, logd_values)

    # Lipophilicity class
    lipo_class = _lipophilicity_class(logd_74)

    notes = (
        f"ionization_type={ionization_type}; logP={logP:.2f}; pKa={pka:.2f}; "
        f"logD(7.4)={logd_74:.2f}; logD(6.8)={logd_68:.2f}; "
        f"logD(1.2)={logd_12:.2f}; class={lipo_class}"
    )

    return LogDResult(
        drug_name=drug_name,
        logP=logP,
        pka=pka,
        ionization_type=ionization_type,
        ph_values=list(ph_values),
        logd_values=logd_values,
        logd_at_ph74=logd_74,
        logd_at_ph68=logd_68,
        logd_at_ph12=logd_12,
        optimal_absorption_ph_range=opt_range,
        lipophilicity_class_at_74=lipo_class,
        notes=notes,
    )


def screen_logd_profile(drugs: list[dict]) -> list[LogDResult]:
    """Screen multiple drugs and return sorted by logD_at_7p4 descending.

    Each dict must have keys: drug_name, logP, pka, ionization_type.
    Optional key: ph_values.

    Parameters
    ----------
    drugs:
        List of dicts with drug parameters.

    Returns
    -------
    list[LogDResult] sorted by logd_at_ph74 descending.
    """
    results: list[LogDResult] = []
    for d in drugs:
        result = predict_logd(
            drug_name=d["drug_name"],
            logP=d["logP"],
            pka=d["pka"],
            ionization_type=d["ionization_type"],
            ph_values=d.get("ph_values"),
        )
        results.append(result)

    results.sort(key=lambda r: r.logd_at_ph74, reverse=True)
    return results
