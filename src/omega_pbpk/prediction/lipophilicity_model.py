"""Drug lipophilicity and biological membrane partitioning prediction.

Extends the logP prediction in logp_predictor.py with ionization-corrected
logD values across pH and biological membrane partition coefficients.

Key references:
- Manners et al. (1988) J Pharm Sci 77:1038-1042 (logD measurement)
- Yalkowsky & Banerjee (1992) Aqueous Solubility: Methods of Estimation
- Austin et al. (2002) J Med Chem 45:1569-1579 (membrane partitioning)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LogDProfileResult:
    """LogD profile across a pH range.

    Attributes
    ----------
    logP : float
        Intrinsic (unionised) logP.
    pka_acid : float or None
        Acidic pKa if applicable.
    pka_base : float or None
        Basic pKa if applicable.
    ph_values : list[float]
        pH values at which logD was computed.
    logd_values : list[float]
        logD values corresponding to ph_values.
    ph_of_min_logd : float
        pH at which logD is minimised (most hydrophilic form).
    logd_at_74 : float
        logD at physiological pH 7.4.
    notes : str
        Human-readable description of ionisation type and profile.
    """

    logP: float
    pka_acid: float | None
    pka_base: float | None
    ph_values: list
    logd_values: list
    ph_of_min_logd: float
    logd_at_74: float
    notes: str = field(default="")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def logd_from_logp(
    logP: float,
    pka_acid: float | None,
    pka_base: float | None,
    ph: float = 7.4,
) -> float:
    """Compute logD at a given pH using Henderson-Hasselbalch corrections.

    For an acid:
        logD = logP - log10(1 + 10^(pH - pKa_acid))

    For a base:
        logD = logP - log10(1 + 10^(pKa_base - pH))

    For a zwitterion (both pKa values present):
        logD = logP - log10(1 + 10^(pH - pKa_acid) + 10^(pKa_base - pH))

    For a neutral compound:
        logD = logP

    Parameters
    ----------
    logP : float
        Intrinsic (unionised) octanol-water partition coefficient.
    pka_acid : float or None
        Acidic pKa of the compound.  None if the compound has no acidic group.
    pka_base : float or None
        Basic pKa of the compound.  None if no basic group.
    ph : float
        Target pH.  Default 7.4 (physiological plasma).

    Returns
    -------
    float
        logD at the specified pH.
    """
    ionisation_sum = 0.0

    if pka_acid is not None:
        ionisation_sum += 10 ** (ph - pka_acid)

    if pka_base is not None:
        ionisation_sum += 10 ** (pka_base - ph)

    if ionisation_sum == 0.0:
        return float(logP)

    return float(logP - math.log10(1.0 + ionisation_sum))


def membrane_partition_coefficient(
    logP: float,
    logD74: float,
    mw: float,
    charge: str = "neutral",
) -> float:
    """Predict biological membrane partition coefficient (Kp_membrane).

    Uses empirical relationships calibrated to phosphatidylcholine liposome
    binding data:

    - Neutral:    logKp = 0.9 * logD74 + 0.2
    - Cationic:   logKp = 0.7 * logD74 + 0.5  (cationic amphiphile accumulation)
    - Anionic:    logKp = 0.8 * logD74 - 0.3  (repulsion by anionic membranes)
    - Zwitterion: logKp = 0.85 * logD74

    Parameters
    ----------
    logP : float
        Intrinsic logP (used implicitly via logD74).
    logD74 : float
        logD at pH 7.4.
    mw : float
        Molecular weight (Da).  Accepted but not used in current equations
        (included for API compatibility and future MW-dependent corrections).
    charge : str
        Charge state at physiological pH: "neutral", "cationic", "anionic",
        or "zwitterion".  Default "neutral".

    Returns
    -------
    float
        Membrane partition coefficient Kp_membrane (dimensionless).

    Raises
    ------
    ValueError
        If charge is not one of the recognised values.
    """
    valid_charges = {"neutral", "cationic", "anionic", "zwitterion"}
    charge_lower = charge.lower()
    if charge_lower not in valid_charges:
        raise ValueError(
            f"charge must be one of {valid_charges}, got '{charge}'."
        )

    if charge_lower == "neutral":
        log_kp = 0.9 * logD74 + 0.2
    elif charge_lower == "cationic":
        log_kp = 0.7 * logD74 + 0.5
    elif charge_lower == "anionic":
        log_kp = 0.8 * logD74 - 0.3
    else:  # zwitterion
        log_kp = 0.85 * logD74

    return float(10**log_kp)


def predict_logd_profile(
    logP: float,
    pka_acid: float | None,
    pka_base: float | None,
    ph_range: tuple = (1.0, 10.0),
    n_points: int = 50,
) -> LogDProfileResult:
    """Compute logD profile across a pH range.

    Parameters
    ----------
    logP : float
        Intrinsic logP.
    pka_acid : float or None
        Acidic pKa.
    pka_base : float or None
        Basic pKa.
    ph_range : tuple of (float, float)
        (pH_min, pH_max).  Default (1.0, 10.0).
    n_points : int
        Number of pH points.  Default 50.

    Returns
    -------
    LogDProfileResult
        Profile data including pH of minimum logD and logD at 7.4.

    Raises
    ------
    ValueError
        If ph_range is invalid (min >= max) or n_points < 2.
    """
    ph_min, ph_max = ph_range
    if ph_min >= ph_max:
        raise ValueError(
            f"ph_range must have min < max, got ({ph_min}, {ph_max})."
        )
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")

    step = (ph_max - ph_min) / (n_points - 1)
    ph_values = [ph_min + i * step for i in range(n_points)]
    logd_values = [
        logd_from_logp(logP, pka_acid, pka_base, ph) for ph in ph_values
    ]

    # Find pH of minimum logD
    min_idx = logd_values.index(min(logd_values))
    ph_of_min = ph_values[min_idx]

    logd_at_74 = logd_from_logp(logP, pka_acid, pka_base, 7.4)

    # Build descriptive notes
    ion_type = _ionisation_type(pka_acid, pka_base)
    notes = (
        f"Ionisation type: {ion_type}; "
        f"logD range [{min(logd_values):.2f}, {max(logd_values):.2f}]; "
        f"most hydrophilic at pH {ph_of_min:.2f}"
    )

    return LogDProfileResult(
        logP=logP,
        pka_acid=pka_acid,
        pka_base=pka_base,
        ph_values=ph_values,
        logd_values=logd_values,
        ph_of_min_logd=ph_of_min,
        logd_at_74=logd_at_74,
        notes=notes,
    )


def aqueous_solubility_from_logd(
    logP: float,
    logD74: float,
    melting_point_C: float = 150.0,
) -> float:
    """Estimate aqueous solubility at pH 7.4 using modified Yalkowsky equation.

    The general solubility equation (GSE):
        log S_intrinsic = 0.5 - logP - 0.01 * (Tm - 25)

    pH correction for ionisation:
        S_at_pH74 = S_intrinsic * (1 + 10^(logP - logD74))

    The ionisation factor 10^(logP - logD74) represents the ratio of
    ionised to unionised species.

    Parameters
    ----------
    logP : float
        Intrinsic logP.
    logD74 : float
        logD at pH 7.4.
    melting_point_C : float
        Melting point in Celsius.  Default 150°C.

    Returns
    -------
    float
        Estimated aqueous solubility at pH 7.4 in mg/mL.
    """
    log_s_intrinsic = 0.5 - logP - 0.01 * (melting_point_C - 25.0)
    s_intrinsic_mg_per_mL = 10**log_s_intrinsic

    # Ionisation enhancement at pH 7.4
    ionisation_factor = 10 ** (logP - logD74)
    s_at_ph74 = s_intrinsic_mg_per_mL * (1.0 + ionisation_factor)

    return float(s_at_ph74)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ionisation_type(pka_acid: float | None, pka_base: float | None) -> str:
    """Return a human-readable ionisation type string."""
    if pka_acid is not None and pka_base is not None:
        return "zwitterion"
    if pka_acid is not None:
        return "acid"
    if pka_base is not None:
        return "base"
    return "neutral"


__all__ = [
    "LogDProfileResult",
    "logd_from_logp",
    "membrane_partition_coefficient",
    "predict_logd_profile",
    "aqueous_solubility_from_logd",
]
