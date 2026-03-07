"""Predict logD (pH-dependent octanol-water distribution coefficient).

logD accounts for ionization of the molecule at a given pH, while logP is the
intrinsic partition coefficient for the neutral species.

Henderson-Hasselbalch corrections:
  Acid:        logD = logP - log10(1 + 10^(pH - pKa))
  Base:        logD = logP - log10(1 + 10^(pKa - pH))
  Neutral:     logD = logP
  Amphoteric:  average of acid and base corrections

References
----------
Manallack DT. The pK(a) Distribution of Drugs: Application to Drug Discovery.
Perspect Medicin Chem. 2007;1:25-38.

Liao C, Nicklaus MC. Comparison of Nine Programs Predicting pKa Values of
Pharmaceutical Substances. J Chem Inf Model. 2009;49(12):2801-12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogDResult:
    """Result of a single-pH logD prediction."""

    logP: float
    pKa: float
    molecule_type: str
    pH: float
    logD: float
    ionization_fraction: float
    fraction_neutral: float
    lipophilicity_class: str
    notes: str


@dataclass
class LogDProfileResult:
    """Result of a logD vs pH profile calculation."""

    logP: float
    pKa: float
    molecule_type: str
    pH_values: list[float] = field(default_factory=list)
    logD_values: list[float] = field(default_factory=list)
    optimal_pH: float = 0.0
    optimal_logD: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_MOLECULE_TYPES = {"acid", "base", "neutral", "amphoteric"}


def _acid_correction(pH: float, pKa: float) -> float:
    """log10(1 + 10^(pH-pKa)) for acid ionization correction."""
    return math.log10(1.0 + 10.0 ** (pH - pKa))


def _base_correction(pH: float, pKa: float) -> float:
    """log10(1 + 10^(pKa-pH)) for base ionization correction."""
    return math.log10(1.0 + 10.0 ** (pKa - pH))


def _compute_logD(logP: float, pKa: float, molecule_type: str, pH: float) -> float:
    """Compute logD for a given molecule type at a specific pH."""
    if molecule_type == "acid":
        return logP - _acid_correction(pH, pKa)
    elif molecule_type == "base":
        return logP - _base_correction(pH, pKa)
    elif molecule_type == "neutral":
        return logP
    else:  # amphoteric — average of acid and base correction
        correction = 0.5 * (_acid_correction(pH, pKa) + _base_correction(pH, pKa))
        return logP - correction


def _ionization_fraction(pKa: float, molecule_type: str, pH: float) -> float:
    """Return fraction of molecules ionized (0–1)."""
    if molecule_type == "neutral":
        return 0.0
    if molecule_type == "acid":
        # Henderson-Hasselbalch: ratio = 10^(pH-pKa) for acid
        ratio = 10.0 ** (pH - pKa)
        return ratio / (1.0 + ratio)
    if molecule_type == "base":
        # ratio = 10^(pKa-pH) for base (protonated/neutral)
        ratio = 10.0 ** (pKa - pH)
        return ratio / (1.0 + ratio)
    # amphoteric: average of both
    ratio_a = 10.0 ** (pH - pKa)
    ratio_b = 10.0 ** (pKa - pH)
    fi_a = ratio_a / (1.0 + ratio_a)
    fi_b = ratio_b / (1.0 + ratio_b)
    return 0.5 * (fi_a + fi_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lipophilicity_class(logD_at_pH74: float) -> str:
    """Classify lipophilicity based on logD at pH 7.4.

    Parameters
    ----------
    logD_at_pH74:
        logD value at physiological pH 7.4.

    Returns
    -------
    str
        One of: very_hydrophilic, hydrophilic, moderate, lipophilic, highly_lipophilic.
    """
    if not math.isfinite(logD_at_pH74):
        raise ValueError(f"logD_at_pH74 must be a finite float, got {logD_at_pH74!r}")
    if logD_at_pH74 < -3.0:
        return "very_hydrophilic"
    elif logD_at_pH74 < 0.0:
        return "hydrophilic"
    elif logD_at_pH74 < 3.0:
        return "moderate"
    elif logD_at_pH74 <= 5.0:
        return "lipophilic"
    else:
        return "highly_lipophilic"


def predict_logD(
    logP: float,
    pka: float,
    molecule_type: str,
    pH: float,
) -> LogDResult:
    """Predict logD at a given pH using Henderson-Hasselbalch correction.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient of the neutral species.
    pka:
        Acid dissociation constant (pKa) of the ionizable group.
    molecule_type:
        One of ``"acid"``, ``"base"``, ``"neutral"``, ``"amphoteric"``.
    pH:
        Target pH for logD prediction (valid range 0–14).

    Returns
    -------
    LogDResult
    """
    if not math.isfinite(logP):
        raise ValueError(f"logP must be a finite float, got {logP!r}")
    if not math.isfinite(pka):
        raise ValueError(f"pka must be a finite float, got {pka!r}")
    if molecule_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {sorted(_VALID_MOLECULE_TYPES)}, got {molecule_type!r}"
        )
    if not (0.0 <= pH <= 14.0):
        raise ValueError(f"pH must be in range [0, 14], got {pH}")

    logD = _compute_logD(logP, pka, molecule_type, pH)
    fi = _ionization_fraction(pka, molecule_type, pH)
    fn = 1.0 - fi
    lclass = lipophilicity_class(logD)

    notes_parts: list[str] = []
    if molecule_type == "amphoteric":
        notes_parts.append("Amphoteric: correction averaged from acid and base models.")
    if abs(pH - pka) < 0.5:
        notes_parts.append(f"pH ({pH:.2f}) is close to pKa ({pka:.2f}); ~50% ionization expected.")
    notes = " ".join(notes_parts) if notes_parts else "Standard Henderson-Hasselbalch prediction."

    return LogDResult(
        logP=logP,
        pKa=pka,
        molecule_type=molecule_type,
        pH=pH,
        logD=round(logD, 4),
        ionization_fraction=round(fi, 4),
        fraction_neutral=round(fn, 4),
        lipophilicity_class=lclass,
        notes=notes,
    )


def logD_profile(
    logP: float,
    pka: float,
    molecule_type: str,
    pH_range: list[float],
) -> LogDProfileResult:
    """Compute logD across a range of pH values.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient of the neutral species.
    pka:
        pKa of the ionizable group.
    molecule_type:
        One of ``"acid"``, ``"base"``, ``"neutral"``, ``"amphoteric"``.
    pH_range:
        List of pH values to evaluate (each must be 0–14).

    Returns
    -------
    LogDProfileResult
    """
    if not math.isfinite(logP):
        raise ValueError(f"logP must be a finite float, got {logP!r}")
    if not math.isfinite(pka):
        raise ValueError(f"pka must be a finite float, got {pka!r}")
    if molecule_type not in _VALID_MOLECULE_TYPES:
        raise ValueError(
            f"molecule_type must be one of {sorted(_VALID_MOLECULE_TYPES)}, got {molecule_type!r}"
        )
    if not pH_range:
        raise ValueError("pH_range must be a non-empty list of pH values.")
    for ph in pH_range:
        if not math.isfinite(ph) or not (0.0 <= ph <= 14.0):
            raise ValueError(f"All pH values must be finite and in [0, 14], got {ph!r}")

    logD_values: list[float] = []
    for ph in pH_range:
        logD_values.append(_compute_logD(logP, pka, molecule_type, ph))

    # Optimal pH for absorption: logD closest to 0 (balanced lipophilicity)
    best_idx = min(range(len(logD_values)), key=lambda i: abs(logD_values[i]))
    optimal_pH = pH_range[best_idx]
    optimal_logD = logD_values[best_idx]

    notes = (
        f"Profile spans pH {min(pH_range):.1f}–{max(pH_range):.1f}. "
        f"Optimal absorption pH (logD~0): {optimal_pH:.2f} (logD={optimal_logD:.4f})."
    )

    return LogDProfileResult(
        logP=logP,
        pKa=pka,
        molecule_type=molecule_type,
        pH_values=list(pH_range),
        logD_values=[round(v, 4) for v in logD_values],
        optimal_pH=optimal_pH,
        optimal_logD=round(optimal_logD, 4),
        notes=notes,
    )
