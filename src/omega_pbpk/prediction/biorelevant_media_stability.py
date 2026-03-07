"""Phase 358 — Drug Stability in Biorelevant Media.

Predicts drug stability and solubility enhancement in biorelevant dissolution
media: FaSSIF (fasted small intestine), FeSSIF (fed small intestine),
and FaSSGF (fasted gastric fluid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VALID_DRUG_TYPES = {"acid", "base", "neutral"}
VALID_MEDIA = {"FaSSIF", "FeSSIF", "FaSSGF"}

# Media properties (informational, used for notes)
FASSIF_PH = 6.5
FESSIF_PH = 5.0
FASSGF_PH = 1.6

# Maximum bile salt solubilization cap
F_BILE_CAP = 5.0


def _classify_stability(remaining: float) -> str:
    """Return stability class based on fraction remaining."""
    if remaining > 0.9:
        return "stable"
    elif remaining >= 0.5:
        return "unstable"
    else:
        return "highly_unstable"


@dataclass(frozen=True)
class BiorelevantStabilityResult:
    """Result of drug stability prediction in biorelevant media."""

    drug_name: str
    logP: float
    fassif_remaining: float
    fessif_remaining: float
    fassgf_remaining: float
    fassif_stability_class: str
    fessif_stability_class: str
    fassgf_stability_class: str
    fassif_sol_enhancement: float
    fessif_sol_enhancement: float
    recommended_medium: str
    overall_stability: str
    notes: str


def predict_biorelevant_stability(
    drug_name: str,
    logP: float,
    pKa: float,
    drug_type: str,
    mw_Da: float,
    k_deg_fassif_per_h: float = 0.0,
    k_deg_fessif_per_h: float = 0.0,
    k_deg_fassgf_per_h: float = 0.0,
    t_end_min: float = 120.0,
) -> BiorelevantStabilityResult:
    """Predict drug stability in biorelevant dissolution media.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logP:
        Octanol-water partition coefficient.
    pKa:
        Ionisation constant of the drug.
    drug_type:
        One of "acid", "base", or "neutral".
    mw_Da:
        Molecular weight in Daltons (must be > 0).
    k_deg_fassif_per_h:
        First-order degradation rate constant in FaSSIF (h^-1, >= 0).
    k_deg_fessif_per_h:
        First-order degradation rate constant in FeSSIF (h^-1, >= 0).
    k_deg_fassgf_per_h:
        First-order degradation rate constant in FaSSGF (h^-1, >= 0).
    t_end_min:
        Incubation time in minutes (must be > 0).

    Returns
    -------
    BiorelevantStabilityResult
    """
    # --- Input validation ---
    if drug_type not in VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {VALID_DRUG_TYPES}, got {drug_type!r}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if t_end_min <= 0:
        raise ValueError(f"t_end_min must be > 0, got {t_end_min}")
    if k_deg_fassif_per_h < 0:
        raise ValueError(f"k_deg_fassif_per_h must be >= 0, got {k_deg_fassif_per_h}")
    if k_deg_fessif_per_h < 0:
        raise ValueError(f"k_deg_fessif_per_h must be >= 0, got {k_deg_fessif_per_h}")
    if k_deg_fassgf_per_h < 0:
        raise ValueError(f"k_deg_fassgf_per_h must be >= 0, got {k_deg_fassgf_per_h}")

    # --- Convert time ---
    t_end_h = t_end_min / 60.0

    # --- First-order degradation ---
    remaining_fassif = math.exp(-k_deg_fassif_per_h * t_end_h)
    remaining_fessif = math.exp(-k_deg_fessif_per_h * t_end_h)
    remaining_fassgf = math.exp(-k_deg_fassgf_per_h * t_end_h)

    # --- Stability classes ---
    fassif_class = _classify_stability(remaining_fassif)
    fessif_class = _classify_stability(remaining_fessif)
    fassgf_class = _classify_stability(remaining_fassgf)

    # --- Bile salt solubilisation factor ---
    f_bile = min(1.0 + 0.5 * logP, F_BILE_CAP)
    # For negative logP, f_bile can be < 1; ensure floor at 1.0 for enhancement
    f_bile_positive = max(1.0, f_bile)

    # --- Solubility enhancement ---
    fassif_sol_enhancement = max(1.0, f_bile_positive * 0.8)
    fessif_sol_enhancement = max(1.0, f_bile_positive * 1.5)
    fassgf_sol_enhancement = 1.0  # no bile salts in gastric fluid

    # --- Recommended medium (highest solubility enhancement) ---
    enhancements = {
        "FaSSIF": fassif_sol_enhancement,
        "FeSSIF": fessif_sol_enhancement,
        "FaSSGF": fassgf_sol_enhancement,
    }
    recommended_medium = max(enhancements, key=lambda k: enhancements[k])

    # --- Overall stability ---
    all_remaining = [remaining_fassif, remaining_fessif, remaining_fassgf]
    if all(r > 0.9 for r in all_remaining):
        overall_stability = "stable"
    elif all(r > 0.5 for r in all_remaining):
        overall_stability = "moderate"
    else:
        overall_stability = "labile"

    # --- Build notes ---
    notes_parts = [
        f"drug_type={drug_type}",
        f"logP={logP:.2f}",
        f"f_bile={f_bile:.2f}",
    ]
    if logP >= 2.0:
        notes_parts.append("lipophilic: bile-salt solubilisation expected")
    elif logP < 0:
        notes_parts.append("hydrophilic: minimal bile-salt effect")
    notes = "; ".join(notes_parts)

    return BiorelevantStabilityResult(
        drug_name=drug_name,
        logP=logP,
        fassif_remaining=remaining_fassif,
        fessif_remaining=remaining_fessif,
        fassgf_remaining=remaining_fassgf,
        fassif_stability_class=fassif_class,
        fessif_stability_class=fessif_class,
        fassgf_stability_class=fassgf_class,
        fassif_sol_enhancement=fassif_sol_enhancement,
        fessif_sol_enhancement=fessif_sol_enhancement,
        recommended_medium=recommended_medium,
        overall_stability=overall_stability,
        notes=notes,
    )
