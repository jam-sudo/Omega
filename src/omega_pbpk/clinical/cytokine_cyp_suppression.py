"""Phase 490 — Cytokine-Mediated CYP Suppression.

Models suppression of CYP enzymes during infection/inflammation via cytokine
signaling. Relevant for COVID-19, sepsis, chronic inflammation.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Cytokine CYP activity factors (relative to normal = 1.0)
# Keys: cyp_state → {CYP3A4, CYP1A2, CYP2C9, CYP2D6} activity fractions
# ---------------------------------------------------------------------------

_CYP_FACTORS: dict[str, dict[str, float]] = {
    "IL-6_high": {
        "CYP3A4": 0.4,
        "CYP1A2": 0.5,
        "CYP2C9": 0.7,
        "CYP2D6": 0.9,
    },
    "IL-6_moderate": {
        "CYP3A4": 0.7,
        "CYP1A2": 0.8,
        "CYP2C9": 0.85,
        "CYP2D6": 0.95,
    },
    "sepsis": {
        "CYP3A4": 0.2,
        "CYP1A2": 0.3,
        "CYP2C9": 0.5,
        "CYP2D6": 0.8,
    },
    "COVID-19_severe": {
        "CYP3A4": 0.3,
        "CYP1A2": 0.4,
        "CYP2C9": 0.6,
        "CYP2D6": 0.85,
    },
    "normal": {
        "CYP3A4": 1.0,
        "CYP1A2": 1.0,
        "CYP2C9": 1.0,
        "CYP2D6": 1.0,
    },
}

_VALID_STATES = set(_CYP_FACTORS.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CytokineCYPSuppressionResult:
    """Result of cytokine-mediated CYP suppression prediction."""

    drug_name: str
    cyp_state: str
    fm_cyp3a4: float
    fm_cyp1a2: float
    fm_cyp2c9: float
    fm_cyp2d6: float
    cyp3a4_factor: float
    cyp1a2_factor: float
    cyp2c9_factor: float
    cyp2d6_factor: float
    cl_ratio: float
    auc_ratio: float
    dose_adjustment_factor: float
    clinical_significance: str
    recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_cytokine_cyp_suppression(
    drug_name: str,
    cyp_state: str,
    fm_cyp3a4: float = 0.5,
    fm_cyp1a2: float = 0.1,
    fm_cyp2c9: float = 0.2,
    fm_cyp2d6: float = 0.1,
) -> CytokineCYPSuppressionResult:
    """Predict CYP suppression effect on drug PK.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    cyp_state:
        One of 'IL-6_high', 'IL-6_moderate', 'sepsis', 'COVID-19_severe', 'normal'.
    fm_cyp3a4:
        Fraction metabolized by CYP3A4 (0–1).
    fm_cyp1a2:
        Fraction metabolized by CYP1A2 (0–1).
    fm_cyp2c9:
        Fraction metabolized by CYP2C9 (0–1).
    fm_cyp2d6:
        Fraction metabolized by CYP2D6 (0–1).

    Returns
    -------
    CytokineCYPSuppressionResult
    """
    if cyp_state not in _VALID_STATES:
        raise ValueError(f"cyp_state must be one of {sorted(_VALID_STATES)}, got '{cyp_state}'")
    if fm_cyp3a4 < 0 or fm_cyp1a2 < 0 or fm_cyp2c9 < 0 or fm_cyp2d6 < 0:
        raise ValueError("All fm values must be >= 0")
    sum_fm = fm_cyp3a4 + fm_cyp1a2 + fm_cyp2c9 + fm_cyp2d6
    if sum_fm > 1.0 + 1e-9:
        raise ValueError(f"Sum of fm values must be <= 1.0, got {sum_fm:.4f}")

    factors = _CYP_FACTORS[cyp_state]
    f3a4 = factors["CYP3A4"]
    f1a2 = factors["CYP1A2"]
    f2c9 = factors["CYP2C9"]
    f2d6 = factors["CYP2D6"]

    # Non-CYP fraction (unaffected pathway)
    fm_non_cyp = 1.0 - sum_fm

    # CL ratio = CL_suppressed / CL_normal
    cl_ratio = (
        fm_cyp3a4 * f3a4 + fm_cyp1a2 * f1a2 + fm_cyp2c9 * f2c9 + fm_cyp2d6 * f2d6 + fm_non_cyp
    )

    # Protect against division by zero (should not happen with valid inputs)
    if cl_ratio <= 0.0:
        cl_ratio = 1e-9

    auc_ratio = 1.0 / cl_ratio
    dose_adjustment_factor = 1.0 / auc_ratio  # = cl_ratio

    # Clinical significance based on AUC increase
    auc_increase_pct = (auc_ratio - 1.0) * 100.0
    if auc_increase_pct < 25.0:
        clinical_significance = "none"
        recommendation = "monitor"
    elif auc_increase_pct <= 100.0:
        clinical_significance = "moderate"
        recommendation = "reduce_dose"
    else:
        clinical_significance = "major"
        recommendation = "avoid"

    # Notes
    if cyp_state == "normal":
        notes = "Normal CYP activity. No dose adjustment required."
    else:
        notes = (
            f"Inflammatory state '{cyp_state}' suppresses CYP enzymes. "
            f"AUC increases {auc_increase_pct:.0f}% vs normal. "
            f"Dose adjustment factor: {dose_adjustment_factor:.2f}."
        )

    return CytokineCYPSuppressionResult(
        drug_name=drug_name,
        cyp_state=cyp_state,
        fm_cyp3a4=fm_cyp3a4,
        fm_cyp1a2=fm_cyp1a2,
        fm_cyp2c9=fm_cyp2c9,
        fm_cyp2d6=fm_cyp2d6,
        cyp3a4_factor=f3a4,
        cyp1a2_factor=f1a2,
        cyp2c9_factor=f2c9,
        cyp2d6_factor=f2d6,
        cl_ratio=round(cl_ratio, 6),
        auc_ratio=round(auc_ratio, 6),
        dose_adjustment_factor=round(dose_adjustment_factor, 6),
        clinical_significance=clinical_significance,
        recommendation=recommendation,
        notes=notes,
    )


def screen_cyp_states(
    drug_name: str,
    fm_cyp3a4: float = 0.5,
    fm_cyp1a2: float = 0.1,
    fm_cyp2c9: float = 0.2,
    fm_cyp2d6: float = 0.1,
) -> list[CytokineCYPSuppressionResult]:
    """Screen all 5 CYP states for a drug.

    Returns list sorted by auc_ratio descending (highest risk first).
    """
    if fm_cyp3a4 < 0 or fm_cyp1a2 < 0 or fm_cyp2c9 < 0 or fm_cyp2d6 < 0:
        raise ValueError("All fm values must be >= 0")
    sum_fm = fm_cyp3a4 + fm_cyp1a2 + fm_cyp2c9 + fm_cyp2d6
    if sum_fm > 1.0 + 1e-9:
        raise ValueError(f"Sum of fm values must be <= 1.0, got {sum_fm:.4f}")

    results = [
        predict_cytokine_cyp_suppression(
            drug_name=drug_name,
            cyp_state=state,
            fm_cyp3a4=fm_cyp3a4,
            fm_cyp1a2=fm_cyp1a2,
            fm_cyp2c9=fm_cyp2c9,
            fm_cyp2d6=fm_cyp2d6,
        )
        for state in _VALID_STATES
    ]

    return sorted(results, key=lambda r: r.auc_ratio, reverse=True)
