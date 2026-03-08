"""Phase 494 — Drug Concentrations in Hair.

Models drug incorporation into hair follicles for forensic and
chronic exposure assessment.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_IONIZATION_TYPES = frozenset({"acid", "base", "neutral"})

_DETECTION_THRESHOLD_MG_KG: float = 0.001  # typical LOD in hair (mg/kg)
_HPR_MIN: float = 0.001
_HPR_MAX: float = 100.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HairDrugConcentrationResult:
    """Results from hair drug concentration prediction."""

    drug_name: str
    logP: float
    pKa: float
    ionization_type: str
    plasma_conc_mg_L: float
    hair_plasma_ratio: float
    hair_conc_mg_kg: float
    detectable_in_hair: bool
    detection_window_months: float
    melanin_effect: bool
    forensic_significance: str
    notes: str


# ---------------------------------------------------------------------------
# HPR calculation
# ---------------------------------------------------------------------------


def _calculate_hpr(logP: float, pKa: float, ionization_type: str) -> float:
    """Calculate Hair-to-Plasma Ratio (HPR)."""
    # Baseline for neutral drugs
    hpr_neutral = 0.01 + 0.001 * max(0.0, logP)

    if ionization_type == "base":
        # Basic drugs benefit from melanin binding
        hpr = hpr_neutral * (10.0 ** ((pKa - 7.0) * 0.5))
    elif ionization_type == "acid":
        # Acid drugs incorporate poorly
        hpr = hpr_neutral * 0.1
    else:
        # Neutral
        hpr = hpr_neutral

    # Clamp to [HPR_MIN, HPR_MAX]
    return max(_HPR_MIN, min(_HPR_MAX, hpr))


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_hair_concentration(
    drug_name: str,
    logP: float,
    pKa: float,
    ionization_type: str,
    plasma_conc_mg_L: float,
) -> HairDrugConcentrationResult:
    """Predict drug concentration in hair from plasma concentration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Lipophilicity (octanol-water partition coefficient).
    pKa:
        Acid dissociation constant.
    ionization_type:
        One of "acid", "base", "neutral".
    plasma_conc_mg_L:
        Plasma drug concentration in mg/L.

    Returns
    -------
    HairDrugConcentrationResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION_TYPES)}, "
            f"got {ionization_type!r}"
        )
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")

    # ------------------------------------------------------------------
    # HPR calculation
    # ------------------------------------------------------------------
    hpr = _calculate_hpr(logP, pKa, ionization_type)

    # Hair concentration (mg/kg), using density ~1 kg/L so mg/L ≈ mg/kg
    hair_conc = plasma_conc_mg_L * hpr

    # Detection
    detectable = hair_conc > _DETECTION_THRESHOLD_MG_KG
    detection_window = 3.0 if detectable else 0.0

    # Melanin effect applies to basic drugs with pKa > 7
    melanin_effect = ionization_type == "base" and pKa > 7.0

    # Forensic significance classification
    if hair_conc <= _DETECTION_THRESHOLD_MG_KG:
        forensic_significance = "not_detected"
    elif hair_conc <= 0.01:
        forensic_significance = "trace"
    elif hair_conc <= 1.0:
        forensic_significance = "positive"
    else:
        forensic_significance = "high"

    # Notes
    notes_parts: list[str] = [
        f"HPR={hpr:.4f}",
        f"hair_conc={hair_conc:.4f} mg/kg",
        f"melanin_effect={melanin_effect}",
        f"significance={forensic_significance}",
    ]
    if melanin_effect:
        notes_parts.append(f"pKa={pKa} > 7 enhances melanin binding")
    notes = "; ".join(notes_parts)

    return HairDrugConcentrationResult(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        ionization_type=ionization_type,
        plasma_conc_mg_L=plasma_conc_mg_L,
        hair_plasma_ratio=hpr,
        hair_conc_mg_kg=hair_conc,
        detectable_in_hair=detectable,
        detection_window_months=detection_window,
        melanin_effect=melanin_effect,
        forensic_significance=forensic_significance,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch forensic screening
# ---------------------------------------------------------------------------


def screen_forensic_detectability(
    compounds: list[dict],
) -> list[HairDrugConcentrationResult]:
    """Screen multiple compounds for forensic detectability in hair.

    Each dict in compounds must have keys:
        drug_name, logP, pKa, ionization_type, plasma_conc_mg_L

    Returns results sorted by hair_plasma_ratio descending.
    """
    results = [
        predict_hair_concentration(
            drug_name=c["drug_name"],
            logP=c["logP"],
            pKa=c["pKa"],
            ionization_type=c["ionization_type"],
            plasma_conc_mg_L=c["plasma_conc_mg_L"],
        )
        for c in compounds
    ]
    results.sort(key=lambda r: r.hair_plasma_ratio, reverse=True)
    return results
