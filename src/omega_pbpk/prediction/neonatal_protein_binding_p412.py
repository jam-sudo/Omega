"""
Phase 412 — Neonatal Protein Binding Changes

Predict plasma protein binding in neonates and young infants due to
age-dependent albumin, AAG, and bilirubin displacement.

Pure Python, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Valid drug types
# ---------------------------------------------------------------------------
_VALID_DRUG_TYPES = {"acid_albumin", "base_aag", "neutral"}
_SCREEN_AGES_DAYS = [0, 7, 30, 90, 180, 360]


# ---------------------------------------------------------------------------
# Result dataclass  (frozen — no list/dict fields)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NeonatalProteinBindingResult:
    drug_name: str
    age_days: float
    gestational_age_weeks: float
    drug_type: str  # "acid_albumin", "base_aag", "neutral"
    adult_fup: float
    neonatal_fup: float
    fup_ratio_neonatal_to_adult: float
    albumin_fraction_of_adult: float  # 0-1
    aag_fraction_of_adult: float  # 0-1
    bilirubin_displacement_effect: float  # extra fup increase from bilirubin
    cl_adjustment_factor: float  # expected CL change relative to adult
    vd_adjustment_factor: float  # Vd change
    clinical_notes: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def predict_neonatal_protein_binding(
    drug_name: str,
    drug_type: str,
    adult_fup: float,
    age_days: float,
    gestational_age_weeks: float = 40.0,
    bilirubin_mg_dL: float = 5.0,
) -> NeonatalProteinBindingResult:
    """Predict neonatal plasma protein binding and PK adjustments.

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    drug_type : str
        One of "acid_albumin", "base_aag", "neutral".
    adult_fup : float
        Adult fraction unbound in plasma (0 < fup <= 1).
    age_days : float
        Postnatal age in days (>= 0).
    gestational_age_weeks : float
        Gestational age at birth in weeks [24, 42].
    bilirubin_mg_dL : float
        Plasma bilirubin concentration (mg/dL); >= 0.

    Returns
    -------
    NeonatalProteinBindingResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if not (0 < adult_fup <= 1.0):
        raise ValueError("adult_fup must be in (0, 1]")
    if age_days < 0:
        raise ValueError("age_days must be >= 0")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}, got '{drug_type}'")
    if not (24.0 <= gestational_age_weeks <= 42.0):
        raise ValueError("gestational_age_weeks must be in [24, 42]")
    if bilirubin_mg_dL < 0:
        raise ValueError("bilirubin_mg_dL must be >= 0")

    # ------------------------------------------------------------------
    # Maturation fractions
    # ------------------------------------------------------------------
    # Albumin: ~50% at term birth, approaches 100% over first 6 months
    albumin_fraction = max(0.3, min(1.0, 0.5 + 0.5 * min(age_days, 180.0) / 180.0))

    # AAG: ~20% at birth, matures over ~12 months
    aag_fraction = max(0.2, min(1.0, 0.2 + 0.8 * min(age_days, 360.0) / 360.0))

    # ------------------------------------------------------------------
    # Bilirubin displacement (acid drugs bound to albumin only)
    # ------------------------------------------------------------------
    bilirubin_displacement = 0.0
    if drug_type == "acid_albumin" and bilirubin_mg_dL > 5.0:
        bilirubin_displacement = (bilirubin_mg_dL - 5.0) * 0.01

    # ------------------------------------------------------------------
    # Neonatal fup
    # ------------------------------------------------------------------
    if drug_type == "acid_albumin":
        fup_neo = 1.0 - (1.0 - adult_fup) * albumin_fraction + bilirubin_displacement
    elif drug_type == "base_aag":
        fup_neo = 1.0 - (1.0 - adult_fup) * aag_fraction
    else:  # neutral
        fup_neo = adult_fup

    # Clamp: cannot be less than half adult fup; cannot exceed 1.0
    fup_neo = min(1.0, max(adult_fup * 0.5, fup_neo))

    fup_ratio = fup_neo / adult_fup

    # ------------------------------------------------------------------
    # PK adjustment factors
    # ------------------------------------------------------------------
    # Low-ER drugs: total CL proportional to fup
    cl_adjustment_factor = fup_neo / adult_fup

    # Vd = Vp/fu + Vt/fut  — simplification: higher fu → lower Vd
    vd_adjustment_factor = max(0.5, adult_fup / fup_neo)

    # ------------------------------------------------------------------
    # Clinical notes
    # ------------------------------------------------------------------
    pma = gestational_age_weeks + age_days / 7.0

    if drug_type == "acid_albumin":
        binding_protein = "albumin"
        maturation_val = albumin_fraction
    elif drug_type == "base_aag":
        binding_protein = "alpha-1-acid glycoprotein (AAG)"
        maturation_val = aag_fraction
    else:
        binding_protein = "none (neutral)"
        maturation_val = 1.0

    clinical_notes_parts = [
        f"PMA {pma:.1f} weeks; {binding_protein} at {maturation_val * 100:.0f}% adult level.",
    ]
    if bilirubin_displacement > 0:
        clinical_notes_parts.append(
            f"Bilirubin {bilirubin_mg_dL:.1f} mg/dL displaces drug from albumin "
            f"(+{bilirubin_displacement * 100:.1f}% fup increase)."
        )
    if fup_ratio > 1.5:
        clinical_notes_parts.append(
            "Substantially elevated free fraction; consider dose reduction for narrow-TI drugs."
        )
    elif fup_ratio > 1.2:
        clinical_notes_parts.append("Moderately elevated free fraction; monitor drug levels.")
    else:
        clinical_notes_parts.append("Free fraction close to adult values.")

    clinical_notes = " ".join(clinical_notes_parts)

    notes = (
        f"Albumin fraction={albumin_fraction:.3f}; "
        f"AAG fraction={aag_fraction:.3f}; "
        f"bilirubin_disp={bilirubin_displacement:.4f}; "
        f"fup_neo={fup_neo:.4f}; fup_ratio={fup_ratio:.3f}"
    )

    return NeonatalProteinBindingResult(
        drug_name=drug_name,
        age_days=age_days,
        gestational_age_weeks=gestational_age_weeks,
        drug_type=drug_type,
        adult_fup=adult_fup,
        neonatal_fup=fup_neo,
        fup_ratio_neonatal_to_adult=fup_ratio,
        albumin_fraction_of_adult=albumin_fraction,
        aag_fraction_of_adult=aag_fraction,
        bilirubin_displacement_effect=bilirubin_displacement,
        cl_adjustment_factor=cl_adjustment_factor,
        vd_adjustment_factor=vd_adjustment_factor,
        clinical_notes=clinical_notes,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Age-screen utility
# ---------------------------------------------------------------------------
def screen_ages_protein_binding(
    drug_name: str,
    drug_type: str,
    adult_fup: float,
    gestational_age_weeks: float = 40.0,
) -> list:
    """Screen postnatal ages [0, 7, 30, 90, 180, 360] days.

    Returns list of NeonatalProteinBindingResult sorted by
    neonatal_fup descending (highest unbound fraction first).
    """
    results = []
    for age in _SCREEN_AGES_DAYS:
        r = predict_neonatal_protein_binding(
            drug_name=drug_name,
            drug_type=drug_type,
            adult_fup=adult_fup,
            age_days=float(age),
            gestational_age_weeks=gestational_age_weeks,
        )
        results.append(r)

    results.sort(key=lambda r: r.neonatal_fup, reverse=True)
    return results
