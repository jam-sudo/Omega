"""Phase 430 — Dissolution medium effect prediction.

Predict drug dissolution in different biorelevant media:
FaSSIF, FeSSIF, FaSSGF, SGF, phosphate buffer, water.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Media database
# ---------------------------------------------------------------------------

MEDIA = {
    "FaSSIF": {"ph": 6.5, "bile_conc_mM": 3.0, "lecithin_mM": 0.75, "volume_mL": 250},
    "FeSSIF": {"ph": 5.0, "bile_conc_mM": 15.0, "lecithin_mM": 3.75, "volume_mL": 250},
    "FaSSGF": {"ph": 1.6, "bile_conc_mM": 0.08, "lecithin_mM": 0.02, "volume_mL": 250},
    "SGF": {"ph": 1.2, "bile_conc_mM": 0.0, "lecithin_mM": 0.0, "volume_mL": 250},
    "phosphate_buffer_pH68": {"ph": 6.8, "bile_conc_mM": 0.0, "lecithin_mM": 0.0, "volume_mL": 250},
    "water": {"ph": 7.0, "bile_conc_mM": 0.0, "lecithin_mM": 0.0, "volume_mL": 250},
}

VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DissolutionMediumResult:
    """Result of dissolution medium effect prediction."""

    drug_name: str
    medium: str
    drug_pka: float
    logp: float
    dose_mg: float
    intrinsic_solubility_mg_mL: float
    medium_solubility_mg_mL: float
    solubility_enhancement: float
    dose_number: float
    dissolution_class: str
    predicted_f_abs: float
    biorelevant_factor: str
    notes: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def predict_dissolution_medium(
    drug_name: str,
    medium: str,
    drug_pka: float,
    logp: float,
    ionization_type: str,
    dose_mg: float,
    intrinsic_solubility_mg_mL: float,
) -> DissolutionMediumResult:
    """Predict drug dissolution in a specific biorelevant medium.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    medium:
        One of the keys in MEDIA.
    drug_pka:
        Drug pKa value.
    logp:
        Octanol-water partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    dose_mg:
        Dose in mg.
    intrinsic_solubility_mg_mL:
        Intrinsic (un-ionized) aqueous solubility in mg/mL.

    Returns
    -------
    DissolutionMediumResult
    """
    # --- Validation ---
    if medium not in MEDIA:
        raise ValueError(f"medium must be one of {list(MEDIA.keys())}, got '{medium}'")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ionization_type not in VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(VALID_IONIZATION_TYPES)}, got '{ionization_type}'"
        )

    props = MEDIA[medium]
    ph = props["ph"]
    bile_conc_mM = props["bile_conc_mM"]

    # --- pH effect (Henderson-Hasselbalch) ---
    if ionization_type == "acid":
        ph_enhancement = 1.0 + 10.0 ** (ph - drug_pka)
        s_ph = intrinsic_solubility_mg_mL * ph_enhancement
    elif ionization_type == "base":
        ph_enhancement = 1.0 + 10.0 ** (drug_pka - ph)
        s_ph = intrinsic_solubility_mg_mL * ph_enhancement
    else:
        s_ph = intrinsic_solubility_mg_mL

    # --- Bile salt micellar solubilization (only for logP > 1) ---
    if logp > 1:
        bile_enhancement = 1.0 + bile_conc_mM * max(0.0, logp - 1.0) * 0.05
    else:
        bile_enhancement = 1.0

    # --- Medium solubility ---
    medium_solubility = s_ph * bile_enhancement
    # Cap at 100x intrinsic
    medium_solubility = min(medium_solubility, intrinsic_solubility_mg_mL * 100.0)

    solubility_enhancement = medium_solubility / intrinsic_solubility_mg_mL

    # --- Dose number ---
    # dose_number = dose_mg / (medium_solubility_mg_mL * 250 mL)
    dose_number = dose_mg / (medium_solubility * 250.0)

    # --- Dissolution class ---
    if dose_number < 1.0:
        dissolution_class = "sink_conditions"
    elif dose_number <= 3.0:
        dissolution_class = "borderline"
    else:
        dissolution_class = "dissolution_limited"

    # --- Predicted fraction absorbed ---
    if dose_number < 1.0:
        predicted_f_abs = 1.0
    elif dose_number <= 3.0:
        predicted_f_abs = 1.0 / (dose_number**0.5)
    else:
        predicted_f_abs = 0.3 + 0.7 / dose_number

    predicted_f_abs = max(0.1, min(1.0, predicted_f_abs))

    # --- Biorelevant factor ---
    if bile_conc_mM > 1.0:
        biorelevant_factor = "bile_salt_enhancement"
    elif abs(ph - 7.0) > 2.0:
        biorelevant_factor = "pH_effect"
    else:
        biorelevant_factor = "none"

    # --- Notes ---
    notes = (
        f"Medium solubility {medium_solubility:.3f} mg/mL "
        f"({solubility_enhancement:.1f}x intrinsic). "
        f"Dose number {dose_number:.2f}: {dissolution_class}. "
        f"Predicted F_abs {predicted_f_abs:.2f}."
    )

    return DissolutionMediumResult(
        drug_name=drug_name,
        medium=medium,
        drug_pka=drug_pka,
        logp=logp,
        dose_mg=dose_mg,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        medium_solubility_mg_mL=medium_solubility,
        solubility_enhancement=solubility_enhancement,
        dose_number=dose_number,
        dissolution_class=dissolution_class,
        predicted_f_abs=predicted_f_abs,
        biorelevant_factor=biorelevant_factor,
        notes=notes,
    )


def compare_media(
    drug_name: str,
    drug_pka: float,
    logp: float,
    ionization_type: str,
    dose_mg: float,
    intrinsic_solubility_mg_mL: float,
) -> list[DissolutionMediumResult]:
    """Compare dissolution across all 6 media, sorted by medium_solubility descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_pka:
        Drug pKa value.
    logp:
        Octanol-water partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    dose_mg:
        Dose in mg.
    intrinsic_solubility_mg_mL:
        Intrinsic aqueous solubility in mg/mL.

    Returns
    -------
    List of DissolutionMediumResult for all 6 media, sorted by medium_solubility descending.
    """
    results = []
    for medium in MEDIA:
        result = predict_dissolution_medium(
            drug_name=drug_name,
            medium=medium,
            drug_pka=drug_pka,
            logp=logp,
            ionization_type=ionization_type,
            dose_mg=dose_mg,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        )
        results.append(result)

    # Sort descending by medium_solubility (highest solubility first)
    results.sort(key=lambda r: r.medium_solubility_mg_mL, reverse=True)
    return results
