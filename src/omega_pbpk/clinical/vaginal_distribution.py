"""Phase 911 — Drug Vaginal Distribution.

Models drug distribution via vaginal route for local antifungals,
antiviral microbicides, and contraceptive drugs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VaginalDistributionResult:
    """Result of vaginal drug distribution prediction."""

    drug_name: str
    pka: float
    logp: float
    ionization_type: str
    vaginal_ph: float
    vaginal_plasma_ratio: float
    predicted_vaginal_conc_ug_mL: float
    systemic_absorption_fraction: float
    plasma_conc_mg_L: float
    local_selectivity_index: float
    dose_mg: float
    antifungal_local_conc_adequate: bool
    notes: str


def predict_vaginal_distribution(
    drug_name: str,
    dose_mg: float,
    pka: float,
    logp: float,
    ionization_type: str = "acid",
    vaginal_ph: float = 4.2,
    vd_L: float = 30.0,
    cl_L_per_h: float = 5.0,
) -> VaginalDistributionResult:
    """Predict vaginal drug distribution parameters.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered vaginally in mg.
    pka:
        Drug pKa value.
    logp:
        Octanol-water partition coefficient (log scale).
    ionization_type:
        One of "acid", "base", "neutral".
    vaginal_ph:
        Vaginal fluid pH (typical range 3.8-4.5, default 4.2).
    vd_L:
        Volume of distribution (L).
    cl_L_per_h:
        Systemic clearance (L/h).

    Returns
    -------
    VaginalDistributionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 <= pka <= 14):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    valid_types = {"acid", "base", "neutral"}
    if ionization_type not in valid_types:
        raise ValueError(f"ionization_type must be one of {valid_types}, got {ionization_type!r}")

    plasma_ph = 7.4

    # Henderson-Hasselbalch local:plasma ratio
    if ionization_type == "base":
        vaginal_plasma_ratio = (1.0 + 10.0 ** (pka - vaginal_ph)) / (
            1.0 + 10.0 ** (pka - plasma_ph)
        )
    elif ionization_type == "acid":
        vaginal_plasma_ratio = (1.0 + 10.0 ** (vaginal_ph - pka)) / (
            1.0 + 10.0 ** (plasma_ph - pka)
        )
    else:  # neutral
        vaginal_plasma_ratio = 1.0

    # Clamp ratio
    vaginal_plasma_ratio = max(0.5, min(200.0, vaginal_plasma_ratio))

    # Predicted vaginal concentration (µg/mL)
    predicted_vaginal_conc_ug_mL = dose_mg * vaginal_plasma_ratio * 50.0
    predicted_vaginal_conc_ug_mL = max(0.001, min(1e5, predicted_vaginal_conc_ug_mL))

    # Systemic absorption fraction
    base_absorption = {"acid": 0.05, "base": 0.2, "neutral": 0.1}[ionization_type]
    logp_modifier = 1.0 + max(0.0, logp - 2.0) * 0.1
    systemic_absorption_fraction = base_absorption * logp_modifier
    systemic_absorption_fraction = max(0.01, min(0.8, systemic_absorption_fraction))

    # Plasma concentration from vaginal dose
    plasma_conc_mg_L = dose_mg * systemic_absorption_fraction / vd_L
    plasma_conc_mg_L = max(0.0, min(100.0, plasma_conc_mg_L))

    # Local selectivity index (vaginal µg/mL vs plasma µg/mL)
    # plasma_conc_mg_L * 1000 converts to µg/mL
    local_selectivity_index = predicted_vaginal_conc_ug_mL / (plasma_conc_mg_L * 1000.0 + 0.001)
    local_selectivity_index = max(0.1, min(1000.0, local_selectivity_index))

    # Adequacy for antifungal therapy
    antifungal_local_conc_adequate = predicted_vaginal_conc_ug_mL >= 1.0

    # Notes
    note_parts = []
    if antifungal_local_conc_adequate:
        note_parts.append("Adequate local vaginal concentrations for antifungal therapy")
    if local_selectivity_index > 10.0:
        note_parts.append("High local selectivity — minimal systemic exposure, good safety profile")
    else:
        note_parts.append("Moderate local:systemic ratio — monitor for systemic effects")
    notes = "; ".join(note_parts)

    return VaginalDistributionResult(
        drug_name=drug_name,
        pka=pka,
        logp=logp,
        ionization_type=ionization_type,
        vaginal_ph=vaginal_ph,
        vaginal_plasma_ratio=vaginal_plasma_ratio,
        predicted_vaginal_conc_ug_mL=predicted_vaginal_conc_ug_mL,
        systemic_absorption_fraction=systemic_absorption_fraction,
        plasma_conc_mg_L=plasma_conc_mg_L,
        local_selectivity_index=local_selectivity_index,
        dose_mg=dose_mg,
        antifungal_local_conc_adequate=antifungal_local_conc_adequate,
        notes=notes,
    )


def screen_vaginal_drugs(candidates: list[dict]) -> list[VaginalDistributionResult]:
    """Screen a list of drug candidates for vaginal delivery suitability.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name", "pka", "logp", "dose_mg",
        and optional "ionization_type" (default "acid").

    Returns
    -------
    List of VaginalDistributionResult sorted by local_selectivity_index descending.
    """
    results = []
    for cand in candidates:
        result = predict_vaginal_distribution(
            drug_name=cand["name"],
            dose_mg=cand["dose_mg"],
            pka=cand["pka"],
            logp=cand["logp"],
            ionization_type=cand.get("ionization_type", "acid"),
        )
        results.append(result)
    results.sort(key=lambda r: r.local_selectivity_index, reverse=True)
    return results
