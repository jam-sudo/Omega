"""Phase 1039 — Drug Corneal Penetration prediction for ophthalmic formulations."""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class CornealPenetrationResult:
    drug_name: str
    mw_Da: float
    logp: float
    pka: float
    ionization_type: str
    corneal_perm_cm_s: float
    aqueous_humor_conc_mg_mL: float
    bioavailability_ocular: float
    penetration_class: str
    lipophilicity_optimal: bool
    ionization_penalty: float
    predicted_tmax_min: float
    notes: str


def predict_corneal_penetration(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    dose_mg_mL: float = 0.5,
    drop_volume_uL: float = 30.0,
    nasolacrimal_fraction: float = 0.7,
    ph_formulation: float = 7.4,
) -> CornealPenetrationResult:
    """Predict drug penetration through the cornea for ophthalmic formulations.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight in Daltons (must be > 0).
    logp:
        Octanol-water partition coefficient.
    pka:
        Acid/base dissociation constant (must be in [0, 14]).
    ionization_type:
        One of "acid", "base", "neutral".
    dose_mg_mL:
        Drop concentration in mg/mL (must be > 0).
    drop_volume_uL:
        Instilled drop volume in microliters (must be > 0).
    nasolacrimal_fraction:
        Fraction of dose lost to nasolacrimal drainage (must be in [0, 1)).
    ph_formulation:
        pH of the formulation.

    Returns
    -------
    CornealPenetrationResult
    """
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if dose_mg_mL <= 0:
        raise ValueError(f"dose_mg_mL must be > 0, got {dose_mg_mL}")
    if drop_volume_uL <= 0:
        raise ValueError(f"drop_volume_uL must be > 0, got {drop_volume_uL}")
    if not (0.0 <= nasolacrimal_fraction < 1.0):
        raise ValueError(f"nasolacrimal_fraction must be in [0, 1), got {nasolacrimal_fraction}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")

    # --- Ionization at formulation pH ---
    if ionization_type == "acid":
        ionized_fraction = 1.0 / (1.0 + 10 ** (pka - ph_formulation))
    elif ionization_type == "base":
        ionized_fraction = 1.0 / (1.0 + 10 ** (ph_formulation - pka))
    else:  # neutral
        ionized_fraction = 0.0

    unionized_fraction = 1.0 - ionized_fraction
    ionization_penalty = ionized_fraction

    # --- Corneal permeability (cm/s) — Prausnitz & Noonan 1998 ---
    papp_neutral = 10 ** (-2.0 - 0.01 * mw_Da + 0.5 * logp)
    papp_neutral = max(1e-8, min(1e-3, papp_neutral))

    corneal_perm = papp_neutral * (unionized_fraction + ionized_fraction * 0.01)
    corneal_perm = max(1e-8, corneal_perm)

    # --- Dose on cornea ---
    absorbed_dose_mg = dose_mg_mL * drop_volume_uL / 1000.0
    available_fraction = 1.0 - nasolacrimal_fraction
    dose_on_cornea_mg = absorbed_dose_mg * available_fraction  # noqa: F841 — kept for clarity

    # --- Simple bioavailability model ---
    # Corneal surface area ~1 cm², corneal volume ~0.05 mL, transit time ~20 min
    k_perm = corneal_perm * 3600.0 * 1.0 / (0.05 * 1.0)  # per hour
    bioavailability_ocular = min(0.1, k_perm / (k_perm + 30.0))

    # --- Aqueous humor concentration (AH volume = 0.25 mL = 0.00025 L) ---
    aqueous_humor_conc = absorbed_dose_mg * available_fraction * bioavailability_ocular / 0.00025

    # --- tmax ---
    predicted_tmax_min = max(5.0, 60.0 / (k_perm + 0.1))

    # --- Penetration class ---
    if bioavailability_ocular > 0.05:
        penetration_class = "excellent"
    elif bioavailability_ocular > 0.02:
        penetration_class = "good"
    elif bioavailability_ocular > 0.005:
        penetration_class = "moderate"
    else:
        penetration_class = "poor"

    # --- Lipophilicity optimal range ---
    lipophilicity_optimal = 1.0 <= logp <= 3.0

    # --- Notes ---
    notes_parts = [
        f"Prausnitz & Noonan permeability model; ionization_type={ionization_type}.",
        f"Penetration class: {penetration_class}.",
    ]
    if not lipophilicity_optimal:
        notes_parts.append("logP outside optimal range [1, 3]; consider formulation adjustment.")
    if ionization_penalty > 0.5:
        notes_parts.append("High ionization penalty; unionized fraction is low at formulation pH.")
    notes = " ".join(notes_parts)

    return CornealPenetrationResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        corneal_perm_cm_s=corneal_perm,
        aqueous_humor_conc_mg_mL=aqueous_humor_conc,
        bioavailability_ocular=bioavailability_ocular,
        penetration_class=penetration_class,
        lipophilicity_optimal=lipophilicity_optimal,
        ionization_penalty=ionization_penalty,
        predicted_tmax_min=predicted_tmax_min,
        notes=notes,
    )


# Keep math in namespace (used implicitly via math.log10 through `10 **`)
_ = math  # prevent unused import warning
