"""
Phase 899 — Tear Fluid Drug Excretion

Models drug excretion via tear fluid for therapeutic drug monitoring,
ocular pharmacology, and contact lens drug delivery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TearFluidExcretionResult:
    """Result of tear fluid drug excretion prediction."""

    drug_name: str
    pka: float
    logp: float
    ionization_type: str
    tear_ph: float
    tear_plasma_ratio: float
    predicted_tear_conc_ug_mL: float
    plasma_conc_input_mg_L: float
    tear_volume_uL: float
    drug_amount_in_tears_ng: float
    contact_lens_accumulation_factor: float
    therapeutic_ocular_potential: bool
    notes: str


def predict_tear_fluid_excretion(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str = "neutral",
    plasma_conc_mg_L: float = 1.0,
    tear_ph: float = 7.4,
    fu_plasma: float | None = None,
) -> TearFluidExcretionResult:
    """
    Predict drug concentration in tear fluid from plasma concentration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pka:
        Acid dissociation constant (0–14).
    logp:
        Octanol-water partition coefficient.
    ionization_type:
        One of "acid", "base", or "neutral".
    plasma_conc_mg_L:
        Free plasma concentration in mg/L (must be > 0).
    tear_ph:
        pH of tear fluid (default 7.4).
    fu_plasma:
        Fraction unbound in plasma; estimated from logP if not provided.

    Returns
    -------
    TearFluidExcretionResult
    """
    # --- Validation ---
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be between 0 and 14, got {pka}")
    if plasma_conc_mg_L <= 0.0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )

    plasma_ph = 7.4

    # --- fu_plasma estimate ---
    if fu_plasma is not None:
        fu_estimate = fu_plasma
    else:
        fu_estimate = max(0.01, min(1.0, 1.0 / (1.0 + 10.0 ** (logp - 2.5))))

    # --- Tear/Plasma ratio via Henderson-Hasselbalch ---
    if math.isclose(tear_ph, plasma_ph, rel_tol=1e-9, abs_tol=1e-12):
        tp_ratio = 1.0
    elif ionization_type == "base":
        tp_ratio = (1.0 + 10.0 ** (pka - tear_ph)) / (1.0 + 10.0 ** (pka - plasma_ph))
    elif ionization_type == "acid":
        tp_ratio = (1.0 + 10.0 ** (tear_ph - pka)) / (1.0 + 10.0 ** (plasma_ph - pka))
    else:
        tp_ratio = 1.0

    # Clamp T/P to [0.1, 10.0]
    tp_ratio = max(0.1, min(10.0, tp_ratio))

    # --- Predicted tear concentration ---
    # plasma_conc_mg_L * 1000 = µg/mL equivalent, then scale by T/P and fu
    predicted_tear_conc_ug_mL = tp_ratio * plasma_conc_mg_L * fu_estimate * 1000.0

    # --- Drug amount in steady-state tear volume ---
    tear_volume_uL = 7.0
    # µg/mL * µL * 1000 → ng
    drug_amount_in_tears_ng = predicted_tear_conc_ug_mL * tear_volume_uL * 1000.0

    # --- Contact lens accumulation (HEMA matrix, lipophilic) ---
    if logp > 0.0:
        contact_lens_accumulation_factor = max(1.0, logp * 2.0)
    else:
        contact_lens_accumulation_factor = 1.0

    # --- Therapeutic ocular potential ---
    therapeutic_ocular_potential = predicted_tear_conc_ug_mL >= 0.1

    # --- Notes ---
    if therapeutic_ocular_potential:
        notes = (
            "Therapeutic tear concentrations achievable — consider as non-invasive "
            "ocular drug monitoring"
        )
    elif contact_lens_accumulation_factor > 5.0:
        notes = "High contact lens accumulation — ocular drug delivery via lens possible"
    else:
        notes = "Tear drug monitoring feasible for systemic TDM purposes"

    return TearFluidExcretionResult(
        drug_name=drug_name,
        pka=pka,
        logp=logp,
        ionization_type=ionization_type,
        tear_ph=tear_ph,
        tear_plasma_ratio=tp_ratio,
        predicted_tear_conc_ug_mL=predicted_tear_conc_ug_mL,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        tear_volume_uL=tear_volume_uL,
        drug_amount_in_tears_ng=drug_amount_in_tears_ng,
        contact_lens_accumulation_factor=contact_lens_accumulation_factor,
        therapeutic_ocular_potential=therapeutic_ocular_potential,
        notes=notes,
    )


def screen_tear_tdm(candidates: list[dict]) -> list[TearFluidExcretionResult]:
    """
    Screen a list of drug candidates for tear fluid TDM suitability.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name", "pka", "logp", and optionally
        "ionization_type" (default "neutral").

    Returns
    -------
    List of TearFluidExcretionResult sorted by tear_plasma_ratio descending.
    """
    results: list[TearFluidExcretionResult] = []
    for c in candidates:
        result = predict_tear_fluid_excretion(
            drug_name=c["name"],
            pka=c["pka"],
            logp=c["logp"],
            ionization_type=c.get("ionization_type", "neutral"),
        )
        results.append(result)

    results.sort(key=lambda r: r.tear_plasma_ratio, reverse=True)
    return results
