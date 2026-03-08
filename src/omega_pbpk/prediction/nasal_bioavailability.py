"""Phase 904 — Drug Nasal Bioavailability Calculator.

Predicts absolute nasal bioavailability from physicochemical properties.
Nasal route can offer higher bioavailability than oral for some drugs
by bypassing first-pass metabolism.

Pure Python only — stdlib math and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class NasalBioavailabilityResult:
    """Result of nasal bioavailability prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    pka: float | None
    ionization_type: str

    nasal_ph: float
    """Nasal mucosal pH (typically 6.8-7.4)."""

    f_nasal: float
    """Predicted nasal bioavailability fraction (0-1)."""

    nasal_vs_oral_advantage: float
    """Ratio of f_nasal to f_oral_estimate (>1 favours nasal route)."""

    f_oral_estimate: float
    """Estimated oral bioavailability for comparison (0-1)."""

    absorption_rate_class: str
    """'fast', 'moderate', or 'slow' based on permeation coefficient."""

    mucociliary_loss_fraction: float
    """Fraction lost to mucociliary clearance before absorption (0-1)."""

    permeation_coefficient_cm_s: float
    """Estimated nasal Papp (cm/s)."""

    suitable_for_systemic_delivery: bool
    """True if f_nasal > 0.3."""

    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Return value clamped to [lo, hi]."""
    return max(lo, min(hi, value))


def predict_nasal_bioavailability(
    drug_name: str,
    logp: float = 2.0,
    mw_Da: float = 300.0,
    pka: float | None = None,
    ionization_type: str = "neutral",
    nasal_ph: float = 7.0,
    fu_plasma: float = 0.5,
) -> NasalBioavailabilityResult:
    """Predict nasal bioavailability for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logp:
        Lipophilicity (log P octanol/water).
    mw_Da:
        Molecular weight in Daltons.
    pka:
        Acid dissociation constant (optional).
    ionization_type:
        One of 'acid', 'base', or 'neutral'.
    nasal_ph:
        Nasal mucosal pH (default 7.0, typical range 6.8-7.4).
    fu_plasma:
        Unbound fraction in plasma (0-1); used for context.

    Returns
    -------
    NasalBioavailabilityResult
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )

    # --- Nasal permeation coefficient (cm/s) ---
    log_papp = -6.5 + logp * 0.5 - mw_Da / 1000.0
    permeation_coefficient_cm_s = _clamp(10.0**log_papp, 1e-8, 1e-3)

    # --- Mucociliary clearance loss ---
    if mw_Da > 1000:
        mucociliary_loss_fraction = 0.8
    elif mw_Da > 500:
        mucociliary_loss_fraction = 0.5
    else:
        mucociliary_loss_fraction = 0.3

    # --- Nasal bioavailability ---
    f_nasal_raw = permeation_coefficient_cm_s * 1e5 * (1.0 - mucociliary_loss_fraction)
    f_nasal = _clamp(f_nasal_raw, 0.01, 0.95)

    # --- Oral bioavailability estimate ---
    # Decreases at very high or very low logP extremes
    f_oral_raw = (1.0 / (1.0 + 10.0 ** (logp - 3.0))) * 0.7
    f_oral_estimate = _clamp(f_oral_raw, 0.02, 0.95)

    # --- Nasal vs oral advantage ---
    nasal_vs_oral_advantage = _clamp(
        f_nasal / f_oral_estimate if f_oral_estimate > 0 else 10.0,
        0.1,
        10.0,
    )

    # --- Absorption rate class ---
    if permeation_coefficient_cm_s > 1e-5:
        absorption_rate_class = "fast"
    elif permeation_coefficient_cm_s > 1e-6:
        absorption_rate_class = "moderate"
    else:
        absorption_rate_class = "slow"

    # --- Suitability flag ---
    suitable_for_systemic_delivery = f_nasal > 0.3

    # --- Notes (order matters: advantage > 2 takes priority) ---
    if nasal_vs_oral_advantage > 2.0:
        notes = "Nasal route advantageous — higher bioavailability than oral, bypasses first-pass"
    elif suitable_for_systemic_delivery:
        notes = "Suitable for systemic nasal delivery"
    else:
        notes = (
            "Limited nasal bioavailability — consider formulation enhancement "
            "(absorption enhancers)"
        )

    return NasalBioavailabilityResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        pka=pka,
        ionization_type=ionization_type,
        nasal_ph=nasal_ph,
        f_nasal=f_nasal,
        nasal_vs_oral_advantage=nasal_vs_oral_advantage,
        f_oral_estimate=f_oral_estimate,
        absorption_rate_class=absorption_rate_class,
        mucociliary_loss_fraction=mucociliary_loss_fraction,
        permeation_coefficient_cm_s=permeation_coefficient_cm_s,
        suitable_for_systemic_delivery=suitable_for_systemic_delivery,
        notes=notes,
    )


def screen_nasal_candidates(candidates: list[dict]) -> list[NasalBioavailabilityResult]:
    """Screen a list of candidate drugs for nasal delivery suitability.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name" (required), "logp" (required),
        "mw_Da" (optional, default 300), "ionization_type" (optional,
        default 'neutral'), "pka" (optional), "nasal_ph" (optional).

    Returns
    -------
    List of NasalBioavailabilityResult sorted by f_nasal descending.
    """
    results = []
    for cand in candidates:
        result = predict_nasal_bioavailability(
            drug_name=cand["name"],
            logp=cand["logp"],
            mw_Da=float(cand.get("mw_Da", 300.0)),
            pka=cand.get("pka"),
            ionization_type=cand.get("ionization_type", "neutral"),
            nasal_ph=float(cand.get("nasal_ph", 7.0)),
        )
        results.append(result)

    results.sort(key=lambda r: r.f_nasal, reverse=True)
    return results
