"""Phase 556 - Blood-Brain Barrier Permeability Prediction.

Predicts BBB permeability (PS product, logPS) from physicochemical
properties using the Clark/Ecker model.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BBBPermeabilityResult:
    """Result of BBB permeability prediction."""

    drug_name: str
    logP: float
    mw_Da: float
    hbd: int
    psa_A2: float
    log_ps_cm3_per_s: float
    ps_cm3_per_s: float
    cns_penetration_class: str
    kp_brain: float
    fraction_unbound_brain: float
    cns_mpo_score: float
    notes: str


def predict_bbb_permeability(
    drug_name: str,
    logP: float,
    mw_Da: float,
    hbd: int,
    psa_A2: float,
    pKa: float = 7.0,
) -> BBBPermeabilityResult:
    """Predict BBB permeability from physicochemical properties.

    Args:
        drug_name: Name of the drug.
        logP: Octanol-water partition coefficient (log scale).
        mw_Da: Molecular weight in Daltons (must be > 0).
        hbd: Number of hydrogen bond donors (must be >= 0).
        psa_A2: Polar surface area in Angstrom^2 (must be >= 0).
        pKa: Acid dissociation constant (default 7.0).

    Returns:
        BBBPermeabilityResult with permeability predictions.
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if hbd < 0:
        raise ValueError(f"hbd must be >= 0, got {hbd}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")

    # Clark model: log_PS
    log_ps_raw = 0.152 * logP - 0.0148 * mw_Da - 0.139 * hbd - 0.138 * psa_A2 + 0.233
    log_ps = max(-6.0, min(1.0, log_ps_raw))

    # PS product
    ps = math.pow(10, log_ps)

    # CNS penetration class
    if log_ps > -1:
        cns_class = "high"
    elif log_ps > -3:
        cns_class = "moderate"
    elif log_ps > -5:
        cns_class = "low"
    else:
        cns_class = "none"

    # Brain-plasma partition (lipophilicity-driven)
    kp_brain_raw = 1.0 + max(0.0, logP - 1.0) * 0.5
    kp_brain = max(0.01, min(100.0, kp_brain_raw))

    # Fraction unbound in brain
    fu_brain_raw = 1.0 / (1.0 + 0.5 * max(0.0, logP))
    fu_brain = max(0.01, min(1.0, fu_brain_raw))

    # CNS MPO score (simplified, 6-point)
    mpo = 0.0
    if logP <= 3:
        mpo += 1.0
    if logP >= 0:
        mpo += 1.0
    if mw_Da <= 360:
        mpo += 1.0
    if psa_A2 <= 90:
        mpo += 1.0
    if hbd <= 0:
        mpo += 1.0
    if pKa <= 10:
        mpo += 1.0

    notes = f"Clark/Ecker model; logP={logP}, MW={mw_Da}, HBD={hbd}, PSA={psa_A2}, pKa={pKa}"

    return BBBPermeabilityResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        hbd=hbd,
        psa_A2=psa_A2,
        log_ps_cm3_per_s=log_ps,
        ps_cm3_per_s=ps,
        cns_penetration_class=cns_class,
        kp_brain=kp_brain,
        fraction_unbound_brain=fu_brain,
        cns_mpo_score=mpo,
        notes=notes,
    )


def screen_bbb_permeability(compounds: list) -> list:
    """Screen multiple compounds for BBB permeability.

    Args:
        compounds: List of dicts with keys: drug_name, logP, mw_Da, hbd,
                   psa_A2, and optionally pKa.

    Returns:
        List of BBBPermeabilityResult sorted by log_ps descending.
    """
    results = []
    for comp in compounds:
        pKa = comp.get("pKa", 7.0)
        result = predict_bbb_permeability(
            drug_name=comp["drug_name"],
            logP=comp["logP"],
            mw_Da=comp["mw_Da"],
            hbd=comp["hbd"],
            psa_A2=comp["psa_A2"],
            pKa=pKa,
        )
        results.append(result)
    results.sort(key=lambda r: r.log_ps_cm3_per_s, reverse=True)
    return results
