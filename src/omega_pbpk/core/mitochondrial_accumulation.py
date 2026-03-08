"""
Mitochondrial Drug Accumulation Model (Phase 881)

Models accumulation of lipophilic cations in mitochondria driven by
mitochondrial membrane potential (ΔΨm). Relevant for cationic amphiphiles
and DILI risk assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants
_F = 96485.0  # Faraday constant, C/mol
_R = 8.314  # Gas constant, J/(mol·K)


@dataclass
class MitochondrialAccumulationResult:
    """Result of mitochondrial drug accumulation prediction."""

    drug_name: str
    dose_mg: float
    delta_psi_mV: float
    charge: int
    logp: float
    nernst_ratio: float
    mitochondrial_conc_factor: float
    dili_risk_score: float
    dili_risk_level: str
    predicted_mitochondrial_auc_ratio: float
    notes: str


def predict_mitochondrial_accumulation(
    drug_name: str,
    dose_mg: float,
    charge: int = 1,
    logp: float = 2.0,
    delta_psi_mV: float = -180.0,
    temp_K: float = 310.0,
) -> MitochondrialAccumulationResult:
    """
    Predict mitochondrial drug accumulation using the Nernst equation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    charge : int
        Cation charge (e.g., +1).
    logp : float
        Lipophilicity (log P).
    delta_psi_mV : float
        Mitochondrial membrane potential in mV (must be negative).
    temp_K : float
        Temperature in Kelvin (default 310 K = 37 °C).

    Returns
    -------
    MitochondrialAccumulationResult
    """
    if not isinstance(charge, int):
        raise ValueError("charge must be an integer")
    if delta_psi_mV >= 0:
        raise ValueError("delta_psi_mV must be negative (polarized membrane)")

    # Nernst ratio: concentration_mito / concentration_cytoplasm
    # exp(charge * F * |ΔΨm| / (R * T))
    nernst_exponent = charge * _F * abs(delta_psi_mV) * 1e-3 / (_R * temp_K)
    nernst_ratio = math.exp(nernst_exponent)

    # logP modifies actual uptake
    if logp > 0:
        mitochondrial_conc_factor = nernst_ratio * (1.0 + logp / 10.0)
    else:
        mitochondrial_conc_factor = nernst_ratio

    # Clamp to [1.0, 1e6]
    mitochondrial_conc_factor = max(1.0, min(1e6, mitochondrial_conc_factor))

    # DILI risk score
    log_factor = math.log10(mitochondrial_conc_factor + 1.0) / math.log10(1e5) * 100.0
    dili_risk_score = log_factor + logp * 2.0
    dili_risk_score = max(0.0, min(100.0, dili_risk_score))

    # Risk level
    if dili_risk_score > 60:
        dili_risk_level = "high"
        notes = "High mitochondrial accumulation — DILI risk. Monitor hepatotoxicity markers."
    elif dili_risk_score > 30:
        dili_risk_level = "moderate"
        notes = "Moderate accumulation. Consider monitoring."
    else:
        dili_risk_level = "low"
        notes = "Low mitochondrial accumulation risk."

    # Predicted mito AUC ratio (10% mitochondrial volume fraction)
    predicted_mitochondrial_auc_ratio = mitochondrial_conc_factor * 0.1

    return MitochondrialAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        delta_psi_mV=delta_psi_mV,
        charge=charge,
        logp=logp,
        nernst_ratio=nernst_ratio,
        mitochondrial_conc_factor=mitochondrial_conc_factor,
        dili_risk_score=dili_risk_score,
        dili_risk_level=dili_risk_level,
        predicted_mitochondrial_auc_ratio=predicted_mitochondrial_auc_ratio,
        notes=notes,
    )


def screen_mito_toxicity(
    candidates: list[dict],
) -> list[MitochondrialAccumulationResult]:
    """
    Screen a list of candidate compounds for mitochondrial toxicity.

    Parameters
    ----------
    candidates : list[dict]
        Each dict must have "name" and "dose_mg"; optionally "charge" (default 1)
        and "logp" (default 1.0).

    Returns
    -------
    list[MitochondrialAccumulationResult]
        Sorted by dili_risk_score descending.
    """
    results = []
    for candidate in candidates:
        name = candidate["name"]
        dose_mg = candidate["dose_mg"]
        charge = int(candidate.get("charge", 1))
        logp = float(candidate.get("logp", 1.0))
        result = predict_mitochondrial_accumulation(
            drug_name=name,
            dose_mg=dose_mg,
            charge=charge,
            logp=logp,
        )
        results.append(result)

    results.sort(key=lambda r: r.dili_risk_score, reverse=True)
    return results
