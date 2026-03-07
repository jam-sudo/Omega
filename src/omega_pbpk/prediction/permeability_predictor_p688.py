"""Membrane permeability prediction from physicochemical properties (Phase 688).

Predicts Caco-2, MDCK, and PAMPA permeability values and provides
BCS-based permeability classification and fraction absorbed estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PermeabilityResult",
    "predict_caco2_papp",
    "classify_permeability",
    "predict_mdck_papp",
    "predict_pampa_papp",
    "predict_permeability",
    "screen_permeability",
]


@dataclass(frozen=True)
class PermeabilityResult:
    """Result of permeability prediction (Phase 688)."""

    compound_name: str
    logP: float
    mw: float
    psa: float
    papp_caco2: float
    papp_mdck: float
    papp_pampa: float
    permeability_class: str
    bcs_permeability_class: str
    fa_predicted: float
    notes: str


def predict_caco2_papp(
    logP: float,
    mw: float,
    psa: float,
    n_h_donors: int = 1,
    n_h_acceptors: int = 2,
    charge: float = 0,
) -> float:
    """Predict Caco-2 apparent permeability (Papp) in units of x10^-6 cm/s.

    Args:
        logP: Lipophilicity (octanol-water partition coefficient).
        mw: Molecular weight in g/mol.
        psa: Polar surface area in Angstrom squared.
        n_h_donors: Number of hydrogen bond donors.
        n_h_acceptors: Number of hydrogen bond acceptors (unused, kept for API).
        charge: Formal molecular charge.

    Returns:
        Predicted Papp in x10^-6 cm/s (float).
    """
    # Base Papp from logP: papp_base = 10^(0.5 * logP - 1.0)
    papp_base = 10 ** (0.5 * logP - 1.0)
    papp_base = max(0.01, min(300.0, papp_base))

    # Penalty factors
    factor_mw = math.exp(-max(0.0, mw - 300.0) * 0.003)
    factor_psa = math.exp(-max(0.0, psa - 60.0) * 0.02)
    factor_hd = math.exp(-max(0.0, n_h_donors - 1) * 0.3)
    factor_charge = 0.3 if abs(charge) >= 1 else 1.0

    papp = papp_base * factor_mw * factor_psa * factor_hd * factor_charge
    return papp


def classify_permeability(papp_1e6_cm_per_s: float) -> str:
    """Classify permeability based on BCS thresholds.

    Args:
        papp_1e6_cm_per_s: Papp in x10^-6 cm/s.

    Returns:
        "low", "moderate", or "high".
    """
    if papp_1e6_cm_per_s < 1.0:
        return "low"
    if papp_1e6_cm_per_s < 10.0:
        return "moderate"
    return "high"


def predict_mdck_papp(logP: float, mw: float, psa: float) -> float:
    """Predict MDCK apparent permeability (x10^-6 cm/s).

    MDCK permeability is typically 2x Caco-2 due to lower efflux.

    Args:
        logP: Lipophilicity.
        mw: Molecular weight in g/mol.
        psa: Polar surface area in Angstrom squared.

    Returns:
        Predicted MDCK Papp in x10^-6 cm/s.
    """
    papp_caco2 = predict_caco2_papp(logP, mw, psa)
    return papp_caco2 * 2.0


def predict_pampa_papp(logP: float, mw: float, psa: float) -> float:
    """Predict PAMPA apparent permeability (x10^-6 cm/s).

    PAMPA measures passive permeability only (no efflux transporters).

    Args:
        logP: Lipophilicity.
        mw: Molecular weight in g/mol.
        psa: Polar surface area in Angstrom squared.

    Returns:
        Predicted PAMPA Papp in x10^-6 cm/s.
    """
    # Slightly higher base (passive only, no P-gp; different intercept)
    papp_base = 10 ** (0.5 * logP - 0.8)
    papp_base = max(0.001, min(500.0, papp_base))

    # Only MW and PSA penalties (no charge penalty for PAMPA)
    factor_mw = math.exp(-max(0.0, mw - 300.0) * 0.003)
    factor_psa = math.exp(-max(0.0, psa - 60.0) * 0.02)

    return papp_base * factor_mw * factor_psa


def predict_permeability(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float = 80.0,
    n_h_donors: int = 1,
    charge: float = 0,
) -> PermeabilityResult:
    """Predict comprehensive membrane permeability profile.

    Args:
        compound_name: Name of the compound.
        logP: Lipophilicity.
        mw: Molecular weight in g/mol.
        psa: Polar surface area in Angstrom squared.
        n_h_donors: Number of hydrogen bond donors.
        charge: Formal molecular charge.

    Returns:
        PermeabilityResult with all permeability metrics.

    Raises:
        ValueError: If mw <= 0 or psa < 0.
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")

    papp_caco2 = predict_caco2_papp(logP, mw, psa, n_h_donors=n_h_donors, charge=charge)
    papp_mdck = predict_mdck_papp(logP, mw, psa)
    papp_pampa = predict_pampa_papp(logP, mw, psa)

    perm_class = classify_permeability(papp_caco2)

    # Simplified sigmoidal fa prediction
    fa = min(0.99, papp_caco2 / (papp_caco2 + 3.0))

    notes = (
        f"Caco-2={papp_caco2:.2f}, MDCK={papp_mdck:.2f}, PAMPA={papp_pampa:.2f} "
        f"(x10^-6 cm/s). Class: {perm_class}. Predicted fa={fa:.2f}."
    )

    return PermeabilityResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        papp_caco2=papp_caco2,
        papp_mdck=papp_mdck,
        papp_pampa=papp_pampa,
        permeability_class=perm_class,
        bcs_permeability_class=perm_class,
        fa_predicted=fa,
        notes=notes,
    )


def screen_permeability(compounds: list) -> list:
    """Screen a list of compounds for permeability, sorted by Caco-2 Papp descending.

    Each compound dict must have keys: compound_name, logP, mw.
    Optional keys: psa (default 80.0), n_h_donors (default 1), charge (default 0).

    Args:
        compounds: List of dicts with compound properties.

    Returns:
        List of PermeabilityResult sorted by papp_caco2 descending.
    """
    if not compounds:
        return []

    results = []
    for cpd in compounds:
        result = predict_permeability(
            compound_name=cpd["compound_name"],
            logP=cpd["logP"],
            mw=cpd["mw"],
            psa=cpd.get("psa", 80.0),
            n_h_donors=cpd.get("n_h_donors", 1),
            charge=cpd.get("charge", 0),
        )
        results.append(result)

    results.sort(key=lambda r: r.papp_caco2, reverse=True)
    return results
