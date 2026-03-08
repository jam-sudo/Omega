"""
Phase 403 — Intestinal Permeability Predictor

Predicts intestinal effective permeability (Peff) from physicochemical properties
using Caco-2 correlation, PAMPA correlation, or physicochemical models.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IntestinalPermeabilityResult:
    """Result of intestinal permeability prediction."""

    drug_name: str
    logp: float
    mw_da: float
    psa_A2: float  # polar surface area in angstrom^2
    n_hbd: int  # H-bond donors
    n_hba: int  # H-bond acceptors
    logpe_predicted: float  # predicted log Peff (Peff in cm/s)
    peff_cm_s: float  # effective permeability in cm/s
    peff_class: str  # "high", "medium", or "low"
    lipinski_violations: int  # count of violations
    lipinski_compliant: bool
    bcs_permeability_class: str  # "high" or "low"
    fa_predicted_pct: float  # predicted fraction absorbed %
    human_intestine_method: str  # method used
    notes: str


def predict_intestinal_permeability(
    drug_name: str,
    logp: float,
    mw_da: float,
    psa_A2: float,
    n_hbd: int = 1,
    n_hba: int = 3,
    caco2_papp_cm_s: float = None,
    pampa_peff_cm_s: float = None,
) -> IntestinalPermeabilityResult:
    """
    Predict intestinal effective permeability (Peff) from physicochemical properties.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Octanol-water partition coefficient (log scale).
    mw_da : float
        Molecular weight in Daltons.
    psa_A2 : float
        Polar surface area in angstrom^2.
    n_hbd : int
        Number of hydrogen bond donors.
    n_hba : int
        Number of hydrogen bond acceptors.
    caco2_papp_cm_s : float, optional
        Caco-2 apparent permeability in cm/s. If provided, uses Sun 2004 correlation.
    pampa_peff_cm_s : float, optional
        PAMPA effective permeability in cm/s. If provided, uses PAMPA correlation.

    Returns
    -------
    IntestinalPermeabilityResult
    """
    # --- Input validation ---
    if mw_da <= 0:
        raise ValueError(f"mw_da must be positive, got {mw_da}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be non-negative, got {psa_A2}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be non-negative, got {n_hbd}")
    if n_hba < 0:
        raise ValueError(f"n_hba must be non-negative, got {n_hba}")
    if caco2_papp_cm_s is not None and caco2_papp_cm_s <= 0:
        raise ValueError(f"caco2_papp_cm_s must be positive, got {caco2_papp_cm_s}")
    if pampa_peff_cm_s is not None and pampa_peff_cm_s <= 0:
        raise ValueError(f"pampa_peff_cm_s must be positive, got {pampa_peff_cm_s}")

    # --- Method selection and logPeff calculation ---
    if caco2_papp_cm_s is not None:
        # Sun 2004: logPeff = 0.292 + 0.6916 * log10(Caco2)
        log_caco2 = math.log10(caco2_papp_cm_s)
        logpe_predicted = 0.292 + 0.6916 * log_caco2
        method = "caco2_correlation"
        notes = "Caco-2 to human Peff correlation (Sun 2004)"
    elif pampa_peff_cm_s is not None:
        log_pampa = math.log10(pampa_peff_cm_s)
        logpe_predicted = 0.1 + 0.8 * log_pampa
        method = "pampa_correlation"
        notes = "PAMPA to human Peff correlation"
    else:
        # Physicochemical model
        logpe_predicted = -6.0 + 0.85 * logp - 0.01 * psa_A2 - 0.003 * mw_da
        method = "physicochemical"
        notes = "Physicochemical model (logP, PSA, MW)"

    # --- Peff calculation with clamping ---
    peff_raw = 10.0**logpe_predicted
    peff_cm_s = max(1e-9, min(1e-3, peff_raw))

    # --- Peff class ---
    if peff_cm_s > 2e-4:
        peff_class = "high"
    elif peff_cm_s > 1e-5:
        peff_class = "medium"
    else:
        peff_class = "low"

    # --- BCS permeability class ---
    bcs_permeability_class = "high" if peff_cm_s > 2e-4 else "low"

    # --- Lipinski violations ---
    violations = 0
    if mw_da > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if n_hbd > 5:
        violations += 1
    if n_hba > 10:
        violations += 1
    lipinski_compliant = violations <= 1

    # --- Fraction absorbed (%) ---
    # Using simplified Peff-based model
    fa_pct = min(100.0, 100.0 / (1.0 + (2e-4 / max(peff_cm_s, 1e-9)) ** 0.7))

    return IntestinalPermeabilityResult(
        drug_name=drug_name,
        logp=logp,
        mw_da=mw_da,
        psa_A2=psa_A2,
        n_hbd=n_hbd,
        n_hba=n_hba,
        logpe_predicted=logpe_predicted,
        peff_cm_s=peff_cm_s,
        peff_class=peff_class,
        lipinski_violations=violations,
        lipinski_compliant=lipinski_compliant,
        bcs_permeability_class=bcs_permeability_class,
        fa_predicted_pct=fa_pct,
        human_intestine_method=method,
        notes=notes,
    )


def screen_permeability(drugs_list: list) -> list:
    """
    Screen a list of drugs for intestinal permeability.

    Parameters
    ----------
    drugs_list : list of dict
        Each dict must contain: name, logp, mw_da, psa_A2.
        Optional keys: n_hbd, n_hba, caco2_papp_cm_s, pampa_peff_cm_s.

    Returns
    -------
    list of IntestinalPermeabilityResult sorted by peff_cm_s descending.
    """
    results = []
    for drug in drugs_list:
        result = predict_intestinal_permeability(
            drug_name=drug["name"],
            logp=drug["logp"],
            mw_da=drug["mw_da"],
            psa_A2=drug["psa_A2"],
            n_hbd=drug.get("n_hbd", 1),
            n_hba=drug.get("n_hba", 3),
            caco2_papp_cm_s=drug.get("caco2_papp_cm_s"),
            pampa_peff_cm_s=drug.get("pampa_peff_cm_s"),
        )
        results.append(result)

    results.sort(key=lambda r: r.peff_cm_s, reverse=True)
    return results
