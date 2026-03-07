"""
Phase 710 — Integrated ADME property calculator.

Combines multiple ADME predictions into a unified profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ADMEProfile", "compute_adme_profile", "screen_adme", "compare_adme_optimization"]


@dataclass(frozen=True)
class ADMEProfile:
    """Unified ADME profile for a compound."""

    compound_name: str
    logP: float
    mw: float
    psa: float

    fa_predicted: float
    vd_L_estimated: float
    cl_int_uL_min_mg: float

    t_half_h_estimated: float
    fe_urine_estimated: float

    f_oral_estimated: float
    fup_estimated: float

    herg_risk: str  # "low", "moderate", "high"
    reactive_alert: str  # "" or description

    lipinski_pass: bool
    overall_developability: str  # "excellent", "good", "moderate", "poor"
    notes: str


def _extract_physicochemical(physicochemical: dict) -> tuple:
    """Extract and validate physicochemical parameters."""
    if "logP" not in physicochemical:
        raise ValueError("physicochemical must contain 'logP'")
    logP = float(physicochemical["logP"])

    mw = physicochemical.get("mw")
    if mw is None:
        raise ValueError("physicochemical must contain 'mw'")
    mw = float(mw)
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    psa = float(physicochemical.get("psa", 80.0))
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")

    n_h_donors = int(physicochemical.get("n_h_donors", 1))
    n_h_acceptors = int(physicochemical.get("n_h_acceptors", 3))
    n_rotatable_bonds = int(physicochemical.get("n_rotatable_bonds", 5))
    pka_acid = physicochemical.get("pka_acid")
    pka_base = physicochemical.get("pka_base")
    charge = int(physicochemical.get("charge", 0))

    return logP, mw, psa, n_h_donors, n_h_acceptors, n_rotatable_bonds, pka_acid, pka_base, charge


def compute_adme_profile(
    compound_name: str,
    smiles_features: dict,
    physicochemical: dict,
) -> ADMEProfile:
    """Compute integrated ADME profile for a compound.

    Args:
        compound_name: Name of the compound.
        smiles_features: SMILES-derived features (optional keys: n_cyp3a4_alerts,
            n_cyp2d6_alerts, n_pgp_substrate_features).
        physicochemical: Physicochemical properties dict (required: logP, mw).

    Returns:
        ADMEProfile dataclass.
    """
    logP, mw, psa, n_h_donors, n_h_acceptors, n_rotatable_bonds, pka_acid, pka_base, charge = (
        _extract_physicochemical(physicochemical)
    )

    n_cyp3a4_alerts = int(smiles_features.get("n_cyp3a4_alerts", 0))
    n_cyp2d6_alerts = int(smiles_features.get("n_cyp2d6_alerts", 0))

    # 1. Absorption (fa estimate)
    papp_raw = (
        math.pow(10, 0.5 * logP - 1.0)
        * math.exp(-max(0, psa - 60) * 0.02)
        * math.exp(-max(0, mw - 300) * 0.003)
        * math.exp(-max(0, n_h_donors - 1) * 0.3)
    )
    papp = max(0.01, min(300.0, papp_raw))  # units: ×10⁻⁶ cm/s (implied)
    fa = min(0.99, papp / (papp + 3.0))

    # 2. Distribution (Vd estimate)
    vd_raw = 5.0 + max(0.0, logP) * 8.0 + (1.0 / max(0.01, psa / 100.0)) * 5.0
    vd_L = max(5.0, min(500.0, vd_raw))

    # 3. Metabolism (CLint estimate)
    base_CLint = 20.0 * (logP if logP > 0 else 0.5) * (300.0 / max(100.0, mw))
    cyp_alerts = n_cyp3a4_alerts + n_cyp2d6_alerts
    cl_int = base_CLint * (1.0 + 0.1 * cyp_alerts)
    cl_int = max(1.0, min(500.0, cl_int))

    # 4. Elimination (t½ estimate)
    cl_hepatic = 5.0 * (1.0 - math.exp(-base_CLint / 100.0))
    cl_hepatic = max(0.1, min(90.0, cl_hepatic))
    t_half_h = math.log(2) * vd_L / cl_hepatic

    # 5. Excretion (fe_urine estimate)
    if logP < 0:
        fe_urine = 0.8
    elif logP < 2:
        fe_urine = 0.4
    else:
        fe_urine = 0.1

    # 6. Toxicity flags
    herg_risk = "low"
    if logP > 3 and mw < 500 and pka_base is not None and float(pka_base) > 7:
        herg_risk = "moderate"

    reactive_alert = ""
    if n_cyp3a4_alerts > 2:
        reactive_alert = "possible reactive metabolite"

    # 7. Oral bioavailability estimate
    fup = max(0.01, 1.0 / (1.0 + 0.1 * max(0.0, logP)))
    q_h = 90.0  # L/h
    er = cl_hepatic / q_h
    f_oral = fa * (1.0 - er)
    f_oral = max(0.0, min(1.0, f_oral))

    # Lipinski rule of 5
    lipinski_pass = mw <= 500 and logP <= 5 and n_h_donors <= 5 and n_h_acceptors <= 10

    # Overall developability
    if fa > 0.7 and f_oral > 0.4 and 4 < t_half_h < 24 and lipinski_pass:
        overall_developability = "excellent"
    elif fa > 0.2 and f_oral > 0.2 and lipinski_pass:
        overall_developability = "good"
    elif fa > 0.1 and lipinski_pass:
        overall_developability = "moderate"
    elif fa > 0.4 and not lipinski_pass:
        overall_developability = "moderate"
    else:
        overall_developability = "poor"

    notes_parts = [
        f"Papp={papp:.2f}×10⁻⁶ cm/s, fa={fa:.2f}, Vd={vd_L:.1f} L.",
        f"CLint={cl_int:.1f} µL/min/mg, CLh={cl_hepatic:.2f} L/h, t½={t_half_h:.1f} h.",
        f"F_oral={f_oral:.2f}, fup={fup:.3f}, fe_urine={fe_urine:.2f}.",
        f"hERG risk: {herg_risk}.",
    ]
    if reactive_alert:
        notes_parts.append(f"Alert: {reactive_alert}.")
    notes = " ".join(notes_parts)

    return ADMEProfile(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        fa_predicted=fa,
        vd_L_estimated=vd_L,
        cl_int_uL_min_mg=cl_int,
        t_half_h_estimated=t_half_h,
        fe_urine_estimated=fe_urine,
        f_oral_estimated=f_oral,
        fup_estimated=fup,
        herg_risk=herg_risk,
        reactive_alert=reactive_alert,
        lipinski_pass=lipinski_pass,
        overall_developability=overall_developability,
        notes=notes,
    )


def screen_adme(compounds: list[dict]) -> list[ADMEProfile]:
    """Screen multiple compounds and rank by oral bioavailability.

    Each compound dict must have keys: compound_name, physicochemical (dict),
    and optionally smiles_features (dict).

    Args:
        compounds: List of compound dicts.

    Returns:
        List of ADMEProfile sorted by f_oral_estimated descending.
    """
    profiles = []
    for cmpd in compounds:
        name = cmpd.get("compound_name", "Unknown")
        phys = cmpd.get("physicochemical", {})
        smiles = cmpd.get("smiles_features", {})
        profile = compute_adme_profile(name, smiles, phys)
        profiles.append(profile)

    profiles.sort(key=lambda p: p.f_oral_estimated, reverse=True)
    return profiles


def compare_adme_optimization(
    base_compound: dict,
    variants: list[dict],
) -> list[ADMEProfile]:
    """Compare base compound against structural variants.

    Args:
        base_compound: Dict with compound_name, physicochemical, optional smiles_features.
        variants: List of variant compound dicts in same format.

    Returns:
        List of ADMEProfile with base compound first, then variants.
    """
    all_compounds = [base_compound] + variants
    results = []
    for cmpd in all_compounds:
        name = cmpd.get("compound_name", "Unknown")
        phys = cmpd.get("physicochemical", {})
        smiles = cmpd.get("smiles_features", {})
        profile = compute_adme_profile(name, smiles, phys)
        results.append(profile)
    return results
