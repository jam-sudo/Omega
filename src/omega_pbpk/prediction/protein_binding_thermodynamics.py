"""Phase 546 — Drug Protein Binding Thermodynamics.

Estimates thermodynamic parameters of drug-protein (albumin) binding
from physicochemical descriptors (logP, MW): Ka, Kd, ΔG, ΔH, ΔS,
fu, and binding classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinBindingThermodynamicsResult:
    """Thermodynamic analysis result for drug-protein binding."""

    drug_name: str
    logP: float
    mw_Da: float

    pKd: float
    kd_M: float
    kd_mg_L: float
    ka_L_per_mol: float

    dg_kJ_per_mol: float
    dh_kJ_per_mol: float
    ds_J_per_mol_K: float

    binding_class: str  # "tight" / "moderate" / "weak"

    fu_predicted: float
    fu_class: str  # "highly_bound" / "moderately_bound" / "low_binding"

    entropy_driven: bool
    notes: str


def predict_binding_thermodynamics(
    drug_name: str,
    logP: float,
    mw_Da: float,
) -> ProteinBindingThermodynamicsResult:
    """Predict drug-protein binding thermodynamics from logP and MW.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight (Da).

    Returns
    -------
    ProteinBindingThermodynamicsResult
    """
    if mw_Da <= 0.0:
        raise ValueError("mw_Da must be positive")

    # Binding affinity
    pKd = 1.5 + 0.8 * max(0.0, logP) - 0.003 * mw_Da
    kd_M = 10.0 ** (-pKd)
    kd_mg_L = kd_M * mw_Da * 1000.0
    ka_L_per_mol = 1.0 / kd_M

    # Thermodynamics at 37°C (310 K)
    R = 8.314  # J/(mol·K)
    T = 310.0  # K

    dG = R * T * math.log(kd_M)  # J/mol  (negative = favorable)
    dG_kJ = dG / 1000.0  # kJ/mol

    dH_kJ = -0.5 * logP * 4.0  # hydrophobic enthalpy contribution (kJ/mol)
    # dG = dH - T*dS  →  dS = (dH - dG) / T
    dS_J_per_K = (dH_kJ * 1000.0 - dG) / T  # J/(mol·K)

    # Binding class
    if pKd > 6.0:
        binding_class = "tight"
    elif pKd >= 4.0:
        binding_class = "moderate"
    else:
        binding_class = "weak"

    # Unbound fraction using albumin binding (1 site per molecule)
    albumin_conc_M = 6.0e-4  # 0.6 mM
    n = 1
    fu_raw = 1.0 / (1.0 + n * ka_L_per_mol * albumin_conc_M)
    fu = max(0.001, min(1.0, fu_raw))

    # fu class
    if fu < 0.1:
        fu_class = "highly_bound"
    elif fu <= 0.5:
        fu_class = "moderately_bound"
    else:
        fu_class = "low_binding"

    # Entropy-driven: entropy favors binding when -T*dS < 0 (dS > 0)
    # i.e. entropic contribution (-T*dS) is negative → drives binding
    # Equivalently, entropy term contribution to dG: -T*dS_J_per_K
    t_ds = T * dS_J_per_K  # = T*dS in J/mol
    # binding is entropy-driven if -T*dS < dG (entropy term dominates over enthalpy)
    entropy_driven = (-t_ds) < dG  # both in J/mol

    # Notes
    notes_parts = []
    if logP < 0:
        notes_parts.append("logP<0: hydrophilic drug; weak protein binding expected")
    if mw_Da > 700:
        notes_parts.append("high MW reduces predicted pKd")
    if binding_class == "tight":
        notes_parts.append("tight binding may limit free drug concentration")
    if fu < 0.05:
        notes_parts.append("very highly bound (fu<5%): significant protein binding")

    return ProteinBindingThermodynamicsResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        pKd=pKd,
        kd_M=kd_M,
        kd_mg_L=kd_mg_L,
        ka_L_per_mol=ka_L_per_mol,
        dg_kJ_per_mol=dG_kJ,
        dh_kJ_per_mol=dH_kJ,
        ds_J_per_mol_K=dS_J_per_K,
        binding_class=binding_class,
        fu_predicted=fu,
        fu_class=fu_class,
        entropy_driven=entropy_driven,
        notes="; ".join(notes_parts),
    )


def screen_binding_affinity(compounds: list) -> list:
    """Screen a list of compounds for protein binding thermodynamics.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys: drug_name (str), logP (float),
        mw_Da (float).

    Returns
    -------
    List of ProteinBindingThermodynamicsResult sorted by pKd descending
    (tightest binders first).
    """
    results = []
    for comp in compounds:
        res = predict_binding_thermodynamics(
            drug_name=comp["drug_name"],
            logP=comp["logP"],
            mw_Da=comp["mw_Da"],
        )
        results.append(res)

    results.sort(key=lambda r: r.pKd, reverse=True)
    return results
