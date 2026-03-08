"""Drug Plasma Stability Prediction (Phase 616).

Predicts drug stability in human plasma based on structural features.
Considers hydrolysis (ester, amide), oxidation (aromatic/lipophilic),
and baseline plasma protein binding effects.

References
----------
- Di L et al., J Med Chem. 2005 — plasma stability assays
- Fasinu PS et al., Curr Drug Metab. 2012 — ester hydrolysis in plasma
- Testa B & Mayer JM, Hydrolysis in Drug and Prodrug Metabolism. 2003
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlasmaStabilityResult:
    """Results from plasma stability prediction.

    Attributes
    ----------
    drug_name : Drug name.
    smiles_features : Simplified structural descriptor string.
    k_deg_per_h : Overall plasma degradation rate (/h).
    t_half_plasma_h : Plasma half-life (h).
    fraction_remaining_1h : Fraction of drug remaining at 1h.
    fraction_remaining_2h : Fraction of drug remaining at 2h.
    fraction_remaining_4h : Fraction of drug remaining at 4h.
    stability_class : Stability category (unstable/labile/moderate/stable).
    degradation_pathways : Comma-separated active degradation pathways.
    ester_content : Whether drug contains ester groups.
    amide_content : Whether drug contains amide groups.
    adjustments : Notes on factors driving stability prediction.
    notes : Summary text.
    """

    drug_name: str
    smiles_features: str
    k_deg_per_h: float
    t_half_plasma_h: float
    fraction_remaining_1h: float
    fraction_remaining_2h: float
    fraction_remaining_4h: float
    stability_class: str
    degradation_pathways: str
    ester_content: bool
    amide_content: bool
    adjustments: str
    notes: str


def predict_plasma_stability(
    drug_name: str,
    mw_Da: float,
    logP: float,
    n_ester_groups: int = 0,
    n_amide_groups: int = 0,
    n_aromatic_rings: int = 0,
    lactone: bool = False,
    pka: float = None,
) -> PlasmaStabilityResult:
    """Predict drug stability in human plasma.

    Parameters
    ----------
    drug_name : Drug name.
    mw_Da : Molecular weight (Da). Must be > 0.
    logP : Lipophilicity (log partition coefficient).
    n_ester_groups : Number of ester groups (default 0). Must be >= 0.
    n_amide_groups : Number of amide groups (default 0). Must be >= 0.
    n_aromatic_rings : Number of aromatic rings (default 0). Must be >= 0.
    lactone : Whether the drug contains a lactone (cyclic ester) (default False).
    pka : pKa of the drug (optional, currently used for future extensions).

    Returns
    -------
    PlasmaStabilityResult (frozen dataclass)
    """
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if n_ester_groups < 0:
        raise ValueError("n_ester_groups must be >= 0")
    if n_amide_groups < 0:
        raise ValueError("n_amide_groups must be >= 0")
    if n_aromatic_rings < 0:
        raise ValueError("n_aromatic_rings must be >= 0")

    # --- Degradation rate contributions ---
    k_base = 0.05  # /h — stable baseline for unknown drug

    # Ester hydrolysis (plasma esterases are abundant)
    k_ester = n_ester_groups * 0.3
    if lactone:
        k_ester += 0.5  # cyclic esters more susceptible

    # Amide hydrolysis (amides more resistant than esters)
    k_amide = n_amide_groups * 0.02

    # Oxidation (lipophilic aromatics more susceptible to CYP-like plasma oxidases)
    k_ox = n_aromatic_rings * 0.01 * max(0.0, logP - 2.0)

    # Total degradation rate, clamped to [0.01, 5.0]
    k_deg_raw = k_base + k_ester + k_amide + k_ox
    k_deg = max(0.01, min(5.0, k_deg_raw))

    # --- Derived quantities ---
    t_half = math.log(2.0) / k_deg

    fr_1h = math.exp(-k_deg * 1.0)
    fr_2h = math.exp(-k_deg * 2.0)
    fr_4h = math.exp(-k_deg * 4.0)

    # --- Stability classification ---
    if t_half < 0.5:
        stability_class = "unstable"
    elif t_half < 2.0:
        stability_class = "labile"
    elif t_half < 6.0:
        stability_class = "moderate"
    else:
        stability_class = "stable"

    # --- Degradation pathways ---
    pathways = []
    # If total is essentially just baseline (within 1% of k_base)
    if k_deg <= k_base * 1.01:
        pathways = ["none"]
    else:
        if n_ester_groups > 0 or lactone:
            pathways.append("ester_hydrolysis")
        if n_amide_groups > 0:
            pathways.append("amide_hydrolysis")
        if k_ox > 0.001:
            pathways.append("oxidation")
        if not pathways:
            pathways = ["none"]

    degradation_pathways = ", ".join(pathways)

    # --- Structural flags ---
    ester_content = (n_ester_groups > 0) or lactone
    amide_content = n_amide_groups > 0

    # --- SMILES features descriptor string ---
    smiles_features = (
        f"esters:{n_ester_groups} amides:{n_amide_groups} aromatics:{n_aromatic_rings}"
    )

    # --- Adjustments text ---
    adj_parts = []
    if k_ester > 0:
        if lactone and n_ester_groups == 0:
            adj_parts.append("lactone ring susceptible to hydrolysis")
        elif lactone:
            adj_parts.append("ester hydrolysis dominant (includes lactone)")
        else:
            adj_parts.append("ester hydrolysis dominant")
    if k_amide > 0:
        adj_parts.append(f"{n_amide_groups} amide(s) contributing minor hydrolysis")
    if k_ox > 0.001:
        adj_parts.append(f"aromatic oxidation (logP={logP:.2f})")
    if not adj_parts:
        adj_parts.append("no significant structural instability drivers")
    adjustments = "; ".join(adj_parts)

    notes = (
        f"Plasma stability for {drug_name}: MW={mw_Da:.1f} Da, logP={logP:.2f}. "
        f"k_deg={k_deg:.4f}/h, t_half={t_half:.2f}h, class={stability_class}. "
        f"Pathways: {degradation_pathways}."
    )

    return PlasmaStabilityResult(
        drug_name=drug_name,
        smiles_features=smiles_features,
        k_deg_per_h=k_deg,
        t_half_plasma_h=t_half,
        fraction_remaining_1h=fr_1h,
        fraction_remaining_2h=fr_2h,
        fraction_remaining_4h=fr_4h,
        stability_class=stability_class,
        degradation_pathways=degradation_pathways,
        ester_content=ester_content,
        amide_content=amide_content,
        adjustments=adjustments,
        notes=notes,
    )


def stability_screening(compounds: list) -> list:
    """Screen multiple compounds for plasma stability.

    Parameters
    ----------
    compounds : List of dicts, each containing keyword arguments for
        predict_plasma_stability (must include 'drug_name', 'mw_Da', 'logP').

    Returns
    -------
    list of PlasmaStabilityResult sorted by t_half_plasma_h descending
    (most stable first).
    """
    results = []
    for c in compounds:
        r = predict_plasma_stability(**c)
        results.append(r)

    results.sort(key=lambda r: r.t_half_plasma_h, reverse=True)
    return results
