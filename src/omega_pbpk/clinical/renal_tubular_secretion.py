"""
Phase 441 — Mechanistic model of renal tubular secretion.

OAT1/OAT3/OCT2 transporter-mediated secretion of drugs using
Michaelis-Menten kinetics simplified to CL-based model.
"""

from dataclasses import dataclass

TRANSPORTER_CL = {
    "OAT1": {"cl_max_L_per_h": 3.0, "substrate_threshold": 0.5, "inhibition_pct": 85},
    "OAT3": {"cl_max_L_per_h": 2.5, "substrate_threshold": 0.4, "inhibition_pct": 80},
    "OCT2": {"cl_max_L_per_h": 2.0, "substrate_threshold": 0.3, "inhibition_pct": 75},
    "combined": {"cl_max_L_per_h": 7.5, "substrate_threshold": 0.6, "inhibition_pct": 90},
}

GFR_NORMAL_L_PER_H = 7.5  # normal GFR for 70 kg adult


@dataclass(frozen=True)
class TubularSecretionResult:
    drug_name: str
    transporter: str
    substrate_class: str
    cl_filtration_L_per_h: float
    cl_secretion_L_per_h: float
    cl_reabsorption_offset_L_per_h: float
    cl_renal_total_L_per_h: float
    fe_urine_fraction: float
    secretion_to_filtration_ratio: float
    inhibitor_effect_reduction_pct: float
    ddi_risk: str
    dose_adjustment_with_ckd: str
    notes: str


def _classify_substrate(substrate_fraction: float) -> str:
    if substrate_fraction > 0.6:
        return "high"
    elif substrate_fraction >= 0.3:
        return "moderate"
    elif substrate_fraction >= 0.1:
        return "low"
    else:
        return "non_substrate"


def _classify_ddi_risk(ratio: float) -> str:
    if ratio > 2.0:
        return "high"
    elif ratio >= 1.0:
        return "moderate"
    else:
        return "low"


def _classify_ckd_adjustment(fe: float) -> str:
    if fe > 0.6:
        return "major"
    elif fe >= 0.3:
        return "moderate"
    elif fe >= 0.1:
        return "minor"
    else:
        return "none"


def assess_tubular_secretion(
    drug_name: str,
    transporter: str,
    substrate_fraction: float,
    fu_plasma: float,
    logp: float,
    cl_nonrenal_L_per_h: float = 8.0,
    gfr_L_per_h: float = 7.5,
) -> TubularSecretionResult:
    """
    Assess renal tubular secretion for a drug via a specified transporter.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    transporter : str
        One of "OAT1", "OAT3", "OCT2", "combined".
    substrate_fraction : float
        Fraction of transporter capacity used by this drug (0-1).
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    logp : float
        Lipophilicity (log P).
    cl_nonrenal_L_per_h : float
        Non-renal clearance in L/h (default 8.0).
    gfr_L_per_h : float
        Glomerular filtration rate in L/h (default 7.5).

    Returns
    -------
    TubularSecretionResult
    """
    if transporter not in TRANSPORTER_CL:
        raise ValueError(
            f"Unknown transporter '{transporter}'. Must be one of {list(TRANSPORTER_CL.keys())}."
        )
    if not (0.0 <= substrate_fraction <= 1.0):
        raise ValueError(f"substrate_fraction must be between 0 and 1, got {substrate_fraction}.")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}.")
    if gfr_L_per_h <= 0.0:
        raise ValueError(f"gfr_L_per_h must be positive, got {gfr_L_per_h}.")

    t_info = TRANSPORTER_CL[transporter]
    cl_max = t_info["cl_max_L_per_h"]
    inhibition_pct = t_info["inhibition_pct"]

    # Filtration clearance
    cl_filtration = gfr_L_per_h * fu_plasma

    # Active secretion clearance (linear approximation)
    cl_secretion = cl_max * substrate_fraction

    # Passive reabsorption: logP > 1 drives reabsorption
    if logp > 1.0:
        cl_reabsorption = cl_filtration * logp * 0.1
        # Cap at 50% of filtration
        max_reab = 0.5 * cl_filtration
        cl_reabsorption = min(cl_reabsorption, max_reab)
    else:
        cl_reabsorption = 0.0

    # Total renal clearance (minimum 0)
    cl_renal_total = max(0.0, cl_filtration + cl_secretion - cl_reabsorption)

    # fe_urine: fraction excreted unchanged in urine
    cl_total = cl_renal_total + cl_nonrenal_L_per_h
    if cl_total > 0.0:
        fe_urine = cl_renal_total / cl_total
    else:
        fe_urine = 0.0
    fe_urine = max(0.0, min(1.0, fe_urine))

    # Secretion to filtration ratio
    sec_to_filt = cl_secretion / max(cl_filtration, 0.001)

    substrate_class = _classify_substrate(substrate_fraction)
    ddi_risk = _classify_ddi_risk(sec_to_filt)
    ckd_adjustment = _classify_ckd_adjustment(fe_urine)

    notes = (
        f"Transporter: {transporter}. "
        f"Secretion CL: {cl_secretion:.3f} L/h. "
        f"DDI risk: {ddi_risk} (inhibition up to {inhibition_pct}% of secretion CL). "
        f"fe_urine: {fe_urine:.3f}. "
        f"CKD dose adjustment: {ckd_adjustment}."
    )

    return TubularSecretionResult(
        drug_name=drug_name,
        transporter=transporter,
        substrate_class=substrate_class,
        cl_filtration_L_per_h=cl_filtration,
        cl_secretion_L_per_h=cl_secretion,
        cl_reabsorption_offset_L_per_h=cl_reabsorption,
        cl_renal_total_L_per_h=cl_renal_total,
        fe_urine_fraction=fe_urine,
        secretion_to_filtration_ratio=sec_to_filt,
        inhibitor_effect_reduction_pct=float(inhibition_pct),
        ddi_risk=ddi_risk,
        dose_adjustment_with_ckd=ckd_adjustment,
        notes=notes,
    )


def compare_transporter_impact(
    drug_name: str,
    substrate_fraction: float,
    fu_plasma: float,
    logp: float,
    transporters_list=None,
) -> list:
    """
    Compare tubular secretion across multiple transporters.

    Parameters
    ----------
    drug_name : str
    substrate_fraction : float
    fu_plasma : float
    logp : float
    transporters_list : list or None
        List of transporter names. Defaults to all four.

    Returns
    -------
    list of TubularSecretionResult sorted by cl_secretion_L_per_h descending.
    """
    if transporters_list is None:
        transporters_list = list(TRANSPORTER_CL.keys())

    results = []
    for t in transporters_list:
        result = assess_tubular_secretion(
            drug_name=drug_name,
            transporter=t,
            substrate_fraction=substrate_fraction,
            fu_plasma=fu_plasma,
            logp=logp,
        )
        results.append(result)

    results.sort(key=lambda r: r.cl_secretion_L_per_h, reverse=True)
    return results
