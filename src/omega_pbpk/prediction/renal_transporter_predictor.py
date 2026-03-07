"""Renal transporter substrate/inhibitor prediction and clearance estimation.

Predicts OAT1/OAT3, OCT2, MATE1/MATE2-K, P-gp, and MRP2 transporter status
and estimates renal clearance contributions.
"""

from __future__ import annotations


def predict_oat_substrate(
    mw: float,
    logP: float,
    pka: float,
    drug_type: str,
    n_carboxyl: int = 0,
    n_sulfonate: int = 0,
) -> dict:
    """Predict OAT1/OAT3 substrate status.

    OAT1 and OAT3 prefer anionic drugs (weak acids, sulfonates), MW < 500, logP < 2.

    Parameters
    ----------
    mw: molecular weight (g/mol)
    logP: lipophilicity
    pka: pKa of ionizable group
    drug_type: 'weak_acid', 'weak_base', 'neutral', 'zwitterion'
    n_carboxyl: number of carboxyl groups
    n_sulfonate: number of sulfonate groups

    Returns
    -------
    dict with oat1_substrate, oat3_substrate, oat_score, confidence
    """
    if mw <= 0:
        raise ValueError("mw must be positive")

    score = 0
    if drug_type == "weak_acid":
        score += 3
    if n_carboxyl > 0:
        score += 2
    if n_sulfonate > 0:
        score += 3
    if mw < 400:
        score += 1
    if logP < 1:
        score += 1

    oat_substrate = score >= 4

    if score > 5:
        confidence = "high"
    elif score > 3:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "oat1_substrate": oat_substrate,
        "oat3_substrate": oat_substrate,
        "oat_score": score,
        "confidence": confidence,
    }


def predict_oct_substrate(
    mw: float,
    logP: float,
    pka: float,
    drug_type: str,
    n_basic_nitrogen: int = 0,
) -> dict:
    """Predict OCT2 substrate status.

    OCT2 prefers cationic drugs (weak bases), MW < 500, modest logP.

    Parameters
    ----------
    mw: molecular weight (g/mol)
    logP: lipophilicity
    pka: pKa of ionizable group
    drug_type: 'weak_acid', 'weak_base', 'neutral', 'zwitterion'
    n_basic_nitrogen: number of basic nitrogen atoms

    Returns
    -------
    dict with oct2_substrate, oct_score, confidence
    """
    if mw <= 0:
        raise ValueError("mw must be positive")

    score = 0
    if drug_type == "weak_base":
        score += 3
    if n_basic_nitrogen > 0:
        score += 2
    if 0 < logP < 4:
        score += 1
    if mw < 400:
        score += 1

    oct_substrate = score >= 4

    if score > 5:
        confidence = "high"
    elif score > 3:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "oct2_substrate": oct_substrate,
        "oct_score": score,
        "confidence": confidence,
    }


def estimate_tubular_secretion(
    oat_substrate: bool,
    oct_substrate: bool,
    gfr_L_per_h: float = 7.2,
    cl_filtration_L_per_h: float | None = None,
    fu_plasma: float = 1.0,
) -> dict:
    """Estimate renal tubular secretion clearance.

    Parameters
    ----------
    oat_substrate: whether the drug is an OAT1/3 substrate
    oct_substrate: whether the drug is an OCT2 substrate
    gfr_L_per_h: glomerular filtration rate (L/h), default 7.2 L/h (120 mL/min)
    cl_filtration_L_per_h: filtration clearance (L/h); computed from gfr*fu if None
    fu_plasma: unbound fraction in plasma

    Returns
    -------
    dict with cl_filtration_L_per_h, cl_secretion_L_per_h, cl_renal_total_L_per_h,
    fe_secretion_fraction
    """
    if gfr_L_per_h < 0:
        raise ValueError("gfr_L_per_h must be non-negative")
    if not (0.0 <= fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be between 0 and 1")

    if cl_filtration_L_per_h is None:
        cl_filtration = gfr_L_per_h * fu_plasma
    else:
        cl_filtration = cl_filtration_L_per_h

    cl_oat = 2.0 if oat_substrate else 0.0
    cl_oct = 1.5 if oct_substrate else 0.0
    cl_secretion_total = cl_oat + cl_oct
    cl_renal_total = cl_filtration + cl_secretion_total

    if cl_renal_total > 0:
        fe_secretion_fraction = cl_secretion_total / cl_renal_total
    else:
        fe_secretion_fraction = 0.0

    return {
        "cl_filtration_L_per_h": cl_filtration,
        "cl_secretion_L_per_h": cl_secretion_total,
        "cl_renal_total_L_per_h": cl_renal_total,
        "fe_secretion_fraction": fe_secretion_fraction,
    }


def assess_transporter_ddi(
    victim_oat_substrate: bool,
    victim_oct_substrate: bool,
    inhibitor_name: str,
    inhibitor_mechanism: str,
) -> dict:
    """Predict renal transporter-mediated DDI.

    Parameters
    ----------
    victim_oat_substrate: whether victim drug is OAT1/3 substrate
    victim_oct_substrate: whether victim drug is OCT2 substrate
    inhibitor_name: name of the inhibitor drug
    inhibitor_mechanism: 'OAT', 'OCT', or 'both'

    Returns
    -------
    dict with aucr, mechanism, ddi_predicted, severity, recommendation
    """
    if inhibitor_mechanism not in ("OAT", "OCT", "both"):
        raise ValueError("inhibitor_mechanism must be 'OAT', 'OCT', or 'both'")

    aucr = 1.0
    mechanism_parts = []

    oat_inhibited = inhibitor_mechanism in ("OAT", "both")
    oct_inhibited = inhibitor_mechanism in ("OCT", "both")

    if oat_inhibited and oct_inhibited and victim_oat_substrate and victim_oct_substrate:
        aucr = 1.7
        mechanism_parts.append("OAT1/3 + OCT2 inhibition")
    elif oat_inhibited and victim_oat_substrate and oct_inhibited and victim_oct_substrate:
        aucr = 1.7
        mechanism_parts.append("OAT1/3 + OCT2 inhibition")
    elif oat_inhibited and victim_oat_substrate:
        aucr = 1.4
        mechanism_parts.append("OAT1/3 inhibition")
    elif oct_inhibited and victim_oct_substrate:
        aucr = 1.3
        mechanism_parts.append("OCT2 inhibition")

    mechanism = "; ".join(mechanism_parts) if mechanism_parts else "no relevant transporter overlap"

    ddi_predicted = aucr > 1.1

    if aucr >= 1.7:
        severity = "moderate"
    elif aucr >= 1.3:
        severity = "mild"
    else:
        severity = "none"

    if ddi_predicted:
        recommendation = (
            f"Monitor for increased exposure of victim drug when co-administered with "
            f"{inhibitor_name} (predicted AUCR {aucr:.1f}). Consider dose adjustment."
        )
    else:
        recommendation = "No clinically relevant renal transporter DDI predicted."

    return {
        "aucr": aucr,
        "mechanism": mechanism,
        "ddi_predicted": ddi_predicted,
        "severity": severity,
        "recommendation": recommendation,
    }
