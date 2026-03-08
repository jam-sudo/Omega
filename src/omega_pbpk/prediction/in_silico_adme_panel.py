"""
Phase 454 — In silico ADME panel prediction.

Rapid prediction of key ADME properties from molecular descriptors
(MW, logP, pKa, ionization type) for early drug candidate screening.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InSilicoADMEResult:
    """Comprehensive in silico ADME prediction result."""

    drug_name: str
    mw_Da: float
    logp: float
    pka: float
    ionization_type: str
    predicted_hia: float
    predicted_f_oral: float
    predicted_vd_L_per_kg: float
    predicted_cl_mL_per_min_per_kg: float
    predicted_t_half_h: float
    predicted_fu_plasma: float
    predicted_bbb_penetration: float
    predicted_p_gp_substrate: bool
    predicted_cyp3a4_substrate: bool
    predicted_cyp2d6_substrate: bool
    lipinski_score: int
    lead_likeness_score: float
    adme_risk_flags: str
    recommendation: str
    notes: str


def _predict_properties(mw_Da: float, logp: float, pka: float, ionization_type: str):
    """Compute all predicted ADME properties, returning a dict."""

    # --- HIA ---
    if mw_Da < 500 and -2.0 < logp < 5.0:
        hia = 0.95 - max(0.0, (mw_Da - 300.0) * 0.001) - max(0.0, (abs(logp - 2.0) - 2.0) * 0.05)
    else:
        # Penalize molecules outside Lipinski space
        hia = 0.95 - max(0.0, (mw_Da - 300.0) * 0.001) - max(0.0, (abs(logp - 2.0) - 2.0) * 0.05)
    hia = min(1.0, max(0.1, hia))

    # --- fu_plasma ---
    fu_plasma = pow(10.0, -0.5 * logp) * 0.5 + 0.05
    fu_plasma = min(1.0, max(0.01, fu_plasma))

    # --- F_oral ---
    if logp > 0:
        first_pass = min(0.8, logp * 0.05 + 0.1)
    else:
        first_pass = 0.1
    f_oral = hia * (1.0 - first_pass)
    f_oral = min(1.0, max(0.0, f_oral))

    # --- Vd ---
    vd = 0.3 + logp * 0.15 + (1.0 - fu_plasma) * 2.0
    vd = min(20.0, max(0.1, vd))

    # --- CL (mL/min/kg) ---
    cl = 0.5 * (1.0 - fu_plasma) * (1.0 + logp * 0.1)
    cl = min(20.0, max(0.01, cl))

    # --- t_half (h) ---
    # t_half = 0.693 * Vd(L/kg) * 16.67 / CL(mL/min/kg)
    # Derivation: 0.693 * Vd(mL/kg) / (CL(mL/min/kg) * 60 min/h)
    #           = 0.693 * Vd(L/kg) * 1000 / (CL * 60)
    #           = 0.693 * Vd * 16.667 / CL
    t_half = 0.693 * vd * 16.667 / cl
    t_half = max(0.0, t_half)

    # --- BBB penetration ---
    bbb = max(0.0, min(1.0, 0.5 - abs(logp - 2.0) * 0.1))

    # --- Substrate predictions ---
    p_gp_substrate = mw_Da > 400.0 and logp > 2.0 and ionization_type in {"base", "neutral"}
    cyp3a4_substrate = mw_Da > 300.0 and logp > 1.0
    cyp2d6_substrate = ionization_type == "base" and 7.0 <= pka <= 10.0

    # --- Lipinski score ---
    lipinski_score = (
        (1 if mw_Da < 500.0 else 0)
        + (1 if logp < 5.0 else 0)
        + (1 if logp > -2.0 else 0)
        + (1 if pka > 0.0 or ionization_type == "neutral" else 0)
        + 1  # 5th rule always passes
    )

    # --- Risk flags ---
    risk_flags = []
    if logp > 4.0:
        risk_flags.append("low_solubility")
    if hia < 0.5:
        risk_flags.append("poor_absorption")
    if cl > 10.0:
        risk_flags.append("high_clearance")
    if bbb > 0.7:
        risk_flags.append("bbb_toxic_risk")
    if cyp3a4_substrate or cyp2d6_substrate:
        risk_flags.append("p450_liability")
    adme_risk_flags = ",".join(risk_flags)

    # --- Lead-likeness score ---
    if mw_Da > 250.0:
        lead_likeness = 100.0 - (mw_Da - 250.0) * 0.1
    else:
        lead_likeness = 100.0
    lead_likeness -= len(risk_flags) * 10.0
    lead_likeness = min(100.0, max(0.0, lead_likeness))

    # --- Recommendation ---
    if lead_likeness > 70.0 and len(risk_flags) < 2:
        recommendation = "progress"
    elif len(risk_flags) < 2:
        recommendation = "optimize"
    else:
        recommendation = "deprioritize"

    return {
        "predicted_hia": hia,
        "predicted_f_oral": f_oral,
        "predicted_vd_L_per_kg": vd,
        "predicted_cl_mL_per_min_per_kg": cl,
        "predicted_t_half_h": t_half,
        "predicted_fu_plasma": fu_plasma,
        "predicted_bbb_penetration": bbb,
        "predicted_p_gp_substrate": p_gp_substrate,
        "predicted_cyp3a4_substrate": cyp3a4_substrate,
        "predicted_cyp2d6_substrate": cyp2d6_substrate,
        "lipinski_score": lipinski_score,
        "lead_likeness_score": lead_likeness,
        "adme_risk_flags": adme_risk_flags,
        "recommendation": recommendation,
        "risk_flags": risk_flags,
    }


def predict_adme_panel(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float,
    ionization_type: str,
) -> InSilicoADMEResult:
    """
    Predict ADME properties from molecular descriptors.

    Parameters
    ----------
    drug_name : str
        Name of the compound.
    mw_Da : float
        Molecular weight in Daltons.
    logp : float
        Lipophilicity (logP).
    pka : float
        pKa of the ionizable group.
    ionization_type : str
        "acid", "base", or "neutral".

    Returns
    -------
    InSilicoADMEResult
    """
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got '{ionization_type}'"
        )
    if not (-5.0 <= logp <= 10.0):
        raise ValueError(f"logp must be between -5 and 10, got {logp}")

    props = _predict_properties(mw_Da, logp, pka, ionization_type)

    key_risks = props["adme_risk_flags"] if props["adme_risk_flags"] else "none"
    notes = (
        f"Lead-likeness score: {props['lead_likeness_score']:.1f}/100. "
        f"Key ADME risks: {key_risks}. "
        f"Recommendation: {props['recommendation']}."
    )

    return InSilicoADMEResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        predicted_hia=props["predicted_hia"],
        predicted_f_oral=props["predicted_f_oral"],
        predicted_vd_L_per_kg=props["predicted_vd_L_per_kg"],
        predicted_cl_mL_per_min_per_kg=props["predicted_cl_mL_per_min_per_kg"],
        predicted_t_half_h=props["predicted_t_half_h"],
        predicted_fu_plasma=props["predicted_fu_plasma"],
        predicted_bbb_penetration=props["predicted_bbb_penetration"],
        predicted_p_gp_substrate=props["predicted_p_gp_substrate"],
        predicted_cyp3a4_substrate=props["predicted_cyp3a4_substrate"],
        predicted_cyp2d6_substrate=props["predicted_cyp2d6_substrate"],
        lipinski_score=props["lipinski_score"],
        lead_likeness_score=props["lead_likeness_score"],
        adme_risk_flags=props["adme_risk_flags"],
        recommendation=props["recommendation"],
        notes=notes,
    )


def screen_adme_panel(compounds_list: list) -> list:
    """
    Screen a list of compounds through the ADME panel.

    Parameters
    ----------
    compounds_list : list of dict
        Each dict must have keys: drug_name, mw_Da, logp, pka, ionization_type.

    Returns
    -------
    list of InSilicoADMEResult sorted by lead_likeness_score descending.
    """
    results = []
    for compound in compounds_list:
        result = predict_adme_panel(
            drug_name=compound["drug_name"],
            mw_Da=compound["mw_Da"],
            logp=compound["logp"],
            pka=compound["pka"],
            ionization_type=compound["ionization_type"],
        )
        results.append(result)
    results.sort(key=lambda r: r.lead_likeness_score, reverse=True)
    return results
