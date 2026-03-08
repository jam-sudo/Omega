"""Phase 901 — Bile Drug Concentration Prediction.

Predicts drug concentration in bile based on bile:plasma ratio (B/P),
driven by active hepatic transporters (MRP2, BCRP, P-gp) and passive
partitioning. High biliary excretion implies fecal elimination pathway.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BileDrugConcentrationResult:
    """Result of bile drug concentration prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    plasma_conc_input_mg_L: float
    bile_plasma_ratio: float
    predicted_bile_conc_mg_L: float
    f_biliary_excretion: float
    biliary_clearance_mL_min: float
    transporter_contribution: str
    micellar_solubilization_factor: float
    dili_biliary_risk: str
    notes: str


def predict_bile_concentration(
    drug_name: str,
    logp: float = 2.0,
    mw_Da: float = 400.0,
    fu_plasma: float = 0.1,
    has_pgp_substrate: bool = False,
    has_mrp2_substrate: bool = False,
    has_bcrp_substrate: bool = False,
    plasma_conc_mg_L: float = 1.0,
) -> BileDrugConcentrationResult:
    """Predict drug concentration in bile.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    fu_plasma:
        Unbound fraction in plasma.
    has_pgp_substrate:
        True if drug is a P-glycoprotein substrate.
    has_mrp2_substrate:
        True if drug is an MRP2 substrate.
    has_bcrp_substrate:
        True if drug is a BCRP substrate.
    plasma_conc_mg_L:
        Plasma drug concentration in mg/L.

    Returns
    -------
    BileDrugConcentrationResult
        Predicted biliary pharmacokinetic parameters.

    Raises
    ------
    ValueError
        If mw_Da <= 0 or plasma_conc_mg_L <= 0.
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")

    # Determine transporter contribution and compute bile:plasma ratio
    any_transporter = has_pgp_substrate or has_mrp2_substrate or has_bcrp_substrate

    bile_plasma_ratio = 1.0
    if any_transporter:
        if has_pgp_substrate:
            bile_plasma_ratio *= 3.0
            transporter = "P-gp"
        if has_mrp2_substrate:
            bile_plasma_ratio *= 5.0
            transporter = "MRP2"
        if has_bcrp_substrate:
            bile_plasma_ratio *= 4.0
            transporter = "BCRP"
        # If multiple transporters, use last assignment (MRP2 > BCRP > P-gp priority)
        # Reset transporter to highest-priority active transporter
        if has_mrp2_substrate:
            transporter = "MRP2"
        elif has_bcrp_substrate:
            transporter = "BCRP"
        else:
            transporter = "P-gp"
    else:
        transporter = "passive"
        bile_plasma_ratio *= 1 + logp * 0.3

    # Clamp bile_plasma_ratio to [0.1, 100.0]
    bile_plasma_ratio = max(0.1, min(100.0, bile_plasma_ratio))

    # Predicted bile concentration
    predicted_bile_conc_mg_L = bile_plasma_ratio * plasma_conc_mg_L

    # Fraction of dose excreted in bile (Hirom rule)
    if mw_Da < 300:
        base_f = 0.1
    elif mw_Da < 500:
        base_f = 0.4
    else:
        base_f = 0.7

    f_biliary_excretion = base_f
    if any_transporter:
        f_biliary_excretion = f_biliary_excretion * 1.5
    f_biliary_excretion = max(0.0, min(1.0, f_biliary_excretion))

    # Biliary clearance (human bile flow ~0.6 mL/min)
    biliary_clearance_mL_min = bile_plasma_ratio * 0.6
    biliary_clearance_mL_min = max(0.01, min(60.0, biliary_clearance_mL_min))

    # Micellar solubilization factor
    if logp > 0:
        micellar_solubilization_factor = 1 + logp * 2.0
    else:
        micellar_solubilization_factor = 1.0
    micellar_solubilization_factor = max(1.0, min(20.0, micellar_solubilization_factor))

    # DILI biliary risk
    if bile_plasma_ratio > 20:
        dili_biliary_risk = "high"
    elif bile_plasma_ratio > 5:
        dili_biliary_risk = "moderate"
    else:
        dili_biliary_risk = "low"

    # Notes
    if dili_biliary_risk == "high":
        notes = "High biliary concentration — risk of cholestatic DILI. Monitor liver enzymes."
    elif dili_biliary_risk == "moderate":
        notes = "Moderate biliary excretion — hepatic monitoring recommended"
    else:
        notes = "Low biliary concentration risk"

    return BileDrugConcentrationResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        bile_plasma_ratio=bile_plasma_ratio,
        predicted_bile_conc_mg_L=predicted_bile_conc_mg_L,
        f_biliary_excretion=f_biliary_excretion,
        biliary_clearance_mL_min=biliary_clearance_mL_min,
        transporter_contribution=transporter,
        micellar_solubilization_factor=micellar_solubilization_factor,
        dili_biliary_risk=dili_biliary_risk,
        notes=notes,
    )


def screen_biliary_excretion(candidates: list) -> list:
    """Screen a list of drug candidates for biliary excretion.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name", "logp", optional "mw_Da" (default 400),
        optional "has_pgp_substrate" (bool), optional "has_mrp2_substrate" (bool),
        optional "has_bcrp_substrate" (bool).

    Returns
    -------
    list
        List of BileDrugConcentrationResult sorted by bile_plasma_ratio descending.
    """
    results = []
    for cand in candidates:
        result = predict_bile_concentration(
            drug_name=cand["name"],
            logp=cand["logp"],
            mw_Da=cand.get("mw_Da", 400.0),
            has_pgp_substrate=cand.get("has_pgp_substrate", False),
            has_mrp2_substrate=cand.get("has_mrp2_substrate", False),
            has_bcrp_substrate=cand.get("has_bcrp_substrate", False),
        )
        results.append(result)

    results.sort(key=lambda r: r.bile_plasma_ratio, reverse=True)
    return results
