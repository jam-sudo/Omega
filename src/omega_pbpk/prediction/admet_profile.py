"""Comprehensive ADMET profiler.

Aggregates physicochemical, ADME, and safety predictions into a single
unified profile for compound assessment and screening.

Estimation models are simplified empirical correlations suitable for
early-stage compound triaging.  For quantitative predictions, use
dedicated QSAR/PBPK models.

References
----------
- Lipinski CA et al., Adv Drug Deliv Rev. 2001;46(1-3):3-26 (Rule of 5)
- Egan WJ et al., J Med Chem. 2000;43(21):3867-77 (BBB permeability)
- Waring MJ, Expert Opin Drug Discov. 2010;5(3):235-48 (hERG liability)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ADMETProfile:
    """Comprehensive ADMET prediction profile.

    Attributes
    ----------
    compound_name : Compound identifier.
    smiles : SMILES string.
    mw : Molecular weight (Da).
    logP : Partition coefficient.
    logS : Aqueous solubility (log mol/L).
    tpsa : Topological polar surface area (A^2).
    hbd : H-bond donors (estimated).
    hba : H-bond acceptors (estimated).
    fup : Fraction unbound in plasma (0-1).
    peff_cm_s : Intestinal permeability (cm/s).
    bcs_class : BCS classification (I/II/III/IV).
    f_oral : Estimated oral bioavailability (0-1).
    clint_L_per_h_per_kg : Intrinsic clearance estimate (L/h/kg).
    vd_L_per_kg : Volume of distribution (L/kg).
    t_half_h : Terminal half-life (h).
    lipinski_pass : Passes Lipinski Rule of 5.
    bbb_penetrant : Predicted BBB penetration.
    herg_risk : hERG channel liability ('low', 'moderate', 'high').
    overall_risk_score : Composite risk (0-100, lower=better).
    drug_likeness_score : Drug-likeness (0-1).
    development_recommendation : 'promising', 'optimize', or 'concern'.
    notes : Summary text.
    """

    compound_name: str
    smiles: str
    mw: float
    logP: float
    logS: float
    tpsa: float
    hbd: int
    hba: int
    fup: float
    peff_cm_s: float
    bcs_class: str
    f_oral: float
    clint_L_per_h_per_kg: float
    vd_L_per_kg: float
    t_half_h: float
    lipinski_pass: bool
    bbb_penetrant: bool
    herg_risk: str
    overall_risk_score: float
    drug_likeness_score: float
    development_recommendation: str
    notes: str


def predict_admet(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    logS: float | None = None,
    tpsa: float = 50.0,
    pka: float = 7.0,
    cmax_uM: float = 1.0,
) -> ADMETProfile:
    """Predict comprehensive ADMET profile for a compound.

    Parameters
    ----------
    compound_name : Compound name.
    smiles : SMILES string.
    mw : Molecular weight (Da).
    logP : Octanol-water partition coefficient.
    logS : Aqueous solubility (log mol/L). Estimated if None.
    tpsa : Topological polar surface area (A^2).
    pka : Acid dissociation constant.
    cmax_uM : Expected Cmax (uM) for safety margin.

    Returns
    -------
    ADMETProfile with all predictions.

    Raises
    ------
    ValueError : If mw <= 0.
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")

    # Estimate logS if not provided
    if logS is None:
        logS = 0.5 - 0.01 * mw - 0.6 * logP

    # H-bond donors/acceptors from SMILES (rough)
    n_heteroatoms = smiles.upper().count("O") + smiles.upper().count("N")
    hbd = min(5, max(0, n_heteroatoms // 2))
    hba = min(10, max(0, n_heteroatoms))

    # Fraction unbound in plasma
    exponent = 0.072 * logP - 0.067 * mw / 100.0 + 0.015 * tpsa / 10.0
    fup = 1.0 / (1.0 + 10.0**exponent)
    fup = max(0.01, min(0.99, fup))

    # Intestinal permeability (cm/s)
    peff_log = 0.32 * logP - 0.0027 * mw - 0.011 * tpsa
    peff = max(1e-6, 10.0**peff_log)

    # BCS classification
    high_perm = peff > 1e-4
    high_sol = logS > -3.0
    if high_perm and high_sol:
        bcs_class = "I"
    elif high_perm:
        bcs_class = "II"
    elif high_sol:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    # Oral bioavailability
    f_oral = max(0.05, min(0.95, peff * 100.0 * (1.0 - 0.3 * max(0.0, logP - 3.0))))

    # Intrinsic clearance (L/h/kg)
    clint = max(0.01, 0.1 * logP + 0.001 * mw)

    # Volume of distribution (L/kg)
    vd = max(0.1, 0.5 + 0.15 * logP)

    # Half-life
    ke = (clint * fup) / vd / 70.0  # well-stirred, 70 kg
    t_half = 0.693 / max(ke, 0.001)

    # Lipinski Rule of 5
    lipinski_pass = mw <= 500 and logP <= 5 and hbd <= 5 and hba <= 10

    # BBB penetration
    bbb_penetrant = logP > 1.0 and mw < 450 and tpsa < 90 and fup > 0.1

    # hERG risk
    if logP > 3.0 and mw > 300 and pka > 7.0:
        herg_risk = "high"
    elif logP > 2.0 and mw > 200:
        herg_risk = "moderate"
    else:
        herg_risk = "low"

    # Drug-likeness score (0-1)
    dl_score = 0.0
    dl_score += 0.4 if lipinski_pass else 0.0
    if -1.0 <= logP <= 4.0:
        dl_score += 0.3
    elif -2.0 <= logP <= 5.0:
        dl_score += 0.15
    if tpsa < 120:
        dl_score += 0.3
    elif tpsa < 140:
        dl_score += 0.15

    # Overall risk score (0-100, lower=better)
    risk = 0.0
    if herg_risk == "high":
        risk += 20.0
    elif herg_risk == "moderate":
        risk += 10.0
    if bcs_class == "IV":
        risk += 15.0
    if not lipinski_pass:
        risk += 10.0
    if t_half < 0.5:
        risk += 10.0
    if vd > 20.0:
        risk += 10.0
    risk = min(100.0, max(0.0, risk))

    # Development recommendation
    if risk < 30 and dl_score > 0.6:
        recommendation = "promising"
    elif risk >= 60 or dl_score < 0.3:
        recommendation = "concern"
    else:
        recommendation = "optimize"

    notes = (
        f"ADMET profile: {compound_name} (MW={mw}, logP={logP}). "
        f"BCS {bcs_class}, {recommendation}. Risk={risk:.0f}/100."
    )

    return ADMETProfile(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        logS=logS,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        fup=fup,
        peff_cm_s=peff,
        bcs_class=bcs_class,
        f_oral=f_oral,
        clint_L_per_h_per_kg=clint,
        vd_L_per_kg=vd,
        t_half_h=t_half,
        lipinski_pass=lipinski_pass,
        bbb_penetrant=bbb_penetrant,
        herg_risk=herg_risk,
        overall_risk_score=risk,
        drug_likeness_score=dl_score,
        development_recommendation=recommendation,
        notes=notes,
    )


def screen_admet(compounds: list[dict]) -> list[ADMETProfile]:
    """Screen multiple compounds and rank by risk.

    Parameters
    ----------
    compounds : List of dicts with keys: name, smiles, mw, logP, and
        optionally logS, tpsa, pka, cmax_uM.

    Returns
    -------
    List of ADMETProfile sorted by overall_risk_score ascending (best first).
    """
    profiles = [
        predict_admet(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            logS=c.get("logS"),
            tpsa=c.get("tpsa", 50.0),
            pka=c.get("pka", 7.0),
            cmax_uM=c.get("cmax_uM", 1.0),
        )
        for c in compounds
    ]
    return sorted(profiles, key=lambda p: p.overall_risk_score)


__all__ = [
    "ADMETProfile",
    "predict_admet",
    "screen_admet",
]
