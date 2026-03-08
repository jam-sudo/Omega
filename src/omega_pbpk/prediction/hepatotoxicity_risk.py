"""Hepatotoxicity (DILI) risk prediction (Phase 978).

Predicts drug-induced liver injury (DILI) risk from physicochemical and
structural features using a rule-based scoring approach.

Key references:
- Xu JJ et al., Toxicol Sci. 2015 (DILI prediction, physicochemical descriptors)
- Lammert C et al., Gastroenterology. 2010;139(4):1280-1287 (dose-DILI rule)
- Dawson S et al., Arch Toxicol. 2012;86(11):1681-8 (BSEP inhibition)
- Dykens JA & Will Y, Drug Discov Today. 2007 (mitochondrial liability)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HepatotoxicityRiskResult:
    """Results from hepatotoxicity risk prediction.

    Attributes
    ----------
    drug_name : str
    logp : float
    mw : float
    daily_dose_mg : float
    dili_score : float — composite DILI score 0–100
    dili_risk : str — "low" (< 30), "moderate" (30–60), "high" (> 60)
    bsep_inhibition_risk : str — "low", "moderate", or "high"
    mitochondrial_liability_risk : str — "low", "moderate", or "high"
    dose_dependent_risk : bool — True if daily_dose_mg > 100
    most_concern_factor : str — highest-scoring risk factor name
    notes : str
    """

    drug_name: str
    logp: float
    mw: float
    daily_dose_mg: float
    dili_score: float
    dili_risk: str
    bsep_inhibition_risk: str
    mitochondrial_liability_risk: str
    dose_dependent_risk: bool
    most_concern_factor: str
    notes: str


def predict_hepatotoxicity_risk(
    drug_name: str,
    logp: float,
    mw: float,
    daily_dose_mg: float = 10.0,
    has_reactive_metabolite: bool = False,
    has_mitochondrial_liability: bool = False,
    has_bsep_inhibition: bool = False,
    ionization_type: str = "neutral",
) -> HepatotoxicityRiskResult:
    """Predict hepatotoxicity risk from physicochemical and structural features.

    Parameters
    ----------
    drug_name : Drug name.
    logp : Octanol-water partition coefficient.
    mw : Molecular weight (g/mol).
    daily_dose_mg : Maximum daily dose (mg/day). Default 10 mg.
    has_reactive_metabolite : Drug forms reactive metabolites. Default False.
    has_mitochondrial_liability : Drug has mitochondrial toxicity potential. Default False.
    has_bsep_inhibition : Drug inhibits bile salt export pump. Default False.
    ionization_type : One of "acid", "base", "neutral". Default "neutral".

    Returns
    -------
    HepatotoxicityRiskResult

    Raises
    ------
    ValueError : On invalid parameters.
    """
    valid_ionization = {"acid", "base", "neutral"}
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")
    if ionization_type not in valid_ionization:
        raise ValueError(
            f"ionization_type must be one of {valid_ionization}, got {ionization_type!r}"
        )

    # DILI score components — track each contribution for most_concern_factor
    contributions: dict[str, float] = {}

    base = 5.0
    contributions["base"] = base

    if logp > 3:
        contributions["logP > 3 (lipophilic)"] = 15.0

    if mw > 500:
        contributions["MW > 500 (large molecule)"] = 10.0

    if daily_dose_mg > 100:
        contributions["daily_dose > 100 mg (Lammert rule)"] = 20.0

    if has_reactive_metabolite:
        contributions["reactive metabolite"] = 25.0

    if has_mitochondrial_liability:
        contributions["mitochondrial liability"] = 20.0

    if has_bsep_inhibition:
        contributions["BSEP inhibition"] = 15.0

    if ionization_type == "acid" and logp > 2:
        contributions["acidic + lipophilic (BSEP risk)"] = 10.0

    # Composite DILI score capped at 100
    raw_score = sum(contributions.values())
    dili_score = min(100.0, raw_score)

    # DILI risk category
    if dili_score < 30:
        dili_risk = "low"
    elif dili_score <= 60:
        dili_risk = "moderate"
    else:
        dili_risk = "high"

    # BSEP inhibition risk
    if has_bsep_inhibition:
        bsep_risk = "high"
    elif logp > 3 and ionization_type == "acid":
        bsep_risk = "moderate"
    else:
        bsep_risk = "low"

    # Mitochondrial liability risk
    if has_mitochondrial_liability:
        mito_risk = "high"
    elif logp > 4:
        mito_risk = "moderate"
    else:
        mito_risk = "low"

    # Dose-dependent risk flag
    dose_dependent = daily_dose_mg > 100

    # Most concern factor: exclude base, find factor with highest contribution
    non_base = {k: v for k, v in contributions.items() if k != "base"}
    if non_base:
        most_concern = max(non_base, key=lambda k: non_base[k])
    else:
        most_concern = "base (no elevated risk factors)"

    notes = (
        f"DILI risk assessment: {drug_name}. "
        f"logP={logp:.2f}, MW={mw:.1f} g/mol, daily_dose={daily_dose_mg:.1f} mg. "
        f"DILI score={dili_score:.1f}/100 ({dili_risk}). "
        f"BSEP={bsep_risk}, mitochondrial={mito_risk}. "
        f"Dose-dependent risk: {dose_dependent}. "
        f"Primary concern: {most_concern}."
    )

    return HepatotoxicityRiskResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        daily_dose_mg=daily_dose_mg,
        dili_score=dili_score,
        dili_risk=dili_risk,
        bsep_inhibition_risk=bsep_risk,
        mitochondrial_liability_risk=mito_risk,
        dose_dependent_risk=dose_dependent,
        most_concern_factor=most_concern,
        notes=notes,
    )


def screen_hepatotoxicity(compounds: list) -> list:
    """Screen a list of compound dicts and return results sorted by dili_score descending.

    Each compound dict should contain keys matching parameters of
    predict_hepatotoxicity_risk(). Required keys: drug_name, logp, mw.
    Optional keys: daily_dose_mg, has_reactive_metabolite,
    has_mitochondrial_liability, has_bsep_inhibition, ionization_type.

    Parameters
    ----------
    compounds : list of dicts, each with compound parameters.

    Returns
    -------
    list of HepatotoxicityRiskResult, sorted descending by dili_score.
    """
    results: list[HepatotoxicityRiskResult] = []
    for cmpd in compounds:
        result = predict_hepatotoxicity_risk(
            drug_name=cmpd["drug_name"],
            logp=cmpd["logp"],
            mw=cmpd["mw"],
            daily_dose_mg=cmpd.get("daily_dose_mg", 10.0),
            has_reactive_metabolite=cmpd.get("has_reactive_metabolite", False),
            has_mitochondrial_liability=cmpd.get("has_mitochondrial_liability", False),
            has_bsep_inhibition=cmpd.get("has_bsep_inhibition", False),
            ionization_type=cmpd.get("ionization_type", "neutral"),
        )
        results.append(result)

    results.sort(key=lambda r: r.dili_score, reverse=True)
    return results


__all__ = [
    "HepatotoxicityRiskResult",
    "predict_hepatotoxicity_risk",
    "screen_hepatotoxicity",
]
