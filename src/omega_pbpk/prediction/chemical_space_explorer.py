"""Phase 215: Chemical space explorer using Lipinski and extended drug-likeness rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrugLikenessResult:
    """Result of drug-likeness evaluation."""

    drug_name: str
    mw_Da: float
    logP: float
    hbd: int
    hba: int
    psa_A2: float
    rotatable_bonds: int
    ro5_violations: int
    veber_pass: bool
    pfizer_flag: bool
    lead_like: bool
    pains_alerts: int
    overall_score: float
    drug_likeness_class: str
    notes: str


def _compute_drug_likeness_class(score: float) -> str:
    if score >= 80:
        return "excellent"
    elif score >= 60:
        return "good"
    elif score >= 40:
        return "moderate"
    else:
        return "poor"


def evaluate_drug_likeness(
    drug_name: str,
    mw_Da: float,
    logP: float,
    hbd: int,
    hba: int,
    psa_A2: float,
    rotatable_bonds: int,
) -> DrugLikenessResult:
    """Evaluate drug-likeness using Lipinski Ro5 and extended rules.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient.
    hbd:
        Hydrogen bond donor count.
    hba:
        Hydrogen bond acceptor count.
    psa_A2:
        Polar surface area in Angstrom squared.
    rotatable_bonds:
        Number of rotatable bonds.

    Returns
    -------
    DrugLikenessResult
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be positive, got {mw_Da}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be non-negative, got {psa_A2}")

    notes_parts: list[str] = []

    # Lipinski Rule of Five violations
    ro5_violations = 0
    if mw_Da > 500:
        ro5_violations += 1
        notes_parts.append("MW>500")
    if logP > 5:
        ro5_violations += 1
        notes_parts.append("logP>5")
    if hbd > 5:
        ro5_violations += 1
        notes_parts.append("HBD>5")
    if hba > 10:
        ro5_violations += 1
        notes_parts.append("HBA>10")

    # Veber rules: oral bioavailability likely
    veber_pass = psa_A2 <= 140 and rotatable_bonds <= 10

    # Pfizer 3/75 rule: toxicity risk
    pfizer_flag = logP > 3 and psa_A2 < 75

    # Lead-like (Oprea criteria)
    lead_like = 200 <= mw_Da <= 350 and logP <= 3 and hbd <= 3 and hba <= 6

    # PAINS alerts (simplified)
    pains_alerts = 0
    if hbd > 3 and logP < 0:
        pains_alerts += 1
    if hba > 8:
        pains_alerts += 1

    # Overall score: start at 100, apply penalties
    score = 100.0
    score -= 15.0 * ro5_violations
    if not veber_pass:
        score -= 10.0
    if pfizer_flag:
        score -= 10.0

    # Clamp to [0, 100]
    score = max(0.0, min(100.0, score))

    drug_likeness_class = _compute_drug_likeness_class(score)

    if not notes_parts:
        notes_parts.append("No Ro5 violations")
    if veber_pass:
        notes_parts.append("Veber pass")
    if pfizer_flag:
        notes_parts.append("Pfizer 3/75 flag")
    if lead_like:
        notes_parts.append("Lead-like")
    if pains_alerts > 0:
        notes_parts.append(f"PAINS alerts: {pains_alerts}")

    notes = "; ".join(notes_parts)

    return DrugLikenessResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        hbd=hbd,
        hba=hba,
        psa_A2=psa_A2,
        rotatable_bonds=rotatable_bonds,
        ro5_violations=ro5_violations,
        veber_pass=veber_pass,
        pfizer_flag=pfizer_flag,
        lead_like=lead_like,
        pains_alerts=pains_alerts,
        overall_score=score,
        drug_likeness_class=drug_likeness_class,
        notes=notes,
    )


def screen_compound_library(
    compounds: list[dict],
) -> list[DrugLikenessResult]:
    """Screen a library of compounds and return results sorted by overall_score descending.

    Each compound dict must have keys: drug_name, mw_Da, logP, hbd, hba, psa_A2, rotatable_bonds.

    Parameters
    ----------
    compounds:
        List of compound parameter dicts.

    Returns
    -------
    List[DrugLikenessResult] sorted by overall_score descending.
    """
    results = []
    for c in compounds:
        result = evaluate_drug_likeness(
            drug_name=c["drug_name"],
            mw_Da=c["mw_Da"],
            logP=c["logP"],
            hbd=c["hbd"],
            hba=c["hba"],
            psa_A2=c["psa_A2"],
            rotatable_bonds=c["rotatable_bonds"],
        )
        results.append(result)

    results.sort(key=lambda r: r.overall_score, reverse=True)
    return results
