"""Multi-drug resistance (MDR) risk prediction for cancer chemotherapy."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MDRRiskResult",
    "mdr_risk_score",
    "efflux_ratio_impact",
    "predict_combination_mdr",
    "screen_mdr_liability",
]

_VALID_DRUG_CLASSES = {"cytotoxic", "targeted", "immunotherapy", "other"}


@dataclass(frozen=True)
class MDRRiskResult:
    """MDR risk result for a single compound."""

    compound_name: str
    smiles: str
    mw: float
    logP: float
    psa: float
    is_pgp_substrate: bool
    is_bcrp_substrate: bool
    is_mrp_substrate: bool
    mdr_score: float  # 0-1
    risk_category: str  # low / moderate / high
    efflux_mechanisms: list[str]
    mitigation: str
    notes: str


def _validate_mdr_inputs(mw: float, psa: float, drug_class: str) -> None:
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(
            f"drug_class must be one of {sorted(_VALID_DRUG_CLASSES)}, got '{drug_class}'"
        )


def mdr_risk_score(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    psa: float,
    is_pgp_substrate: bool,
    is_bcrp_substrate: bool,
    is_mrp_substrate: bool,
    drug_class: str = "cytotoxic",
) -> MDRRiskResult:
    """Compute MDR risk score from transporter substrate flags and physicochemistry.

    Scoring contributions
    ---------------------
    P-gp substrate  : 0.4
    BCRP substrate  : 0.3
    MRP substrate   : 0.2
    logP > 3        : 0.1
    Total range     : 0.0 – 1.0

    Risk categories
    ---------------
    low      : score < 0.3
    moderate : 0.3 <= score <= 0.6
    high     : score > 0.6
    """
    _validate_mdr_inputs(mw, psa, drug_class)

    pgp_contrib = 0.4 if is_pgp_substrate else 0.0
    bcrp_contrib = 0.3 if is_bcrp_substrate else 0.0
    mrp_contrib = 0.2 if is_mrp_substrate else 0.0
    pc_contrib = 0.1 if logP > 3 else 0.0

    score = round(pgp_contrib + bcrp_contrib + mrp_contrib + pc_contrib, 4)

    if score < 0.3:
        risk_category = "low"
    elif score <= 0.6:
        risk_category = "moderate"
    else:
        risk_category = "high"

    mechanisms: list[str] = []
    if is_pgp_substrate:
        mechanisms.append("P-gp (ABCB1)")
    if is_bcrp_substrate:
        mechanisms.append("BCRP (ABCG2)")
    if is_mrp_substrate:
        mechanisms.append("MRP1/2/3 (ABCC)")

    # Mitigation strategy
    if risk_category == "high":
        mitigation = (
            "Consider P-gp/BCRP inhibitor co-administration or structural "
            "modification to reduce efflux liability. Monitor intracellular accumulation."
        )
    elif risk_category == "moderate":
        mitigation = (
            "Monitor clinical resistance development. "
            "Combination regimens may overcome partial efflux."
        )
    else:
        mitigation = "Low MDR liability; standard dosing regimen expected to be adequate."

    notes = (
        f"{compound_name} ({drug_class}): MDR score={score:.2f} ({risk_category}). "
        f"Active efflux via: {', '.join(mechanisms) if mechanisms else 'none detected'}. "
        f"logP={logP:.1f}, MW={mw:.0f}, PSA={psa:.0f}."
    )

    return MDRRiskResult(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        psa=psa,
        is_pgp_substrate=is_pgp_substrate,
        is_bcrp_substrate=is_bcrp_substrate,
        is_mrp_substrate=is_mrp_substrate,
        mdr_score=score,
        risk_category=risk_category,
        efflux_mechanisms=mechanisms,
        mitigation=mitigation,
        notes=notes,
    )


def efflux_ratio_impact(
    er_pgp: float,
    er_bcrp: float,
    fm_pgp: float = 0.7,
    fm_bcrp: float = 0.3,
) -> dict:
    """Estimate intracellular drug penetration from efflux ratio data.

    Parameters
    ----------
    er_pgp  : Efflux ratio measured in a P-gp-expressing assay (e.g. MDR1-MDCK).
    er_bcrp : Efflux ratio measured in a BCRP-expressing assay.
    fm_pgp  : Fractional contribution of P-gp to total efflux (default 0.7).
    fm_bcrp : Fractional contribution of BCRP to total efflux (default 0.3).

    Returns
    -------
    dict with:
        effective_efflux  - weighted efflux burden (dimensionless)
        intracellular_fraction - estimated intracellular/extracellular ratio (0-1)
        resistance_fold_change - fold reduction in effective intracellular concentration
    """
    if er_pgp < 0:
        raise ValueError("er_pgp must be >= 0")
    if er_bcrp < 0:
        raise ValueError("er_bcrp must be >= 0")
    if not (0 <= fm_pgp <= 1):
        raise ValueError("fm_pgp must be in [0, 1]")
    if not (0 <= fm_bcrp <= 1):
        raise ValueError("fm_bcrp must be in [0, 1]")

    effective_efflux = fm_pgp * er_pgp + fm_bcrp * er_bcrp

    # Penetration = 1 / effective_efflux; clamp to [0, 1]
    if effective_efflux <= 0:
        intracellular_fraction = 1.0
        resistance_fold_change = 1.0
    else:
        raw_penetration = 1.0 / effective_efflux
        intracellular_fraction = min(1.0, raw_penetration)
        resistance_fold_change = max(1.0, effective_efflux)

    return {
        "effective_efflux": round(effective_efflux, 4),
        "intracellular_fraction": round(intracellular_fraction, 4),
        "resistance_fold_change": round(resistance_fold_change, 4),
    }


def predict_combination_mdr(drug1: dict, drug2: dict) -> dict:
    """Predict MDR outcome for a two-drug combination.

    Each drug dict should contain:
        name            : str
        is_pgp_substrate: bool
        is_pgp_inhibitor: bool  (optional, default False)
        mdr_score       : float (0-1, optional — computed if not provided)

    Logic
    -----
    - If both are P-gp substrates: additive competition for efflux → combined MDR risk
      is the average of both scores plus a 0.1 synergy penalty (capped at 1.0).
    - If one inhibits P-gp and the other is a P-gp substrate: the inhibitor reduces the
      substrate's effective MDR score by 50%.
    - Otherwise: combined score is the maximum of individual scores.
    """
    name1 = drug1.get("name", "Drug1")
    name2 = drug2.get("name", "Drug2")
    sub1 = bool(drug1.get("is_pgp_substrate", False))
    sub2 = bool(drug2.get("is_pgp_substrate", False))
    inh1 = bool(drug1.get("is_pgp_inhibitor", False))
    inh2 = bool(drug2.get("is_pgp_inhibitor", False))
    score1 = float(drug1.get("mdr_score", 0.0))
    score2 = float(drug2.get("mdr_score", 0.0))

    if sub1 and sub2:
        # Additive/synergistic efflux competition
        combined_score = min(1.0, (score1 + score2) / 2.0 + 0.1)
        interaction = "synergistic_efflux_competition"
        note = (
            f"{name1} and {name2} both compete for P-gp efflux; "
            "combined MDR risk is elevated (additive + 0.1 synergy penalty)."
        )
    elif (inh1 and sub2) or (inh2 and sub1):
        # One drug inhibits P-gp → reduces MDR of the substrate
        if inh1 and sub2:
            inhibitor, substrate = name1, name2
            reduced_score = score2 * 0.5
            combined_score = max(score1, reduced_score)
        else:
            inhibitor, substrate = name2, name1
            reduced_score = score1 * 0.5
            combined_score = max(score2, reduced_score)
        interaction = "pgp_inhibition_reduces_mdr"
        note = (
            f"{inhibitor} inhibits P-gp, reducing effective MDR score of {substrate} by 50%. "
            f"Effective combined MDR score = {combined_score:.2f}."
        )
        combined_score = round(combined_score, 4)
    else:
        combined_score = max(score1, score2)
        interaction = "independent"
        note = (
            f"No P-gp substrate-inhibitor interaction detected. "
            f"Combined MDR risk = max of individual scores = {combined_score:.2f}."
        )

    combined_score = round(combined_score, 4)

    if combined_score < 0.3:
        combined_category = "low"
    elif combined_score <= 0.6:
        combined_category = "moderate"
    else:
        combined_category = "high"

    return {
        "drug1_name": name1,
        "drug2_name": name2,
        "drug1_mdr_score": score1,
        "drug2_mdr_score": score2,
        "combined_mdr_score": combined_score,
        "combined_risk_category": combined_category,
        "interaction_type": interaction,
        "notes": note,
    }


def screen_mdr_liability(compounds: list[dict]) -> list[MDRRiskResult]:
    """Screen a list of compound dicts for MDR liability.

    Each dict must contain the same keyword arguments as ``mdr_risk_score``.

    Required keys: compound_name, smiles, mw, logP, psa,
                   is_pgp_substrate, is_bcrp_substrate, is_mrp_substrate.
    Optional key : drug_class (default "cytotoxic").

    Returns results sorted by descending mdr_score.
    """
    results: list[MDRRiskResult] = []
    for cpd in compounds:
        result = mdr_risk_score(
            compound_name=cpd["compound_name"],
            smiles=cpd.get("smiles", ""),
            mw=cpd["mw"],
            logP=cpd["logP"],
            psa=cpd["psa"],
            is_pgp_substrate=bool(cpd.get("is_pgp_substrate", False)),
            is_bcrp_substrate=bool(cpd.get("is_bcrp_substrate", False)),
            is_mrp_substrate=bool(cpd.get("is_mrp_substrate", False)),
            drug_class=cpd.get("drug_class", "cytotoxic"),
        )
        results.append(result)

    return sorted(results, key=lambda r: r.mdr_score, reverse=True)
