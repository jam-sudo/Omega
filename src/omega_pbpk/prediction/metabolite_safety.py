"""Safety assessment of major drug metabolites."""

from __future__ import annotations

# Known reactive metabolite structural alerts
_KNOWN_ALERTS = frozenset(
    {
        "epoxide",
        "quinone",
        "michael_acceptor",
        "acyl_halide",
        "nitroaromatic",
        "aniline",
    }
)

_FDA_MIST_THRESHOLD_PCT = 10.0


def mist_assessment(
    parent_auc_ng_h_mL: float,
    metabolite_auc_ng_h_mL: float,
    metabolite_mw: float,
    parent_mw: float,
    is_unique_to_humans: bool = False,
) -> dict:
    """FDA MIST metabolite safety assessment.

    Converts mass-based AUC ratios to molar ratios using molecular weights.
    Threshold for qualification: metabolite molar fraction > 10%.

    Returns dict with:
      metabolite_auc_mass_fraction, metabolite_molar_fraction,
      requires_qualification, human_unique_concern,
      fda_guidance_threshold_pct, recommendation
    """
    if parent_auc_ng_h_mL <= 0:
        raise ValueError("parent_auc_ng_h_mL must be > 0")
    if metabolite_auc_ng_h_mL < 0:
        raise ValueError("metabolite_auc_ng_h_mL must be >= 0")
    if metabolite_mw <= 0:
        raise ValueError("metabolite_mw must be > 0")
    if parent_mw <= 0:
        raise ValueError("parent_mw must be > 0")

    # Mass-based fraction
    metabolite_auc_mass_fraction = metabolite_auc_ng_h_mL / parent_auc_ng_h_mL

    # Convert to molar: molar_AUC = mass_AUC / MW
    # molar_fraction = (met_AUC/met_MW) / (parent_AUC/parent_MW)
    #                = mass_fraction * parent_MW / met_MW
    metabolite_molar_fraction = metabolite_auc_mass_fraction * (parent_mw / metabolite_mw)

    requires_qualification = metabolite_molar_fraction > (_FDA_MIST_THRESHOLD_PCT / 100.0)
    human_unique_concern = is_unique_to_humans and requires_qualification

    if human_unique_concern:
        recommendation = (
            "Metabolite is unique to humans and exceeds MIST threshold. "
            "Dedicated safety studies required before Phase 2/3."
        )
    elif requires_qualification:
        recommendation = (
            "Metabolite exceeds 10% molar MIST threshold. "
            "Qualification studies in nonclinical species recommended."
        )
    else:
        recommendation = (
            "Metabolite below MIST threshold. "
            "No additional qualification required based on exposure alone."
        )

    return {
        "metabolite_auc_mass_fraction": metabolite_auc_mass_fraction,
        "metabolite_molar_fraction": metabolite_molar_fraction,
        "requires_qualification": requires_qualification,
        "human_unique_concern": human_unique_concern,
        "fda_guidance_threshold_pct": _FDA_MIST_THRESHOLD_PCT,
        "recommendation": recommendation,
    }


def reactive_metabolite_risk(structural_alerts: list[str]) -> dict:
    """Assess reactive metabolite risk from structural alerts.

    Known alerts: 'epoxide', 'quinone', 'michael_acceptor',
                  'acyl_halide', 'nitroaromatic', 'aniline'

    Risk level:
      - 0 alerts  -> 'low'
      - 1 alert   -> 'moderate'
      - >= 2 alerts -> 'high'

    Returns dict with: alerts_found, risk_score, risk_level, mitigation_suggestions
    """
    if not isinstance(structural_alerts, list):
        raise TypeError("structural_alerts must be a list")

    alerts_found = [a for a in structural_alerts if a in _KNOWN_ALERTS]
    risk_score = len(alerts_found)

    if risk_score >= 2:
        risk_level = "high"
    elif risk_score == 1:
        risk_level = "moderate"
    else:
        risk_level = "low"

    mitigation_suggestions: list[str] = []
    if "epoxide" in alerts_found:
        mitigation_suggestions.append(
            "Consider structural modification to avoid epoxide formation (e.g., fluorine substitution)."
        )
    if "quinone" in alerts_found:
        mitigation_suggestions.append("Block para/ortho positions to prevent quinone formation.")
    if "michael_acceptor" in alerts_found:
        mitigation_suggestions.append(
            "Reduce alpha,beta-unsaturated carbonyl reactivity via saturation or substitution."
        )
    if "acyl_halide" in alerts_found:
        mitigation_suggestions.append(
            "Avoid halogenated acyl groups; consider ester or amide bioisosteres."
        )
    if "nitroaromatic" in alerts_found:
        mitigation_suggestions.append(
            "Replace nitro group with alternative electron-withdrawing group (e.g., CN, CF3)."
        )
    if "aniline" in alerts_found:
        mitigation_suggestions.append(
            "Mask or remove aniline motif; consider N-alkylation or ring substitution."
        )
    if not mitigation_suggestions:
        mitigation_suggestions.append(
            "No structural modifications required based on reactive metabolite alerts."
        )

    return {
        "alerts_found": alerts_found,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "mitigation_suggestions": mitigation_suggestions,
    }


def gssg_trapping_prediction(
    logP: float,
    mw: float,
    contains_sulfur: bool = False,
) -> dict:
    """Estimate GSH trapping propensity for electrophilic metabolites.

    High risk:    logP > 2 AND mw < 450 AND contains_sulfur
    Moderate:     logP > 1 OR mw < 400
    Low:          otherwise

    Returns dict with: gsh_trapping_risk, score, notes
    """
    score = 0
    notes: list[str] = []

    if logP > 2:
        score += 1
        notes.append(
            "logP > 2: increased membrane permeability facilitates intracellular reactive metabolite exposure."
        )
    if mw < 450:
        score += 1
        notes.append(
            "MW < 450: smaller molecules more readily form reactive species that can be trapped by GSH."
        )
    if contains_sulfur:
        score += 1
        notes.append(
            "Sulfur-containing compound: thiol/thioether groups can participate in reactive metabolite formation."
        )

    # Determine risk
    high_risk = logP > 2 and mw < 450 and contains_sulfur
    if high_risk:
        gsh_trapping_risk = "high"
    elif logP > 1 or mw < 400:
        gsh_trapping_risk = "moderate"
    else:
        gsh_trapping_risk = "low"

    if not notes:
        notes.append("No high-risk physicochemical features identified for GSH trapping.")

    return {
        "gsh_trapping_risk": gsh_trapping_risk,
        "score": score,
        "notes": notes,
    }


def metabolite_exposure_margin(
    parent_noael_mg_kg: float,
    metabolite_fraction: float,
    safety_factor: float = 10.0,
) -> dict:
    """Estimate metabolite safety margin from parent NOAEL.

    metabolite_NOAEL_equivalent = parent_noael / metabolite_fraction
    safety_margin_fold = metabolite_NOAEL_equivalent / safety_factor
    adequate = safety_margin_fold > 10.0

    Returns dict with: metabolite_noael_equivalent, safety_margin_fold, adequate
    """
    if parent_noael_mg_kg <= 0:
        raise ValueError("parent_noael_mg_kg must be > 0")
    if metabolite_fraction <= 0 or metabolite_fraction > 1.0:
        raise ValueError("metabolite_fraction must be in (0, 1]")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be > 0")

    metabolite_noael_equivalent = parent_noael_mg_kg / metabolite_fraction
    safety_margin_fold = metabolite_noael_equivalent / safety_factor
    adequate = safety_margin_fold > 10.0

    return {
        "metabolite_noael_equivalent": metabolite_noael_equivalent,
        "safety_margin_fold": safety_margin_fold,
        "adequate": adequate,
    }
