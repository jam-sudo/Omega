"""Metabolite Safety Profiling — Phase 1048.

Predict safety risks of drug metabolites based on structural and exposure characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_MET_TYPES = {"phase1", "phase2", "reactive"}
_VALID_ORGANS = {"liver", "kidney", "heart", "bone_marrow"}


# ---------------------------------------------------------------------------
# Result dataclass  (frozen=True — no list fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaboliteSafetyResult:
    drug_name: str
    metabolite_name: str
    metabolite_type: str
    mw_Da: float
    exposure_ratio: float
    reactive_risk: float
    organ_toxicity_risk: float
    genotoxic_risk: float
    overall_safety_score: float
    fda_mist_required: bool
    safety_class: str
    structural_alerts: tuple  # tuple so frozen=True works; behaves like list for iteration
    notes: str


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_metabolite_safety(
    drug_name: str,
    metabolite_name: str,
    parent_mw_Da: float,
    metabolite_mw_Da: float,
    metabolite_type: str,
    exposure_ratio: float,
    has_reactive_group: bool = False,
    has_quinone_methide: bool = False,
    has_arene_oxide: bool = False,
    has_nitroso: bool = False,
    has_acyl_glucuronide: bool = False,
    metabolite_logp: float = 1.0,
    organ_target: str = "liver",
) -> MetaboliteSafetyResult:
    """Predict safety risk profile of a drug metabolite.

    Parameters
    ----------
    drug_name:
        Parent drug name.
    metabolite_name:
        Metabolite identifier.
    parent_mw_Da:
        Parent drug molecular weight (Da).
    metabolite_mw_Da:
        Metabolite molecular weight (Da).
    metabolite_type:
        "phase1", "phase2", or "reactive".
    exposure_ratio:
        Metabolite AUC / parent AUC.
    has_reactive_group:
        Presence of electrophilic/reactive moiety.
    has_quinone_methide:
        Quinone methide structural alert.
    has_arene_oxide:
        Arene oxide structural alert.
    has_nitroso:
        Nitroso structural alert.
    has_acyl_glucuronide:
        Acyl glucuronide (phase 2 reactive) alert.
    metabolite_logp:
        Metabolite log octanol-water partition coefficient.
    organ_target:
        Primary organ of concern: "liver", "kidney", "heart", or "bone_marrow".

    Returns
    -------
    MetaboliteSafetyResult
    """
    # --- validation ---
    if exposure_ratio < 0:
        raise ValueError(f"exposure_ratio must be >= 0, got {exposure_ratio}")
    if parent_mw_Da <= 0:
        raise ValueError(f"parent_mw_Da must be > 0, got {parent_mw_Da}")
    if metabolite_mw_Da <= 0:
        raise ValueError(f"metabolite_mw_Da must be > 0, got {metabolite_mw_Da}")
    if metabolite_type not in _VALID_MET_TYPES:
        raise ValueError(
            f"metabolite_type must be one of {_VALID_MET_TYPES}, got {metabolite_type!r}"
        )
    if organ_target not in _VALID_ORGANS:
        raise ValueError(f"organ_target must be one of {_VALID_ORGANS}, got {organ_target!r}")

    # --- reactive_risk (0-100) ---
    if metabolite_type == "reactive":
        rr_base = 50.0
    elif metabolite_type == "phase1" and not has_reactive_group:
        rr_base = 10.0
    elif metabolite_type == "phase2" and not has_acyl_glucuronide:
        rr_base = 5.0
    else:
        rr_base = 0.0  # will be overridden below by alert additions

    # For phase1/phase2, start from their base unless reactive flags apply
    reactive_risk = rr_base
    if has_reactive_group:
        reactive_risk += 20.0
    if has_quinone_methide:
        reactive_risk += 30.0
    if has_arene_oxide:
        reactive_risk += 25.0
    if has_nitroso:
        reactive_risk += 35.0
    if has_acyl_glucuronide:
        reactive_risk += 15.0

    # Exposure penalty
    exposure_mult = min(2.0, 1.0 + exposure_ratio)
    reactive_risk = reactive_risk * exposure_mult
    reactive_risk = max(0.0, min(100.0, reactive_risk))

    # --- organ_toxicity_risk (0-100) ---
    organ_risk = 15.0
    organ_add = {
        "liver": 10.0,
        "kidney": 5.0,
        "heart": 15.0,
        "bone_marrow": 20.0,
    }
    organ_risk += organ_add[organ_target]
    if exposure_ratio > 1.0:
        organ_risk += 15.0
    if has_reactive_group:
        organ_risk += 20.0
    if metabolite_logp > 3.0:
        organ_risk += 10.0
    organ_risk = max(0.0, min(100.0, organ_risk))

    # --- genotoxic_risk (0-100) ---
    geo_risk = 5.0
    if has_arene_oxide:
        geo_risk += 30.0
    if has_nitroso:
        geo_risk += 25.0
    if has_quinone_methide:
        geo_risk += 20.0
    if metabolite_type == "reactive":
        geo_risk += 15.0
    if has_reactive_group:
        geo_risk += 10.0
    geo_risk = max(0.0, min(100.0, geo_risk))

    # --- overall_safety_score ---
    weighted = reactive_risk * 0.4 + organ_risk * 0.3 + geo_risk * 0.3
    overall_safety_score = max(0.0, 100.0 - weighted)

    # --- FDA MIST ---
    fda_mist_required = exposure_ratio >= 0.1

    # --- safety_class ---
    if overall_safety_score >= 75.0:
        safety_class = "acceptable"
    elif overall_safety_score >= 50.0:
        safety_class = "monitor"
    elif overall_safety_score >= 30.0:
        safety_class = "concern"
    else:
        safety_class = "high_risk"

    # --- structural_alerts ---
    alerts: list[str] = []
    if has_reactive_group:
        alerts.append("Reactive electrophile")
    if has_quinone_methide:
        alerts.append("Quinone methide")
    if has_arene_oxide:
        alerts.append("Arene oxide")
    if has_nitroso:
        alerts.append("Nitroso compound")
    if has_acyl_glucuronide:
        alerts.append("Acyl glucuronide")
    if not alerts:
        alerts = ["No structural alerts"]

    notes = (
        f"Metabolite type: {metabolite_type}. Exposure ratio: {exposure_ratio:.3f}. "
        f"Organ target: {organ_target}. "
        f"FDA MIST required: {fda_mist_required}. "
        f"Safety class: {safety_class}."
    )

    return MetaboliteSafetyResult(
        drug_name=drug_name,
        metabolite_name=metabolite_name,
        metabolite_type=metabolite_type,
        mw_Da=metabolite_mw_Da,
        exposure_ratio=exposure_ratio,
        reactive_risk=reactive_risk,
        organ_toxicity_risk=organ_risk,
        genotoxic_risk=geo_risk,
        overall_safety_score=overall_safety_score,
        fda_mist_required=fda_mist_required,
        safety_class=safety_class,
        structural_alerts=tuple(alerts),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------


def screen_metabolites(metabolites: list) -> list:
    """Screen a list of metabolite parameter dicts through predict_metabolite_safety.

    Parameters
    ----------
    metabolites:
        List of dicts with keys matching predict_metabolite_safety parameters.

    Returns
    -------
    list[MetaboliteSafetyResult] sorted by overall_safety_score descending (safest first).
    """
    results = [predict_metabolite_safety(**m) for m in metabolites]
    return sorted(results, key=lambda r: r.overall_safety_score, reverse=True)
