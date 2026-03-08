"""Phase 420 — Drug-Excipient Interaction Predictor.

Predicts drug-excipient interactions based on physicochemical properties.
Excipients can interact with APIs via pH changes, complex formation,
oxidation, hydrolysis. Critical for formulation stability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExcipientInteractionResult:
    """Result of a drug-excipient interaction prediction."""

    drug_name: str
    excipient_name: str
    interaction_type: str  # "none", "pH_shift", "complexation", "oxidation_risk",
    # "hydrolysis_risk", "adsorption"
    severity: str  # "none", "low", "moderate", "high"
    severity_score: float  # 0-100
    mechanism: str
    solubility_change_factor: float  # 1.0 = no change
    stability_impact_pct: float  # % drug loss at 25C/60%RH over 6 months
    compatibility: str  # "compatible", "conditionally_compatible", "incompatible"
    mitigation: str
    notes: str


# ---------------------------------------------------------------------------
# Excipient database
# ---------------------------------------------------------------------------

_EXCIPIENT_DB: dict = {
    "MCC": {
        "type": "filler",
        "ph": 6.5,
        "reducing": False,
        "acidic": False,
        "basic": False,
    },
    "lactose": {
        "type": "filler",
        "ph": 5.5,
        "reducing": True,
        "acidic": False,
        "basic": False,
    },
    "mannitol": {
        "type": "filler",
        "ph": 6.5,
        "reducing": False,
        "acidic": False,
        "basic": False,
    },
    "citric_acid": {
        "type": "acidifier",
        "ph": 3.0,
        "reducing": False,
        "acidic": True,
        "basic": False,
    },
    "Na_bicarbonate": {
        "type": "alkalizer",
        "ph": 8.5,
        "reducing": False,
        "acidic": False,
        "basic": True,
    },
    "Mg_stearate": {
        "type": "lubricant",
        "ph": 7.0,
        "reducing": False,
        "acidic": False,
        "basic": False,
    },
    "HPMC": {
        "type": "binder",
        "ph": 6.8,
        "reducing": False,
        "acidic": False,
        "basic": False,
    },
    "PVP": {
        "type": "binder",
        "ph": 5.5,
        "reducing": False,
        "acidic": False,
        "basic": False,
    },
    "SDS": {
        "type": "surfactant",
        "ph": 7.0,
        "reducing": False,
        "acidic": True,
        "basic": False,
    },
    "polysorbate_80": {
        "type": "surfactant",
        "ph": 6.0,
        "reducing": False,
        "acidic": False,
        "basic": False,
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _compatibility_from_score(score: float) -> str:
    """Derive compatibility category from severity score."""
    if score > 60:
        return "incompatible"
    elif score > 30:
        return "conditionally_compatible"
    else:
        return "compatible"


def _mitigation_for_type(interaction_type: str, severity: str) -> str:
    """Return a mitigation recommendation based on interaction type."""
    if interaction_type == "complexation":
        return "Avoid lactose with primary amine drugs; substitute with MCC or mannitol."
    elif interaction_type == "pH_shift":
        if severity in ("high", "moderate"):
            return (
                "Adjust formulation pH or use a buffering excipient to maintain "
                "drug stability. Consider enteric coating."
            )
        else:
            return "Monitor pH; minor shift unlikely to impact stability."
    elif interaction_type == "adsorption":
        return (
            "SDS may adsorb basic drugs; optimize concentration and consider "
            "alternative surfactants (e.g., polysorbate 80)."
        )
    elif interaction_type == "hydrolysis_risk":
        return (
            "Reduce moisture; use desiccant; consider anhydrous excipients. "
            "Monitor drug content at accelerated stability conditions."
        )
    else:
        return "No specific mitigation required."


def predict_excipient_interaction(
    drug_name: str,
    excipient_name: str,
    drug_pka: float,
    drug_logp: float,
    ionization_type: str,
    drug_mw_Da: float,
    drug_primary_amine: bool = False,
) -> ExcipientInteractionResult:
    """Predict the interaction between a drug and an excipient.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    excipient_name:
        Excipient name (must be in database).
    drug_pka:
        Drug pKa value.
    drug_logp:
        Drug logP value.
    ionization_type:
        'acid', 'base', or 'neutral'.
    drug_mw_Da:
        Drug molecular weight in Da.
    drug_primary_amine:
        Whether the drug contains a primary amine group (Maillard risk).

    Returns
    -------
    ExcipientInteractionResult

    Raises
    ------
    ValueError
        If excipient_name is not in the database.
    """
    if excipient_name not in _EXCIPIENT_DB:
        supported = ", ".join(sorted(_EXCIPIENT_DB.keys()))
        raise ValueError(f"Unknown excipient '{excipient_name}'. Supported: {supported}")

    exc = _EXCIPIENT_DB[excipient_name]
    exc_ph = exc["ph"]

    interaction_type = "none"
    severity = "none"
    severity_score = 0.0
    stability_impact_pct = 0.0
    mechanism = "No significant interaction identified."

    # --- Rule 1: Maillard reaction (reducing excipient + primary amine) ---
    if exc["reducing"] and drug_primary_amine:
        interaction_type = "complexation"
        severity = "high"
        severity_score = 75.0
        stability_impact_pct = 20.0
        mechanism = (
            "Maillard condensation between reducing sugar (lactose) and primary "
            "amine group of drug. Produces brown discoloration and drug loss."
        )

    # --- Rule 2: Acid shift (acidic excipient + base drug) ---
    elif exc["acidic"] and ionization_type == "base":
        delta_ph = abs(drug_pka - exc_ph)
        interaction_type = "pH_shift"
        if delta_ph > 3:
            severity = "high"
            severity_score = 70.0
            stability_impact_pct = 15.0
        elif delta_ph >= 1:
            severity = "moderate"
            severity_score = 40.0
            stability_impact_pct = 8.0
        else:
            severity = "low"
            severity_score = 15.0
            stability_impact_pct = 2.0
        mechanism = (
            f"Acidic excipient (pH {exc_ph}) shifts microenvironmental pH, "
            f"potentially ionizing base drug (pKa {drug_pka}). "
            f"pH difference = {delta_ph:.1f}."
        )

    # --- Rule 3: Basic shift (basic excipient + acid drug) ---
    elif exc["basic"] and ionization_type == "acid":
        delta_ph = abs(drug_pka - exc_ph)
        interaction_type = "pH_shift"
        if delta_ph > 3:
            severity = "high"
            severity_score = 70.0
            stability_impact_pct = 15.0
        elif delta_ph >= 1:
            severity = "moderate"
            severity_score = 40.0
            stability_impact_pct = 8.0
        else:
            severity = "low"
            severity_score = 15.0
            stability_impact_pct = 2.0
        mechanism = (
            f"Basic excipient (pH {exc_ph}) shifts microenvironmental pH, "
            f"potentially ionizing acid drug (pKa {drug_pka}). "
            f"pH difference = {delta_ph:.1f}."
        )

    # --- Rule 4: Adsorption (SDS + basic drug) ---
    elif excipient_name == "SDS" and ionization_type == "base":
        interaction_type = "adsorption"
        severity = "moderate"
        severity_score = 45.0
        stability_impact_pct = 5.0
        mechanism = (
            "Anionic SDS can electrostatically adsorb cationic (basic) drug, "
            "reducing free drug concentration and bioavailability."
        )

    # --- Rule 5: Hydrolysis risk (high pH excipient + ester/amide-like drug) ---
    elif exc_ph > 7.5 and drug_mw_Da > 300 and drug_logp > 2:
        interaction_type = "hydrolysis_risk"
        severity = "low"
        severity_score = 20.0
        stability_impact_pct = 3.0
        mechanism = (
            f"Alkaline microenvironment (excipient pH {exc_ph}) may catalyze "
            "hydrolysis of susceptible bonds in high-MW lipophilic drugs."
        )

    # --- Solubility change factor ---
    solubility_change_factor = 1.0
    if interaction_type == "pH_shift":
        if ionization_type == "acid" and exc["basic"]:
            # Basic excipient ionizes acid drug -> higher solubility
            solubility_change_factor = 1.0 + (exc_ph - drug_pka) * 0.3
        # Otherwise neutral
        solubility_change_factor = _clamp(solubility_change_factor, 0.1, 10.0)

    # --- Compatibility ---
    compatibility = _compatibility_from_score(severity_score)

    # --- Mitigation ---
    mitigation = _mitigation_for_type(interaction_type, severity)

    # --- Notes ---
    notes = (
        f"Drug '{drug_name}' with excipient '{excipient_name}': "
        f"interaction_type='{interaction_type}', severity='{severity}', "
        f"score={severity_score:.0f}, stability_impact={stability_impact_pct:.0f}%, "
        f"solubility_change_factor={solubility_change_factor:.2f}."
    )

    return ExcipientInteractionResult(
        drug_name=drug_name,
        excipient_name=excipient_name,
        interaction_type=interaction_type,
        severity=severity,
        severity_score=severity_score,
        mechanism=mechanism,
        solubility_change_factor=solubility_change_factor,
        stability_impact_pct=stability_impact_pct,
        compatibility=compatibility,
        mitigation=mitigation,
        notes=notes,
    )


def screen_excipient_compatibility(
    drug_name: str,
    drug_pka: float,
    drug_logp: float,
    ionization_type: str,
    drug_mw_Da: float,
    excipient_names: list,
    drug_primary_amine: bool = False,
) -> list:
    """Screen multiple excipients for compatibility with a drug.

    Parameters
    ----------
    drug_name:
        Drug name.
    drug_pka:
        Drug pKa.
    drug_logp:
        Drug logP.
    ionization_type:
        'acid', 'base', or 'neutral'.
    drug_mw_Da:
        Drug molecular weight (Da).
    excipient_names:
        List of excipient names to screen.
    drug_primary_amine:
        Whether drug contains a primary amine group.

    Returns
    -------
    List of ExcipientInteractionResult sorted by severity_score ascending (best first).
    """
    results = []
    for exc_name in excipient_names:
        result = predict_excipient_interaction(
            drug_name=drug_name,
            excipient_name=exc_name,
            drug_pka=drug_pka,
            drug_logp=drug_logp,
            ionization_type=ionization_type,
            drug_mw_Da=drug_mw_Da,
            drug_primary_amine=drug_primary_amine,
        )
        results.append(result)

    results.sort(key=lambda r: r.severity_score)
    return results
