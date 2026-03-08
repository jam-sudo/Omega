"""
Phase 450: Excipient Dissolution Impact Prediction
Predict how specific excipients (HPMC, PVP, surfactants) impact dissolution
rate enhancement of poorly soluble drugs.

Background: Polymer-surfactant combinations can synergistically enhance
dissolution via wetting (SDS, polysorbate), solubilization (bile salts),
and nucleation inhibition (PVP, HPMC).
"""

from dataclasses import dataclass

EXCIPIENTS = {
    "SDS": {
        "mechanism": "wetting",
        "max_factor": 5.0,
        "optimal_pct": 0.5,
        "supersaturation_h": 0.5,
        "spring": True,
        "parachute": False,
    },
    "polysorbate_80": {
        "mechanism": "solubilization",
        "max_factor": 4.0,
        "optimal_pct": 1.0,
        "supersaturation_h": 1.0,
        "spring": True,
        "parachute": False,
    },
    "PVP_K30": {
        "mechanism": "nucleation_inhibition",
        "max_factor": 3.0,
        "optimal_pct": 2.0,
        "supersaturation_h": 4.0,
        "spring": False,
        "parachute": True,
    },
    "HPMC_E5": {
        "mechanism": "nucleation_inhibition",
        "max_factor": 2.5,
        "optimal_pct": 1.0,
        "supersaturation_h": 6.0,
        "spring": False,
        "parachute": True,
    },
    "HPMC_AS": {
        "mechanism": "nucleation_inhibition",
        "max_factor": 4.0,
        "optimal_pct": 5.0,
        "supersaturation_h": 8.0,
        "spring": False,
        "parachute": True,
    },
    "cyclodextrin_beta": {
        "mechanism": "complexation",
        "max_factor": 10.0,
        "optimal_pct": 10.0,
        "supersaturation_h": 2.0,
        "spring": True,
        "parachute": True,
    },
    "lecithin": {
        "mechanism": "combined",
        "max_factor": 3.5,
        "optimal_pct": 2.0,
        "supersaturation_h": 3.0,
        "spring": True,
        "parachute": True,
    },
    "cremophor_EL": {
        "mechanism": "solubilization",
        "max_factor": 6.0,
        "optimal_pct": 5.0,
        "supersaturation_h": 1.5,
        "spring": True,
        "parachute": False,
    },
}


@dataclass(frozen=True)
class ExcipientDissolutionResult:
    """Results from excipient dissolution impact prediction."""

    drug_name: str
    excipient_name: str
    excipient_concentration_pct: float
    mechanism: str
    dissolution_enhancement_factor: float
    supersaturation_maintenance_h: float
    spring_effect: bool
    parachute_effect: bool
    t50_reduction_pct: float
    optimal_concentration_pct: float
    combination_synergy: str
    notes: str


def predict_excipient_dissolution(
    drug_name: str,
    excipient_name: str,
    excipient_concentration_pct: float,
    drug_logp: float = 2.0,
    drug_bcs_class: int = 2,
) -> ExcipientDissolutionResult:
    """
    Predict the dissolution enhancement from a specific excipient.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    excipient_name : str
        Name of the excipient (must be in EXCIPIENTS database).
    excipient_concentration_pct : float
        Excipient concentration as % w/w (must be > 0).
    drug_logp : float
        Octanol-water partition coefficient of the drug.
    drug_bcs_class : int
        BCS class of the drug (1-4).

    Returns
    -------
    ExcipientDissolutionResult
    """
    if excipient_name not in EXCIPIENTS:
        raise ValueError(
            f"Excipient '{excipient_name}' not found. Valid options: {sorted(EXCIPIENTS.keys())}"
        )
    if excipient_concentration_pct <= 0:
        raise ValueError(
            f"excipient_concentration_pct must be > 0, got {excipient_concentration_pct}"
        )

    exc_data = EXCIPIENTS[excipient_name]
    max_factor = exc_data["max_factor"]
    optimal_pct = exc_data["optimal_pct"]
    mechanism = exc_data["mechanism"]
    supersaturation_base_h = exc_data["supersaturation_h"]
    spring = exc_data["spring"]
    parachute = exc_data["parachute"]

    # Hill equation: factor = 1 + (max_factor - 1) * conc^2 / (optimal_pct^2 + conc^2)
    conc = excipient_concentration_pct
    conc_sq = conc * conc
    opt_sq = optimal_pct * optimal_pct
    dissolution_enhancement_factor = 1.0 + (max_factor - 1.0) * conc_sq / (opt_sq + conc_sq)

    # t50_reduction_pct = (1 - 1/factor) * 100, clamped [0, 95]
    t50_reduction_pct = (1.0 - 1.0 / dissolution_enhancement_factor) * 100.0
    t50_reduction_pct = max(0.0, min(95.0, t50_reduction_pct))

    # supersaturation_maintenance_h scales with concentration ratio, clamped [0.1, 24]
    conc_ratio = conc / optimal_pct if optimal_pct > 0 else 1.0
    scale = min(2.0, conc_ratio)
    supersaturation_maintenance_h = max(0.1, min(24.0, supersaturation_base_h * scale))

    # combination_synergy determination
    if drug_bcs_class == 2 and mechanism in {"nucleation_inhibition", "complexation"}:
        combination_synergy = "synergistic"
    elif dissolution_enhancement_factor > 2.0:
        combination_synergy = "additive"
    else:
        combination_synergy = "none"

    # Notes
    notes = (
        f"Excipient: {excipient_name} at {excipient_concentration_pct:.1f}% w/w. "
        f"Mechanism: {mechanism}. "
        f"Dissolution enhancement factor: {dissolution_enhancement_factor:.2f}x "
        f"(t50 reduction: {t50_reduction_pct:.1f}%). "
        f"Supersaturation maintained for {supersaturation_maintenance_h:.1f} h. "
        f"Synergy with BCS {drug_bcs_class} drug: {combination_synergy}."
    )

    return ExcipientDissolutionResult(
        drug_name=drug_name,
        excipient_name=excipient_name,
        excipient_concentration_pct=excipient_concentration_pct,
        mechanism=mechanism,
        dissolution_enhancement_factor=dissolution_enhancement_factor,
        supersaturation_maintenance_h=supersaturation_maintenance_h,
        spring_effect=spring,
        parachute_effect=parachute,
        t50_reduction_pct=t50_reduction_pct,
        optimal_concentration_pct=optimal_pct,
        combination_synergy=combination_synergy,
        notes=notes,
    )


def screen_excipients_dissolution(
    drug_name: str,
    drug_logp: float = 2.0,
    drug_bcs_class: int = 2,
) -> list:
    """
    Screen all excipients at their optimal concentrations for a given drug,
    sorted by dissolution enhancement factor in descending order.

    Parameters
    ----------
    drug_name : str
    drug_logp : float
    drug_bcs_class : int

    Returns
    -------
    list of ExcipientDissolutionResult sorted by dissolution_enhancement_factor descending
    """
    results = []
    for excipient_name, exc_data in EXCIPIENTS.items():
        result = predict_excipient_dissolution(
            drug_name=drug_name,
            excipient_name=excipient_name,
            excipient_concentration_pct=exc_data["optimal_pct"],
            drug_logp=drug_logp,
            drug_bcs_class=drug_bcs_class,
        )
        results.append(result)

    results.sort(key=lambda r: r.dissolution_enhancement_factor, reverse=True)
    return results
