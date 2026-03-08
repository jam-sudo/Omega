"""Amorphous solid dispersion (ASD) system modeling — Phase 243.

Models ASD formulations for poorly soluble drugs, predicting supersaturation,
spring-parachute behavior, and physical stability.
"""

from dataclasses import dataclass

# Polymer property lookup table
# Keys: polymer_type
# Values: dict with supersaturation_factor, crystallization_inhibitor, solubility_enhancement
_POLYMER_TABLE: dict[str, dict[str, float]] = {
    "HPMC_AS": {
        "supersaturation_factor": 10.0,
        "crystallization_inhibitor": 0.85,
        "solubility_enhancement": 8.0,
    },
    "PVP_VA": {
        "supersaturation_factor": 8.0,
        "crystallization_inhibitor": 0.75,
        "solubility_enhancement": 6.0,
    },
    "Eudragit_L100": {
        "supersaturation_factor": 6.0,
        "crystallization_inhibitor": 0.70,
        "solubility_enhancement": 5.0,
    },
    "HPMC": {
        "supersaturation_factor": 7.0,
        "crystallization_inhibitor": 0.80,
        "solubility_enhancement": 7.0,
    },
    "PEG_6000": {
        "supersaturation_factor": 4.0,
        "crystallization_inhibitor": 0.50,
        "solubility_enhancement": 3.0,
    },
}

_VALID_POLYMERS = set(_POLYMER_TABLE.keys())

_FORMULATION_CLASSES = ("excellent", "good", "acceptable", "poor")


def _formulation_class(score: float) -> str:
    if score >= 80.0:
        return "excellent"
    elif score >= 60.0:
        return "good"
    elif score >= 40.0:
        return "acceptable"
    else:
        return "poor"


@dataclass(frozen=True)
class ASDResult:
    """Result of ASD system design — Phase 243."""

    drug_name: str
    polymer_type: str
    drug_loading_pct: float
    crystalline_solubility_mg_mL: float
    amorphous_solubility_mg_mL: float
    apparent_solubility_mg_mL: float
    spring_duration_min: float
    parachute_duration_min: float
    relative_bioavailability_increase_pct: float
    physical_stability_score: float
    formulation_class: str
    notes: str


def design_asd(
    drug_name: str,
    crystalline_solubility_mg_mL: float,
    logp: float,
    drug_loading_pct: float,
    polymer_type: str = "HPMC_AS",
) -> ASDResult:
    """Design an ASD formulation for a poorly soluble drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    crystalline_solubility_mg_mL:
        Crystalline solubility of the drug (mg/mL). Must be > 0.
    logp:
        Octanol-water partition coefficient (unused in core model but stored).
    drug_loading_pct:
        Drug loading percentage in ASD (0 < x < 100).
    polymer_type:
        One of: HPMC_AS, PVP_VA, Eudragit_L100, HPMC, PEG_6000.

    Returns
    -------
    ASDResult
    """
    # Validation
    if crystalline_solubility_mg_mL <= 0.0:
        raise ValueError(
            f"crystalline_solubility_mg_mL must be > 0, got {crystalline_solubility_mg_mL}"
        )
    if not (0.0 < drug_loading_pct < 100.0):
        raise ValueError(f"drug_loading_pct must be in (0, 100), got {drug_loading_pct}")
    if polymer_type not in _VALID_POLYMERS:
        raise ValueError(
            f"polymer_type must be one of {sorted(_VALID_POLYMERS)}, got {polymer_type!r}"
        )

    props = _POLYMER_TABLE[polymer_type]
    sf = props["supersaturation_factor"]
    ci = props["crystallization_inhibitor"]

    # Amorphous solubility
    amorphous_solubility = crystalline_solubility_mg_mL * sf

    # Loading factor
    lf = max(0.3, 1.0 - 0.005 * drug_loading_pct)

    # Apparent solubility accounting for polymer loading
    apparent_solubility = amorphous_solubility * lf

    # Spring duration (supersaturation maintenance time)
    spring_duration = 60.0 * ci * (1.0 + 0.01 * (100.0 - drug_loading_pct))

    # Parachute duration (before recrystallization)
    parachute_duration = spring_duration * ci

    # Relative bioavailability increase
    relative_ba_increase = (apparent_solubility / crystalline_solubility_mg_mL - 1.0) * 100.0

    # Physical stability score
    raw_score = ci * 100.0 * lf * (1.0 - drug_loading_pct / 200.0)
    physical_stability = max(0.0, min(100.0, raw_score))

    fc = _formulation_class(physical_stability)

    notes = (
        f"ASD design: polymer={polymer_type}, loading={drug_loading_pct:.1f}%, "
        f"logP={logp:.2f}, supersaturation_factor={sf:.1f}, "
        f"crystallization_inhibitor={ci:.2f}"
    )

    return ASDResult(
        drug_name=drug_name,
        polymer_type=polymer_type,
        drug_loading_pct=drug_loading_pct,
        crystalline_solubility_mg_mL=crystalline_solubility_mg_mL,
        amorphous_solubility_mg_mL=amorphous_solubility,
        apparent_solubility_mg_mL=apparent_solubility,
        spring_duration_min=spring_duration,
        parachute_duration_min=parachute_duration,
        relative_bioavailability_increase_pct=relative_ba_increase,
        physical_stability_score=physical_stability,
        formulation_class=fc,
        notes=notes,
    )


def screen_asd_polymers(
    drug_name: str,
    crystalline_solubility_mg_mL: float,
    logp: float,
    drug_loading_pct: float = 20.0,
) -> list:
    """Screen all 5 polymers for a drug, sorted by physical_stability_score descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    crystalline_solubility_mg_mL:
        Crystalline solubility of the drug (mg/mL). Must be > 0.
    logp:
        Octanol-water partition coefficient.
    drug_loading_pct:
        Drug loading percentage in ASD (0 < x < 100). Default 20.0.

    Returns
    -------
    list of ASDResult sorted by physical_stability_score descending.
    """
    results = []
    for polymer in _VALID_POLYMERS:
        result = design_asd(
            drug_name=drug_name,
            crystalline_solubility_mg_mL=crystalline_solubility_mg_mL,
            logp=logp,
            drug_loading_pct=drug_loading_pct,
            polymer_type=polymer,
        )
        results.append(result)

    results.sort(key=lambda r: r.physical_stability_score, reverse=True)
    return results
