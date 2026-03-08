"""Amorphous solid dispersion (ASD) performance prediction.

Predicts drug release and supersaturation behavior from ASD formulations,
which enhance bioavailability of BCS II/IV drugs by maintaining supersaturation.
"""

from dataclasses import dataclass

# Polymer property table: name -> (parachute_factor, solubility_boost_x)
_POLYMER_TABLE: dict[str, tuple[float, float]] = {
    "HPMC-AS": (0.85, 15.0),
    "PVP-VA": (0.70, 10.0),
    "HPMC": (0.60, 8.0),
    "Eudragit_L100": (0.65, 12.0),
    "Soluplus": (0.75, 10.0),
    "PVP": (0.55, 7.0),
}

_VALID_POLYMERS = set(_POLYMER_TABLE.keys())
_VALID_IONIZATION = {"acid", "base", "neutral"}

# Best polymer per ionization type
_RECOMMENDED_POLYMER: dict[str, str] = {
    "acid": "HPMC-AS",
    "base": "PVP-VA",
    "neutral": "Soluplus",
}


@dataclass(frozen=True)
class ASDResult:
    """Result of ASD performance prediction."""

    drug_name: str
    polymer_name: str
    drug_loading_pct: float
    logp: float
    intrinsic_crystalline_solubility_mg_mL: float
    asd_apparent_solubility_mg_mL: float
    supersaturation_ratio: float
    crystallization_induction_time_min: float
    spring_parachute_duration_min: float
    spring_factor: float
    parachute_factor: float
    bioavailability_enhancement_factor: float
    recommended_polymer: str
    notes: str


def predict_asd_performance(
    drug_name: str,
    polymer_name: str,
    drug_loading_pct: float,
    logp: float = 3.0,
    drug_pka: float = 7.0,
    ionization_type: str = "neutral",
) -> ASDResult:
    """Predict ASD formulation performance for a drug-polymer combination.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    polymer_name:
        One of: HPMC-AS, PVP-VA, HPMC, Eudragit_L100, Soluplus, PVP.
    drug_loading_pct:
        Drug loading percentage in ASD (0 < x < 100).
    logp:
        Octanol-water partition coefficient.
    drug_pka:
        Drug pKa value.
    ionization_type:
        One of: "acid", "base", "neutral".

    Returns
    -------
    ASDResult
    """
    # Validation
    if polymer_name not in _VALID_POLYMERS:
        raise ValueError(
            f"polymer_name must be one of {sorted(_VALID_POLYMERS)}, got {polymer_name!r}"
        )
    if not (0.0 < drug_loading_pct < 100.0):
        raise ValueError(f"drug_loading_pct must be in (0, 100), got {drug_loading_pct}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    parachute_factor, solubility_boost = _POLYMER_TABLE[polymer_name]

    # Intrinsic (crystalline) solubility from logP
    intrinsic_solubility = 10 ** (-0.7 * logp + 0.44)

    # Spring factor: reduced at high drug loading
    spring_factor = solubility_boost * (1.0 - drug_loading_pct / 100.0 * 0.5)
    # Ensure spring_factor stays above 1.0 for meaningful ASD (clamp at minimum)
    spring_factor = max(1.01, spring_factor)

    # ASD apparent (supersaturated) solubility
    asd_apparent_solubility = intrinsic_solubility * spring_factor

    # Crystallization induction time
    loading_factor = drug_loading_pct / 20.0
    cryst_time = 60.0 * parachute_factor / loading_factor
    cryst_time = max(5.0, min(240.0, cryst_time))

    # Spring-parachute duration
    spring_parachute_duration = cryst_time * (1.0 + parachute_factor)

    # Supersaturation ratio equals spring_factor
    supersaturation_ratio = spring_factor

    # Bioavailability enhancement factor
    baf = min(5.0, spring_factor * parachute_factor * 0.5 + 0.5)

    # Recommended polymer for this ionization type
    recommended = _RECOMMENDED_POLYMER[ionization_type]

    notes = (
        f"ASD prediction: logP={logp:.1f}, loading={drug_loading_pct:.1f}%, "
        f"polymer={polymer_name}, ionization={ionization_type}, pKa={drug_pka:.1f}"
    )

    return ASDResult(
        drug_name=drug_name,
        polymer_name=polymer_name,
        drug_loading_pct=drug_loading_pct,
        logp=logp,
        intrinsic_crystalline_solubility_mg_mL=intrinsic_solubility,
        asd_apparent_solubility_mg_mL=asd_apparent_solubility,
        supersaturation_ratio=supersaturation_ratio,
        crystallization_induction_time_min=cryst_time,
        spring_parachute_duration_min=spring_parachute_duration,
        spring_factor=spring_factor,
        parachute_factor=parachute_factor,
        bioavailability_enhancement_factor=baf,
        recommended_polymer=recommended,
        notes=notes,
    )


def screen_asd_polymers(
    drug_name: str,
    drug_loading_pct: float,
    logp: float,
    drug_pka: float,
    ionization_type: str,
) -> list:
    """Screen all 6 ASD polymers for a drug, sorted by bioavailability enhancement.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_loading_pct:
        Drug loading percentage in ASD (0 < x < 100).
    logp:
        Octanol-water partition coefficient.
    drug_pka:
        Drug pKa value.
    ionization_type:
        One of: "acid", "base", "neutral".

    Returns
    -------
    list of ASDResult sorted by bioavailability_enhancement_factor descending.
    """
    results = []
    for polymer in _VALID_POLYMERS:
        result = predict_asd_performance(
            drug_name=drug_name,
            polymer_name=polymer,
            drug_loading_pct=drug_loading_pct,
            logp=logp,
            drug_pka=drug_pka,
            ionization_type=ionization_type,
        )
        results.append(result)

    results.sort(key=lambda r: r.bioavailability_enhancement_factor, reverse=True)
    return results
