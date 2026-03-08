"""
Phase 265: Surfactant Micelle Solubilization
Models drug solubilization by surfactant micelles (pharmaceutical excipients).
"""

from dataclasses import dataclass

# Surfactant lookup table: CMC_mM, Kv (micellar partition), aggregation_number
_SURFACTANTS = {
    "polysorbate_80": {"CMC_mM": 0.012, "Kv": 1000.0, "aggregation_number": 60},
    "polysorbate_20": {"CMC_mM": 0.059, "Kv": 500.0, "aggregation_number": 55},
    "SDS": {"CMC_mM": 8.0, "Kv": 200.0, "aggregation_number": 62},
    "CTAB": {"CMC_mM": 0.9, "Kv": 300.0, "aggregation_number": 78},
    "poloxamer_407": {"CMC_mM": 0.001, "Kv": 2000.0, "aggregation_number": 50},
}


@dataclass(frozen=True)
class MicelleResult:
    drug_name: str
    surfactant: str
    logp: float
    surfactant_conc_mM: float
    cmc_mM: float
    micellar_solubilization_ratio: float
    total_solubility_mg_mL: float
    intrinsic_solubility_mg_mL: float
    micelle_contribution_pct: float
    partition_coefficient_Kv: float
    precipitation_risk: str
    bioavailability_impact: str
    notes: str


def predict_micellar_solubilization(
    drug_name: str,
    logp: float,
    intrinsic_solubility_mg_mL: float,
    surfactant: str,
    surfactant_conc_mM: float,
) -> MicelleResult:
    """
    Predict micellar solubilization of a drug by a given surfactant.

    Parameters
    ----------
    drug_name : str
        Name identifier for the drug.
    logp : float
        Log partition coefficient (octanol/water). Must be in (-5, 10).
    intrinsic_solubility_mg_mL : float
        Aqueous intrinsic solubility in mg/mL. Must be > 0.
    surfactant : str
        One of the supported surfactant types.
    surfactant_conc_mM : float
        Total surfactant concentration in mM. Must be >= 0.

    Returns
    -------
    MicelleResult
    """
    # Validation
    if logp <= -5 or logp >= 10:
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if surfactant_conc_mM < 0:
        raise ValueError(f"surfactant_conc_mM must be >= 0, got {surfactant_conc_mM}")
    if surfactant not in _SURFACTANTS:
        raise ValueError(
            f"Unknown surfactant '{surfactant}'. Valid options: {sorted(_SURFACTANTS.keys())}"
        )

    props = _SURFACTANTS[surfactant]
    cmc_mM = props["CMC_mM"]
    Kv_base = props["Kv"]

    # Below CMC: no micelles
    if surfactant_conc_mM < cmc_mM:
        ratio = 1.0
        total_solubility = intrinsic_solubility_mg_mL
        micelle_contribution_pct = 0.0
        Kv_eff = Kv_base
        precipitation_risk = "high"
        notes = (
            f"Surfactant concentration ({surfactant_conc_mM:.4f} mM) is below CMC "
            f"({cmc_mM:.4f} mM). No micelles formed; no solubilization enhancement."
        )
    else:
        # Effective free micelle concentration
        effective_conc = surfactant_conc_mM - cmc_mM

        # Kv modified by logP
        if logp > 0:
            Kv_eff = Kv_base * (logp / 3.0)
        else:
            Kv_eff = Kv_base * 0.1

        # Solubilization ratio (unit conversion: mM -> M via /1000)
        ratio = 1.0 + Kv_eff * effective_conc / 1000.0
        total_solubility = intrinsic_solubility_mg_mL * ratio
        micelle_contribution_pct = (ratio - 1.0) / ratio * 100.0
        precipitation_risk = "none"
        notes = (
            f"Above CMC: {effective_conc:.4f} mM free micellar surfactant. "
            f"Kv_eff={Kv_eff:.2f}. Solubilization ratio={ratio:.3f}."
        )

    # Bioavailability impact
    if ratio > 2.0:
        bioavailability_impact = "enhanced"
    elif ratio >= 1.0:
        bioavailability_impact = "neutral"
    else:
        bioavailability_impact = "reduced"

    return MicelleResult(
        drug_name=drug_name,
        surfactant=surfactant,
        logp=logp,
        surfactant_conc_mM=surfactant_conc_mM,
        cmc_mM=cmc_mM,
        micellar_solubilization_ratio=ratio,
        total_solubility_mg_mL=total_solubility,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        micelle_contribution_pct=micelle_contribution_pct,
        partition_coefficient_Kv=Kv_eff,
        precipitation_risk=precipitation_risk,
        bioavailability_impact=bioavailability_impact,
        notes=notes,
    )


def screen_surfactants(
    drug_name: str,
    logp: float,
    intrinsic_solubility_mg_mL: float,
    surfactant_conc_mM: float = 1.0,
) -> list[MicelleResult]:
    """
    Screen all supported surfactants and rank by micellar_solubilization_ratio descending.

    Parameters
    ----------
    drug_name : str
    logp : float
    intrinsic_solubility_mg_mL : float
    surfactant_conc_mM : float
        Surfactant concentration to use for all surfactants (default 1.0 mM).

    Returns
    -------
    list of MicelleResult, sorted by micellar_solubilization_ratio descending
    """
    results = []
    for surf_name in _SURFACTANTS:
        result = predict_micellar_solubilization(
            drug_name=drug_name,
            logp=logp,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            surfactant=surf_name,
            surfactant_conc_mM=surfactant_conc_mM,
        )
        results.append(result)

    results.sort(key=lambda r: r.micellar_solubilization_ratio, reverse=True)
    return results
