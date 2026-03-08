"""Phase 583 - Drug Solubility Enhancement by Cyclodextrins."""

from dataclasses import dataclass

# Cyclodextrin molecular weights (Da)
CD_MW = {
    "HP_beta_CD": 1460.0,
    "SBE_beta_CD": 2163.0,
    "alpha_CD": 972.0,
}

# Ka base values and logP coefficients per CD type
_KA_PARAMS = {
    "HP_beta_CD": (500.0, 0.5),
    "SBE_beta_CD": (400.0, 0.4),
    "alpha_CD": (200.0, 0.2),
}


@dataclass(frozen=True)
class CyclodextrinSolubilityResult:
    drug_name: str
    cyclodextrin_type: str
    drug_logP: float
    intrinsic_solubility_mg_mL: float
    cyclodextrin_conc_mM: float
    apparent_solubility_mg_mL: float
    solubility_enhancement_ratio: float
    complexation_efficiency_pct: float
    ka_complex_per_mM: float
    drug_loading_pct: float
    formulation_feasibility: str
    notes: str


def _classify_feasibility(se_ratio: float) -> str:
    if se_ratio > 10.0:
        return "excellent"
    if se_ratio >= 5.0:
        return "good"
    if se_ratio >= 2.0:
        return "moderate"
    return "poor"


def predict_cd_solubility(
    drug_name: str,
    drug_logP: float,
    intrinsic_solubility_mg_mL: float,
    mw_drug_Da: float,
    cyclodextrin_type: str,
    cyclodextrin_conc_mM: float = 10.0,
) -> CyclodextrinSolubilityResult:
    """Predict drug solubility enhancement by cyclodextrin complexation."""
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")
    if cyclodextrin_conc_mM <= 0:
        raise ValueError("cyclodextrin_conc_mM must be > 0")
    if mw_drug_Da <= 0:
        raise ValueError("mw_drug_Da must be > 0")
    if cyclodextrin_type not in _KA_PARAMS:
        raise ValueError(
            f"Invalid cyclodextrin_type '{cyclodextrin_type}'. "
            f"Must be one of {list(_KA_PARAMS.keys())}"
        )

    base_ka, coeff = _KA_PARAMS[cyclodextrin_type]
    ka = base_ka * (1.0 + coeff * max(0.0, drug_logP - 1.0))

    s0 = intrinsic_solubility_mg_mL
    sapp = s0 * (1.0 + ka * cyclodextrin_conc_mM)
    se_ratio = sapp / s0

    delta_s = sapp - s0
    ce = delta_s / cyclodextrin_conc_mM  # mg/mL per mM

    mw_cd = CD_MW[cyclodextrin_type]
    cd_mass = cyclodextrin_conc_mM * mw_cd / 1000.0  # mg/mL
    if delta_s + cd_mass > 0:
        loading = min(50.0, delta_s / (delta_s + cd_mass) * 100.0)
    else:
        loading = 0.0

    feasibility = _classify_feasibility(se_ratio)

    notes_parts = []
    if se_ratio > 10:
        notes_parts.append("Excellent solubility enhancement achieved")
    if drug_logP > 3:
        notes_parts.append("Highly lipophilic drug benefits from CD complexation")
    notes = "; ".join(notes_parts) if notes_parts else "Standard CD complexation"

    return CyclodextrinSolubilityResult(
        drug_name=drug_name,
        cyclodextrin_type=cyclodextrin_type,
        drug_logP=drug_logP,
        intrinsic_solubility_mg_mL=s0,
        cyclodextrin_conc_mM=cyclodextrin_conc_mM,
        apparent_solubility_mg_mL=sapp,
        solubility_enhancement_ratio=se_ratio,
        complexation_efficiency_pct=ce * 100.0,
        ka_complex_per_mM=ka,
        drug_loading_pct=loading,
        formulation_feasibility=feasibility,
        notes=notes,
    )


def compare_cyclodextrins(
    drug_name: str,
    drug_logP: float,
    intrinsic_solubility: float,
    mw_drug: float,
) -> list:
    """Compare all 3 CD types, return sorted by SE ratio descending."""
    results = []
    for cd_type in _KA_PARAMS:
        r = predict_cd_solubility(
            drug_name=drug_name,
            drug_logP=drug_logP,
            intrinsic_solubility_mg_mL=intrinsic_solubility,
            mw_drug_Da=mw_drug,
            cyclodextrin_type=cd_type,
        )
        results.append(r)
    results.sort(key=lambda x: x.solubility_enhancement_ratio, reverse=True)
    return results
