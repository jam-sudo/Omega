"""Phase 347 — Solubility Enhancement by Cyclodextrin Complexation.

Models solubility enhancement of poorly soluble drugs via cyclodextrin (CD)
complexation using the Higuchi-Connors AL-type phase solubility model.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_CD_TYPES = ("alpha", "beta", "gamma", "HP_beta")
_VALID_DRUG_TYPES = ("acid", "base", "neutral")

_CD_K11_CORRECTIONS = {
    "alpha": 1.0,
    "beta": 1.5,
    "gamma": 0.8,
    "HP_beta": 1.3,
}


@dataclass(frozen=True)
class CyclodextrinComplexationResult:
    """Result of cyclodextrin complexation solubility enhancement model."""

    drug_name: str
    cd_type: str
    cd_conc_mM: float
    intrinsic_solubility_mg_mL: float
    k11_effective_per_mM: float
    solubility_with_cd_mg_mL: float
    enhancement_ratio: float
    dose_number_with_cd: float
    bcs_dissolution_class: str
    formulation_feasibility: str
    optimal_cd_conc_mM: float
    notes: str


def _compute_k11_effective(k11_base: float, cd_type: str, logP: float) -> float:
    """Apply CD-type correction and logP correction to K11."""
    k11 = k11_base * _CD_K11_CORRECTIONS[cd_type]
    if logP < 0.0 or logP > 6.0:
        k11 *= 0.5
    return k11


def _classify_feasibility(enhancement_ratio: float) -> str:
    if enhancement_ratio >= 2.0:
        return "feasible"
    if enhancement_ratio >= 1.0:
        return "marginal"
    return "not_beneficial"


def _classify_bcs(dose_number: float) -> str:
    if dose_number < 1.0:
        return "complete_dissolution"
    return "dissolution_limited"


def _build_notes(
    cd_type: str,
    logP: float,
    enhancement_ratio: float,
    drug_type: str,
) -> str:
    parts: list[str] = []
    if logP < 0.0 or logP > 6.0:
        parts.append("logP outside 0-6 range; K11 halved")
    if 2.0 <= logP <= 4.0:
        parts.append("logP 2-4: optimal complexation range")
    if cd_type == "beta":
        parts.append("beta-CD: most common, best for MW 200-400")
    elif cd_type == "HP_beta":
        parts.append("HP-beta-CD: hydroxypropyl modification improves aqueous solubility")
    elif cd_type == "alpha":
        parts.append("alpha-CD: suitable for small molecules")
    elif cd_type == "gamma":
        parts.append("gamma-CD: larger cavity, lower K11 correction")
    if drug_type in ("acid", "base"):
        parts.append(f"ionizable {drug_type}: ionization state may affect complexation")
    if enhancement_ratio < 1.0:
        parts.append("enhancement_ratio < 1: CD may not be beneficial")
    return "; ".join(parts) if parts else "standard complexation model applied"


def predict_cyclodextrin_complexation(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    logP: float,
    pKa: float,
    drug_type: str,
    cd_type: str,
    cd_conc_mM: float,
    K11_per_mM: float = 10.0,
    dose_mg: float = 100.0,
) -> CyclodextrinComplexationResult:
    """Predict solubility enhancement via cyclodextrin complexation.

    Uses Higuchi-Connors AL-type phase solubility model:
        S_total = S0 * (1 + K11_eff * [CD])

    Parameters
    ----------
    drug_name : str
    intrinsic_solubility_mg_mL : float  intrinsic (uncomplexed) aqueous solubility
    logP : float
    pKa : float
    drug_type : str  "acid", "base", or "neutral"
    cd_type : str  "alpha", "beta", "gamma", or "HP_beta"
    cd_conc_mM : float  cyclodextrin concentration in mM
    K11_per_mM : float  1:1 complexation stability constant (default 10.0 mM^-1)
    dose_mg : float  dose in mg for BCS dose number calculation (default 100)
    """
    if intrinsic_solubility_mg_mL <= 0.0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")
    if cd_conc_mM <= 0.0:
        raise ValueError("cd_conc_mM must be > 0")
    if K11_per_mM <= 0.0:
        raise ValueError("K11_per_mM must be > 0")
    if cd_type not in _VALID_CD_TYPES:
        raise ValueError(f"cd_type must be one of {_VALID_CD_TYPES}")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}")

    k11_eff = _compute_k11_effective(K11_per_mM, cd_type, logP)

    # Phase solubility AL model
    enhancement_ratio = 1.0 + k11_eff * cd_conc_mM
    sol_with_cd = intrinsic_solubility_mg_mL * enhancement_ratio

    # BCS dose number: Do = dose / (sol * 250 mL)
    dose_number = dose_mg / (sol_with_cd * 250.0)

    bcs_class = _classify_bcs(dose_number)
    feasibility = _classify_feasibility(enhancement_ratio)

    # Optimal CD concentration for 10x enhancement: S_total = 10 * S0
    # => 1 + K11_eff * [CD]_opt = 10  => [CD]_opt = 9 / K11_eff
    optimal_cd_conc = 9.0 / k11_eff

    notes = _build_notes(cd_type, logP, enhancement_ratio, drug_type)

    return CyclodextrinComplexationResult(
        drug_name=drug_name,
        cd_type=cd_type,
        cd_conc_mM=cd_conc_mM,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        k11_effective_per_mM=k11_eff,
        solubility_with_cd_mg_mL=sol_with_cd,
        enhancement_ratio=enhancement_ratio,
        dose_number_with_cd=dose_number,
        bcs_dissolution_class=bcs_class,
        formulation_feasibility=feasibility,
        optimal_cd_conc_mM=optimal_cd_conc,
        notes=notes,
    )


def screen_cd_types(
    drug_name: str,
    intrinsic_sol: float,
    logP: float,
    pKa: float,
    drug_type: str,
    cd_conc_mM: float,
    K11_per_mM: float = 10.0,
    dose_mg: float = 100.0,
) -> list[CyclodextrinComplexationResult]:
    """Screen all four CD types and return results sorted by enhancement_ratio descending.

    Parameters
    ----------
    drug_name : str
    intrinsic_sol : float  intrinsic solubility in mg/mL
    logP : float
    pKa : float
    drug_type : str  "acid", "base", or "neutral"
    cd_conc_mM : float  CD concentration in mM
    K11_per_mM : float  base K11 (default 10.0 mM^-1)
    dose_mg : float  dose in mg (default 100)
    """
    results = [
        predict_cyclodextrin_complexation(
            drug_name=drug_name,
            intrinsic_solubility_mg_mL=intrinsic_sol,
            logP=logP,
            pKa=pKa,
            drug_type=drug_type,
            cd_type=ct,
            cd_conc_mM=cd_conc_mM,
            K11_per_mM=K11_per_mM,
            dose_mg=dose_mg,
        )
        for ct in _VALID_CD_TYPES
    ]
    return sorted(results, key=lambda r: r.enhancement_ratio, reverse=True)
