"""Allometric covariate scaling for patient-specific PK prediction.

Applies body-weight allometric scaling and CYP genotype activity factors
to population-level PK parameters.
"""

from __future__ import annotations

REF_WEIGHT_KG = 70.0

# CYP genotype activity factors relative to extensive metabolizer (EM = 1.0)
CYP_FACTORS: dict[str, dict[str, float]] = {
    "CYP2D6": {
        "UM": 1.5,  # ultra-rapid metabolizer
        "EM": 1.0,  # extensive (normal)
        "IM": 0.5,  # intermediate
        "PM": 0.1,  # poor metabolizer
    },
    "CYP2C9": {
        "*1/*1": 1.0,
        "*1/*2": 0.8,
        "*1/*3": 0.6,
        "*2/*2": 0.5,
        "*2/*3": 0.3,
        "*3/*3": 0.1,
    },
    "CYP2C19": {
        "UM": 1.5,
        "EM": 1.0,
        "IM": 0.5,
        "PM": 0.2,
    },
}


def scale_clearance(cl_pop: float, weight_kg: float = REF_WEIGHT_KG) -> float:
    """Scale population clearance by allometric exponent 0.75."""
    return cl_pop * (weight_kg / REF_WEIGHT_KG) ** 0.75


def scale_volume(vd_pop: float, weight_kg: float = REF_WEIGHT_KG) -> float:
    """Scale population volume of distribution linearly with weight."""
    return vd_pop * (weight_kg / REF_WEIGHT_KG)


def cyp_genotype_factor(enzyme: str, phenotype: str) -> float:
    """Return activity factor for a CYP enzyme/phenotype pair.

    Returns 1.0 (no effect) for unknown enzyme/phenotype combinations.
    """
    return CYP_FACTORS.get(enzyme, {}).get(phenotype, 1.0)


def apply_covariates(
    base_params: dict[str, float],
    covariates: dict[str, object],
) -> dict[str, float]:
    """Apply covariate adjustments to base PK parameters.

    Parameters
    ----------
    base_params : dict
        Must contain ``cl_L_h`` (clearance, L/h) and ``vd_L`` (volume, L).
    covariates : dict
        Optional keys: ``weight_kg`` (float), and any CYP enzyme name
        (e.g. ``CYP2D6``) mapped to its phenotype string.

    Returns
    -------
    dict with adjusted ``cl_L_h`` and ``vd_L`` (plus any other keys unchanged).
    """
    adjusted = dict(base_params)
    weight_kg = float(covariates.get("weight_kg", REF_WEIGHT_KG))

    # Allometric scaling
    if "cl_L_h" in adjusted:
        adjusted["cl_L_h"] = scale_clearance(adjusted["cl_L_h"], weight_kg)
    if "vd_L" in adjusted:
        adjusted["vd_L"] = scale_volume(adjusted["vd_L"], weight_kg)

    # CYP genotype factors — apply to clearance
    for enzyme in CYP_FACTORS:
        if enzyme in covariates:
            phenotype = str(covariates[enzyme])
            factor = cyp_genotype_factor(enzyme, phenotype)
            if "cl_L_h" in adjusted:
                adjusted["cl_L_h"] *= factor

    return adjusted
