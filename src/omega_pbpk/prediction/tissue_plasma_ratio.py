"""
Phase 377 — Tissue-to-Plasma Ratio Calculator

Calculates tissue-to-plasma concentration ratios (Kp values) for major organs
using a simplified Rodgers-Rowland approach based on physicochemical properties.
"""

from dataclasses import dataclass

# Tissue composition constants (fractions of lipid, water, protein)
_TISSUE_PARAMS: dict[str, dict[str, float]] = {
    "liver": {"f_lipid": 0.0348, "f_water": 0.711, "f_protein": 0.18},
    "kidney": {"f_lipid": 0.0285, "f_water": 0.783, "f_protein": 0.13},
    "muscle": {"f_lipid": 0.0100, "f_water": 0.760, "f_protein": 0.18},
    "adipose": {"f_lipid": 0.853, "f_water": 0.150, "f_protein": 0.02},
    "brain": {"f_lipid": 0.0393, "f_water": 0.770, "f_protein": 0.10},
    "lung": {"f_lipid": 0.0220, "f_water": 0.811, "f_protein": 0.18},
    "heart": {"f_lipid": 0.0135, "f_water": 0.758, "f_protein": 0.18},
    "gut": {"f_lipid": 0.0197, "f_water": 0.718, "f_protein": 0.18},
}

# Plasma composition
_F_LIPID_PLASMA = 0.00194
_F_WATER_PLASMA = 0.937
_F_PROTEIN_PLASMA = 0.072

# Tissue volumes (L/kg body weight) for Vd estimation
_TISSUE_VOLUMES_L_PER_KG: dict[str, float] = {
    "liver": 0.026,
    "kidney": 0.004,
    "muscle": 0.400,
    "adipose": 0.210,
    "brain": 0.020,
}

_VALID_DRUG_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class TissueRatioResult:
    """Result of tissue-to-plasma ratio calculation."""

    drug_name: str
    logp: float
    pka: float
    drug_type: str
    fup: float
    fut: float
    kp_liver: float
    kp_kidney: float
    kp_muscle: float
    kp_adipose: float
    kp_brain: float
    kp_lung: float
    kp_heart: float
    kp_gut: float
    vd_predicted_L_per_kg: float
    dominant_tissue: str
    distribution_class: str
    notes: str


def _validate_inputs(
    drug_name: str,
    logp: float,
    pka: float,
    drug_type: str,
    fup: float,
    molecular_weight: float,
) -> None:
    """Validate all input parameters and raise ValueError if invalid."""
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'")
    if not (0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if not (-5 <= logp <= 10):
        raise ValueError(f"logp must be in [-5, 10], got {logp}")
    if not (-5 <= pka <= 14):
        raise ValueError(f"pka must be in [-5, 14], got {pka}")
    if molecular_weight <= 0:
        raise ValueError(f"molecular_weight must be > 0, got {molecular_weight}")


def _compute_kp(
    tissue_params: dict[str, float],
    partition_coeff: float,
    fu_factor: float,
) -> float:
    """Compute tissue-to-plasma Kp for a single tissue."""
    numerator = (
        tissue_params["f_water"]
        + tissue_params["f_lipid"] * partition_coeff
        + tissue_params["f_protein"] * fu_factor
    )
    denominator = (
        _F_WATER_PLASMA + _F_LIPID_PLASMA * partition_coeff + _F_PROTEIN_PLASMA * fu_factor
    )
    return numerator / denominator


def _compute_fut(partition_coeff: float) -> float:
    """Estimate fraction unbound in tissue."""
    avg_lipid = sum(p["f_lipid"] for p in _TISSUE_PARAMS.values()) / 8.0
    fut = 1.0 / (1.0 + 0.5 * partition_coeff * avg_lipid)
    # Clamp to [0.01, 1.0]
    if fut < 0.01:
        fut = 0.01
    if fut > 1.0:
        fut = 1.0
    return fut


def calculate_tissue_ratios(
    drug_name: str,
    logp: float,
    pka: float,
    drug_type: str,
    fup: float,
    molecular_weight: float = 400.0,
) -> TissueRatioResult:
    """
    Calculate tissue-to-plasma concentration ratios for major organs.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Octanol-water partition coefficient (log scale). Range [-5, 10].
    pka : float
        Acid dissociation constant. Range [-5, 14].
    drug_type : str
        Drug ionization type: "acid", "base", or "neutral".
    fup : float
        Fraction unbound in plasma. Range (0, 1].
    molecular_weight : float
        Molecular weight in g/mol (must be > 0). Default 400.0.

    Returns
    -------
    TissueRatioResult
        Frozen dataclass with all Kp values and derived properties.
    """
    _validate_inputs(drug_name, logp, pka, drug_type, fup, molecular_weight)

    partition_coeff = 10.0**logp
    fu_factor = (1.0 / fup) * 0.1
    fut = _compute_fut(partition_coeff)

    # Compute Kp for each tissue
    kp_values: dict[str, float] = {}
    for tissue, params in _TISSUE_PARAMS.items():
        kp_values[tissue] = _compute_kp(params, partition_coeff, fu_factor)

    # Weighted Vd estimate (L/kg)
    vd = 0.04 + (
        kp_values["liver"] * _TISSUE_VOLUMES_L_PER_KG["liver"]
        + kp_values["kidney"] * _TISSUE_VOLUMES_L_PER_KG["kidney"]
        + kp_values["muscle"] * _TISSUE_VOLUMES_L_PER_KG["muscle"]
        + kp_values["adipose"] * _TISSUE_VOLUMES_L_PER_KG["adipose"]
        + kp_values["brain"] * _TISSUE_VOLUMES_L_PER_KG["brain"]
    )

    # Distribution class
    if vd < 0.3:
        distribution_class = "restricted"
    elif vd < 2.0:
        distribution_class = "moderate"
    else:
        distribution_class = "extensive"

    # Dominant tissue (highest Kp)
    dominant_tissue = max(kp_values, key=lambda t: kp_values[t])

    # Build notes
    notes_parts = []
    notes_parts.append(f"drug_type={drug_type}")
    notes_parts.append(f"P={partition_coeff:.2f}")
    if logp > 3:
        notes_parts.append("lipophilic compound; expect high adipose distribution")
    elif logp < 0:
        notes_parts.append("hydrophilic compound; restricted tissue distribution")
    if fup < 0.1:
        notes_parts.append("highly protein-bound")
    notes = "; ".join(notes_parts)

    return TissueRatioResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        drug_type=drug_type,
        fup=fup,
        fut=fut,
        kp_liver=kp_values["liver"],
        kp_kidney=kp_values["kidney"],
        kp_muscle=kp_values["muscle"],
        kp_adipose=kp_values["adipose"],
        kp_brain=kp_values["brain"],
        kp_lung=kp_values["lung"],
        kp_heart=kp_values["heart"],
        kp_gut=kp_values["gut"],
        vd_predicted_L_per_kg=vd,
        dominant_tissue=dominant_tissue,
        distribution_class=distribution_class,
        notes=notes,
    )


def screen_drugs_tissue_distribution(drugs_list: list[dict]) -> list[TissueRatioResult]:
    """
    Screen multiple drugs for tissue distribution properties.

    Parameters
    ----------
    drugs_list : list of dict
        Each dict must contain keys: name, logp, pka, drug_type, fup.
        Optional key: molecular_weight (default 400.0).

    Returns
    -------
    list of TissueRatioResult
        Results sorted by vd_predicted_L_per_kg descending.
    """
    results = []
    for drug in drugs_list:
        result = calculate_tissue_ratios(
            drug_name=drug["name"],
            logp=drug["logp"],
            pka=drug["pka"],
            drug_type=drug["drug_type"],
            fup=drug["fup"],
            molecular_weight=drug.get("molecular_weight", 400.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.vd_predicted_L_per_kg, reverse=True)
    return results
