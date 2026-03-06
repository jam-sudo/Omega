"""Drug product characterization and formulation classification."""

from dataclasses import dataclass

__all__ = [
    "DrugProductResult",
    "classify_formulation",
    "formulation_comparison",
]

_VALID_FORMULATION_TYPES = {
    "tablet",
    "capsule",
    "solution",
    "suspension",
    "patch",
    "iv_bolus",
    "iv_infusion",
}

_R_INTESTINE_CM = 1.75  # cm, radius of intestine
_LUMINAL_VOLUME_ML = 250.0  # mL, volume for dose number calculation
_T_ABS_DEFAULT_H = 3.5  # h, default absorption transit time
_PEFF_DEFAULT_CM_S = 1e-4  # cm/s


@dataclass
class DrugProductResult:
    drug_name: str
    formulation_type: str
    dose_mg: float
    dose_number: float
    dissolution_number: float
    absorption_number: float
    bcs_class: str
    solubility_limited: bool
    permeability_limited: bool
    recommended_strategy: str


def classify_formulation(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    Peff_cm_s: float = _PEFF_DEFAULT_CM_S,
    ka_per_h: float = 1.0,
    dissolution_time_h: float = 0.5,
    formulation_type: str = "tablet",
) -> DrugProductResult:
    """Classify a drug product using BCS framework.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    Peff_cm_s:
        Effective intestinal permeability in cm/s.
    ka_per_h:
        Absorption rate constant in 1/h.
    dissolution_time_h:
        Dissolution time in hours.
    formulation_type:
        One of tablet/capsule/solution/suspension/patch/iv_bolus/iv_infusion.

    Returns
    -------
    DrugProductResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be positive, got {solubility_mg_mL}")

    # Dose number: Do = dose / (Cs * 250 mL)
    dose_number = dose_mg / (solubility_mg_mL * _LUMINAL_VOLUME_ML)

    # Absorption time: t_abs = 1 / ka  [hours]
    t_abs_h = 1.0 / ka_per_h

    # Dissolution number: Dn = t_abs / t_dis
    dissolution_number = t_abs_h / dissolution_time_h

    # Absorption number: An = Peff * 2 * t_abs * 3600 / R
    # Peff in cm/s -> convert t_abs to seconds: t_abs_h * 3600
    absorption_number = (Peff_cm_s * 2.0 * t_abs_h * 3600.0) / _R_INTESTINE_CM

    # BCS classification
    solubility_limited = dose_number > 1.0
    permeability_limited = absorption_number < 1.0

    if not solubility_limited and not permeability_limited:
        bcs_class = "I"
        recommended_strategy = "no major formulation challenge"
    elif solubility_limited and not permeability_limited:
        bcs_class = "II"
        recommended_strategy = "enhance solubility (nano, amorphous, cyclodextrin)"
    elif not solubility_limited and permeability_limited:
        bcs_class = "III"
        recommended_strategy = "enhance permeability or use prodrug"
    else:
        bcs_class = "IV"
        recommended_strategy = "combination enhancement"

    return DrugProductResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        dose_mg=dose_mg,
        dose_number=dose_number,
        dissolution_number=dissolution_number,
        absorption_number=absorption_number,
        bcs_class=bcs_class,
        solubility_limited=solubility_limited,
        permeability_limited=permeability_limited,
        recommended_strategy=recommended_strategy,
    )


def formulation_comparison(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    formulation_types: list[str],
) -> list[DrugProductResult]:
    """Compare a drug across multiple formulation types.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    formulation_types:
        List of formulation type strings to compare.

    Returns
    -------
    List of DrugProductResult, one per formulation type.
    """
    return [
        classify_formulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            solubility_mg_mL=solubility_mg_mL,
            formulation_type=ft,
        )
        for ft in formulation_types
    ]
