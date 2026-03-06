"""Tissue binding correction for PBPK models.

Provides utilities for estimating unbound fractions in tissues (fut) and
correcting tissue-to-plasma partition coefficients (Kp) using empirical
logP-based relationships and the Rodgers & Rowland approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TissueBindingResult",
    "predict_tissue_binding",
    "compare_binding_species",
]


@dataclass(frozen=True)
class TissueBindingResult:
    """Results of tissue binding prediction and Kp correction.

    Attributes:
        drug_name: Name of the drug compound.
        fup: Unbound fraction in plasma (0 < fup <= 1).
        fut_muscle: Unbound fraction in muscle tissue.
        fut_adipose: Unbound fraction in adipose tissue.
        fut_liver: Unbound fraction in liver tissue.
        fut_brain: Unbound fraction in brain tissue.
        kpu_to_kp_ratio: Kpu/Kp scaling factor = fup / fut_muscle for muscle tissue.
        corrected_kp: Tissue name to corrected Kp value mapping.
    """

    drug_name: str
    fup: float
    fut_muscle: float
    fut_adipose: float
    fut_liver: float
    fut_brain: float
    kpu_to_kp_ratio: float
    corrected_kp: dict[str, float] = field(default_factory=dict)


def _fut_from_kp_and_fup(kp_in_vivo: float, fup: float) -> float:
    """Estimate unbound tissue fraction using Rodgers & Rowland approximation.

    Calculates fut = fup / kp_in_vivo, clamped to [0.001, 1.0].

    Args:
        kp_in_vivo: In vivo tissue-to-plasma partition coefficient.
        fup: Unbound fraction in plasma.

    Returns:
        Estimated unbound fraction in tissue, clamped to [0.001, 1.0].
    """
    if kp_in_vivo <= 0:
        return 1.0
    fut = fup / kp_in_vivo
    return max(0.001, min(1.0, fut))


def _tissue_binding_correction(kpu: float, fup: float, fut: float) -> float:
    """Apply tissue binding correction to partition coefficient.

    Computes corrected Kp = kpu * fup / fut.

    Args:
        kpu: Unbound tissue-to-plasma partition coefficient.
        fup: Unbound fraction in plasma.
        fut: Unbound fraction in tissue.

    Returns:
        Corrected Kp value.
    """
    return kpu * fup / fut


def predict_tissue_binding(
    drug_name: str,
    logP: float,
    fup: float,
    kp_dict: dict[str, float] | None = None,
) -> TissueBindingResult:
    """Predict tissue binding and correct Kp values.

    Uses empirical logP-based relationships to estimate the unbound fraction
    in each tissue, then optionally applies corrections to provided Kp values.

    Args:
        drug_name: Name of the drug compound (must not be empty).
        logP: Logarithm of octanol-water partition coefficient.
        fup: Unbound fraction in plasma (0 < fup <= 1).
        kp_dict: Optional mapping of tissue name to Kp value. If provided,
            corrected Kp values will be computed per tissue. If None, returns {}.

    Returns:
        TissueBindingResult with estimated fut values, kpu_to_kp_ratio, and
        optionally corrected_kp.

    Raises:
        ValueError: If fup <= 0 or fup > 1 or drug_name is empty.
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if fup <= 0 or fup > 1:
        raise ValueError(f"fup must be in (0, 1], got {fup}")

    logP_pos = max(logP, 0.0)

    fut_muscle = max(0.001, min(1.0, fup / (1 + 0.3 * logP_pos)))
    fut_adipose = max(0.001, min(1.0, fup / (1 + 0.5 * logP_pos)))
    fut_liver = max(0.001, min(1.0, fup * 0.8))
    fut_brain = max(0.001, min(1.0, fup / (1 + 0.2 * logP_pos)))

    kpu_to_kp_ratio = fup / fut_muscle

    _tissue_fut_map = {
        "muscle": fut_muscle,
        "adipose": fut_adipose,
        "liver": fut_liver,
        "brain": fut_brain,
    }

    corrected_kp: dict[str, float] = {}
    if kp_dict is not None:
        for tissue, kp_value in kp_dict.items():
            fut = _tissue_fut_map.get(tissue, fut_muscle)
            corrected_kp[tissue] = _tissue_binding_correction(kp_value, fup, fut)

    return TissueBindingResult(
        drug_name=drug_name,
        fup=fup,
        fut_muscle=fut_muscle,
        fut_adipose=fut_adipose,
        fut_liver=fut_liver,
        fut_brain=fut_brain,
        kpu_to_kp_ratio=kpu_to_kp_ratio,
        corrected_kp=corrected_kp,
    )


def compare_binding_species(
    drug_name: str,
    logP: float,
    fup_human: float,
    fup_rat: float | None = None,
    fup_dog: float | None = None,
) -> dict[str, TissueBindingResult]:
    """Compare tissue binding predictions across species.

    Runs predict_tissue_binding for each species where fup is provided.

    Args:
        drug_name: Name of the drug compound.
        logP: Logarithm of octanol-water partition coefficient.
        fup_human: Unbound fraction in human plasma.
        fup_rat: Unbound fraction in rat plasma. If None, rat is excluded.
        fup_dog: Unbound fraction in dog plasma. If None, dog is excluded.

    Returns:
        Dictionary mapping species name ('human', 'rat', 'dog') to
        TissueBindingResult. Only species with provided fup values are included.
    """
    species_fup: dict[str, float] = {"human": fup_human}
    if fup_rat is not None:
        species_fup["rat"] = fup_rat
    if fup_dog is not None:
        species_fup["dog"] = fup_dog

    return {
        species: predict_tissue_binding(drug_name, logP, fup)
        for species, fup in species_fup.items()
    }
