"""Phase 1140: Solubility enhancer screening for poorly water-soluble drugs.

Screens cosolvents, cyclodextrins, surfactants, pH adjusters, and salt forms
to identify optimal solubility enhancement strategies for BCS II/IV drugs.
"""

from __future__ import annotations

from dataclasses import dataclass

_ENHANCERS: dict = {
    "PEG400": {"type": "cosolvent", "max_fold": 10.0, "sigma": 0.8},
    "HPBCD": {"type": "cyclodextrin", "max_fold": 50.0, "sigma": 1.2},
    "polysorbate80": {"type": "surfactant", "max_fold": 20.0, "sigma": 1.0},
    "SDS": {"type": "surfactant", "max_fold": 30.0, "sigma": 1.5},
    "NaOH_pH": {"type": "pH_adjustment", "max_fold": 100.0, "sigma": 1.5},
    "HCl_pH": {"type": "pH_adjustment", "max_fold": 100.0, "sigma": 1.5},
    "meglumine_salt": {"type": "salt", "max_fold": 200.0, "sigma": 1.8},
    "HCl_salt": {"type": "salt", "max_fold": 150.0, "sigma": 1.6},
}

_REGULATORY_STATUS: dict = {
    "cyclodextrin": "established",
    "cosolvent": "established",
    "surfactant": "established",
    "salt": "generally_recognized",
    "pH_adjustment": "generally_recognized",
}

# pH adjusters and salts applicable to specific ionization types
_PH_ADJUSTER_ION_MAP: dict = {
    "NaOH_pH": "acid",
    "HCl_pH": "base",
    "meglumine_salt": "acid",
    "HCl_salt": "base",
}


@dataclass(frozen=True)
class SolubilityEnhancerResult:
    """Result for a single solubility enhancer evaluation."""

    drug_name: str
    enhancer_name: str
    enhancer_type: str
    intrinsic_solubility_mg_mL: float
    logP: float
    ionization_type: str
    fold_enhancement: float
    enhanced_solubility_mg_mL: float
    target_met: bool
    regulatory_status: str
    notes: str


def _validate_inputs(
    intrinsic_solubility_mg_mL: float,
    logP: float,
    ionization_type: str,
) -> None:
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )


def _compute_fold_enhancement(
    enhancer_name: str,
    enhancer_info: dict,
    logP: float,
    ionization_type: str,
) -> float:
    """Compute fold enhancement for a given enhancer."""
    enh_type = enhancer_info["type"]
    max_fold = enhancer_info["max_fold"]
    sigma = enhancer_info["sigma"]

    # pH adjusters and salts: only applicable to matching ionization type
    if enh_type in {"pH_adjustment", "salt"}:
        required_ion = _PH_ADJUSTER_ION_MAP.get(enhancer_name)
        if required_ion is None or ionization_type != required_ion:
            return 1.0
        return max_fold

    # Cosolvents and surfactants/cyclodextrins: logP-dependent
    base_fold = sigma * logP if logP > 0.0 else 1.0
    logp_factor = max(1.0, logP / 2.0)
    fold = base_fold * logp_factor
    fold = min(fold, max_fold)
    fold = max(fold, 1.0)
    return fold


def _build_result(
    drug_name: str,
    enhancer_name: str,
    enhancer_info: dict,
    intrinsic_solubility_mg_mL: float,
    logP: float,
    ionization_type: str,
    fold_enhancement: float,
    target_solubility_mg_mL: float,
) -> SolubilityEnhancerResult:
    enhanced = intrinsic_solubility_mg_mL * fold_enhancement
    enh_type = enhancer_info["type"]
    reg_status = _REGULATORY_STATUS.get(enh_type, "established")
    target_met = enhanced >= target_solubility_mg_mL

    notes_parts = [
        f"Type: {enh_type}",
        f"Fold enhancement: {fold_enhancement:.2f}x",
        f"Enhanced solubility: {enhanced:.4f} mg/mL",
    ]
    if enh_type in {"pH_adjustment", "salt"}:
        required_ion = _PH_ADJUSTER_ION_MAP.get(enhancer_name)
        if required_ion and ionization_type == required_ion:
            notes_parts.append(f"Applicable to {ionization_type} drugs")
        else:
            notes_parts.append(f"Not applicable (requires {required_ion} drug)")

    return SolubilityEnhancerResult(
        drug_name=drug_name,
        enhancer_name=enhancer_name,
        enhancer_type=enh_type,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        logP=logP,
        ionization_type=ionization_type,
        fold_enhancement=fold_enhancement,
        enhanced_solubility_mg_mL=enhanced,
        target_met=target_met,
        regulatory_status=reg_status,
        notes="; ".join(notes_parts),
    )


def screen_solubility_enhancers(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    logP: float,
    ionization_type: str,
) -> list:
    """Screen all enhancers and return sorted list by enhanced solubility (descending).

    Args:
        drug_name: Name of the drug.
        intrinsic_solubility_mg_mL: Aqueous solubility without enhancement (mg/mL).
        logP: Octanol-water partition coefficient.
        ionization_type: One of 'acid', 'base', 'neutral'.

    Returns:
        List of SolubilityEnhancerResult sorted descending by enhanced_solubility_mg_mL.
    """
    _validate_inputs(intrinsic_solubility_mg_mL, logP, ionization_type)

    results = []
    for enhancer_name, enhancer_info in _ENHANCERS.items():
        fold = _compute_fold_enhancement(enhancer_name, enhancer_info, logP, ionization_type)
        result = _build_result(
            drug_name=drug_name,
            enhancer_name=enhancer_name,
            enhancer_info=enhancer_info,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            logP=logP,
            ionization_type=ionization_type,
            fold_enhancement=fold,
            target_solubility_mg_mL=0.0,  # target_met always False here
        )
        results.append(result)

    results.sort(key=lambda r: r.enhanced_solubility_mg_mL, reverse=True)
    return results


def find_optimal_enhancer(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    logP: float,
    ionization_type: str,
    target_solubility_mg_mL: float,
) -> SolubilityEnhancerResult:
    """Find the first enhancer that meets the target solubility, or the best available.

    Args:
        drug_name: Name of the drug.
        intrinsic_solubility_mg_mL: Aqueous solubility without enhancement (mg/mL).
        logP: Octanol-water partition coefficient.
        ionization_type: One of 'acid', 'base', 'neutral'.
        target_solubility_mg_mL: Desired minimum solubility (mg/mL).

    Returns:
        SolubilityEnhancerResult for the first enhancer meeting the target,
        or the best available if none meet the target.
    """
    _validate_inputs(intrinsic_solubility_mg_mL, logP, ionization_type)

    results = []
    for enhancer_name, enhancer_info in _ENHANCERS.items():
        fold = _compute_fold_enhancement(enhancer_name, enhancer_info, logP, ionization_type)
        result = _build_result(
            drug_name=drug_name,
            enhancer_name=enhancer_name,
            enhancer_info=enhancer_info,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            logP=logP,
            ionization_type=ionization_type,
            fold_enhancement=fold,
            target_solubility_mg_mL=target_solubility_mg_mL,
        )
        results.append(result)

    results.sort(key=lambda r: r.enhanced_solubility_mg_mL, reverse=True)

    # Return first that meets target, else best available
    for result in results:
        if result.target_met:
            return result
    return results[0]
