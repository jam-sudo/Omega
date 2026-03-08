"""Models for drug half-life extension strategies (PEGylation, albumin fusion, etc.)."""

from __future__ import annotations

from dataclasses import dataclass

_LN2 = 0.693147

# Strategy parameters: (cl_factor, vd_factor, notes)
_STRATEGIES: dict[str, tuple[float, float, str]] = {
    "PEGylation": (0.2, 3.0, "MW increase reduces renal/hepatic CL"),
    "albumin_fusion": (0.05, 1.5, "FcRn recycling, ~3 week t½"),
    "FcRn_engineering": (0.03, 1.2, "Optimized FcRn binding, t½ ~4 weeks"),
    "nanoparticle": (0.3, 5.0, "Depot effect, slow release"),
    "prodrug_depot": (0.1, 8.0, "IM/SC depot, slow absorption"),
}


def _classify_frequency(extension_factor: float) -> str:
    """Map t½ extension factor to dosing frequency reduction label."""
    if extension_factor < 2.0:
        return "no_change"
    if extension_factor < 4.0:
        return "2x_longer"
    if extension_factor < 7.0:
        return "weekly"
    if extension_factor < 14.0:
        return "biweekly"
    if extension_factor < 30.0:
        return "monthly"
    return "quarterly"


@dataclass(frozen=True)
class HalfLifeExtensionResult:
    """Result of a half-life extension strategy prediction."""

    drug_name: str
    strategy: str
    parent_cl_l_per_h: float
    parent_vd_l: float
    modified_cl_l_per_h: float
    modified_vd_l: float
    parent_t_half_h: float
    modified_t_half_h: float
    t_half_extension_factor: float
    dosing_frequency_reduction: str
    target_dose_mg: float
    notes: str


def predict_half_life_extension(
    drug_name: str,
    strategy: str,
    parent_cl_l_per_h: float,
    parent_vd_l: float,
    parent_dose_mg: float,
) -> HalfLifeExtensionResult:
    """Predict PK changes from a half-life extension strategy.

    Parameters
    ----------
    drug_name : str
        Name of the parent drug.
    strategy : str
        One of: PEGylation, albumin_fusion, FcRn_engineering, nanoparticle, prodrug_depot.
    parent_cl_l_per_h : float
        Parent drug clearance (L/h). Must be > 0.
    parent_vd_l : float
        Parent drug volume of distribution (L). Must be > 0.
    parent_dose_mg : float
        Parent drug dose (mg). Must be > 0.

    Returns
    -------
    HalfLifeExtensionResult
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {set(_STRATEGIES.keys())}, got '{strategy}'")
    if parent_cl_l_per_h <= 0:
        raise ValueError(f"parent_cl_l_per_h must be > 0, got {parent_cl_l_per_h}")
    if parent_vd_l <= 0:
        raise ValueError(f"parent_vd_l must be > 0, got {parent_vd_l}")
    if parent_dose_mg <= 0:
        raise ValueError(f"parent_dose_mg must be > 0, got {parent_dose_mg}")

    cl_factor, vd_factor, strategy_notes = _STRATEGIES[strategy]

    modified_cl = parent_cl_l_per_h * cl_factor
    modified_vd = parent_vd_l * vd_factor

    parent_t_half = _LN2 * parent_vd_l / parent_cl_l_per_h
    modified_t_half = _LN2 * modified_vd / modified_cl

    extension_factor = modified_t_half / parent_t_half
    freq_label = _classify_frequency(extension_factor)

    # Target dose for same Css: dose proportional to t½ extension
    target_dose = parent_dose_mg * (modified_t_half / parent_t_half)

    return HalfLifeExtensionResult(
        drug_name=drug_name,
        strategy=strategy,
        parent_cl_l_per_h=parent_cl_l_per_h,
        parent_vd_l=parent_vd_l,
        modified_cl_l_per_h=modified_cl,
        modified_vd_l=modified_vd,
        parent_t_half_h=parent_t_half,
        modified_t_half_h=modified_t_half,
        t_half_extension_factor=extension_factor,
        dosing_frequency_reduction=freq_label,
        target_dose_mg=target_dose,
        notes=strategy_notes,
    )


def compare_strategies(
    drug_name: str,
    parent_cl_l_per_h: float,
    parent_vd_l: float,
    parent_dose_mg: float,
) -> list[HalfLifeExtensionResult]:
    """Compare all 5 half-life extension strategies, sorted by modified_t_half_h descending."""
    results = [
        predict_half_life_extension(
            drug_name=drug_name,
            strategy=s,
            parent_cl_l_per_h=parent_cl_l_per_h,
            parent_vd_l=parent_vd_l,
            parent_dose_mg=parent_dose_mg,
        )
        for s in _STRATEGIES
    ]
    return sorted(results, key=lambda r: r.modified_t_half_h, reverse=True)
