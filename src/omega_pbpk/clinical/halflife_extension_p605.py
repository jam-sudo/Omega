"""
Phase 605 — Drug Half-Life Extension Strategies
Models and compares different strategies for extending drug half-life.
"""

from dataclasses import dataclass

VALID_STRATEGIES = {
    "none",
    "pegylation",
    "albumin_fusion",
    "nano_encapsulation",
    "sustained_release",
}


@dataclass(frozen=True)
class HalflifeExtensionResult:
    drug_name: str
    strategy: str
    mw_original_kDa: float
    t_half_original_h: float
    t_half_extended_h: float
    extension_fold: float
    cl_reduction_factor: float
    vd_change_factor: float
    dosing_frequency_reduction: str
    immunogenicity_risk_change: str
    manufacturing_complexity: str
    notes: str


def _get_dosing_frequency_reduction(extension_fold: float) -> str:
    if extension_fold < 2.0:
        return "same"
    elif extension_fold < 4.0:
        return "2x_less_frequent"
    elif extension_fold < 12.0:
        return "4x_less_frequent"
    elif extension_fold < 84.0:
        return "weekly"
    else:
        return "monthly"


def predict_halflife_extension(
    drug_name: str,
    t_half_original_h: float,
    mw_kDa: float,
    strategy: str,
    pegylation_kDa: float = 20.0,
    albumin_binding_affinity_uM: float = 1.0,
) -> HalflifeExtensionResult:
    """
    Predict half-life extension for a drug given a specific strategy.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    t_half_original_h : float
        Original half-life in hours.
    mw_kDa : float
        Molecular weight of the original drug in kDa.
    strategy : str
        Extension strategy: "none", "pegylation", "albumin_fusion",
        "nano_encapsulation", or "sustained_release".
    pegylation_kDa : float
        Mass of PEG added in kDa (used for pegylation strategy).
    albumin_binding_affinity_uM : float
        Albumin binding affinity in uM (not directly used in formula but
        included for future extensions).

    Returns
    -------
    HalflifeExtensionResult
    """
    # Validation
    if t_half_original_h <= 0:
        raise ValueError("t_half_original_h must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}")
    if pegylation_kDa <= 0:
        raise ValueError("pegylation_kDa must be > 0")

    if strategy == "none":
        cl_reduction = 1.0
        vd_change = 1.0
        t_half_new = t_half_original_h
        immunogenicity_risk_change = "none"
        manufacturing_complexity = "low"
        notes = "No modification applied; original half-life retained."

    elif strategy == "pegylation":
        cl_reduction = 1.0 / (1.0 + pegylation_kDa / mw_kDa * 2.0)
        vd_change = 1.0 + pegylation_kDa / mw_kDa * 0.5
        t_half_new = t_half_original_h * vd_change / cl_reduction
        immunogenicity_risk_change = "reduced"
        manufacturing_complexity = "medium"
        notes = (
            f"PEGylation with {pegylation_kDa:.1f} kDa PEG chain. "
            "CL reduced via steric shielding; Vd slightly increased."
        )

    elif strategy == "albumin_fusion":
        cl_reduction = 0.05
        vd_change = 0.1
        t_half_new = t_half_original_h * vd_change / cl_reduction
        # Cap at 720 h (30 days)
        t_half_new = min(t_half_new, 720.0)
        immunogenicity_risk_change = "variable"
        manufacturing_complexity = "high"
        notes = (
            "Albumin fusion leverages FcRn recycling mechanism. "
            "Half-life capped at 720 h (30 days)."
        )

    elif strategy == "nano_encapsulation":
        cl_reduction = 0.3
        vd_change = 2.0
        t_half_new = t_half_original_h * vd_change / cl_reduction
        immunogenicity_risk_change = "increased"
        manufacturing_complexity = "high"
        notes = (
            "Nanoencapsulation slows MPS clearance and promotes tissue accumulation. "
            "Immunogenicity risk increased due to nanoparticle recognition."
        )

    elif strategy == "sustained_release":
        cl_reduction = 1.0
        vd_change = 1.0
        t_half_new = t_half_original_h * 5.0
        immunogenicity_risk_change = "none"
        manufacturing_complexity = "medium"
        notes = (
            "Sustained release (SC depot): effective half-life extended ~5x "
            "via rate-limited absorption. Drug CL and Vd unchanged."
        )

    extension_fold = t_half_new / t_half_original_h
    dosing_frequency_reduction = _get_dosing_frequency_reduction(extension_fold)

    return HalflifeExtensionResult(
        drug_name=drug_name,
        strategy=strategy,
        mw_original_kDa=mw_kDa,
        t_half_original_h=t_half_original_h,
        t_half_extended_h=t_half_new,
        extension_fold=extension_fold,
        cl_reduction_factor=cl_reduction,
        vd_change_factor=vd_change,
        dosing_frequency_reduction=dosing_frequency_reduction,
        immunogenicity_risk_change=immunogenicity_risk_change,
        manufacturing_complexity=manufacturing_complexity,
        notes=notes,
    )


def compare_extension_strategies(
    drug_name: str,
    t_half_h: float,
    mw_kDa: float,
) -> list:
    """
    Compare all half-life extension strategies for a given drug.

    Returns a list of HalflifeExtensionResult sorted by t_half_extended_h
    in descending order.
    """
    results = []
    for strategy in VALID_STRATEGIES:
        result = predict_halflife_extension(
            drug_name=drug_name,
            t_half_original_h=t_half_h,
            mw_kDa=mw_kDa,
            strategy=strategy,
        )
        results.append(result)

    results.sort(key=lambda r: r.t_half_extended_h, reverse=True)
    return results
