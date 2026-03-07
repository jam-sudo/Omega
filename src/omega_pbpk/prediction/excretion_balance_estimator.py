"""Phase 282 — Excretion Balance Estimator.

Estimate mass balance of drug excretion across renal, hepatic/biliary,
and pulmonary routes. Uses clearance partitioning to compute fraction
excreted via each route.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExcretionBalanceResult:
    """Result of excretion balance estimation."""

    drug_name: str
    dose_mg: float
    fu_plasma: float
    cl_total_L_per_h: float
    ke_per_h: float
    t_half_h: float
    fe_renal: float
    fe_hepatic: float
    fe_pulmonary: float
    fe_other: float
    amount_excreted_renal_mg: float
    amount_excreted_hepatic_mg: float
    amount_excreted_pulmonary_mg: float
    cl_renal_gfr_estimate_L_per_h: float
    dominant_route: str
    excretion_class: str
    notes: str


def _validate_excretion_inputs(
    cl_renal_L_per_h: float,
    cl_hepatic_L_per_h: float,
    cl_pulmonary_L_per_h: float,
    cl_other_L_per_h: float,
    dose_mg: float,
    vd_L: float,
    fu_plasma: float,
) -> None:
    """Validate inputs for excretion balance estimation."""
    if cl_renal_L_per_h < 0:
        raise ValueError(f"cl_renal_L_per_h must be >= 0, got {cl_renal_L_per_h}")
    if cl_hepatic_L_per_h < 0:
        raise ValueError(f"cl_hepatic_L_per_h must be >= 0, got {cl_hepatic_L_per_h}")
    if cl_pulmonary_L_per_h < 0:
        raise ValueError(f"cl_pulmonary_L_per_h must be >= 0, got {cl_pulmonary_L_per_h}")
    if cl_other_L_per_h < 0:
        raise ValueError(f"cl_other_L_per_h must be >= 0, got {cl_other_L_per_h}")

    cl_total = cl_renal_L_per_h + cl_hepatic_L_per_h + cl_pulmonary_L_per_h + cl_other_L_per_h
    if cl_total <= 0:
        raise ValueError("At least one clearance value must be > 0")

    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")


def estimate_excretion_balance(
    drug_name: str,
    cl_renal_L_per_h: float,
    cl_hepatic_L_per_h: float,
    cl_pulmonary_L_per_h: float,
    cl_other_L_per_h: float,
    dose_mg: float,
    vd_L: float,
    fu_plasma: float,
) -> ExcretionBalanceResult:
    """Estimate mass balance of drug excretion across routes.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    cl_renal_L_per_h:
        Renal clearance (L/h). Must be >= 0.
    cl_hepatic_L_per_h:
        Hepatic/biliary clearance (L/h). Must be >= 0.
    cl_pulmonary_L_per_h:
        Pulmonary clearance (L/h). Must be >= 0.
    cl_other_L_per_h:
        Other clearance routes (L/h). Must be >= 0.
    dose_mg:
        Administered dose (mg). Must be > 0.
    vd_L:
        Volume of distribution (L). Must be > 0.
    fu_plasma:
        Unbound fraction in plasma. Must be in (0, 1].

    Returns
    -------
    ExcretionBalanceResult
        Frozen dataclass with excretion fractions and derived metrics.
    """
    _validate_excretion_inputs(
        cl_renal_L_per_h,
        cl_hepatic_L_per_h,
        cl_pulmonary_L_per_h,
        cl_other_L_per_h,
        dose_mg,
        vd_L,
        fu_plasma,
    )

    cl_total = cl_renal_L_per_h + cl_hepatic_L_per_h + cl_pulmonary_L_per_h + cl_other_L_per_h
    ke = cl_total / vd_L
    t_half = 0.693 / ke

    # Fraction excreted via each route
    fe_renal = cl_renal_L_per_h / cl_total
    fe_hepatic = cl_hepatic_L_per_h / cl_total
    fe_pulmonary = cl_pulmonary_L_per_h / cl_total
    fe_other = cl_other_L_per_h / cl_total

    # Estimated amounts excreted (simplified AUC-based)
    amount_renal = dose_mg * fe_renal
    amount_hepatic = dose_mg * fe_hepatic
    amount_pulmonary = dose_mg * fe_pulmonary

    # GFR-based reference estimate (fu_plasma * 7.5 L/h)
    cl_renal_gfr_estimate = fu_plasma * 7.5

    # Dominant route
    route_fractions = {
        "renal": fe_renal,
        "hepatic": fe_hepatic,
        "pulmonary": fe_pulmonary,
        "other": fe_other,
    }
    dominant_route = max(route_fractions, key=lambda k: route_fractions[k])

    # Excretion class
    if fe_renal > 0.5:
        excretion_class = "renally_cleared"
    elif fe_hepatic > 0.5:
        excretion_class = "hepatically_cleared"
    else:
        excretion_class = "mixed"

    notes = (
        f"CL_total={cl_total:.3f} L/h. "
        f"t_half={t_half:.2f} h. "
        f"Dominant route: {dominant_route}. "
        f"Class: {excretion_class}."
    )

    return ExcretionBalanceResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fu_plasma=fu_plasma,
        cl_total_L_per_h=cl_total,
        ke_per_h=ke,
        t_half_h=t_half,
        fe_renal=fe_renal,
        fe_hepatic=fe_hepatic,
        fe_pulmonary=fe_pulmonary,
        fe_other=fe_other,
        amount_excreted_renal_mg=amount_renal,
        amount_excreted_hepatic_mg=amount_hepatic,
        amount_excreted_pulmonary_mg=amount_pulmonary,
        cl_renal_gfr_estimate_L_per_h=cl_renal_gfr_estimate,
        dominant_route=dominant_route,
        excretion_class=excretion_class,
        notes=notes,
    )


def screen_excretion_routes(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    fu_plasma: float,
    cl_renal_list: list,
    cl_hepatic_list: list,
    cl_pulmonary_L_per_h: float = 0.0,
    cl_other_L_per_h: float = 0.0,
) -> list:
    """Screen multiple renal/hepatic clearance combinations.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Administered dose (mg).
    vd_L:
        Volume of distribution (L).
    fu_plasma:
        Unbound fraction in plasma.
    cl_renal_list:
        List of renal clearance values (L/h) to screen.
    cl_hepatic_list:
        List of hepatic clearance values (L/h) to screen.
    cl_pulmonary_L_per_h:
        Fixed pulmonary clearance (L/h).
    cl_other_L_per_h:
        Fixed other clearance (L/h).

    Returns
    -------
    list[ExcretionBalanceResult]
        Results sorted by fe_renal descending.
    """
    results = []
    for cl_r in cl_renal_list:
        for cl_h in cl_hepatic_list:
            result = estimate_excretion_balance(
                drug_name=drug_name,
                cl_renal_L_per_h=cl_r,
                cl_hepatic_L_per_h=cl_h,
                cl_pulmonary_L_per_h=cl_pulmonary_L_per_h,
                cl_other_L_per_h=cl_other_L_per_h,
                dose_mg=dose_mg,
                vd_L=vd_L,
                fu_plasma=fu_plasma,
            )
            results.append(result)

    results.sort(key=lambda r: r.fe_renal, reverse=True)
    return results
