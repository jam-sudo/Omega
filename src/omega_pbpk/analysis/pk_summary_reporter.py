"""PK summary reporter — Phase 779.

Generates a structured PK report from simulation results including
therapeutic window assessment, accumulation, and clinical recommendations.
Pure Python only (stdlib math + dataclasses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PKSummaryReport",
    "generate_pk_summary",
    "batch_generate_pk_reports",
]


@dataclass(frozen=True)
class PKSummaryReport:
    """Structured pharmacokinetic report from simulation results."""

    drug_name: str
    route: str
    dose_mg: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    vd_L: float
    cl_L_per_h: float
    f_oral: float
    accumulation_ratio: float
    steady_state_flag: bool
    subtherapeutic: bool
    supratherapeutic: bool
    mec_mg_L: float
    mtc_mg_L: float
    time_in_therapeutic_pct: float
    summary_text: str
    recommendations: str


def _compute_t_half(vd_L: float, cl_L_per_h: float) -> float:
    """Compute elimination half-life from Vd and CL."""
    if cl_L_per_h <= 0.0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0.0:
        raise ValueError("vd_L must be positive")
    return 0.693 * vd_L / cl_L_per_h


def _compute_accumulation_ratio(tau_h: float, t_half_h: float) -> float:
    """Compute accumulation ratio at steady state, clamped to [1.0, 10.0]."""
    if t_half_h <= 0.0:
        return 1.0
    exponent = -0.693 * tau_h / t_half_h
    # Avoid math domain errors for very large exponents
    if exponent < -100.0:
        return 1.0
    denom = 1.0 - math.exp(exponent)
    if denom <= 0.0:
        return 10.0
    ratio = 1.0 / denom
    return max(1.0, min(10.0, ratio))


def _build_summary_text(
    drug_name: str,
    route: str,
    dose_mg: float,
    cmax_mg_L: float,
    tmax_h: float,
    auc_mg_h_per_L: float,
    t_half_h: float,
    vd_L: float,
    cl_L_per_h: float,
    f_oral: float,
    accumulation_ratio: float,
    steady_state_flag: bool,
    subtherapeutic: bool,
    supratherapeutic: bool,
    mec_mg_L: float,
    mtc_mg_L: float,
    time_in_therapeutic_pct: float,
) -> str:
    """Build a human-readable PK summary paragraph."""
    route_label = route.upper()
    lines = [
        f"PK Report: {drug_name} ({route_label}, {dose_mg:.1f} mg)",
        f"  Cmax = {cmax_mg_L:.3f} mg/L at Tmax = {tmax_h:.2f} h",
        f"  AUC  = {auc_mg_h_per_L:.3f} mg·h/L",
        f"  t½   = {t_half_h:.2f} h",
        f"  Vd   = {vd_L:.2f} L  |  CL = {cl_L_per_h:.3f} L/h",
        f"  F    = {f_oral * 100:.1f}%  |  Accumulation ratio = {accumulation_ratio:.2f}x",
    ]
    if steady_state_flag:
        lines.append("  Steady-state: YES (dosing interval < t½; drug accumulates to true SS)")
    else:
        lines.append("  Steady-state: NO (dosing interval >= t½; minimal accumulation expected)")

    window_label = "WITHIN"
    if subtherapeutic:
        window_label = "BELOW (subtherapeutic)"
    elif supratherapeutic:
        window_label = "ABOVE (supratherapeutic)"
    lines.append(
        f"  Therapeutic window [{mec_mg_L:.3f}–{mtc_mg_L:.3f} mg/L]: Cmax is {window_label}"
    )
    lines.append(f"  Time in therapeutic range = {time_in_therapeutic_pct:.1f}%")
    return "\n".join(lines)


def _build_recommendations(
    subtherapeutic: bool,
    supratherapeutic: bool,
    steady_state_flag: bool,
    accumulation_ratio: float,
    time_in_therapeutic_pct: float,
    dose_mg: float,
    f_oral: float,
    route: str,
) -> str:
    """Generate clinical recommendation strings."""
    recs: list[str] = []

    if subtherapeutic:
        recs.append(
            "Dose increase recommended: Cmax is below the MEC. Consider increasing dose "
            "or reducing dosing interval to achieve therapeutic exposure."
        )
    elif supratherapeutic:
        recs.append(
            "Dose reduction recommended: Cmax exceeds the MTC. Reduce dose or extend "
            "dosing interval to avoid toxicity."
        )
    else:
        recs.append("Current dose appears to be within the therapeutic window.")

    if steady_state_flag and accumulation_ratio > 3.0:
        recs.append(
            f"High accumulation ratio ({accumulation_ratio:.2f}x) at steady state. "
            "Monitor for delayed toxicity with repeated dosing."
        )
    elif steady_state_flag:
        recs.append(f"Moderate accumulation ({accumulation_ratio:.2f}x) expected at steady state.")

    if time_in_therapeutic_pct > 0.0 and time_in_therapeutic_pct < 50.0:
        recs.append(
            f"Low time in therapeutic range ({time_in_therapeutic_pct:.1f}%). "
            "Consider modified-release formulation or more frequent dosing."
        )
    elif time_in_therapeutic_pct >= 80.0:
        recs.append(f"Adequate time in therapeutic range ({time_in_therapeutic_pct:.1f}%).")

    if route.lower() == "oral" and f_oral < 0.3:
        recs.append(
            f"Low oral bioavailability (F = {f_oral * 100:.1f}%). Consider IV or "
            "alternative routes, or formulation optimization."
        )

    return " ".join(recs) if recs else "No specific recommendations."


def generate_pk_summary(
    drug_name: str,
    route: str,
    dose_mg: float,
    cmax_mg_L: float,
    tmax_h: float,
    auc_mg_h_per_L: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    tau_h: float = 24.0,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    time_in_therapeutic_pct: float = 0.0,
) -> PKSummaryReport:
    """Generate a structured PK summary report.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    route:
        Administration route (e.g. "oral", "iv").
    dose_mg:
        Administered dose in mg.
    cmax_mg_L:
        Peak plasma concentration in mg/L.
    tmax_h:
        Time to peak in hours.
    auc_mg_h_per_L:
        Area under the curve in mg·h/L.
    cl_L_per_h:
        Total clearance in L/h.
    vd_L:
        Volume of distribution in L.
    f_oral:
        Oral bioavailability fraction [0, 1].
    tau_h:
        Dosing interval in hours.
    mec_mg_L:
        Minimum effective concentration in mg/L.
    mtc_mg_L:
        Maximum tolerated concentration in mg/L.
    time_in_therapeutic_pct:
        Percentage of dosing interval within therapeutic window.

    Returns
    -------
    PKSummaryReport
        Frozen dataclass with all computed PK descriptors.
    """
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0.0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0.0:
        raise ValueError("vd_L must be positive")
    if not (0.0 <= f_oral <= 1.0):
        raise ValueError("f_oral must be in [0, 1]")
    if tau_h <= 0.0:
        raise ValueError("tau_h must be positive")
    if mec_mg_L < 0.0:
        raise ValueError("mec_mg_L must be non-negative")
    if mtc_mg_L <= mec_mg_L:
        raise ValueError("mtc_mg_L must be greater than mec_mg_L")

    t_half_h = _compute_t_half(vd_L, cl_L_per_h)
    accumulation_ratio = _compute_accumulation_ratio(tau_h, t_half_h)
    steady_state_flag = tau_h < t_half_h
    subtherapeutic = cmax_mg_L < mec_mg_L
    supratherapeutic = cmax_mg_L > mtc_mg_L

    summary_text = _build_summary_text(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        t_half_h=t_half_h,
        vd_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        f_oral=f_oral,
        accumulation_ratio=accumulation_ratio,
        steady_state_flag=steady_state_flag,
        subtherapeutic=subtherapeutic,
        supratherapeutic=supratherapeutic,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        time_in_therapeutic_pct=time_in_therapeutic_pct,
    )

    recommendations = _build_recommendations(
        subtherapeutic=subtherapeutic,
        supratherapeutic=supratherapeutic,
        steady_state_flag=steady_state_flag,
        accumulation_ratio=accumulation_ratio,
        time_in_therapeutic_pct=time_in_therapeutic_pct,
        dose_mg=dose_mg,
        f_oral=f_oral,
        route=route,
    )

    return PKSummaryReport(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        t_half_h=t_half_h,
        vd_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        f_oral=f_oral,
        accumulation_ratio=accumulation_ratio,
        steady_state_flag=steady_state_flag,
        subtherapeutic=subtherapeutic,
        supratherapeutic=supratherapeutic,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        time_in_therapeutic_pct=time_in_therapeutic_pct,
        summary_text=summary_text,
        recommendations=recommendations,
    )


def batch_generate_pk_reports(reports_params: list) -> list:
    """Generate multiple PK reports from a list of parameter dicts.

    Each dict must have the required keys matching generate_pk_summary's
    parameters. Optional keys default to generate_pk_summary defaults.

    Parameters
    ----------
    reports_params:
        List of parameter dictionaries.

    Returns
    -------
    list[PKSummaryReport]
        Ordered list of PKSummaryReport instances.
    """
    results = []
    for params in reports_params:
        report = generate_pk_summary(**params)
        results.append(report)
    return results
