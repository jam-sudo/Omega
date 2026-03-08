"""Phase 561 — Cardiac Output Effect on Drug Distribution.

Models how changes in cardiac output affect drug distribution and elimination,
relevant for heart failure, exercise, and critical illness.
"""

from dataclasses import dataclass

NORMAL_CO = 90.0  # L/h
Q_HEPATIC_NORMAL = 72.0  # L/h
Q_RENAL_NORMAL = 72.0  # L/h
GFR_NORMAL_L_PER_H = 7.2  # ~120 mL/min


@dataclass(frozen=True)
class CardiacOutputDistributionResult:
    drug_name: str
    cardiac_output_L_per_h: float
    cl_hepatic_L_per_h: float
    cl_renal_L_per_h: float
    vd_L: float
    t_half_h: float
    e_hepatic: float
    distribution_rate_constant_h: float
    auc_ratio_vs_normal: float
    co_class: str
    clinical_impact: str
    notes: str


def _classify_co(co: float) -> str:
    if co < 60.0:
        return "low"
    elif co > 120.0:
        return "high"
    return "normal"


def _classify_impact(auc_ratio: float) -> str:
    if auc_ratio < 0.7 or auc_ratio > 1.3:
        return "significant"
    if auc_ratio < 0.8 or auc_ratio > 1.2:
        return "moderate"
    return "minimal"


def predict_co_distribution_effect(
    drug_name: str,
    cardiac_output_L_per_h: float,
    baseline_cl_hepatic_normal_L_per_h: float,
    fup: float,
    baseline_vd_L: float,
    e_hepatic_normal: float = 0.3,
) -> CardiacOutputDistributionResult:
    """Predict how cardiac output changes affect drug distribution."""
    if cardiac_output_L_per_h <= 0:
        raise ValueError("cardiac_output_L_per_h must be > 0")
    if fup <= 0 or fup > 1:
        raise ValueError("fup must be in (0, 1]")
    if baseline_vd_L <= 0:
        raise ValueError("baseline_vd_L must be > 0")
    if e_hepatic_normal < 0 or e_hepatic_normal > 1:
        raise ValueError("e_hepatic_normal must be in [0, 1]")

    co_ratio = cardiac_output_L_per_h / NORMAL_CO

    # Hepatic clearance scaling based on extraction ratio
    q_hepatic = Q_HEPATIC_NORMAL * co_ratio
    if e_hepatic_normal > 0.7:
        cl_hepatic = q_hepatic * e_hepatic_normal
    elif e_hepatic_normal < 0.3:
        cl_hepatic = baseline_cl_hepatic_normal_L_per_h
    else:
        cl_hepatic = baseline_cl_hepatic_normal_L_per_h * (0.5 + 0.5 * co_ratio)

    # Renal clearance scaling
    cl_renal = GFR_NORMAL_L_PER_H * co_ratio * fup

    # Normal clearances for AUC ratio
    cl_hepatic_normal = baseline_cl_hepatic_normal_L_per_h
    cl_renal_normal = GFR_NORMAL_L_PER_H * fup

    cl_total = cl_hepatic + cl_renal
    cl_total_normal = cl_hepatic_normal + cl_renal_normal

    vd = baseline_vd_L
    t_half = 0.693 * vd / cl_total if cl_total > 0 else float("inf")

    # AUC ratio: normal CL / current CL (higher CL → lower AUC)
    auc_ratio = cl_total_normal / cl_total if cl_total > 0 else float("inf")

    # Distribution rate constant (simplified k12)
    k12 = cl_total / vd if vd > 0 else 0.0

    co_class = _classify_co(cardiac_output_L_per_h)
    clinical_impact = _classify_impact(auc_ratio)

    notes_parts = []
    if co_class == "low":
        notes_parts.append("Reduced CO may increase drug exposure")
    elif co_class == "high":
        notes_parts.append("Elevated CO may decrease drug exposure")
    if clinical_impact == "significant":
        notes_parts.append("Dose adjustment recommended")
    notes = "; ".join(notes_parts) if notes_parts else "No significant impact expected"

    return CardiacOutputDistributionResult(
        drug_name=drug_name,
        cardiac_output_L_per_h=cardiac_output_L_per_h,
        cl_hepatic_L_per_h=cl_hepatic,
        cl_renal_L_per_h=cl_renal,
        vd_L=vd,
        t_half_h=t_half,
        e_hepatic=e_hepatic_normal,
        distribution_rate_constant_h=k12,
        auc_ratio_vs_normal=auc_ratio,
        co_class=co_class,
        clinical_impact=clinical_impact,
        notes=notes,
    )


def compare_cardiac_outputs(
    drug_name: str,
    co_values: list,
    baseline_cl_hepatic_normal_L_per_h: float,
    fup: float,
    baseline_vd_L: float,
) -> list:
    """Compare distribution effects across multiple cardiac output values.

    Returns results sorted by auc_ratio_vs_normal descending.
    """
    results = [
        predict_co_distribution_effect(
            drug_name=drug_name,
            cardiac_output_L_per_h=co,
            baseline_cl_hepatic_normal_L_per_h=baseline_cl_hepatic_normal_L_per_h,
            fup=fup,
            baseline_vd_L=baseline_vd_L,
        )
        for co in co_values
    ]
    results.sort(key=lambda r: r.auc_ratio_vs_normal, reverse=True)
    return results
