"""Phase 350 — Hepatic Blood Flow Effect on CL.

Models the effect of hepatic blood flow changes on drug clearance using the
well-stirred liver model and simulates flow-dependent PK (IV bolus, 1-cpt).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HepaticFlowEffectResult:
    """Result of hepatic blood flow effect simulation."""

    times_h: list
    c_baseline_mg_L: list
    c_modified_mg_L: list
    cl_baseline_L_per_h: float
    cl_modified_L_per_h: float
    e_baseline: float
    e_modified: float
    auc_baseline_mg_h_per_L: float
    auc_modified_mg_h_per_L: float
    auc_ratio: float
    flow_sensitivity_class: str
    clinical_significance: str
    notes: str


def _well_stirred_cl(q_h: float, fu: float, cl_int: float) -> float:
    """Compute hepatic CL using the well-stirred model.

    CL_hepatic = (Qh * fu * CLint) / (Qh + fu * CLint)
    """
    return (q_h * fu * cl_int) / (q_h + fu * cl_int)


def _extraction_ratio(cl_h: float, q_h: float) -> float:
    """Extraction ratio E = CL_hepatic / Qh."""
    return cl_h / q_h


def _simulate_iv_bolus(
    dose_mg: float,
    vd_L: float,
    ke: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Simulate IV bolus PK with Forward Euler.

    Returns (times_h, concentrations_mg_L).
    """
    c0 = dose_mg / vd_L
    times = [0.0]
    concs = [c0]

    t = 0.0
    c = c0
    while t < t_end_h - dt_h * 0.5:
        dc_dt = -ke * c
        c = c + dc_dt * dt_h
        if c < 0.0:
            c = 0.0
        t = t + dt_h
        times.append(t)
        concs.append(c)

    return times, concs


def _trapezoidal_auc(x: list[float], y: list[float]) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _flow_sensitivity_class(e: float) -> str:
    """Classify flow sensitivity based on extraction ratio."""
    if e > 0.7:
        return "flow_sensitive"
    if e < 0.3:
        return "flow_insensitive"
    return "intermediate"


def _clinical_significance(auc_ratio: float) -> str:
    """Classify clinical significance based on AUC ratio."""
    deviation = abs(auc_ratio - 1.0)
    if deviation > 0.3:
        return "major"
    if deviation >= 0.1:
        return "moderate"
    return "minor"


def simulate_hepatic_flow_effect(
    drug_name: str,
    dose_mg: float,
    cl_intrinsic_L_per_h: float,
    fu_plasma: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
    q_hepatic_baseline_L_per_h: float = 90.0,
    flow_change_pct: float = 0.0,
) -> HepaticFlowEffectResult:
    """Simulate the effect of hepatic blood flow change on drug clearance and PK.

    Parameters
    ----------
    drug_name:
        Name of the drug being evaluated.
    dose_mg:
        IV bolus dose (mg).
    cl_intrinsic_L_per_h:
        Intrinsic clearance (L/h), unbound hepatic microsomal.
    fu_plasma:
        Fraction unbound in plasma (0 < fu <= 1).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step for Forward Euler (h).
    q_hepatic_baseline_L_per_h:
        Baseline hepatic blood flow (L/h), default 90.0.
    flow_change_pct:
        Percent change in Qh (e.g. -30 for 30% reduction). Must be > -100.

    Returns
    -------
    HepaticFlowEffectResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_intrinsic_L_per_h <= 0:
        raise ValueError("cl_intrinsic_L_per_h must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if flow_change_pct <= -100.0:
        raise ValueError("flow_change_pct must be > -100 (cannot eliminate all flow)")
    if q_hepatic_baseline_L_per_h <= 0:
        raise ValueError("q_hepatic_baseline_L_per_h must be > 0")

    # Baseline and modified hepatic blood flow
    q_baseline = q_hepatic_baseline_L_per_h
    q_modified = q_baseline * (1.0 + flow_change_pct / 100.0)

    # Well-stirred CL
    cl_baseline = _well_stirred_cl(q_baseline, fu_plasma, cl_intrinsic_L_per_h)
    cl_modified = _well_stirred_cl(q_modified, fu_plasma, cl_intrinsic_L_per_h)

    # Extraction ratios
    e_baseline = _extraction_ratio(cl_baseline, q_baseline)
    e_modified = _extraction_ratio(cl_modified, q_modified)

    # Elimination rate constants
    ke_baseline = cl_baseline / vd_L
    ke_modified = cl_modified / vd_L

    # Simulate IV bolus PK
    times, c_baseline = _simulate_iv_bolus(dose_mg, vd_L, ke_baseline, t_end_h, dt_h)
    _, c_modified = _simulate_iv_bolus(dose_mg, vd_L, ke_modified, t_end_h, dt_h)

    # AUC via trapezoidal rule
    auc_baseline = _trapezoidal_auc(times, c_baseline)
    auc_modified = _trapezoidal_auc(times, c_modified)

    auc_ratio = auc_modified / auc_baseline if auc_baseline > 0 else 1.0

    # Classification
    flow_class = _flow_sensitivity_class(e_baseline)
    clin_sig = _clinical_significance(auc_ratio)

    # Notes
    notes_parts = [
        f"{drug_name}: E={e_baseline:.3f} ({flow_class}).",
        f"Flow change {flow_change_pct:+.1f}%: CL {cl_baseline:.2f} -> {cl_modified:.2f} L/h.",
        f"AUC ratio (modified/baseline) = {auc_ratio:.3f} ({clin_sig} clinical significance).",
    ]
    if flow_class == "flow_sensitive" and abs(flow_change_pct) > 10:
        notes_parts.append("High extraction drug: CL is sensitive to hepatic blood flow changes.")
    elif flow_class == "flow_insensitive":
        notes_parts.append(
            "Low extraction drug: CL is largely insensitive to hepatic blood flow changes."
        )

    return HepaticFlowEffectResult(
        times_h=times,
        c_baseline_mg_L=c_baseline,
        c_modified_mg_L=c_modified,
        cl_baseline_L_per_h=cl_baseline,
        cl_modified_L_per_h=cl_modified,
        e_baseline=e_baseline,
        e_modified=e_modified,
        auc_baseline_mg_h_per_L=auc_baseline,
        auc_modified_mg_h_per_L=auc_modified,
        auc_ratio=auc_ratio,
        flow_sensitivity_class=flow_class,
        clinical_significance=clin_sig,
        notes=" ".join(notes_parts),
    )
