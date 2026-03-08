"""
Phase 940 — Perioperative PK model.

PK model for perioperative (surgical) settings — drug behavior before,
during, and after surgery. Time-varying CL and Vd based on surgical phase.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PerioperativePKResult", "simulate_perioperative_pk", "compare_hypothermia_levels"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PerioperativePKResult:
    """Results from a perioperative PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    auc_intraop_mg_h_per_L: float
    cl_intraop_L_per_h: float
    cl_fold_intraop: float
    hypothermia_cl_reduction_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list, conc: list) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (conc[i - 1] + conc[i]) * (times[i] - times[i - 1])
    return auc


def _get_phase_multipliers(
    t: float,
    t_incision_h: float,
    surgery_duration_h: float,
) -> tuple:
    """Return (cl_mult, vd_mult) for the current time point."""
    t_surgery_end = t_incision_h + surgery_duration_h
    t_recovery_start = t_surgery_end + 24.0

    if t < t_incision_h:
        # pre_op: normal
        return 1.0, 1.0
    elif t < t_surgery_end:
        # intra_op: reduced CL and Vd
        return 0.6, 0.8
    elif t < t_recovery_start:
        # post_op: partially reduced
        return 0.8, 0.9
    else:
        # recovery: normal
        return 1.0, 1.0


def _hypothermia_factor(temp_C: float) -> tuple:
    """Return (cl_factor, reduction_pct) for given temperature."""
    reduction_pct = max(0.0, (37.0 - temp_C) * 7.0)
    cl_factor = 1.0 - reduction_pct / 100.0
    # Clamp to [0.3, 1.0]
    cl_factor = max(0.3, min(1.0, cl_factor))
    return cl_factor, reduction_pct


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_perioperative_pk(
    drug_name: str,
    dose_mg: float,
    t_incision_h: float = 1.0,
    surgery_duration_h: float = 3.0,
    cl_normal_L_per_h: float = 10.0,
    vd_normal_L: float = 50.0,
    route: str = "iv_bolus",
    infusion_rate_mg_h: float = 0.0,
    blood_loss_fraction: float = 0.0,
    hypothermia_temp_C: float = 37.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> PerioperativePKResult:
    """Simulate perioperative PK with time-varying parameters."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be positive")
    if vd_normal_L <= 0:
        raise ValueError("vd_normal_L must be positive")
    if not (0.0 <= blood_loss_fraction <= 0.5):
        raise ValueError("blood_loss_fraction must be between 0 and 0.5")
    if route not in {"iv_bolus", "iv_infusion"}:
        raise ValueError("route must be 'iv_bolus' or 'iv_infusion'")

    hypo_cl_factor, hypo_reduction_pct = _hypothermia_factor(hypothermia_temp_C)

    # Blood loss time (occurs at midpoint of surgery)
    t_surgery_end = t_incision_h + surgery_duration_h
    blood_loss_time = t_incision_h + surgery_duration_h * 0.5
    blood_loss_applied = False

    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # Initial conditions
    c = 0.0
    if route == "iv_bolus":
        c = dose_mg / vd_normal_L

    concentrations = []
    cl_intraop_sum = 0.0
    cl_intraop_count = 0

    for step_i in range(n_steps):
        t = times[step_i]

        # Current phase multipliers
        phase_cl_mult, phase_vd_mult = _get_phase_multipliers(t, t_incision_h, surgery_duration_h)

        # Current effective parameters
        # Vd changes affect concentration when Vd expands/contracts —
        # here we use instantaneous ke with current Vd and CL
        vd_eff = vd_normal_L * phase_vd_mult
        cl_eff = cl_normal_L_per_h * phase_cl_mult * hypo_cl_factor
        # Hypothermia only applies during intra_op
        if t < t_incision_h or t >= t_surgery_end:
            cl_eff = cl_normal_L_per_h * phase_cl_mult

        ke_eff = cl_eff / vd_eff

        # Record current concentration
        concentrations.append(c)

        # Track intraop CL
        if t_incision_h <= t < t_surgery_end:
            cl_intraop_sum += cl_eff
            cl_intraop_count += 1

        # Apply blood loss dilution
        if blood_loss_fraction > 0 and not blood_loss_applied:
            if t >= blood_loss_time - dt_h * 0.5:
                c = c * (1.0 - blood_loss_fraction)
                blood_loss_applied = True

        # IV infusion input
        infusion_in = 0.0
        if route == "iv_infusion":
            infusion_in = infusion_rate_mg_h / vd_eff

        # Forward Euler update
        dc_dt = infusion_in - ke_eff * c
        c = max(0.0, c + dc_dt * dt_h)

    # Derived metrics
    cmax = max(concentrations) if concentrations else 0.0
    tmax = times[concentrations.index(cmax)] if concentrations else 0.0
    auc_total = _trapezoidal_auc(times, concentrations)

    # Intraop AUC
    intraop_times = []
    intraop_conc = []
    for i, t in enumerate(times):
        if t_incision_h <= t <= t_surgery_end:
            intraop_times.append(t)
            intraop_conc.append(concentrations[i])
    auc_intraop = _trapezoidal_auc(intraop_times, intraop_conc) if len(intraop_times) > 1 else 0.0

    # Average intraop CL
    _cl_intraop = (
        cl_intraop_sum / cl_intraop_count
        if cl_intraop_count > 0
        else cl_normal_L_per_h * 0.6 * hypo_cl_factor
    )
    # For the intraop phase, we apply both phase multiplier (0.6) and hypothermia
    cl_intraop_for_result = cl_normal_L_per_h * 0.6 * hypo_cl_factor
    cl_fold_intraop = cl_intraop_for_result / cl_normal_L_per_h

    notes_parts = [
        f"Drug: {drug_name}, dose: {dose_mg} mg ({route}).",
        f"Surgery at t={t_incision_h}h for {surgery_duration_h}h.",
        f"Intraop CL reduced to {cl_fold_intraop * 100:.0f}% of normal.",
    ]
    if hypothermia_temp_C < 37.0:
        notes_parts.append(
            f"Hypothermia at {hypothermia_temp_C}C: CL reduced by {hypo_reduction_pct:.1f}%."
        )
    if blood_loss_fraction > 0:
        notes_parts.append(
            f"Blood loss: {blood_loss_fraction * 100:.0f}% dilution at t={blood_loss_time:.1f}h."
        )
    notes = " ".join(notes_parts)

    return PerioperativePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=concentrations,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc_total,
        auc_intraop_mg_h_per_L=auc_intraop,
        cl_intraop_L_per_h=cl_intraop_for_result,
        cl_fold_intraop=cl_fold_intraop,
        hypothermia_cl_reduction_pct=hypo_reduction_pct,
        notes=notes,
    )


def compare_hypothermia_levels(
    drug_name: str,
    dose_mg: float,
    temps_C: list = None,
    t_incision_h: float = 1.0,
    surgery_duration_h: float = 3.0,
    cl_normal_L_per_h: float = 10.0,
    vd_normal_L: float = 50.0,
    route: str = "iv_bolus",
    infusion_rate_mg_h: float = 0.0,
    blood_loss_fraction: float = 0.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> list:
    """Compare PK across hypothermia levels, sorted by AUC descending."""
    if temps_C is None:
        temps_C = [37.0, 35.0, 33.0, 32.0]

    results = []
    for temp in temps_C:
        r = simulate_perioperative_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            t_incision_h=t_incision_h,
            surgery_duration_h=surgery_duration_h,
            cl_normal_L_per_h=cl_normal_L_per_h,
            vd_normal_L=vd_normal_L,
            route=route,
            infusion_rate_mg_h=infusion_rate_mg_h,
            blood_loss_fraction=blood_loss_fraction,
            hypothermia_temp_C=temp,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)

    # Sort by AUC descending (more hypothermia → higher AUC)
    results.sort(key=lambda x: x.auc_mg_h_per_L, reverse=True)
    return results
