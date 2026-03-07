"""Cyclosporine PK model for transplant patients (Phase 979).

Narrow therapeutic index immunosuppressant with highly variable PK.
Monitored by C0 (trough) and C2 (2h post-dose).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CyclosporinePKResult:
    """Result of cyclosporine PK simulation."""

    drug_name: str = "Cyclosporine"
    dose_mg: float = 0.0
    patient_weight_kg: float = 0.0
    times_h: list = field(default_factory=list)
    c_whole_blood_ng_mL: list = field(default_factory=list)
    c0_trough_ng_mL: float = 0.0
    c2_ng_mL: float = 0.0
    cmax_ng_mL: float = 0.0
    tmax_h: float = 0.0
    auc_0_12h_ng_h_per_mL: float = 0.0
    within_therapeutic_range: bool = False
    dose_recommendation: str = "maintain"
    notes: str = ""


def simulate_cyclosporine_pk(
    dose_mg: float,
    patient_weight_kg: float = 70.0,
    hematocrit: float = 0.45,
    creatinine_umol_L: float = 100.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> CyclosporinePKResult:
    """Simulate cyclosporine PK in a transplant patient.

    Uses 1-compartment oral absorption with first-order kinetics.

    Parameters
    ----------
    dose_mg:
        Oral cyclosporine dose (mg).
    patient_weight_kg:
        Patient body weight (kg).
    hematocrit:
        Fractional hematocrit (0-1).
    creatinine_umol_L:
        Serum creatinine (µmol/L); elevated values indicate renal impairment.
    t_end_h:
        Simulation end time (h); 12 h for BID dosing.
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    CyclosporinePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if patient_weight_kg <= 0:
        raise ValueError(f"patient_weight_kg must be > 0, got {patient_weight_kg}")
    if not (0 < hematocrit < 1):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")

    # --- Population PK parameters ---
    vd_L = 5.0 * patient_weight_kg  # L
    cl_L_per_h = 0.3 * patient_weight_kg  # L/h
    ka = 1.0  # 1/h
    f_oral = 0.30  # oral bioavailability
    blood_plasma_ratio = 1.5 + 3.0 * hematocrit

    # Creatinine correction (renal impairment reduces CL slightly)
    if creatinine_umol_L > 200:
        cl_L_per_h *= 0.8

    ke = cl_L_per_h / vd_L  # 1/h

    # --- Forward Euler simulation ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list[float] = []
    c_plasma: list[float] = []

    a_gut = f_oral * dose_mg  # mg in gut
    c_p = 0.0  # mg/L plasma

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma.append(c_p)

        # derivatives
        da_gut = -ka * a_gut
        dc_p = ka * a_gut / vd_L - ke * c_p

        # update
        a_gut += da_gut * dt_h
        c_p += dc_p * dt_h

    # --- Convert plasma (mg/L) to whole blood (ng/mL) ---
    # 1 mg/L = 1000 ng/mL; multiply by blood_plasma_ratio
    c_blood = [cp * blood_plasma_ratio * 1000.0 for cp in c_plasma]

    # --- Derived PK metrics ---
    # Trough = end of dosing interval (steady-state approximation)
    c0_trough = c_blood[-1]
    c2_ng_mL = _interpolate_at_time(times_h, c_blood, 2.0)
    cmax = max(c_blood)
    tmax = times_h[c_blood.index(cmax)]

    # AUC 0-12h (trapezoidal)
    auc = _trapz(times_h, c_blood)

    # Therapeutic range: C0 trough 100-400 ng/mL (stable transplant)
    within_range = 100.0 <= c0_trough <= 400.0

    if c0_trough < 100.0:
        recommendation = "increase"
    elif c0_trough > 400.0:
        recommendation = "decrease"
    else:
        recommendation = "maintain"

    notes_parts = [
        f"Blood:plasma ratio = {blood_plasma_ratio:.2f}",
        f"ke = {ke:.4f} /h",
        f"Vd = {vd_L:.1f} L",
        f"CL = {cl_L_per_h:.2f} L/h",
    ]
    if creatinine_umol_L > 200:
        notes_parts.append("CL reduced 20% for elevated creatinine")

    return CyclosporinePKResult(
        drug_name="Cyclosporine",
        dose_mg=dose_mg,
        patient_weight_kg=patient_weight_kg,
        times_h=times_h,
        c_whole_blood_ng_mL=c_blood,
        c0_trough_ng_mL=c0_trough,
        c2_ng_mL=c2_ng_mL,
        cmax_ng_mL=cmax,
        tmax_h=tmax,
        auc_0_12h_ng_h_per_mL=auc,
        within_therapeutic_range=within_range,
        dose_recommendation=recommendation,
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i - 1] + values[i]) * dt
    return total


def _interpolate_at_time(times: list[float], values: list[float], target_h: float) -> float:
    """Linear interpolation to get value at target_h."""
    for i in range(1, len(times)):
        if times[i] >= target_h:
            t0, t1 = times[i - 1], times[i]
            v0, v1 = values[i - 1], values[i]
            frac = (target_h - t0) / (t1 - t0) if t1 != t0 else 0.0
            return v0 + frac * (v1 - v0)
    return values[-1]
