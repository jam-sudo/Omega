"""Phase 461 — Uterine Blood Flow PK.

3-compartment PK model for drug distribution in the uterus during pregnancy,
accounting for gestational-week-dependent uterine blood flow changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UterinePKResult:
    """Result of uterine PK simulation."""

    drug_name: str
    dose_mg: float
    gestational_week: int
    times_h: list
    c_systemic_mg_L: list
    c_uterus_mg_L: list
    cmax_uterus_mg_L: float
    tmax_uterus_h: float
    auc_uterus_mg_h_per_L: float
    uterine_blood_flow_L_per_h: float
    uterine_to_systemic_ratio: float
    notes: str


def _compute_ubf_L_per_h(gestational_week: int) -> float:
    """Uterine blood flow in L/h.

    UBF (mL/min) = 50 + 6 * gestational_week
    Convert: * 60 / 1000 to get L/h
    """
    ubf_mL_per_min = 50.0 + 6.0 * gestational_week
    return ubf_mL_per_min * 60.0 / 1000.0


def _validate_inputs(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    Kp_uterus: float,
    vd_uterus_L: float,
    gestational_week: int,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if Kp_uterus <= 0:
        raise ValueError("Kp_uterus must be > 0")
    if vd_uterus_L <= 0:
        raise ValueError("vd_uterus_L must be > 0")
    if not (0 <= gestational_week <= 42):
        raise ValueError("gestational_week must be between 0 and 42")


def simulate_uterine_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    Kp_uterus: float = 1.5,
    vd_uterus_L: float = 0.5,
    gestational_week: int = 20,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> UterinePKResult:
    """Simulate uterine PK with gestational-week-dependent blood flow.

    IV bolus into maternal systemic compartment. Drug distributes into
    uterine tissue via uterine blood flow.

    ODE:
        dC_sys/dt  = -CL_sys/Vd_sys * C_sys
        dC_uterus/dt = UBF/Vd_uterus * (C_sys - C_uterus / Kp_uterus)
    """
    _validate_inputs(dose_mg, cl_sys_L_per_h, vd_sys_L, Kp_uterus, vd_uterus_L, gestational_week)

    ubf = _compute_ubf_L_per_h(gestational_week)

    # Initial conditions: IV bolus
    c_sys = dose_mg / vd_sys_L
    c_uterus = 0.0

    ke_sys = cl_sys_L_per_h / vd_sys_L  # elimination rate constant

    times_h: list[float] = []
    c_systemic_list: list[float] = []
    c_uterus_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _i in range(n_steps + 1):
        times_h.append(t)
        c_systemic_list.append(c_sys)
        c_uterus_list.append(c_uterus)

        if _i < n_steps:
            # Forward Euler
            dc_sys = -ke_sys * c_sys
            dc_uterus = (ubf / vd_uterus_L) * (c_sys - c_uterus / Kp_uterus)

            c_sys = c_sys + dc_sys * dt_h
            c_uterus = c_uterus + dc_uterus * dt_h

            # Clamp to avoid floating-point negatives
            if c_sys < 0.0:
                c_sys = 0.0
            if c_uterus < 0.0:
                c_uterus = 0.0

            t = t + dt_h

    # Compute derived metrics
    cmax_uterus = max(c_uterus_list)
    tmax_idx = c_uterus_list.index(cmax_uterus)
    tmax_uterus = times_h[tmax_idx]

    # Trapezoidal AUC for uterus
    auc_uterus = sum(
        0.5 * (c_uterus_list[i] + c_uterus_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Uterine-to-systemic ratio at end of simulation
    c_sys_end = c_systemic_list[-1]
    c_uterus_end = c_uterus_list[-1]
    if c_sys_end > 0.0:
        uterine_to_systemic = c_uterus_end / c_sys_end
    else:
        uterine_to_systemic = 0.0

    notes = (
        f"Gestational week {gestational_week}: UBF={ubf:.2f} L/h "
        f"({50 + 6 * gestational_week:.0f} mL/min). "
        f"Kp_uterus={Kp_uterus}."
    )

    return UterinePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        gestational_week=gestational_week,
        times_h=times_h,
        c_systemic_mg_L=c_systemic_list,
        c_uterus_mg_L=c_uterus_list,
        cmax_uterus_mg_L=cmax_uterus,
        tmax_uterus_h=tmax_uterus,
        auc_uterus_mg_h_per_L=auc_uterus,
        uterine_blood_flow_L_per_h=ubf,
        uterine_to_systemic_ratio=uterine_to_systemic,
        notes=notes,
    )


def compare_gestational_weeks(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    weeks: list,
) -> list:
    """Simulate uterine PK for multiple gestational weeks.

    Returns list of UterinePKResult sorted by uterine_blood_flow_L_per_h ascending.
    """
    results = [
        simulate_uterine_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
            gestational_week=w,
        )
        for w in weeks
    ]
    return sorted(results, key=lambda r: r.uterine_blood_flow_L_per_h)
