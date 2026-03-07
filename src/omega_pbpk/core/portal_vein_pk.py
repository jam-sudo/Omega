"""Portal vein pharmacokinetics — first-pass metabolism model with portal vein concentration tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PortalVeinPKResult", "simulate_portal_vein_pk", "compare_first_pass_effect"]


@dataclass
class PortalVeinPKResult:
    """Result of portal vein PK simulation."""

    drug_name: str
    dose_mg: float
    fup: float
    er: float  # extraction ratio
    times_h: list = field(default_factory=list)
    c_portal_mg_L: list = field(default_factory=list)
    c_sys_mg_L: list = field(default_factory=list)
    cmax_portal_mg_L: float = 0.0
    tmax_portal_h: float = 0.0
    cmax_sys_mg_L: float = 0.0
    tmax_sys_h: float = 0.0
    auc_portal: float = 0.0
    auc_sys: float = 0.0
    portal_to_systemic_ratio: float = 0.0
    f_bioavailable: float = 0.0
    notes: str = ""


def _validate_portal_params(
    dose_mg: float,
    f_abs: float,
    fup: float,
    cl_int_L_per_h: float,
    qh_L_per_h: float,
    vd_portal_L: float,
    vd_sys_L: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < f_abs <= 1):
        raise ValueError(f"f_abs must be in (0, 1], got {f_abs}")
    if not (0 < fup <= 1):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if cl_int_L_per_h <= 0:
        raise ValueError(f"cl_int_L_per_h must be > 0, got {cl_int_L_per_h}")
    if qh_L_per_h <= 0:
        raise ValueError(f"qh_L_per_h must be > 0, got {qh_L_per_h}")
    if vd_portal_L <= 0:
        raise ValueError(f"vd_portal_L must be > 0, got {vd_portal_L}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_portal_vein_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    f_abs: float = 0.9,
    fup: float = 0.1,
    cl_int_L_per_h: float = 20.0,
    qh_L_per_h: float = 90.0,
    vd_portal_L: float = 1.0,
    vd_sys_L: float = 50.0,
    cl_sys_L_per_h: float = 2.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> PortalVeinPKResult:
    """Simulate portal vein PK with first-pass metabolism.

    Uses a 3-compartment Forward Euler ODE:
      GI lumen -> Portal vein -> Systemic circulation
    with well-stirred liver model for hepatic extraction.
    """
    _validate_portal_params(dose_mg, f_abs, fup, cl_int_L_per_h, qh_L_per_h, vd_portal_L, vd_sys_L)

    # Well-stirred liver model: CLh = Qh * fup * CLint / (Qh + fup * CLint)
    cl_h = qh_L_per_h * fup * cl_int_L_per_h / (qh_L_per_h + fup * cl_int_L_per_h)
    er = cl_h / qh_L_per_h  # extraction ratio

    # ke_portal: transit rate through portal vein = Qh / Vd_portal
    ke_portal = qh_L_per_h / vd_portal_L

    # Initial conditions
    a_gut = dose_mg * f_abs  # mg in GI
    c_portal = 0.0  # mg/L
    c_sys = 0.0  # mg/L

    times_h: list[float] = []
    c_portal_list: list[float] = []
    c_sys_list: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _step in range(n_steps):
        times_h.append(t)
        c_portal_list.append(c_portal)
        c_sys_list.append(c_sys)

        # Forward Euler derivatives
        # dA_gut/dt = -ka * A_gut
        da_gut = -ka_per_h * a_gut

        # Portal vein: drug enters from GI absorption, exits via hepatic transit
        # Input rate = ka * A_gut (mg/h), converted to concentration by Vd_portal
        # dC_portal/dt = ka * A_gut / Vd_portal - ke_portal * C_portal
        dc_portal = ka_per_h * a_gut / vd_portal_L - ke_portal * c_portal

        # Systemic: receives drug from liver after first-pass extraction
        # Drug reaching systemic = Qh * (1 - ER) * C_portal (hepatic output)
        # Converted to rate: Qh * (1 - ER) * C_portal / Vd_sys
        # dC_sys/dt = Qh * (1-ER) * C_portal / Vd_sys - CL_sys * C_sys / Vd_sys
        dc_sys = qh_L_per_h * (1.0 - er) * c_portal / vd_sys_L - cl_sys_L_per_h * c_sys / vd_sys_L

        # Update state
        a_gut += da_gut * dt_h
        c_portal += dc_portal * dt_h
        c_sys += dc_sys * dt_h

        # Clamp negatives
        if a_gut < 0.0:
            a_gut = 0.0
        if c_portal < 0.0:
            c_portal = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

        t += dt_h

    # Compute summary stats
    cmax_portal = max(c_portal_list)
    tmax_portal = times_h[c_portal_list.index(cmax_portal)]

    cmax_sys = max(c_sys_list)
    tmax_sys = times_h[c_sys_list.index(cmax_sys)]

    auc_portal = _trapezoidal_auc(times_h, c_portal_list)
    auc_sys = _trapezoidal_auc(times_h, c_sys_list)

    portal_to_sys_ratio = cmax_portal / cmax_sys if cmax_sys > 0 else float("inf")
    f_bioavailable = (1.0 - er) * f_abs

    notes = (
        f"Well-stirred liver: CLh={cl_h:.2f} L/h, ER={er:.3f}, "
        f"F={f_bioavailable:.3f}, Qh={qh_L_per_h} L/h, "
        f"CLint={cl_int_L_per_h} L/h"
    )

    return PortalVeinPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fup=fup,
        er=er,
        times_h=times_h,
        c_portal_mg_L=c_portal_list,
        c_sys_mg_L=c_sys_list,
        cmax_portal_mg_L=cmax_portal,
        tmax_portal_h=tmax_portal,
        cmax_sys_mg_L=cmax_sys,
        tmax_sys_h=tmax_sys,
        auc_portal=auc_portal,
        auc_sys=auc_sys,
        portal_to_systemic_ratio=portal_to_sys_ratio,
        f_bioavailable=f_bioavailable,
        notes=notes,
    )


def compare_first_pass_effect(
    drug_name: str,
    dose_mg: float,
    cl_int_scenarios: list[float],
) -> list[PortalVeinPKResult]:
    """Compare first-pass effect across different CLint values.

    Returns list of PortalVeinPKResult sorted by ER ascending.
    """
    results = []
    for cl_int in cl_int_scenarios:
        result = simulate_portal_vein_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_int_L_per_h=cl_int,
        )
        results.append(result)

    # Sort by extraction ratio ascending
    results.sort(key=lambda r: r.er)
    return results
