"""Phase 322 — Prodrug Liver-Targeted Delivery.

Models liver-targeted prodrug delivery where activation occurs preferentially
in hepatocytes. Models selective hepatic conversion and systemic escape of
active drug.
"""

from __future__ import annotations

from dataclasses import dataclass

_LIVER_VOLUME_L = 10.0
_K_BILE_PER_H = 0.3
_F_ORAL = 0.8
_KA_ORAL_PER_H = 1.0


@dataclass
class LiverTargetedProdrugResult:
    """Result of liver-targeted prodrug PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    f_hepatic_activation: float
    times_h: list
    c_prodrug_mg_L: list
    c_hepatic_active_mg_L: list
    c_systemic_active_mg_L: list
    auc_hepatic_mg_h_per_L: float
    cmax_hepatic_mg_L: float
    auc_systemic_active_mg_h_per_L: float
    cmax_systemic_active_mg_L: float
    hepatic_to_systemic_ratio: float
    targeting_efficiency_pct: float
    notes: str


def _validate_prodrug_params(
    dose_mg: float,
    route: str,
    cl_prodrug_sys_L_per_h: float,
    vd_prodrug_L: float,
    f_hepatic_activation: float,
    k_hepatic_activation_per_h: float,
    cl_active_sys_L_per_h: float,
    vd_active_L: float,
    t_end_h: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if route not in {"oral", "iv_bolus"}:
        raise ValueError("route must be 'oral' or 'iv_bolus'")
    if cl_prodrug_sys_L_per_h <= 0:
        raise ValueError("cl_prodrug_sys_L_per_h must be > 0")
    if vd_prodrug_L <= 0:
        raise ValueError("vd_prodrug_L must be > 0")
    if not (0 <= f_hepatic_activation <= 1):
        raise ValueError("f_hepatic_activation must be in [0, 1]")
    if k_hepatic_activation_per_h <= 0:
        raise ValueError("k_hepatic_activation_per_h must be > 0")
    if cl_active_sys_L_per_h <= 0:
        raise ValueError("cl_active_sys_L_per_h must be > 0")
    if vd_active_L <= 0:
        raise ValueError("vd_active_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")


def simulate_liver_targeted_prodrug(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_prodrug_sys_L_per_h: float,
    vd_prodrug_L: float,
    f_hepatic_activation: float,
    k_hepatic_activation_per_h: float,
    cl_active_sys_L_per_h: float,
    vd_active_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> LiverTargetedProdrugResult:
    """Simulate liver-targeted prodrug PK.

    Parameters
    ----------
    drug_name:
        Name of the prodrug.
    dose_mg:
        Dose (mg).
    route:
        Route of administration: 'oral' or 'iv_bolus'.
    cl_prodrug_sys_L_per_h:
        Systemic clearance of prodrug (L/h).
    vd_prodrug_L:
        Volume of distribution of prodrug (L).
    f_hepatic_activation:
        Fraction of prodrug activated in the liver (0–1).
    k_hepatic_activation_per_h:
        Rate constant for hepatic activation (1/h).
    cl_active_sys_L_per_h:
        Systemic clearance of active drug (L/h).
    vd_active_L:
        Volume of distribution of active drug (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler time step (h).

    Returns
    -------
    LiverTargetedProdrugResult
    """
    _validate_prodrug_params(
        dose_mg,
        route,
        cl_prodrug_sys_L_per_h,
        vd_prodrug_L,
        f_hepatic_activation,
        k_hepatic_activation_per_h,
        cl_active_sys_L_per_h,
        vd_active_L,
        t_end_h,
    )

    k_elim_prod = cl_prodrug_sys_L_per_h / vd_prodrug_L
    k_elim_act = cl_active_sys_L_per_h / vd_active_L

    # Initial conditions
    if route == "iv_bolus":
        a_prod = dose_mg
        a_gut = 0.0
    else:
        # oral: prodrug absorbed via gut
        a_prod = 0.0
        a_gut = dose_mg

    a_hep = 0.0
    a_act_sys = 0.0

    n_steps = max(1, int(t_end_h / dt_h))
    times_h: list = [0.0]
    c_prod_list: list = [a_prod / vd_prodrug_L]
    c_hep_list: list = [a_hep / _LIVER_VOLUME_L]
    c_act_sys_list: list = [a_act_sys / vd_active_L]

    for _ in range(n_steps):
        # Oral absorption input
        if route == "oral":
            gut_input = _F_ORAL * _KA_ORAL_PER_H * a_gut
            d_gut = -_KA_ORAL_PER_H * a_gut
        else:
            gut_input = 0.0
            d_gut = 0.0

        # Prodrug systemic compartment
        # Total prodrug loss from systemic: elimination + all hepatic activation
        d_prod = gut_input - k_elim_prod * a_prod - k_hepatic_activation_per_h * a_prod

        # Hepatic active compartment: only hepatically-activated prodrug arrives
        d_hep = k_hepatic_activation_per_h * f_hepatic_activation * a_prod - _K_BILE_PER_H * a_hep

        # Systemic active compartment: only non-hepatic activation escapes to systemic
        d_act_sys = (
            k_hepatic_activation_per_h * (1.0 - f_hepatic_activation) * a_prod
            - k_elim_act * a_act_sys
        )

        if route == "oral":
            a_gut = max(0.0, a_gut + d_gut * dt_h)

        a_prod = max(0.0, a_prod + d_prod * dt_h)
        a_hep = max(0.0, a_hep + d_hep * dt_h)
        a_act_sys = max(0.0, a_act_sys + d_act_sys * dt_h)

        times_h.append(times_h[-1] + dt_h)
        c_prod_list.append(a_prod / vd_prodrug_L)
        c_hep_list.append(a_hep / _LIVER_VOLUME_L)
        c_act_sys_list.append(a_act_sys / vd_active_L)

    # AUC via trapezoidal rule
    def _auc(t: list, c: list) -> float:
        return sum(0.5 * (c[i] + c[i - 1]) * (t[i] - t[i - 1]) for i in range(1, len(t)))

    auc_hep = _auc(times_h, c_hep_list)
    auc_act_sys = _auc(times_h, c_act_sys_list)

    cmax_hep = max(c_hep_list)
    cmax_act_sys = max(c_act_sys_list)

    if auc_act_sys > 0:
        hepatic_to_systemic_ratio = auc_hep / auc_act_sys
    else:
        hepatic_to_systemic_ratio = float("inf")

    targeting_efficiency_pct = f_hepatic_activation * 100.0

    notes = (
        f"Route: {route}; "
        f"Hepatic targeting: {targeting_efficiency_pct:.1f}%; "
        f"AUC hepatic: {auc_hep:.4f} mg*h/L; "
        f"AUC systemic active: {auc_act_sys:.4f} mg*h/L; "
        f"Hepatic/systemic ratio: {hepatic_to_systemic_ratio:.2f}"
    )

    return LiverTargetedProdrugResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        f_hepatic_activation=f_hepatic_activation,
        times_h=times_h,
        c_prodrug_mg_L=c_prod_list,
        c_hepatic_active_mg_L=c_hep_list,
        c_systemic_active_mg_L=c_act_sys_list,
        auc_hepatic_mg_h_per_L=auc_hep,
        cmax_hepatic_mg_L=cmax_hep,
        auc_systemic_active_mg_h_per_L=auc_act_sys,
        cmax_systemic_active_mg_L=cmax_act_sys,
        hepatic_to_systemic_ratio=hepatic_to_systemic_ratio,
        targeting_efficiency_pct=targeting_efficiency_pct,
        notes=notes,
    )


def compare_targeting_efficiencies(
    drug_name: str,
    dose_mg: float,
    f_hepatic_list: list,
    k_activation_per_h: float,
    cl_prodrug_sys_L_per_h: float,
    vd_prodrug_L: float,
    cl_active_sys_L_per_h: float,
    vd_active_L: float,
    route: str = "iv_bolus",
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list:
    """Compare different hepatic targeting efficiencies.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose (mg).
    f_hepatic_list:
        List of f_hepatic_activation values to compare.
    k_activation_per_h:
        Hepatic activation rate constant (1/h).
    cl_prodrug_sys_L_per_h:
        Prodrug systemic clearance (L/h).
    vd_prodrug_L:
        Prodrug volume of distribution (L).
    cl_active_sys_L_per_h:
        Active drug systemic clearance (L/h).
    vd_active_L:
        Active drug volume of distribution (L).
    route:
        Route of administration; default 'iv_bolus'.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step (h).

    Returns
    -------
    list of dicts sorted by hepatic_to_systemic_ratio descending, each with:
    f_hepatic_activation, hepatic_to_systemic_ratio, auc_hepatic_mg_h_per_L,
    auc_systemic_active_mg_h_per_L, targeting_efficiency_pct, result
    """
    results = []
    for f_hep in f_hepatic_list:
        res = simulate_liver_targeted_prodrug(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_prodrug_sys_L_per_h=cl_prodrug_sys_L_per_h,
            vd_prodrug_L=vd_prodrug_L,
            f_hepatic_activation=f_hep,
            k_hepatic_activation_per_h=k_activation_per_h,
            cl_active_sys_L_per_h=cl_active_sys_L_per_h,
            vd_active_L=vd_active_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(
            {
                "f_hepatic_activation": f_hep,
                "hepatic_to_systemic_ratio": res.hepatic_to_systemic_ratio,
                "auc_hepatic_mg_h_per_L": res.auc_hepatic_mg_h_per_L,
                "auc_systemic_active_mg_h_per_L": res.auc_systemic_active_mg_h_per_L,
                "targeting_efficiency_pct": res.targeting_efficiency_pct,
                "result": res,
            }
        )

    # Sort descending by hepatic_to_systemic_ratio; treat inf as very large
    results.sort(
        key=lambda x: (
            x["hepatic_to_systemic_ratio"]
            if x["hepatic_to_systemic_ratio"] != float("inf")
            else 1e18
        ),
        reverse=True,
    )
    return results
