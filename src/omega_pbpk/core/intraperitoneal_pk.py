"""Phase 524 — Intraperitoneal Drug Delivery PK.

Models pharmacokinetics of intraperitoneally administered drugs.
IP drug is absorbed via the peritoneal membrane into portal blood,
undergoing hepatic first-pass extraction before reaching systemic circulation.
A small direct fraction bypasses the portal route.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IntraperitonealPKResult:
    """Result of an intraperitoneal PK simulation."""

    drug_name: str
    dose_mg: float
    ka_ip_per_h: float
    e_hepatic: float

    # Time-series (list fields — plain dataclass)
    times_h: list
    c_systemic_mg_L: list

    # PK metrics
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float

    # Bioavailability and first-pass
    bioavailability_pct: float  # (f_portal*(1 - E_h) + f_direct) * 100
    first_pass_effect_pct: float  # E_h * f_portal * 100
    t_half_h: float

    notes: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_Q_PORTAL_L_PER_H = 72.0  # portal blood flow (L/h)
_K_PORTAL_CLEAR = 1.0  # portal blood turnover (h⁻¹)
_F_DIRECT = 0.1  # fraction absorbed directly to systemic
_F_PORTAL = 0.9  # fraction absorbed via portal blood


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_intraperitoneal_pk(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    ka_ip_per_h: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> IntraperitonealPKResult:
    """Simulate intraperitoneal drug delivery PK.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    dose_mg:
        IP dose in mg.
    cl_hepatic_L_per_h:
        Hepatic intrinsic clearance (L/h), used to compute extraction ratio.
    cl_sys_L_per_h:
        Systemic clearance (L/h), drives terminal elimination.
    vd_sys_L:
        Systemic volume of distribution (L).
    ka_ip_per_h:
        Peritoneal absorption rate constant (h⁻¹).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    IntraperitonealPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if ka_ip_per_h <= 0:
        raise ValueError("ka_ip_per_h must be positive")

    # --- hepatic extraction ratio ---
    e_h = cl_hepatic_L_per_h / (cl_hepatic_L_per_h + _Q_PORTAL_L_PER_H)
    e_h = min(e_h, 1.0)

    # --- rate constants ---
    k_el = cl_sys_L_per_h / vd_sys_L  # systemic elimination (h⁻¹)

    # --- initial conditions ---
    a_peritoneal = dose_mg
    a_portal = 0.0
    c_sys = 0.0

    # --- storage ---
    times_h: list = [0.0]
    c_sys_list: list = [c_sys]

    # --- Forward Euler integration ---
    t = 0.0
    n_steps = int(round(t_end_h / dt_h))
    for _ in range(n_steps):
        # ODEs
        d_a_peritoneal = -ka_ip_per_h * a_peritoneal
        # portal: receives portal fraction from peritoneal absorption; cleared by liver + portal turnover
        d_a_portal = ka_ip_per_h * _F_PORTAL * a_peritoneal - _K_PORTAL_CLEAR * a_portal
        # systemic: receives post-hepatic portal (1 - E_h) + direct fraction
        d_c_sys = (
            _K_PORTAL_CLEAR * a_portal * (1.0 - e_h) / vd_sys_L
            + ka_ip_per_h * _F_DIRECT * a_peritoneal / vd_sys_L
            - k_el * c_sys
        )

        a_peritoneal += d_a_peritoneal * dt_h
        a_peritoneal = max(0.0, a_peritoneal)
        a_portal += d_a_portal * dt_h
        a_portal = max(0.0, a_portal)
        c_sys += d_c_sys * dt_h
        c_sys = max(0.0, c_sys)

        t += dt_h
        times_h.append(round(t, 10))
        c_sys_list.append(c_sys)

    # --- derived metrics ---
    cmax_mg_L = max(c_sys_list)
    tmax_h = times_h[c_sys_list.index(cmax_mg_L)]
    auc_mg_h_per_L = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Bioavailability estimate
    bioavailability_pct = (_F_PORTAL * (1.0 - e_h) + _F_DIRECT) * 100.0
    first_pass_effect_pct = e_h * _F_PORTAL * 100.0

    # Half-life from terminal elimination
    t_half_h = 0.693 / k_el if k_el > 0 else float("inf")

    notes = (
        f"E_h={e_h:.3f}; F_portal={_F_PORTAL}; F_direct={_F_DIRECT}; "
        f"Bioavailability≈{bioavailability_pct:.1f}%"
    )

    return IntraperitonealPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_ip_per_h=ka_ip_per_h,
        e_hepatic=e_h,
        times_h=times_h,
        c_systemic_mg_L=c_sys_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        bioavailability_pct=bioavailability_pct,
        first_pass_effect_pct=first_pass_effect_pct,
        t_half_h=t_half_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_ka_ip(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    ka_values: list,
) -> list:
    """Simulate across a range of ka_ip values and rank by Cmax descending.

    Parameters
    ----------
    drug_name:
        Base compound name.
    dose_mg:
        IP dose in mg.
    cl_hepatic_L_per_h:
        Hepatic clearance (L/h).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    ka_values:
        List of peritoneal absorption rate constants to evaluate (h⁻¹).

    Returns
    -------
    list of IntraperitonealPKResult sorted by cmax_mg_L descending.
    """
    results = []
    for ka in ka_values:
        r = simulate_intraperitoneal_pk(
            drug_name=f"{drug_name}_ka{ka}",
            dose_mg=dose_mg,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
            ka_ip_per_h=ka,
        )
        results.append(r)

    results.sort(key=lambda r: r.cmax_mg_L, reverse=True)
    return results
