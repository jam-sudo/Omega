"""Phase 329 — Intradermal Drug Delivery Model.

3-compartment Forward Euler ODE:
  dermis depot -> local tissue -> systemic circulation
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntradermalPKResult:
    """Result of an intradermal PK simulation."""

    times_h: list
    c_dermis_mg_mL: list
    c_local_mg_mL: list
    c_systemic_mg_L: list
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    bioavailability_pct: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_intradermal_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    k_abs_dermis_per_h: float,
    k_local_distribution_per_h: float,
    v_dermis_mL: float = 0.1,
    v_local_mL: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> IntradermalPKResult:
    """Simulate intradermal drug delivery PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Injected dose in mg.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L.
    k_abs_dermis_per_h:
        Absorption rate constant from dermis to systemic (1/h).
    k_local_distribution_per_h:
        Local tissue distribution rate constant from dermis (1/h).
    v_dermis_mL:
        Volume of dermis depot in mL (default 0.1 mL).
    v_local_mL:
        Volume of local tissue compartment in mL (default 2.0 mL).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    IntradermalPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if k_abs_dermis_per_h <= 0:
        raise ValueError(f"k_abs_dermis_per_h must be > 0, got {k_abs_dermis_per_h}")
    if k_local_distribution_per_h < 0:
        raise ValueError(
            f"k_local_distribution_per_h must be >= 0, got {k_local_distribution_per_h}"
        )
    if v_dermis_mL <= 0:
        raise ValueError(f"v_dermis_mL must be > 0, got {v_dermis_mL}")
    if v_local_mL <= 0:
        raise ValueError(f"v_local_mL must be > 0, got {v_local_mL}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # Derived parameter
    k_abs_local = k_abs_dermis_per_h * 0.3  # absorption from local tissue

    # Initial conditions
    # A_dermis in mg (amount), C_local in mg/mL, C_sys in mg/L
    a_dermis = dose_mg
    c_local = 0.0
    c_sys = 0.0

    times_h: list = []
    c_dermis_list: list = []
    c_local_list: list = []
    c_sys_list: list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        # Record state
        times_h.append(t)
        c_dermis_list.append(a_dermis / v_dermis_mL)  # mg/mL
        c_local_list.append(c_local)
        c_sys_list.append(c_sys)

        # ODEs (Forward Euler)
        # dermis: dA_dermis/dt = -(k_abs_dermis + k_local) * A_dermis
        da_dermis_dt = -(k_abs_dermis_per_h + k_local_distribution_per_h) * a_dermis

        # local: dC_local/dt = k_local * A_dermis / v_local - k_abs_local * C_local
        dc_local_dt = k_local_distribution_per_h * a_dermis / v_local_mL - k_abs_local * c_local

        # systemic: dC_sys/dt = (k_abs_dermis * A_dermis + k_abs_local * v_local * C_local)
        #                        / vd_sys - (cl / vd_sys) * C_sys
        # Note: vd_sys is in L; convert mg/mL * mL = mg -> mg/L after dividing by vd_sys (L)
        # k_abs_local * v_local_mL * c_local gives mg/h (rate into systemic from local)
        absorption_rate_mg_per_h = (
            k_abs_dermis_per_h * a_dermis + k_abs_local * v_local_mL * c_local
        )
        dc_sys_dt = absorption_rate_mg_per_h / vd_sys_L - (cl_sys_L_per_h / vd_sys_L) * c_sys

        # Update state
        a_dermis += da_dermis_dt * dt_h
        c_local += dc_local_dt * dt_h
        c_sys += dc_sys_dt * dt_h

        # Clamp to non-negative
        a_dermis = max(0.0, a_dermis)
        c_local = max(0.0, c_local)
        c_sys = max(0.0, c_sys)

        t += dt_h
        # Prevent floating-point overshoot
        if t > t_end_h + dt_h * 0.5:
            break

    # PK metrics
    auc_systemic = _trapezoidal_auc(times_h, c_sys_list)
    cmax_systemic = max(c_sys_list)
    tmax_systemic = times_h[c_sys_list.index(cmax_systemic)]

    # Bioavailability: compare to IV AUC = dose / CL (in mg/L * h)
    auc_iv = dose_mg / cl_sys_L_per_h
    bioavailability_pct = min(100.0, (auc_systemic / auc_iv) * 100.0) if auc_iv > 0 else 0.0

    notes = (
        f"Intradermal simulation for {drug_name}: "
        f"dose={dose_mg} mg, k_abs_dermis={k_abs_dermis_per_h}/h, "
        f"k_local={k_local_distribution_per_h}/h, "
        f"F={bioavailability_pct:.1f}%"
    )

    return IntradermalPKResult(
        times_h=times_h,
        c_dermis_mg_mL=c_dermis_list,
        c_local_mg_mL=c_local_list,
        c_systemic_mg_L=c_sys_list,
        auc_systemic_mg_h_per_L=auc_systemic,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        bioavailability_pct=bioavailability_pct,
        notes=notes,
    )
