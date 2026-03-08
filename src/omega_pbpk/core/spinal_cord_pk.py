"""Phase 427: Drug distribution in spinal cord tissue.

3-compartment model: plasma -> CSF -> spinal cord tissue
Relevant for ALS, SMA, and neuropathic pain drugs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpinalCordPKResult:
    """Result of spinal cord PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_csf_mg_mL: list
    c_spinal_cord_mg_L: list
    c_systemic_mg_L: list
    cmax_csf_mg_mL: float
    cmax_spinal_cord_mg_L: float
    tmax_spinal_cord_h: float
    auc_csf_mg_h_per_mL: float
    auc_spinal_cord_mg_h_per_L: float
    csf_to_plasma_ratio: float
    spinal_cord_to_csf_ratio: float
    t_half_csf_h: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_spinal_cord_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    v_csf_L: float = 0.15,
    v_sc_L: float = 0.03,
    k_plasma_to_csf: float = 0.05,
    k_csf_to_plasma: float = 0.3,
    k_csf_to_sc: float = 0.1,
    k_sc_elim: float = 0.2,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SpinalCordPKResult:
    """Simulate drug distribution in spinal cord tissue.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    route : str
        Administration route: "intrathecal", "iv", or "oral".
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_plasma_L : float
        Plasma volume of distribution in L.
    v_csf_L : float
        CSF volume in L (default ~150 mL).
    v_sc_L : float
        Spinal cord volume in L (default ~30 mL).
    k_plasma_to_csf : float
        Rate constant plasma -> CSF (1/h).
    k_csf_to_plasma : float
        Rate constant CSF -> plasma (1/h).
    k_csf_to_sc : float
        Rate constant CSF -> spinal cord (1/h).
    k_sc_elim : float
        Spinal cord local elimination rate (1/h).
    ka_per_h : float
        Oral absorption rate constant (1/h).
    f_oral : float
        Oral bioavailability fraction.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        ODE integration step size in hours.

    Returns
    -------
    SpinalCordPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    valid_routes = {"intrathecal", "iv", "oral"}
    if route not in valid_routes:
        raise ValueError(f"route must be one of {valid_routes}, got '{route}'")

    # Derived elimination rate for plasma
    k_elim_plasma = cl_sys_L_per_h / vd_plasma_L

    # Initial conditions: amounts in mg
    if route == "iv":
        plasma_amt = dose_mg
        csf_amt = 0.0
        sc_amt = 0.0
        depot_amt = 0.0
    elif route == "intrathecal":
        plasma_amt = 0.0
        csf_amt = dose_mg
        sc_amt = 0.0
        depot_amt = 0.0
    else:  # oral
        plasma_amt = 0.0
        csf_amt = 0.0
        sc_amt = 0.0
        depot_amt = dose_mg

    # Forward Euler integration
    n_steps = max(int(t_end_h / dt_h), 1)

    times = [0.0]
    c_plasma_list = [plasma_amt / vd_plasma_L]  # mg/L
    c_csf_list = [csf_amt / v_csf_L / 1000.0]  # mg/mL
    c_sc_list = [sc_amt / v_sc_L]  # mg/L

    for step in range(n_steps):
        # Oral depot absorption (only for oral route)
        if route == "oral":
            depot_to_plasma = ka_per_h * depot_amt * f_oral
            depot_loss = ka_per_h * depot_amt
        else:
            depot_to_plasma = 0.0
            depot_loss = 0.0

        # Plasma ODEs
        plasma_to_csf = k_plasma_to_csf * plasma_amt
        csf_to_plasma_flux = k_csf_to_plasma * csf_amt
        plasma_elim = k_elim_plasma * plasma_amt

        d_plasma = depot_to_plasma + csf_to_plasma_flux - plasma_to_csf - plasma_elim

        # CSF ODEs
        csf_to_sc_flux = k_csf_to_sc * csf_amt
        d_csf = plasma_to_csf - csf_to_plasma_flux - csf_to_sc_flux

        # Spinal cord ODEs
        sc_elim = k_sc_elim * sc_amt
        d_sc = csf_to_sc_flux - sc_elim

        # Update amounts (Forward Euler)
        depot_amt = max(depot_amt - depot_loss * dt_h, 0.0)
        plasma_amt = max(plasma_amt + d_plasma * dt_h, 0.0)
        csf_amt = max(csf_amt + d_csf * dt_h, 0.0)
        sc_amt = max(sc_amt + d_sc * dt_h, 0.0)

        t = (step + 1) * dt_h
        times.append(t)
        c_plasma_list.append(plasma_amt / vd_plasma_L)
        c_csf_list.append(csf_amt / v_csf_L / 1000.0)
        c_sc_list.append(sc_amt / v_sc_L)

    # Compute PK metrics
    auc_csf = _trapezoidal_auc(times, c_csf_list)
    auc_sc = _trapezoidal_auc(times, c_sc_list)
    auc_plasma_mg_L = _trapezoidal_auc(times, c_plasma_list)
    auc_plasma_mg_mL = auc_plasma_mg_L / 1000.0

    cmax_csf = max(c_csf_list)
    cmax_sc = max(c_sc_list)

    tmax_sc_idx = c_sc_list.index(cmax_sc)
    tmax_sc = times[tmax_sc_idx]

    # Half-life in CSF
    total_csf_rate = k_csf_to_plasma + k_csf_to_sc
    if total_csf_rate > 0:
        t_half_csf = 0.693 / total_csf_rate
    else:
        t_half_csf = float("inf")

    # CSF-to-plasma ratio (both in mg/mL)
    csf_to_plasma = auc_csf / max(auc_plasma_mg_mL, 1e-12)

    # Spinal cord-to-CSF ratio (sc in mg/L, csf in mg/mL -> multiply csf by 1000 for mg/L)
    spinal_cord_to_csf = auc_sc / max(auc_csf * 1000.0, 1e-12)

    notes = (
        f"Route: {route}. CSF-to-plasma AUC ratio: {csf_to_plasma:.4f}. "
        f"Spinal cord-to-CSF AUC ratio: {spinal_cord_to_csf:.4f}. "
        f"t1/2 CSF: {t_half_csf:.2f} h."
    )

    return SpinalCordPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_csf_mg_mL=c_csf_list,
        c_spinal_cord_mg_L=c_sc_list,
        c_systemic_mg_L=c_plasma_list,
        cmax_csf_mg_mL=cmax_csf,
        cmax_spinal_cord_mg_L=cmax_sc,
        tmax_spinal_cord_h=tmax_sc,
        auc_csf_mg_h_per_mL=auc_csf,
        auc_spinal_cord_mg_h_per_L=auc_sc,
        csf_to_plasma_ratio=csf_to_plasma,
        spinal_cord_to_csf_ratio=spinal_cord_to_csf,
        t_half_csf_h=t_half_csf,
        notes=notes,
    )


def compare_spinal_routes(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
) -> list[SpinalCordPKResult]:
    """Compare all three administration routes for spinal cord drug delivery.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_plasma_L : float
        Plasma volume of distribution in L.

    Returns
    -------
    list of SpinalCordPKResult
        Results for all 3 routes sorted by auc_spinal_cord_mg_h_per_L descending.
    """
    routes = ["intrathecal", "iv", "oral"]
    results = []
    for route in routes:
        result = simulate_spinal_cord_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_plasma_L=vd_plasma_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_spinal_cord_mg_h_per_L, reverse=True)
    return results
