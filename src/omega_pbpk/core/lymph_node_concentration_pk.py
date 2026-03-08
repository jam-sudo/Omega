"""
Phase 449: Lymph Node Concentration PK
Drug distribution to lymph nodes — important for HIV antiretrovirals,
immunosuppressants, and cancer immunotherapy.

Lymph nodes are pharmacological sanctuaries. Drug concentrations in lymph
nodes can be 2-10x plasma for lipophilic drugs due to lymphatic uptake.
"""

from dataclasses import dataclass


@dataclass
class LymphNodePKResult:
    """Results from lymph node PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_lymph_node_mg_L: list
    cmax_plasma_mg_L: float
    cmax_lymph_node_mg_L: float
    tmax_lymph_node_h: float
    auc_plasma_mg_h_per_L: float
    auc_lymph_node_mg_h_per_L: float
    lymph_to_plasma_ratio: float
    t_half_lymph_node_h: float
    lymph_node_kp: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_lymph_node_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    logp: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 50.0,
    vd_lymph_L: float = 0.15,
    q_lymph_L_per_h: float = 0.01,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> LymphNodePKResult:
    """
    Simulate drug distribution to lymph nodes using a 2-compartment model
    (plasma + lymph node).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams (must be > 0).
    route : str
        Administration route: "iv", "oral", or "subcutaneous".
    logp : float
        Octanol-water partition coefficient.
    cl_sys_L_per_h : float
        Systemic clearance in L/h (must be > 0).
    vd_plasma_L : float
        Volume of distribution of plasma compartment in L.
    vd_lymph_L : float
        Volume of lymphoid tissue compartment in L.
    q_lymph_L_per_h : float
        Lymph flow rate in L/h.
    ka_per_h : float
        Absorption rate constant for oral route in 1/h.
    f_oral : float
        Bioavailability for oral route (0-1).
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step in hours.

    Returns
    -------
    LymphNodePKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    valid_routes = {"iv", "oral", "subcutaneous"}
    if route not in valid_routes:
        raise ValueError(f"route must be one of {valid_routes}, got '{route}'")

    # Compute lymph node Kp from logP
    lymph_node_kp = max(1.0, min(20.0, 1.0 + logp * 0.8))

    # Rate constants
    k_elim = cl_sys_L_per_h / vd_plasma_L
    k_in = q_lymph_L_per_h / vd_plasma_L
    k_out = q_lymph_L_per_h / (lymph_node_kp * vd_lymph_L)
    t_half_lymph_node_h = 0.693 / k_out if k_out > 0 else float("inf")

    # Initial conditions
    # Plasma compartment: C_plasma (mg/L)
    # Lymph compartment: C_lymph (mg/L)
    # Depot (oral/SC): A_depot (mg)

    # Determine route-specific parameters
    if route == "iv":
        # Instantaneous IV bolus: C_plasma(0) = dose_mg / vd_plasma_L
        c_plasma = dose_mg / vd_plasma_L
        a_depot = 0.0
        effective_ka = 0.0
        _effective_f = 1.0
    elif route == "oral":
        c_plasma = 0.0
        a_depot = dose_mg * f_oral
        effective_ka = ka_per_h
        _effective_f = 1.0  # bioavailability already applied to depot
    else:  # subcutaneous
        # SC: slow absorption, full bioavailability
        c_plasma = 0.0
        a_depot = dose_mg * 1.0  # full bioavailability for SC
        effective_ka = 0.3  # slow SC absorption rate
        _effective_f = 1.0

    c_lymph = 0.0

    # Simulation via forward Euler
    times = []
    c_plasma_list = []
    c_lymph_list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    for _step in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_lymph_list.append(c_lymph)

        # Derivatives
        if route in {"oral", "subcutaneous"}:
            # Absorption from depot
            absorption_rate = effective_ka * a_depot
            dc_plasma_dt = (
                absorption_rate / vd_plasma_L
                + k_in * (c_lymph - c_plasma)  # lymph return, net balance
                - k_elim * c_plasma
                - k_in * c_plasma
                + k_out * c_lymph
            )
            # Simpler: plasma gains from depot, loses to elimination and lymph exchange
            # lymph exchange: net flux = k_in * C_plasma - k_out * C_lymph
            dc_plasma_dt = (
                absorption_rate / vd_plasma_L
                - k_elim * c_plasma
                - k_in * c_plasma
                + k_out * c_lymph
            )
            da_depot_dt = -effective_ka * a_depot
        else:  # IV
            dc_plasma_dt = -k_elim * c_plasma - k_in * c_plasma + k_out * c_lymph
            da_depot_dt = 0.0

        dc_lymph_dt = k_in * c_plasma - k_out * c_lymph

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)
        c_lymph = max(0.0, c_lymph + dc_lymph_dt * dt_h)
        if route in {"oral", "subcutaneous"}:
            a_depot = max(0.0, a_depot + da_depot_dt * dt_h)
        t += dt_h

    # Compute metrics
    cmax_plasma = max(c_plasma_list)
    cmax_lymph = max(c_lymph_list)

    # tmax for lymph node
    tmax_lymph_h = times[c_lymph_list.index(cmax_lymph)]

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_lymph = _trapezoidal_auc(times, c_lymph_list)

    lymph_to_plasma_ratio = auc_lymph / max(auc_plasma, 1e-12)

    # Notes
    kp_desc = "low" if lymph_node_kp < 3 else ("moderate" if lymph_node_kp < 8 else "high")
    penetration = (
        "poor"
        if lymph_to_plasma_ratio < 0.5
        else ("moderate" if lymph_to_plasma_ratio < 2.0 else "high")
    )
    notes = (
        f"Route: {route}. Lymph node Kp={lymph_node_kp:.2f} ({kp_desc}, logP={logp:.1f}). "
        f"Lymph penetration: {penetration} (AUC ratio={lymph_to_plasma_ratio:.2f}). "
        f"Lymph node t1/2={t_half_lymph_node_h:.1f} h."
    )

    return LymphNodePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_lymph_node_mg_L=c_lymph_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_lymph_node_mg_L=cmax_lymph,
        tmax_lymph_node_h=tmax_lymph_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_lymph_node_mg_h_per_L=auc_lymph,
        lymph_to_plasma_ratio=lymph_to_plasma_ratio,
        t_half_lymph_node_h=t_half_lymph_node_h,
        lymph_node_kp=lymph_node_kp,
        notes=notes,
    )


def compare_lymph_routes(
    drug_name: str,
    dose_mg: float,
    logp: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
) -> list[LymphNodePKResult]:
    """
    Compare lymph node PK for all three administration routes (IV, oral, SC),
    sorted by lymph node AUC in descending order.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    logp : float
    cl_sys_L_per_h : float
    vd_plasma_L : float

    Returns
    -------
    list of LymphNodePKResult sorted by auc_lymph_node_mg_h_per_L descending
    """
    results = []
    for route in ("iv", "oral", "subcutaneous"):
        result = simulate_lymph_node_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            logp=logp,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_plasma_L=vd_plasma_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_lymph_node_mg_h_per_L, reverse=True)
    return results
