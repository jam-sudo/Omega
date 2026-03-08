"""
Phase 431 — Synovial Joint PK
Drug pharmacokinetics in synovial fluid for NSAIDs, corticosteroids, and biologics.
"""

from dataclasses import dataclass


@dataclass
class SynovialPKResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_synovial_mg_L: list
    cmax_plasma_mg_L: float
    cmax_synovial_mg_L: float
    tmax_synovial_h: float
    auc_plasma_mg_h_per_L: float
    auc_synovial_mg_h_per_L: float
    synovial_to_plasma_ratio: float
    t_half_synovial_h: float
    inflamed_ratio_boost: float
    notes: str


def _trapezoidal_auc(times, concs):
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_synovial_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    v_syn_L: float = 0.003,
    q_syn_L_per_h: float = 0.02,
    k_elim_syn_per_h: float = 0.1,
    inflamed_factor: float = 1.5,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SynovialPKResult:
    """
    Simulate drug PK in plasma and synovial fluid using a 2-compartment model.

    Plasma compartment: systemic 1-cpt with optional oral absorption.
    Synovial compartment: perfusion-limited transfer from plasma.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if route not in {"iv", "oral", "intraarticular"}:
        raise ValueError(f"route must be 'iv', 'oral', or 'intraarticular', got '{route}'")

    # Transfer rate constants
    # k_in (inflamed): perfusion from plasma into synovial fluid
    k_in_base = q_syn_L_per_h / v_syn_L  # base transfer rate (1/h)
    k_in = k_in_base * inflamed_factor  # inflammation-boosted k_in
    # k_out: efflux from synovial fluid
    k_out = q_syn_L_per_h / v_syn_L + k_elim_syn_per_h

    # Elimination rate constant for plasma
    k_elim_plasma = cl_sys_L_per_h / vd_plasma_L

    # t_half_synovial
    denom = q_syn_L_per_h / v_syn_L + k_elim_syn_per_h
    t_half_synovial_h = 0.693 / denom if denom > 0 else 0.0

    # Initialize state
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma = []
    c_synovial = []

    if route == "iv":
        # Bolus IV: C_plasma(0) = dose / Vd, C_syn(0) = 0
        cp = dose_mg / vd_plasma_L
        cs = 0.0
        depot = 0.0
    elif route == "oral":
        # Oral: depot starts at dose * F, absorption via ka
        cp = 0.0
        cs = 0.0
        depot = dose_mg * f_oral
    else:  # intraarticular
        # IA: C_syn(0) = dose / V_syn, C_plasma(0) = 0
        cp = 0.0
        cs = dose_mg / v_syn_L
        depot = 0.0

    times_h.append(0.0)
    c_plasma.append(cp)
    c_synovial.append(cs)

    t = 0.0
    for _ in range(n_steps - 1):
        if route == "oral":
            # Oral absorption from depot into plasma
            dcp_dt = ka_per_h * depot / vd_plasma_L - k_elim_plasma * cp
            ddepot_dt = -ka_per_h * depot
            depot_new = depot + ddepot_dt * dt_h
            depot = max(depot_new, 0.0)
        elif route == "iv":
            dcp_dt = -k_elim_plasma * cp
        else:  # intraarticular: systemic absorption from synovial
            # Drug from synovial goes into plasma via efflux
            dcp_dt = (k_out * cs * v_syn_L) / vd_plasma_L - k_elim_plasma * cp

        # Synovial ODE: dC_syn/dt = k_in * C_plasma - k_out * C_syn
        # For intraarticular: also subtract the portion absorbed systemically
        if route == "intraarticular":
            dcs_dt = k_in * cp - k_out * cs
        else:
            dcs_dt = k_in * cp - k_out * cs

        cp_new = cp + dcp_dt * dt_h
        cs_new = cs + dcs_dt * dt_h

        cp = max(cp_new, 0.0)
        cs = max(cs_new, 0.0)

        t += dt_h
        times_h.append(t)
        c_plasma.append(cp)
        c_synovial.append(cs)

    # Compute PK parameters
    cmax_plasma = max(c_plasma)
    cmax_synovial = max(c_synovial)

    # tmax_synovial
    tmax_synovial_h = times_h[c_synovial.index(cmax_synovial)]

    auc_plasma = _trapezoidal_auc(times_h, c_plasma)
    auc_synovial = _trapezoidal_auc(times_h, c_synovial)

    synovial_to_plasma_ratio = auc_synovial / max(auc_plasma, 1e-12)

    notes = (
        f"Route: {route}. Synovial penetration AUC ratio = {synovial_to_plasma_ratio:.3f}. "
        f"Inflammation factor {inflamed_factor:.1f}x applied to transfer rate. "
        f"Synovial t½ = {t_half_synovial_h:.2f} h."
    )

    return SynovialPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        c_synovial_mg_L=c_synovial,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_synovial_mg_L=cmax_synovial,
        tmax_synovial_h=tmax_synovial_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_synovial_mg_h_per_L=auc_synovial,
        synovial_to_plasma_ratio=synovial_to_plasma_ratio,
        t_half_synovial_h=t_half_synovial_h,
        inflamed_ratio_boost=inflamed_factor,
        notes=notes,
    )


def compare_routes_synovial(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
) -> list:
    """
    Compare all 3 routes (iv, oral, intraarticular), sorted by auc_synovial descending.
    """
    results = []
    for route in ("iv", "oral", "intraarticular"):
        result = simulate_synovial_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_plasma_L=vd_plasma_L,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_synovial_mg_h_per_L, reverse=True)
    return results
