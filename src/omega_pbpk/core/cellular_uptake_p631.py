"""
Phase 631 — Drug Cellular Uptake Kinetics
Models drug uptake into cells via passive diffusion and active transport,
accounting for intracellular binding.
"""

from dataclasses import dataclass


@dataclass
class CellularUptakeResult:
    drug_name: str
    cell_type: str
    dose_conc_mg_L: float
    times_h: list
    c_cytoplasm_mg_L: list
    c_bound_mg_L: list
    c_total_intracell_mg_L: list
    cellular_accumulation_ratio: list
    steady_state_accumulation: float
    t_half_uptake_h: float
    kp_cell: float
    active_transport_contribution: float
    notes: str


def simulate_cellular_uptake(
    drug_name: str,
    cell_type: str,
    c_extracell_mg_L: float,
    logP: float,
    mw_Da: float,
    vmax_mg_L_per_h: float = 0.0,
    km_mg_L: float = 1.0,
    k_binding: float = 0.5,
    bmax_mg_L: float = 5.0,
    t_end_h: float = 6.0,
    dt_h: float = 0.02,
) -> CellularUptakeResult:
    """Simulate cellular uptake kinetics via passive diffusion and active transport."""
    # Validation
    if c_extracell_mg_L <= 0:
        raise ValueError("c_extracell_mg_L must be > 0")
    if vmax_mg_L_per_h < 0:
        raise ValueError("vmax_mg_L_per_h must be >= 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if k_binding <= 0:
        raise ValueError("k_binding must be > 0")
    if bmax_mg_L <= 0:
        raise ValueError("bmax_mg_L must be > 0")

    # Passive diffusion rate: k_passive = 0.3 * max(0.01, logP) / max(1.0, mw_Da/300.0)
    # clamp [0.01, 2.0] /h
    k_passive = 0.3 * max(0.01, logP) / max(1.0, mw_Da / 300.0)
    k_passive = max(0.01, min(2.0, k_passive))

    # Efflux rate (slight asymmetry, allowing accumulation)
    k_efflux = k_passive * 0.8

    # Intracellular binding rate constants
    k_on = k_binding  # /h
    k_off = k_binding / 10.0  # /h (high affinity)

    # Initial conditions
    C_free = 0.0
    C_bound = 0.0

    # Tracking accumulators
    A_active = 0.0
    A_passive = 0.0

    # Output lists
    times_h = []
    c_cytoplasm = []
    c_bound = []
    c_total = []
    accum_ratio = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h

        c_tot = C_free + C_bound
        ratio = c_tot / c_extracell_mg_L if c_extracell_mg_L > 0 else 0.0

        times_h.append(t)
        c_cytoplasm.append(C_free)
        c_bound.append(C_bound)
        c_total.append(c_tot)
        accum_ratio.append(ratio)

        if step == n_steps:
            break

        # Active transport (Michaelis-Menten)
        v_active = 0.0
        if vmax_mg_L_per_h > 0:
            v_active = vmax_mg_L_per_h * c_extracell_mg_L / (km_mg_L + c_extracell_mg_L)

        # Influx via passive and active
        influx_passive = k_passive * c_extracell_mg_L
        influx_active = v_active
        efflux = k_efflux * C_free
        binding_on = k_on * C_free * (bmax_mg_L - C_bound)
        binding_off = k_off * C_bound

        # ODEs
        dC_free = influx_passive - efflux - binding_on + binding_off + influx_active
        dC_bound = binding_on - binding_off

        # Forward Euler
        C_free = C_free + dC_free * dt_h
        C_bound = C_bound + dC_bound * dt_h

        # Clamp
        C_free = max(0.0, C_free)
        C_bound = max(0.0, min(bmax_mg_L, C_bound))

        # Track cumulative uptake contributions
        A_passive += influx_passive * dt_h
        A_active += influx_active * dt_h

    # Steady-state accumulation ratio
    steady_state_accumulation = accum_ratio[-1]

    # kp_cell: max free drug / extracellular concentration
    kp_cell = max(c_cytoplasm) / c_extracell_mg_L if c_extracell_mg_L > 0 else 0.0

    # t_half_uptake: time to reach 50% of final intracellular concentration
    final_total = c_total[-1]
    half_target = 0.5 * final_total
    t_half_uptake_h = t_end_h  # default if never reached
    for i, ct in enumerate(c_total):
        if ct >= half_target:
            t_half_uptake_h = times_h[i]
            break

    # Active transport contribution
    total_uptake = A_active + A_passive
    active_transport_contribution = A_active / total_uptake if total_uptake > 0 else 0.0

    notes = (
        f"k_passive={k_passive:.4f}/h, k_efflux={k_efflux:.4f}/h, "
        f"k_on={k_on:.4f}/h, k_off={k_off:.4f}/h"
    )

    return CellularUptakeResult(
        drug_name=drug_name,
        cell_type=cell_type,
        dose_conc_mg_L=c_extracell_mg_L,
        times_h=times_h,
        c_cytoplasm_mg_L=c_cytoplasm,
        c_bound_mg_L=c_bound,
        c_total_intracell_mg_L=c_total,
        cellular_accumulation_ratio=accum_ratio,
        steady_state_accumulation=steady_state_accumulation,
        t_half_uptake_h=t_half_uptake_h,
        kp_cell=kp_cell,
        active_transport_contribution=active_transport_contribution,
        notes=notes,
    )
