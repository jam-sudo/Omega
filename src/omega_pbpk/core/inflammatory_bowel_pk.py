"""
Phase 248: PK model for drugs targeting inflammatory bowel disease (IBD).

Simulates local colonic delivery with multiple formulation types using a
2-compartment ODE (colon absorption site + plasma), integrated via forward Euler.
"""

from dataclasses import dataclass

# Formulation lookup table
# Keys: f_colon, lag_time_h, ka_colon (/h), ka_plasma (/h)
_FORMULATION_PARAMS = {
    "standard_oral": {"f_colon": 0.05, "lag_time_h": 4.0, "ka_colon": 0.5, "ka_plasma": 0.8},
    "pH_triggered": {"f_colon": 0.60, "lag_time_h": 5.0, "ka_colon": 0.3, "ka_plasma": 0.2},
    "time_controlled": {"f_colon": 0.50, "lag_time_h": 6.0, "ka_colon": 0.4, "ka_plasma": 0.1},
    "enema": {"f_colon": 0.80, "lag_time_h": 0.5, "ka_colon": 1.0, "ka_plasma": 0.3},
    "suppository": {"f_colon": 0.70, "lag_time_h": 0.3, "ka_colon": 1.5, "ka_plasma": 0.2},
}

# Fixed physiological constants
_V_COLON_L = 1.0  # Volume of colon compartment (L)
_V_PLASMA_L = 3.0  # Volume of plasma compartment (L)
_K_COLON_OUT = 0.1  # First-order loss from colon (/h)
_K_COLON_ABS = 0.05  # Transfer from colon to plasma (/h)


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


@dataclass
class IBDPKResult:
    """PK simulation result for an IBD drug formulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list[float]
    c_colon_mg_L: list[float]
    c_plasma_mg_L: list[float]
    cmax_colon_mg_L: float
    tmax_colon_h: float
    cmax_plasma_mg_L: float
    auc_colon_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    local_systemic_ratio: float
    colon_targeting_efficiency_pct: float
    notes: str


def simulate_ibd_pk(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_L_per_h: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> IBDPKResult:
    """
    Simulate IBD drug PK for a given formulation using forward Euler ODEs.

    ODEs (2-compartment: depot -> colon + plasma):
        dA_depot/dt = -ka_total * A_depot   (after lag time)
        dC_colon/dt = f_colon * ka_colon * A_depot / V_colon - k_colon_out * C_colon
        dC_plasma/dt = (1-f_colon) * ka_plasma * A_depot / V_plasma
                       + k_colon_abs * C_colon * V_colon / V_plasma
                       - ke * C_plasma

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    formulation_type : str
        One of "standard_oral", "pH_triggered", "time_controlled", "enema", "suppository".
    cl_L_per_h : float
        Systemic clearance in L/h.
    t_end_h : float
        Simulation end time in hours (default 48.0).
    dt_h : float
        Forward Euler step size in hours (default 0.1).

    Returns
    -------
    IBDPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if formulation_type not in _FORMULATION_PARAMS:
        raise ValueError(
            f"formulation_type must be one of {list(_FORMULATION_PARAMS.keys())}, "
            f"got '{formulation_type}'"
        )

    params = _FORMULATION_PARAMS[formulation_type]
    f_colon = params["f_colon"]
    lag_time_h = params["lag_time_h"]
    ka_colon = params["ka_colon"]
    ka_plasma = params["ka_plasma"]
    ka_total = ka_colon + ka_plasma

    ke = cl_L_per_h / _V_PLASMA_L

    # State variables
    a_depot = dose_mg  # amount in depot (mg)
    c_colon = 0.0  # concentration in colon (mg/L)
    c_plasma = 0.0  # concentration in plasma (mg/L)

    # Output storage
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    c_colon_list: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_colon_list.append(c_colon)
        c_plasma_list.append(c_plasma)

        # Absorption only starts after lag time
        if t >= lag_time_h:
            da_depot = -ka_total * a_depot
            dc_colon = f_colon * ka_colon * a_depot / _V_COLON_L - _K_COLON_OUT * c_colon
            dc_plasma = (
                (1.0 - f_colon) * ka_plasma * a_depot / _V_PLASMA_L
                + _K_COLON_ABS * c_colon * _V_COLON_L / _V_PLASMA_L
                - ke * c_plasma
            )
        else:
            da_depot = 0.0
            dc_colon = 0.0
            dc_plasma = 0.0

        # Forward Euler update
        a_depot = max(0.0, a_depot + da_depot * dt_h)
        c_colon = max(0.0, c_colon + dc_colon * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

        t += dt_h

    # Compute summary statistics
    cmax_colon_mg_L = max(c_colon_list)
    tmax_colon_h = times_h[c_colon_list.index(cmax_colon_mg_L)]
    cmax_plasma_mg_L = max(c_plasma_list)

    auc_colon = _trapz(times_h, c_colon_list)
    auc_plasma = _trapz(times_h, c_plasma_list)

    if auc_plasma > 0.0:
        local_systemic_ratio = auc_colon / auc_plasma
    else:
        local_systemic_ratio = float("inf")

    # Colon targeting efficiency: fraction of total exposure delivered locally
    total_auc = auc_colon + auc_plasma
    if total_auc > 0.0:
        colon_targeting_efficiency_pct = 100.0 * auc_colon / total_auc
    else:
        colon_targeting_efficiency_pct = 0.0

    notes = (
        f"{formulation_type}: f_colon={f_colon:.2f}, lag={lag_time_h:.1f}h; "
        f"targeting_eff={colon_targeting_efficiency_pct:.1f}%"
    )

    return IBDPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times_h,
        c_colon_mg_L=c_colon_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_colon_mg_L=cmax_colon_mg_L,
        tmax_colon_h=tmax_colon_h,
        cmax_plasma_mg_L=cmax_plasma_mg_L,
        auc_colon_mg_h_per_L=auc_colon,
        auc_plasma_mg_h_per_L=auc_plasma,
        local_systemic_ratio=local_systemic_ratio,
        colon_targeting_efficiency_pct=colon_targeting_efficiency_pct,
        notes=notes,
    )


def compare_ibd_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
) -> list[IBDPKResult]:
    """
    Compare all 5 IBD formulation types for a given drug.

    Returns results sorted by colon_targeting_efficiency_pct descending.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    cl_L_per_h : float
        Systemic clearance in L/h.

    Returns
    -------
    list of IBDPKResult, sorted by colon_targeting_efficiency_pct descending.
    """
    results = []
    for formulation in _FORMULATION_PARAMS:
        result = simulate_ibd_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type=formulation,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.colon_targeting_efficiency_pct, reverse=True)
    return results
