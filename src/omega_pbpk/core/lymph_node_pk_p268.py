"""Lymph node PK model — Phase 268.

2-compartment model: plasma + lymph node.
Relevant for immunotherapies and vaccines delivered by SC/IM/IV routes.

ODEs (Forward Euler):
  V_plasma = 3.0 L, V_lymph = 0.05 L
  SC/IM: depot absorption with ka=0.3/h
  IV: direct plasma loading, no depot

  dC_plasma/dt = (1-f_lymph)*ka*A_depot/V_plasma
                 - ke*C_plasma
                 - k_lymph_uptake*C_plasma
                 + k_lymph_return*C_lymph*V_lymph/V_plasma

  dC_lymph/dt  = k_lymph_uptake*C_plasma*V_plasma/V_lymph
                 + f_lymph*ka*A_depot/V_lymph
                 - k_lymph_return*C_lymph
                 - k_deg_lymph*C_lymph
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_ROUTES = ("sc_inguinal", "sc_axillary", "im", "iv")

# Route-specific uptake parameters
_ROUTE_PARAMS: dict = {
    "sc_inguinal": {"k_lymph_uptake": 0.3, "f_lymph": 0.15},
    "sc_axillary": {"k_lymph_uptake": 0.2, "f_lymph": 0.10},
    "im": {"k_lymph_uptake": 0.1, "f_lymph": 0.05},
    "iv": {"k_lymph_uptake": 0.001, "f_lymph": 0.001},
}

_V_PLASMA = 3.0  # L
_V_LYMPH = 0.05  # L
_KA = 0.3  # /h  absorption rate constant (depot for SC/IM)
_K_LYMPH_RETURN = 0.02  # /h  lymph → plasma return
_K_DEG_LYMPH = 0.05  # /h  degradation/retention in lymph node


@dataclass
class LymphNodePKResult:
    """Result of lymph node PK simulation — Phase 268."""

    drug_name: str
    dose_mg: float
    administration_route: str
    times_h: list
    c_plasma_mg_L: list
    c_lymph_mg_L: list
    cmax_plasma_mg_L: float
    cmax_lymph_mg_L: float
    tmax_lymph_h: float
    auc_plasma_mg_h_per_L: float
    auc_lymph_mg_h_per_L: float
    lymph_plasma_ratio: float
    lymph_targeting_efficiency_pct: float
    notes: str


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_lymph_node_pk(
    drug_name: str,
    dose_mg: float,
    administration_route: str,
    cl_L_per_h: float,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> LymphNodePKResult:
    """Simulate drug PK in plasma and lymph nodes after administration.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg. Must be > 0.
    administration_route : str
        Route: sc_inguinal, sc_axillary, im, or iv.
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    LymphNodePKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if administration_route not in _VALID_ROUTES:
        raise ValueError(
            f"administration_route must be one of {_VALID_ROUTES}, got '{administration_route}'"
        )

    params = _ROUTE_PARAMS[administration_route]
    k_lymph_uptake = params["k_lymph_uptake"]
    f_lymph = params["f_lymph"]

    ke = cl_L_per_h / _V_PLASMA

    # Initial conditions
    if administration_route == "iv":
        a_depot = 0.0
        c_plasma = dose_mg / _V_PLASMA
    else:
        a_depot = dose_mg
        c_plasma = 0.0

    c_lymph = 0.0

    times_h: list = []
    c_plasma_list: list = []
    c_lymph_list: list = []

    n_steps = max(1, int(round(t_end_h / dt_h)))
    t = 0.0

    for i in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_lymph_list.append(c_lymph)

        if i < n_steps:
            # ODEs
            da_depot = -_KA * a_depot

            dc_plasma = (
                (1.0 - f_lymph) * _KA * a_depot / _V_PLASMA
                - ke * c_plasma
                - k_lymph_uptake * c_plasma
                + _K_LYMPH_RETURN * c_lymph * _V_LYMPH / _V_PLASMA
            )

            dc_lymph = (
                k_lymph_uptake * c_plasma * _V_PLASMA / _V_LYMPH
                + f_lymph * _KA * a_depot / _V_LYMPH
                - _K_LYMPH_RETURN * c_lymph
                - _K_DEG_LYMPH * c_lymph
            )

            a_depot = max(0.0, a_depot + da_depot * dt_h)
            c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
            c_lymph = max(0.0, c_lymph + dc_lymph * dt_h)
            t += dt_h

    # Metrics
    cmax_plasma_mg_L = max(c_plasma_list)
    cmax_lymph_mg_L = max(c_lymph_list)
    tmax_lymph_idx = c_lymph_list.index(cmax_lymph_mg_L)
    tmax_lymph_h = times_h[tmax_lymph_idx]

    auc_plasma = _trapz(times_h, c_plasma_list)
    auc_lymph = _trapz(times_h, c_lymph_list)

    lymph_plasma_ratio = auc_lymph / auc_plasma if auc_plasma > 0 else 0.0

    # Lymph targeting efficiency: AUC_lymph*(V_lymph) / dose * 100
    lymph_exposure_mg_h = auc_lymph * _V_LYMPH
    lymph_targeting_efficiency_pct = (lymph_exposure_mg_h / dose_mg) * 100.0

    # Notes
    if lymph_targeting_efficiency_pct >= 5.0:
        notes = (
            f"{administration_route}: strong lymph node targeting "
            f"({lymph_targeting_efficiency_pct:.1f}% efficiency)."
        )
    elif lymph_targeting_efficiency_pct >= 1.0:
        notes = (
            f"{administration_route}: moderate lymph node exposure "
            f"({lymph_targeting_efficiency_pct:.1f}% efficiency)."
        )
    else:
        notes = (
            f"{administration_route}: minimal lymph node targeting "
            f"({lymph_targeting_efficiency_pct:.2f}% efficiency)."
        )

    return LymphNodePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        administration_route=administration_route,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_lymph_mg_L=c_lymph_list,
        cmax_plasma_mg_L=cmax_plasma_mg_L,
        cmax_lymph_mg_L=cmax_lymph_mg_L,
        tmax_lymph_h=tmax_lymph_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_lymph_mg_h_per_L=auc_lymph,
        lymph_plasma_ratio=lymph_plasma_ratio,
        lymph_targeting_efficiency_pct=lymph_targeting_efficiency_pct,
        notes=notes,
    )


def compare_admin_routes(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
) -> list:
    """Compare all 4 administration routes, sorted by lymph_targeting_efficiency_pct descending.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg.
    cl_L_per_h : float
        Systemic clearance (L/h).

    Returns
    -------
    list[LymphNodePKResult] sorted by lymph_targeting_efficiency_pct descending.
    """
    results = []
    for route in _VALID_ROUTES:
        r = simulate_lymph_node_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            administration_route=route,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(r)
    results.sort(key=lambda x: x.lymph_targeting_efficiency_pct, reverse=True)
    return results


__all__ = [
    "LymphNodePKResult",
    "compare_admin_routes",
    "simulate_lymph_node_pk",
]
