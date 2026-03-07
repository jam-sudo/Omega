"""Ocular drug delivery extended model for subconjunctival and suprachoroidal injection routes.

Phase 839: Models drug distribution from injection depot through choroid, vitreous, retina,
and systemic circulation for three routes of administration.

Routes modeled:
  - subconjunctival: depot under conjunctiva, slow transfer to choroid, some systemic loss
  - suprachoroidal: depot in suprachoroidal space, rapid transfer to choroid, minimal systemic
  - intravitreal: direct injection into vitreous, reference route

References
----------
- Kang-Mieler JJ et al., Expert Rev Ophthalmol. 2020;15:23-37 (suprachoroidal delivery)
- Edelhauser HF et al., Invest Ophthalmol Vis Sci. 2010;51:5403-5420 (ocular pharmacokinetics)
- Olsen TW et al., Invest Ophthalmol Vis Sci. 2011;52:338-344 (subconjunctival injection)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "OcularDeliveryResult",
    "simulate_ocular_delivery",
    "compare_ocular_routes",
]

# ---------------------------------------------------------------------------
# Compartment volumes (mL)
# ---------------------------------------------------------------------------
_V_CHOROID_ML = 0.5
_V_VITREOUS_ML = 4.0
_V_RETINA_ML = 0.1

# Transfer rate constants (1/h)
_K_CHOROID_TO_VITREOUS = 0.2
_K_CHOROID_TO_RETINA = 0.1
_KE_VITREOUS = 0.05  # slow vitreous clearance
_KE_RETINA = 0.1

# Route parameters: (V_depot_mL, ka_to_choroid, ka_to_systemic)
_ROUTE_PARAMS: dict[str, tuple[float, float, float]] = {
    "subconjunctival": (0.05, 0.3, 0.5),
    "suprachoroidal": (0.10, 2.0, 0.1),
    "intravitreal": (0.05, 0.1, 0.05),
}


@dataclass
class OcularDeliveryResult:
    """Results from an ocular drug delivery simulation."""

    drug_name: str
    route: str
    dose_mg: float
    # Time-course arrays
    times_h: list
    c_injection_site_mg_L: list
    c_choroid_mg_L: list
    c_vitreous_mg_L: list
    c_retina_mg_L: list
    c_systemic_mg_L: list
    # Summary metrics
    cmax_vitreous_mg_L: float
    tmax_vitreous_h: float
    auc_vitreous_mg_h_per_L: float
    cmax_retina_mg_L: float
    auc_retina_mg_h_per_L: float
    cmax_systemic_mg_L: float
    ocular_bioavailability: float
    notes: str


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def simulate_ocular_delivery(
    drug_name: str,
    dose_mg: float,
    route: str = "subconjunctival",
    cl_sys_L_per_h: float = 0.5,
    vd_sys_L: float = 5.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> OcularDeliveryResult:
    """Simulate ocular drug delivery via subconjunctival, suprachoroidal, or intravitreal route.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Administered dose in milligrams.
    route:
        Injection route: "subconjunctival", "suprachoroidal", or "intravitreal".
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L.
    t_end_h:
        Simulation end time in hours (default 168 h = 7 days).
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    OcularDeliveryResult
        Time-course concentrations and summary PK metrics.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be positive, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be positive, got {vd_sys_L}")
    if route not in _ROUTE_PARAMS:
        raise ValueError(f"Invalid route '{route}'. Choose from: {list(_ROUTE_PARAMS.keys())}")

    v_depot_ml, ka_to_choroid, ka_to_systemic = _ROUTE_PARAMS[route]

    # Derived rate constant
    ke_sys = cl_sys_L_per_h / vd_sys_L  # 1/h
    vd_sys_ml = vd_sys_L * 1000.0  # mL

    # ---------------------------------------------------------------------------
    # Initial conditions
    # All concentrations in mg/mL; amounts in mg
    # ---------------------------------------------------------------------------
    if route == "intravitreal":
        # Drug starts directly in vitreous
        depot_mg = 0.0
        c_choroid = 0.0  # mg/mL
        c_vitreous = dose_mg / _V_VITREOUS_ML  # mg/mL
        c_retina = 0.0  # mg/mL
        c_sys = 0.0  # mg/L (different unit for systemic)
    else:
        depot_mg = dose_mg
        c_choroid = 0.0
        c_vitreous = 0.0
        c_retina = 0.0
        c_sys = 0.0

    # Output lists (store concentration at injection site in mg/L for consistent units)
    times: list[float] = []
    c_inj_list: list[float] = []  # injection site: depot_mg / v_depot_ml * 1000 [mg/L]
    c_cho_list: list[float] = []  # choroid mg/L
    c_vit_list: list[float] = []  # vitreous mg/L
    c_ret_list: list[float] = []  # retina mg/L
    c_sys_list: list[float] = []  # systemic mg/L

    n_steps = int(math.ceil(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps + 1):
        # Record current state (convert mg/mL → mg/L for output)
        times.append(t)
        c_inj_list.append(depot_mg / v_depot_ml * 1000.0)  # mg/mL * 1000 = mg/L
        c_cho_list.append(c_choroid * 1000.0)  # mg/mL → mg/L
        c_vit_list.append(c_vitreous * 1000.0)  # mg/mL → mg/L
        c_ret_list.append(c_retina * 1000.0)  # mg/mL → mg/L
        c_sys_list.append(c_sys)  # already mg/L

        if t >= t_end_h:
            break

        # ---------------------------------------------------------------------------
        # ODEs (Forward Euler)
        # ---------------------------------------------------------------------------
        # 1. Depot (amount in mg)
        d_depot = -(ka_to_choroid + ka_to_systemic) * depot_mg

        # 2. Choroid (mg/mL): input from depot, output to vitreous and retina
        d_choroid = (
            ka_to_choroid * depot_mg / _V_CHOROID_ML
            - _K_CHOROID_TO_VITREOUS * c_choroid
            - _K_CHOROID_TO_RETINA * c_choroid
        )

        # 3. Vitreous (mg/mL): input from choroid (scaled by volume ratio)
        d_vitreous = (
            _K_CHOROID_TO_VITREOUS * c_choroid * _V_CHOROID_ML / _V_VITREOUS_ML
            - _KE_VITREOUS * c_vitreous
        )

        # 4. Retina (mg/mL): input from choroid
        d_retina = (
            _K_CHOROID_TO_RETINA * c_choroid * _V_CHOROID_ML / _V_RETINA_ML - _KE_RETINA * c_retina
        )

        # 5. Systemic (mg/L): input from depot leak, clearance
        #    ka_to_systemic * depot_mg [mg/h] → divide by vd_sys_ml [mL] then * 1000 for L
        d_sys = ka_to_systemic * depot_mg / vd_sys_ml * 1000.0 - ke_sys * c_sys

        # Update state
        depot_mg += d_depot * dt_h
        c_choroid += d_choroid * dt_h
        c_vitreous += d_vitreous * dt_h
        c_retina += d_retina * dt_h
        c_sys += d_sys * dt_h

        # Clamp negatives
        if depot_mg < 0.0:
            depot_mg = 0.0
        if c_choroid < 0.0:
            c_choroid = 0.0
        if c_vitreous < 0.0:
            c_vitreous = 0.0
        if c_retina < 0.0:
            c_retina = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

        t = min(t + dt_h, t_end_h)

    # ---------------------------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------------------------
    cmax_vitreous = max(c_vit_list)
    tmax_vitreous = times[c_vit_list.index(cmax_vitreous)]
    auc_vitreous = _trapezoidal_auc(times, c_vit_list)

    cmax_retina = max(c_ret_list)
    auc_retina = _trapezoidal_auc(times, c_ret_list)

    cmax_systemic = max(c_sys_list)

    # Ocular bioavailability: fraction of dose reaching vitreous
    # Estimated as drug mass in vitreous (AUC * ke_vitreous * V_vit) relative to dose
    # total drug mass cleared from vitreous ≈ AUC_vit(mg/L) * ke_vit * V_vit(L)
    v_vit_L = _V_VITREOUS_ML / 1000.0
    drug_reached_vitreous_mg = auc_vitreous * _KE_VITREOUS * v_vit_L
    ocular_bioavailability = min(drug_reached_vitreous_mg / dose_mg, 1.0)

    notes = (
        f"Route: {route}; ka_choroid={ka_to_choroid}/h; ka_sys={ka_to_systemic}/h; "
        f"ke_vit={_KE_VITREOUS}/h; simulation t_end={t_end_h}h dt={dt_h}h"
    )

    return OcularDeliveryResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=times,
        c_injection_site_mg_L=c_inj_list,
        c_choroid_mg_L=c_cho_list,
        c_vitreous_mg_L=c_vit_list,
        c_retina_mg_L=c_ret_list,
        c_systemic_mg_L=c_sys_list,
        cmax_vitreous_mg_L=cmax_vitreous,
        tmax_vitreous_h=tmax_vitreous,
        auc_vitreous_mg_h_per_L=auc_vitreous,
        cmax_retina_mg_L=cmax_retina,
        auc_retina_mg_h_per_L=auc_retina,
        cmax_systemic_mg_L=cmax_systemic,
        ocular_bioavailability=ocular_bioavailability,
        notes=notes,
    )


def compare_ocular_routes(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float = 0.5,
    vd_sys_L: float = 5.0,
) -> list[OcularDeliveryResult]:
    """Simulate all three ocular routes and return sorted by vitreous AUC (descending).

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Administered dose in milligrams.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L.

    Returns
    -------
    list[OcularDeliveryResult]
        Three results sorted by auc_vitreous_mg_h_per_L descending.
    """
    results = [
        simulate_ocular_delivery(drug_name, dose_mg, route, cl_sys_L_per_h, vd_sys_L)
        for route in ("subconjunctival", "suprachoroidal", "intravitreal")
    ]
    results.sort(key=lambda r: r.auc_vitreous_mg_h_per_L, reverse=True)
    return results
