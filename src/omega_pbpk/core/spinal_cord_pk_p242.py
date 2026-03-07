"""Phase 242: PK model for drug distribution in spinal cord tissue."""

from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


_VALID_ROUTES = {"iv_bolus", "intrathecal"}

# Default volume and rate constants
_V_CSF_L = 0.15
_V_SPINAL_L = 0.05
_K_CSF = 0.05  # plasma -> CSF transfer (1/h)
_K_CSF_OUT = 0.02  # CSF elimination (1/h)
_K_SPINAL = 0.1  # CSF -> spinal transfer (1/h)
_K_SPINAL_OUT = 0.05  # spinal elimination (1/h)


@dataclass
class SpinalCordPKResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_csf_mg_L: list
    c_spinal_mg_L: list
    cmax_plasma_mg_L: float
    cmax_csf_mg_L: float
    cmax_spinal_mg_L: float
    auc_plasma_mg_h_per_L: float
    auc_csf_mg_h_per_L: float
    auc_spinal_mg_h_per_L: float
    penetration_ratio_csf: float
    penetration_ratio_spinal: float
    notes: str


def simulate_spinal_cord_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_h: float,
    vp_L: float = 3.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SpinalCordPKResult:
    """Simulate spinal cord PK using a 3-compartment forward Euler model.

    Compartments: plasma, CSF, spinal cord.

    Args:
        drug_name: Drug name.
        dose_mg: Dose in mg.
        route: "iv_bolus" or "intrathecal".
        cl_L_per_h: Plasma clearance (L/h).
        vp_L: Plasma volume (L). Default 3.0.
        t_end_h: Simulation end time (h). Default 24.0.
        dt_h: Time step (h). Default 0.05.

    Returns:
        SpinalCordPKResult with full PK profiles and summary metrics.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")

    ke = cl_L_per_h / vp_L
    v_csf = _V_CSF_L
    v_spinal = _V_SPINAL_L

    # Initial conditions
    if route == "iv_bolus":
        c_plasma = dose_mg / vp_L
        c_csf = 0.0
        c_spinal = 0.0
    else:  # intrathecal
        c_plasma = 0.0
        c_csf = dose_mg / v_csf
        c_spinal = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times = []
    cp_list = []
    ccsf_list = []
    cspinal_list = []

    t = 0.0
    for _ in range(n_steps):
        times.append(t)
        cp_list.append(c_plasma)
        ccsf_list.append(c_csf)
        cspinal_list.append(c_spinal)

        # ODEs
        dcp = -ke * c_plasma - _K_CSF * c_plasma
        dccsf = _K_CSF * c_plasma * vp_L / v_csf - _K_CSF_OUT * c_csf - _K_SPINAL * c_csf
        dcspinal = _K_SPINAL * c_csf * v_csf / v_spinal - _K_SPINAL_OUT * c_spinal

        # Forward Euler update (clamp negatives to zero)
        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_csf = max(0.0, c_csf + dccsf * dt_h)
        c_spinal = max(0.0, c_spinal + dcspinal * dt_h)

        t += dt_h

    # Summary metrics
    cmax_plasma = max(cp_list)
    cmax_csf = max(ccsf_list)
    cmax_spinal = max(cspinal_list)

    auc_plasma = _trapz(times, cp_list)
    auc_csf = _trapz(times, ccsf_list)
    auc_spinal = _trapz(times, cspinal_list)

    if auc_plasma > 0:
        pen_csf = auc_csf / auc_plasma
        pen_spinal = auc_spinal / auc_plasma
    else:
        pen_csf = 0.0
        pen_spinal = 0.0

    notes = (
        f"Route: {route}; ke={ke:.4f}/h; k_csf={_K_CSF}/h; "
        f"k_csf_out={_K_CSF_OUT}/h; k_spinal={_K_SPINAL}/h; "
        f"k_spinal_out={_K_SPINAL_OUT}/h"
    )

    return SpinalCordPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_csf_mg_L=ccsf_list,
        c_spinal_mg_L=cspinal_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_csf_mg_L=cmax_csf,
        cmax_spinal_mg_L=cmax_spinal,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_csf_mg_h_per_L=auc_csf,
        auc_spinal_mg_h_per_L=auc_spinal,
        penetration_ratio_csf=pen_csf,
        penetration_ratio_spinal=pen_spinal,
        notes=notes,
    )


def compare_spinal_routes(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
) -> list:
    """Compare iv_bolus and intrathecal routes, sorted by auc_spinal descending.

    Returns:
        List of SpinalCordPKResult sorted by auc_spinal_mg_h_per_L descending.
    """
    results = [
        simulate_spinal_cord_pk(drug_name, dose_mg, route, cl_L_per_h)
        for route in ("iv_bolus", "intrathecal")
    ]
    results.sort(key=lambda r: r.auc_spinal_mg_h_per_L, reverse=True)
    return results
