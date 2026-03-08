"""Phase 264 -- Bone Marrow PK Model.

PK model for drug distribution in bone marrow, relevant for chemotherapy.
Two-compartment model: plasma + bone marrow.

Pure Python, no external dependencies.  Forward Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_V_PLASMA = 3.0  # L
_V_MARROW = 1.5  # L
_K_PM = 0.08  # /h  plasma -> marrow
_K_MP = 0.04  # /h  marrow -> plasma
_K_DEG = 0.01  # /h  metabolic degradation in marrow
_KA = 1.0  # /h  oral absorption rate constant
_F = 0.8  # oral bioavailability fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _myelosuppression_risk(ratio: float) -> str:
    if ratio > 0.3:
        return "high"
    if ratio >= 0.1:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BoneMarrowPKResult:
    """Result of bone marrow PK simulation (Phase 264)."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_marrow_mg_L: list
    cmax_plasma_mg_L: float
    cmax_marrow_mg_L: float
    tmax_marrow_h: float
    auc_plasma_mg_h_per_L: float
    auc_marrow_mg_h_per_L: float
    marrow_plasma_ratio: float
    myelosuppression_risk: str
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_bone_marrow_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_h: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> BoneMarrowPKResult:
    """Simulate drug distribution between plasma and bone marrow.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg (must be > 0).
    route:
        Either "iv_bolus" or "oral".
    cl_L_per_h:
        Systemic clearance in L/h (must be > 0).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    BoneMarrowPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if route not in {"iv_bolus", "oral"}:
        raise ValueError("route must be 'iv_bolus' or 'oral'")

    ke = cl_L_per_h / _V_PLASMA

    # Initial conditions
    if route == "iv_bolus":
        c_plasma = dose_mg / _V_PLASMA  # mg/L
        a_depot = 0.0
    else:
        c_plasma = 0.0
        a_depot = dose_mg  # mg in depot

    c_marrow = 0.0  # mg/L

    times = [0.0]
    c_plasma_list = [c_plasma]
    c_marrow_list = [c_marrow]

    n_steps = int(round(t_end_h / dt_h))
    for _ in range(n_steps):
        # Oral input term (amount absorbed per unit time, converted to conc/time)
        if route == "oral":
            oral_input = _F * _KA * a_depot / _V_PLASMA  # mg/L/h
            da_depot = -_KA * a_depot
        else:
            oral_input = 0.0
            da_depot = 0.0

        # ODEs
        dc_plasma = (
            -ke * c_plasma
            - _K_PM * c_plasma
            + _K_MP * c_marrow * _V_MARROW / _V_PLASMA
            + oral_input
        )
        dc_marrow = _K_PM * c_plasma * _V_PLASMA / _V_MARROW - _K_MP * c_marrow - _K_DEG * c_marrow

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_marrow = max(0.0, c_marrow + dc_marrow * dt_h)
        if route == "oral":
            a_depot = max(0.0, a_depot + da_depot * dt_h)

        t = times[-1] + dt_h
        times.append(round(t, 10))
        c_plasma_list.append(c_plasma)
        c_marrow_list.append(c_marrow)

    # PK metrics
    cmax_plasma = max(c_plasma_list)
    cmax_marrow = max(c_marrow_list)

    tmax_marrow_idx = c_marrow_list.index(cmax_marrow)
    tmax_marrow_h = times[tmax_marrow_idx]

    auc_plasma = _trapz(times, c_plasma_list)
    auc_marrow = _trapz(times, c_marrow_list)

    ratio = auc_marrow / auc_plasma if auc_plasma > 0.0 else 0.0
    risk = _myelosuppression_risk(ratio)

    if risk == "high":
        notes = "High bone marrow exposure; significant myelosuppression risk. Monitor CBC closely."
    elif risk == "moderate":
        notes = "Moderate bone marrow exposure; routine CBC monitoring recommended."
    else:
        notes = "Low bone marrow exposure; myelosuppression risk is low."

    return BoneMarrowPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_marrow_mg_L=c_marrow_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_marrow_mg_L=cmax_marrow,
        tmax_marrow_h=tmax_marrow_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_marrow_mg_h_per_L=auc_marrow,
        marrow_plasma_ratio=ratio,
        myelosuppression_risk=risk,
        notes=notes,
    )


def compare_routes(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
) -> list:
    """Compare IV bolus and oral routes, sorted by marrow_plasma_ratio descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    cl_L_per_h:
        Systemic clearance in L/h.

    Returns
    -------
    list[BoneMarrowPKResult] sorted by marrow_plasma_ratio descending.
    """
    results = [
        simulate_bone_marrow_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=r,
            cl_L_per_h=cl_L_per_h,
        )
        for r in ("iv_bolus", "oral")
    ]
    results.sort(key=lambda x: x.marrow_plasma_ratio, reverse=True)
    return results
