"""
Phase 266: Synovial Fluid PK Model (p266)
PK model for drug distribution into synovial fluid (joint space).

Note: core/joint_synovial_pk.py already exists; this is a separate,
more detailed synovial fluid model named _p266 to avoid collision.
"""

from dataclasses import dataclass, field


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


@dataclass
class SynovialFluidPKResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_synovial_mg_L: list[float] = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_synovial_mg_L: float = 0.0
    tmax_synovial_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_synovial_mg_h_per_L: float = 0.0
    synovial_plasma_ratio: float = 0.0
    effective_half_life_synovial_h: float = 0.0
    notes: str = ""


# Physiological constants
_V_PLASMA = 3.0  # L
_V_SYNOVIAL = 0.004  # L (4 mL human knee joint)
_K_PS = 0.1  # /h  plasma -> synovial
_K_SP = 0.05  # /h  synovial -> plasma (slow efflux)
_K_ELIM_S = 0.02  # /h  local elimination in synovial
_KA = 1.0  # /h  oral absorption
_F_ORAL = 0.8  # oral bioavailability fraction

_VALID_ROUTES = {"iv_bolus", "oral", "intra_articular"}


def simulate_synovial_fluid_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_h: float,
    t_end_h: float = 72.0,
    dt_h: float = 0.05,
) -> SynovialFluidPKResult:
    """
    Simulate drug PK in plasma and synovial fluid using a 2-compartment ODE.

    Compartments: plasma, synovial fluid
    Optional depot compartment for oral dosing.

    Parameters
    ----------
    drug_name : str
    dose_mg : float  -- must be > 0
    route : str      -- "iv_bolus", "oral", or "intra_articular"
    cl_L_per_h : float -- systemic clearance (L/h), must be > 0
    t_end_h : float  -- simulation duration (hours)
    dt_h : float     -- time step (hours)

    Returns
    -------
    SynovialFluidPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")

    ke = cl_L_per_h / _V_PLASMA

    # Initial conditions
    if route == "iv_bolus":
        C_plasma = dose_mg / _V_PLASMA
        C_synovial = 0.0
        A_depot = 0.0
    elif route == "oral":
        C_plasma = 0.0
        C_synovial = 0.0
        A_depot = dose_mg * _F_ORAL  # absorbed dose in depot (mg)
    else:  # intra_articular
        C_plasma = 0.0
        C_synovial = dose_mg / _V_SYNOVIAL
        A_depot = 0.0

    times = []
    c_plasma_list = []
    c_synovial_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_synovial_list.append(C_synovial)

        if step == n_steps:
            break

        # Absorption from depot (oral only)
        if route == "oral" and A_depot > 0.0:
            absorbed = _KA * A_depot * dt_h
            # Clamp to available depot
            if absorbed > A_depot:
                absorbed = A_depot
            A_depot -= absorbed
            flux_into_plasma = absorbed / _V_PLASMA  # mg/L per step
        else:
            flux_into_plasma = 0.0

        # ODEs (Forward Euler)
        # dC_plasma/dt = -ke*C_plasma - k_ps*C_plasma + k_sp*C_synovial*(V_synovial/V_plasma)
        dCp_dt = -ke * C_plasma - _K_PS * C_plasma + _K_SP * C_synovial * (_V_SYNOVIAL / _V_PLASMA)
        # dC_synovial/dt = k_ps*C_plasma*(V_plasma/V_synovial) - k_sp*C_synovial - k_elim_s*C_synovial
        dCs_dt = (
            _K_PS * C_plasma * (_V_PLASMA / _V_SYNOVIAL)
            - _K_SP * C_synovial
            - _K_ELIM_S * C_synovial
        )

        C_plasma = C_plasma + dCp_dt * dt_h + flux_into_plasma
        C_synovial = C_synovial + dCs_dt * dt_h

        # Clamp negatives (numerical stability)
        if C_plasma < 0.0:
            C_plasma = 0.0
        if C_synovial < 0.0:
            C_synovial = 0.0

        t += dt_h

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    cmax_synovial = max(c_synovial_list)

    idx_tmax_syn = c_synovial_list.index(cmax_synovial)
    tmax_synovial = times[idx_tmax_syn]

    auc_plasma = _trapz(times, c_plasma_list)
    auc_synovial = _trapz(times, c_synovial_list)

    if auc_plasma > 0.0:
        syn_plasma_ratio = auc_synovial / auc_plasma
    else:
        syn_plasma_ratio = 0.0

    import math

    eff_half_life = math.log(2) / (_K_SP + _K_ELIM_S)

    notes = (
        f"Route: {route}. ke={ke:.4f}/h. "
        f"Synovial equilibration half-life={eff_half_life:.2f}h. "
        f"Synovial/plasma AUC ratio={syn_plasma_ratio:.4f}."
    )

    return SynovialFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_synovial_mg_L=c_synovial_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_synovial_mg_L=cmax_synovial,
        tmax_synovial_h=tmax_synovial,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_synovial_mg_h_per_L=auc_synovial,
        synovial_plasma_ratio=syn_plasma_ratio,
        effective_half_life_synovial_h=eff_half_life,
        notes=notes,
    )


def compare_routes(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
) -> list[SynovialFluidPKResult]:
    """
    Compare all three dosing routes and return results sorted by
    auc_synovial_mg_h_per_L descending (IA expected highest).

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_L_per_h : float

    Returns
    -------
    list of SynovialFluidPKResult, sorted by auc_synovial descending
    """
    results = []
    for route in _VALID_ROUTES:
        result = simulate_synovial_fluid_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_synovial_mg_h_per_L, reverse=True)
    return results
