"""
Phase 373 — Mucosal Drug Absorption Model

Models vaginal, rectal, and nasal mucosal absorption with pH-dependent
permeability and mucus layer diffusion using Forward Euler ODE integration.
"""

import math
from dataclasses import dataclass


@dataclass
class MucosalAbsorptionResult:
    """Result of mucosal drug absorption simulation."""

    drug_name: str
    route: str
    dose_mg: float
    times_h: list
    c_mucosal_mg_mL: list
    c_systemic_mg_L: list
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_h: float
    f_absorbed_pct: float
    mucus_barrier_factor: float
    ph_factor: float
    notes: str


# Route-specific baseline volumes in mL
_ROUTE_VOLUMES_mL = {
    "vaginal": 5.0,
    "rectal": 2.0,
    "nasal": 1.0,
}

_VALID_ROUTES = frozenset(_ROUTE_VOLUMES_mL.keys())


def _compute_mucus_barrier_factor(mw_da: float, logp: float, mucus_thickness_um: float) -> float:
    """
    Mucus barrier factor = exp(-0.01 * mw_da / (logp + 2.0 + 0.01) * mucus_thickness_um / 100.0)
    Clamped to [0.05, 1.0].
    """
    denom = logp + 2.0 + 0.01
    exponent = -0.01 * mw_da / denom * mucus_thickness_um / 100.0
    factor = math.exp(exponent)
    return max(0.05, min(1.0, factor))


def _compute_ph_factor(pka: float, ph_mucosal: float, drug_class: str) -> float:
    """
    Compute pH factor based on Henderson-Hasselbalch ionization.

    drug_class: "acid" if pka > 0, "base" if pka < 0, "neutral" if pka == 0
    Convention used here: pka > 0 => acid-like, pka < 0 => base-like, pka == 0 => neutral
    """
    if pka == 0.0:
        return 1.0
    if pka > 0:
        # Acid: fraction neutral = 1 / (1 + 10^(pH - pKa))
        return 1.0 / (1.0 + 10.0 ** (ph_mucosal - pka))
    else:
        # Base: pka < 0, treat abs(pka) as the base pKa
        base_pka = abs(pka)
        return 1.0 / (1.0 + 10.0 ** (base_pka - ph_mucosal))


def simulate_mucosal_absorption(
    drug_name: str,
    route: str,
    dose_mg: float,
    pka: float,
    logp: float,
    mw_da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
    mucus_thickness_um: float = 100.0,
    ph_mucosal: float = 7.0,
) -> MucosalAbsorptionResult:
    """
    Simulate mucosal drug absorption via vaginal, rectal, or nasal route.

    Parameters
    ----------
    drug_name : str
    route : str — "vaginal", "rectal", or "nasal"
    dose_mg : float — administered dose in mg
    pka : float — drug pKa (>0 acid, <0 base, 0 neutral); range [-5, 14]
    logp : float — octanol-water partition coefficient
    mw_da : float — molecular weight in Daltons
    cl_sys_L_per_h : float — systemic clearance in L/h
    vd_sys_L : float — systemic volume of distribution in L
    t_end_h : float — simulation end time in hours (default 12)
    dt_h : float — Forward Euler step in hours (default 0.05)
    mucus_thickness_um : float — mucus layer thickness in micrometers (default 100)
    ph_mucosal : float — mucosal pH (default 7.0)

    Returns
    -------
    MucosalAbsorptionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {sorted(_VALID_ROUTES)}, got '{route}'")
    if not (-5.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [-5, 14], got {pka}")

    v_route_mL = _ROUTE_VOLUMES_mL[route]

    mucus_barrier_factor = _compute_mucus_barrier_factor(mw_da, logp, mucus_thickness_um)
    ph_factor = _compute_ph_factor(pka, ph_mucosal, "")

    # Effective absorption rate (h^-1)
    ka_raw = 0.3 * mucus_barrier_factor * ph_factor * (logp + 5.0) / 6.0
    ka_eff = max(0.01, min(2.0, ka_raw))

    # Forward Euler simulation
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list[float] = []
    c_mucosal_mg_mL: list[float] = []
    c_systemic_mg_L: list[float] = []

    m_mucosal = dose_mg  # mg
    c_sys = 0.0  # mg/L

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)
        c_mucosal_mg_mL.append(m_mucosal / v_route_mL)
        c_systemic_mg_L.append(c_sys)

        # Derivatives
        dm_mucosal_dt = -ka_eff * m_mucosal
        dc_sys_dt = (ka_eff * m_mucosal / vd_sys_L) - (cl_sys_L_per_h / vd_sys_L) * c_sys

        # Forward Euler update
        m_mucosal += dm_mucosal_dt * dt_h
        c_sys += dc_sys_dt * dt_h

        # Clamp to non-negative
        if m_mucosal < 0.0:
            m_mucosal = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

    # AUC via manual trapezoidal rule
    x = times_h
    y = c_systemic_mg_L
    auc = sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))

    # Cmax and Tmax
    cmax = max(c_systemic_mg_L)
    tmax = times_h[c_systemic_mg_L.index(cmax)]

    # Fraction absorbed: systemic amount recovered / dose
    # Systemic amount at end = c_sys * vd_sys_L, but integrated over time
    # Approximate as: mass absorbed = dose - m_mucosal_final (remaining in mucosal)
    # m_mucosal at last step
    m_mucosal_final = m_mucosal
    mass_absorbed_mg = dose_mg - max(0.0, m_mucosal_final)
    f_absorbed_pct = (mass_absorbed_mg / dose_mg) * 100.0

    notes = (
        f"Route: {route}, ka_eff={ka_eff:.4f} h^-1, "
        f"mucus_barrier={mucus_barrier_factor:.3f}, ph_factor={ph_factor:.3f}"
    )

    return MucosalAbsorptionResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=times_h,
        c_mucosal_mg_mL=c_mucosal_mg_mL,
        c_systemic_mg_L=c_systemic_mg_L,
        auc_systemic_mg_h_per_L=auc,
        cmax_systemic_mg_L=cmax,
        tmax_h=tmax,
        f_absorbed_pct=f_absorbed_pct,
        mucus_barrier_factor=mucus_barrier_factor,
        ph_factor=ph_factor,
        notes=notes,
    )


def compare_mucosal_routes(
    drug_name: str,
    dose_mg: float,
    pka: float,
    logp: float,
    mw_da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> list:
    """
    Simulate all three mucosal routes and return results sorted by AUC descending.

    Returns
    -------
    list of MucosalAbsorptionResult — length 3, sorted descending by auc_systemic_mg_h_per_L
    """
    results = []
    for route in ("vaginal", "rectal", "nasal"):
        result = simulate_mucosal_absorption(
            drug_name=drug_name,
            route=route,
            dose_mg=dose_mg,
            pka=pka,
            logp=logp,
            mw_da=mw_da,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_systemic_mg_h_per_L, reverse=True)
    return results
