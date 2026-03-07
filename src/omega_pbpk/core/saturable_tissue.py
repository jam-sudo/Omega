"""Saturable tissue distribution — dose-dependent nonlinear Kp via Langmuir binding.

At high doses, finite tissue binding sites (Bmax) become saturated, causing:
- Decreased apparent tissue:plasma ratio (Kp)
- Decreased apparent volume of distribution (Vd)
- Nonlinear dose-exposure relationship
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PLASMA_VOLUME_L = 3.0  # approximate human plasma volume


@dataclass(frozen=True)
class SaturableTissueResult:
    """Result from a saturable tissue distribution simulation."""

    drug_name: str
    dose_mg: float
    route: str

    # Binding parameters
    bmax_tissue_umol_per_kg: float
    kd_tissue_uM: float

    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_tissue_free_uM: list[float]
    c_tissue_bound_uM: list[float]
    c_tissue_total_uM: list[float]

    # Dose-dependent metrics
    vd_apparent_L: float
    vd_linear_L: float
    vd_nonlinearity_pct: float

    saturation_pct_peak: float
    kp_apparent: float
    kp_linear: float

    dose_proportional: bool
    notes: str


def simulate_saturable_tissue(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    kp_linear: float,
    mw: float,
    bmax_tissue_umol_per_kg: float = 100.0,
    kd_tissue_uM: float = 1.0,
    tissue_mass_kg: float = 5.0,
    route: str = "iv",
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SaturableTissueResult:
    """Simulate saturable tissue distribution with Langmuir binding (QSS).

    Plasma PK is simulated as a standard 1-compartment model with linear Vd.
    Tissue concentration at each time point is calculated from plasma via
    quasi-steady-state (QSS) Langmuir binding.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Administered dose (mg).
    cl_L_per_h:
        Systemic clearance (L/h).
    kp_linear:
        Linear (low-dose) tissue:plasma partition coefficient (unbound basis).
    mw:
        Molecular weight (g/mol).
    bmax_tissue_umol_per_kg:
        Maximum binding site density in tissue (µmol/kg tissue).
    kd_tissue_uM:
        Dissociation constant for tissue binding (µM).
    tissue_mass_kg:
        Total tissue mass involved in distribution (kg).
    route:
        "iv" or "oral".
    ka_per_h:
        First-order oral absorption rate constant (1/h); ignored for IV.
    f_oral:
        Oral bioavailability fraction (0-1).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    SaturableTissueResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if kp_linear <= 0:
        raise ValueError(f"kp_linear must be > 0, got {kp_linear}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if bmax_tissue_umol_per_kg <= 0:
        raise ValueError(f"bmax_tissue_umol_per_kg must be > 0, got {bmax_tissue_umol_per_kg}")
    if kd_tissue_uM <= 0:
        raise ValueError(f"kd_tissue_uM must be > 0, got {kd_tissue_uM}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    # --- Linear Vd estimate ---
    # In the linear (low-dose) range the Langmuir isotherm approaches:
    # c_bound ≈ Bmax * c_free / Kd  (when c_free << Kd)
    # c_total_tissue = c_free * (1 + Bmax / Kd) = kp_linear * c_plasma * (1 + Bmax / Kd)
    # → Kp_linear_total = kp_linear * (1 + Bmax / Kd)
    # vd_linear = V_plasma + tissue_mass * Kp_linear_total
    kp_linear_total = kp_linear * (1.0 + bmax_tissue_umol_per_kg / kd_tissue_uM)
    vd_linear_L = _PLASMA_VOLUME_L + tissue_mass_kg * kp_linear_total

    # --- Build time array ---
    n_steps = max(int(t_end_h / dt_h) + 1, 2)
    times = np.linspace(0.0, t_end_h, n_steps)

    # --- Plasma PK simulation (1-cpt forward Euler) ---
    c_plasma = np.zeros(n_steps)
    a_gut = 0.0  # gut amount (mg) for oral

    ke = cl_L_per_h / vd_linear_L  # elimination rate constant

    if route == "iv":
        c_plasma[0] = dose_mg / vd_linear_L
    else:
        # Oral: start absorption into gut
        a_gut = dose_mg * f_oral
        c_plasma[0] = 0.0

    for i in range(1, n_steps):
        cp = c_plasma[i - 1]
        if route == "oral":
            abs_rate = ka_per_h * a_gut  # mg/h
            a_gut = max(0.0, a_gut - abs_rate * dt_h)
        else:
            abs_rate = 0.0

        dcp_dt = (abs_rate - ke * cp * vd_linear_L) / vd_linear_L
        c_plasma[i] = max(0.0, cp + dcp_dt * dt_h)

    # --- QSS Langmuir tissue binding at each time point ---
    c_tissue_free = np.zeros(n_steps)
    c_tissue_bound = np.zeros(n_steps)
    c_tissue_total = np.zeros(n_steps)

    bmax_total = bmax_tissue_umol_per_kg  # µmol/kg (per kg tissue)

    for i in range(n_steps):
        # Convert plasma to µM
        c_plasma_uM = c_plasma[i] * 1000.0 / mw  # mg/L → µmol/L = µM

        # Free tissue conc (rapid equilibrium assumption)
        c_free = kp_linear * c_plasma_uM  # µM

        # Langmuir bound
        c_bound = (bmax_total * c_free) / (kd_tissue_uM + c_free)  # µmol/kg

        c_tissue_free[i] = c_free
        c_tissue_bound[i] = c_bound
        c_tissue_total[i] = c_free + c_bound

    # --- Apparent Vd ---
    # Vd_app = V_plasma + tissue_mass_kg * Kp_app
    # where Kp_app = c_tissue_total_uM / c_plasma_uM (concentration ratio, same units).
    # This is the standard pharmacokinetic definition.
    idx_cmax_plasma = int(np.argmax(c_plasma))
    c_p_at_cmax = c_plasma[idx_cmax_plasma]
    c_plasma_uM_at_cmax = c_p_at_cmax * 1000.0 / mw

    if c_plasma_uM_at_cmax > 0:
        kp_app_at_cmax = c_tissue_total[idx_cmax_plasma] / c_plasma_uM_at_cmax
        vd_apparent_L = _PLASMA_VOLUME_L + tissue_mass_kg * kp_app_at_cmax
    else:
        vd_apparent_L = vd_linear_L

    vd_nonlinearity_pct = abs(vd_apparent_L - vd_linear_L) / vd_linear_L * 100.0
    dose_proportional = bool(vd_nonlinearity_pct < 20.0)

    # --- Peak saturation ---
    c_peak_uM = max(c_tissue_free) if max(c_tissue_free) > 0 else 0.0
    saturation_pct_peak = (c_peak_uM / (kd_tissue_uM + c_peak_uM)) * 100.0

    # --- Apparent Kp at Cmax ---
    idx_cmax = int(np.argmax(c_plasma))
    c_plasma_peak_uM = c_plasma[idx_cmax] * 1000.0 / mw
    c_tissue_peak_uM = c_tissue_total[idx_cmax]

    if c_plasma_peak_uM > 0:
        kp_apparent = c_tissue_peak_uM / c_plasma_peak_uM
    else:
        kp_apparent = kp_linear

    notes = (
        f"{drug_name} | {route.upper()} {dose_mg} mg | "
        f"Vd_linear={vd_linear_L:.1f} L | Vd_app={vd_apparent_L:.1f} L | "
        f"sat_peak={saturation_pct_peak:.1f}% | "
        f"{'linear' if dose_proportional else 'nonlinear'} PK"
    )

    return SaturableTissueResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        bmax_tissue_umol_per_kg=bmax_tissue_umol_per_kg,
        kd_tissue_uM=kd_tissue_uM,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        c_tissue_free_uM=c_tissue_free.tolist(),
        c_tissue_bound_uM=c_tissue_bound.tolist(),
        c_tissue_total_uM=c_tissue_total.tolist(),
        vd_apparent_L=vd_apparent_L,
        vd_linear_L=vd_linear_L,
        vd_nonlinearity_pct=vd_nonlinearity_pct,
        saturation_pct_peak=saturation_pct_peak,
        kp_apparent=kp_apparent,
        kp_linear=kp_linear,
        dose_proportional=dose_proportional,
        notes=notes,
    )


def dose_linearity_analysis(
    drug_name: str,
    doses_mg: list[float],
    cl_L_per_h: float,
    kp_linear: float,
    mw: float,
    **kwargs,
) -> dict:
    """Analyse dose linearity across a range of doses.

    Parameters
    ----------
    drug_name:
        Drug name.
    doses_mg:
        List of doses to simulate (mg).
    cl_L_per_h:
        Systemic clearance (L/h).
    kp_linear:
        Linear tissue:plasma Kp.
    mw:
        Molecular weight (g/mol).
    **kwargs:
        Passed to :func:`simulate_saturable_tissue`.

    Returns
    -------
    dict with keys:
        "doses_mg", "vd_apparent_L", "saturation_pct", "linear" (bool list),
        "overall_linear" (bool).
    """
    vd_list: list[float] = []
    sat_list: list[float] = []
    linear_list: list[bool] = []

    for dose in doses_mg:
        result = simulate_saturable_tissue(
            drug_name=drug_name,
            dose_mg=dose,
            cl_L_per_h=cl_L_per_h,
            kp_linear=kp_linear,
            mw=mw,
            **kwargs,
        )
        vd_list.append(result.vd_apparent_L)
        sat_list.append(result.saturation_pct_peak)
        linear_list.append(result.dose_proportional)

    overall_linear = all(linear_list)

    return {
        "doses_mg": list(doses_mg),
        "vd_apparent_L": vd_list,
        "saturation_pct": sat_list,
        "linear": linear_list,
        "overall_linear": overall_linear,
    }
