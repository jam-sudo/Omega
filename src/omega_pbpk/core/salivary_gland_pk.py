"""Salivary gland PK model — drug distribution into saliva.

Models passive diffusion of drug from plasma into saliva. Used for
TDM of lithium, theophylline, carbamazepine where salivary drug
concentration reflects unbound plasma concentration for non-ionic drugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SalivaryPKResult:
    """Result of salivary gland PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_saliva_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_saliva_mg_L: float = 0.0
    tmax_plasma_h: float = 0.0
    tmax_saliva_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_saliva_mg_h_per_L: float = 0.0
    saliva_to_plasma_ratio: float = 0.0
    tdm_feasibility: str = "poor"
    notes: str = ""


def simulate_salivary_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 0.5,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    fup: float = 0.5,
    route: str = "oral",
    vd_L: float = 30.0,
    cl_L_per_h: float = 3.0,
    saliva_flow_mL_per_h: float = 30.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SalivaryPKResult:
    """Simulate drug distribution into saliva using a 2-compartment model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Drug pKa value.
    ionization_type:
        One of "acid", "base", or "neutral".
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    route:
        Administration route: "oral" or "iv".
    vd_L:
        Volume of distribution in litres.
    cl_L_per_h:
        Clearance in L/h.
    saliva_flow_mL_per_h:
        Salivary flow rate in mL/h.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        ODE integration step size in hours.

    Returns
    -------
    SalivaryPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if not (0 < fup <= 1):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    # --- Salivary pH ionization correction ---
    salivary_ph = 6.8
    if ionization_type == "acid":
        f_neutral = 1.0 / (1.0 + 10.0 ** (salivary_ph - pka))
    elif ionization_type == "base":
        f_neutral = 1.0 / (1.0 + 10.0 ** (pka - salivary_ph))
    else:
        f_neutral = 1.0

    # --- Model parameters ---
    v_saliva_L = 0.05  # saliva compartment volume (L)
    flow_L_h = saliva_flow_mL_per_h / 1000.0

    # Salivary equilibrium target fraction: unbound, ionization-corrected drug
    # that diffuses passively into saliva.  At steady state C_saliva ≈ f_eq * C_plasma.
    f_eq = fup * f_neutral  # equilibrium saliva/plasma concentration ratio target

    # Permeability-surface-area product (PS, L/h).
    # Rate in saliva compartment = PS / v_saliva_L (/h).
    # For stability of Forward Euler we need PS / v_saliva_L * dt_h < 1.
    # Sub-stepping (dt_sub) is used to guarantee this.
    ps = 5.0  # L/h — salivary gland permeability-area product

    ke = cl_L_per_h / vd_L

    # --- Internal sub-step size for numerical stability ---
    # Maximum safe dt: 1 / (ps/v_saliva + flow/v_saliva + ke) with safety factor 0.4
    max_rate = ps / v_saliva_L + flow_L_h / v_saliva_L + ke
    dt_sub = min(dt_h, 0.4 / max_rate)
    # Ensure dt_sub evenly divides dt_h (round up sub-steps)
    n_sub = max(1, int(math.ceil(dt_h / dt_sub)))
    dt_sub = dt_h / n_sub

    # --- Initial conditions ---
    if route == "iv":
        c_plasma = dose_mg / vd_L
        a_gut = 0.0
        ka = 0.0
    else:
        c_plasma = 0.0
        a_gut = dose_mg * 0.9
        ka = 0.7  # /h

    c_saliva = 0.0

    # --- Time integration (Forward Euler with sub-stepping) ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list[float] = []
    c_plasma_arr: list[float] = []
    c_saliva_arr: list[float] = []

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma_arr.append(c_plasma)
        c_saliva_arr.append(c_saliva)

        # Sub-step the ODE for numerical stability
        for _sub in range(n_sub):
            # Oral absorption input (mg/L/h from gut → plasma)
            if route == "oral":
                absorption_input = ka * a_gut / vd_L
                da_gut_dt = -ka * a_gut
                a_gut += da_gut_dt * dt_sub
            else:
                absorption_input = 0.0

            # Passive diffusion driving force:
            # C_saliva equilibrates to f_eq * C_plasma.
            # Mass flux (mg/h) = PS * (f_eq * C_plasma - C_saliva)
            # Driving force in concentration units:
            transfer_flux = ps * (f_eq * c_plasma - c_saliva)  # mg/h

            # ODE system
            # Plasma: loses drug to saliva, gains back if C_saliva > f_eq*C_plasma
            dc_plasma_dt = absorption_input - ke * c_plasma - transfer_flux / vd_L
            # Saliva: gains from plasma, loses via salivary flow washout
            dc_saliva_dt = transfer_flux / v_saliva_L - (flow_L_h / v_saliva_L) * c_saliva

            c_plasma += dc_plasma_dt * dt_sub
            c_saliva += dc_saliva_dt * dt_sub

            # Clamp to non-negative
            if c_plasma < 0.0:
                c_plasma = 0.0
            if c_saliva < 0.0:
                c_saliva = 0.0

    # --- Derived PK metrics ---
    cmax_plasma = max(c_plasma_arr)
    cmax_saliva = max(c_saliva_arr)
    tmax_plasma = times_h[c_plasma_arr.index(cmax_plasma)]
    tmax_saliva = times_h[c_saliva_arr.index(cmax_saliva)]

    # Manual trapezoidal AUC
    auc_plasma = 0.0
    auc_saliva = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc_plasma += 0.5 * (c_plasma_arr[i - 1] + c_plasma_arr[i]) * dt
        auc_saliva += 0.5 * (c_saliva_arr[i - 1] + c_saliva_arr[i]) * dt

    if auc_plasma > 0:
        saliva_to_plasma_ratio = auc_saliva / auc_plasma
    else:
        saliva_to_plasma_ratio = 0.0

    # TDM feasibility assessment
    if 0.8 <= saliva_to_plasma_ratio <= 1.2:
        tdm_feasibility = "excellent"
    elif 0.5 <= saliva_to_plasma_ratio <= 1.5:
        tdm_feasibility = "good"
    else:
        tdm_feasibility = "poor"

    notes = (
        f"Ionization type: {ionization_type}, f_neutral at pH {salivary_ph}: {f_neutral:.3f}. "
        f"Equilibrium saliva/plasma fraction (f_eq): {f_eq:.4f}. "
        f"For neutral drugs, saliva/plasma ratio should approximate fup ({fup:.2f})."
    )

    return SalivaryPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_arr,
        c_saliva_mg_L=c_saliva_arr,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_saliva_mg_L=cmax_saliva,
        tmax_plasma_h=tmax_plasma,
        tmax_saliva_h=tmax_saliva,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_saliva_mg_h_per_L=auc_saliva,
        saliva_to_plasma_ratio=saliva_to_plasma_ratio,
        tdm_feasibility=tdm_feasibility,
        notes=notes,
    )
