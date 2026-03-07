"""Synovial fluid pharmacokinetics — 2-compartment plasma + synovial model.

Useful for anti-inflammatory / anti-arthritic drugs where joint penetration
is the therapeutic target.

Model
-----
Plasma compartment (1-cpt, IV or oral with depot):
    dA_gut/dt     = -ka * A_gut           (oral only)
    dA_plasma/dt  = ka * A_gut            (oral)   or  [dose at t=0 for IV]
                    - (CL/Vd) * A_plasma
                    - k_sf_in * A_plasma  (transfer into synovial)
                    + k_sf_out * A_synovial  (drainage back to plasma)

Synovial compartment (volume v_synovial_L):
    dA_synovial/dt = k_sf_in * A_plasma
                     - k_sf_out * A_synovial

Concentrations:
    C_plasma   = A_plasma  / Vd        (mg/L)
    C_synovial = A_synovial / v_synovial_L  (mg/L)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = ["JointPKResult", "simulate_joint_pk"]


@dataclass(frozen=True)
class JointPKResult:
    """Result of a synovial fluid PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_synovial_mg_L: list[float]
    cmax_plasma: float
    cmax_synovial: float
    auc_plasma: float
    auc_synovial: float
    synovial_plasma_ratio: float  # AUC-based ratio (proxy for Css ratio)
    t_half_synovial_h: float  # effective half-life in synovial space
    notes: str


def simulate_joint_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    k_sf_in_per_h: float = 0.05,  # transfer rate into synovial fluid (1/h)
    k_sf_out_per_h: float = 0.1,  # drainage rate from synovial fluid (1/h)
    v_synovial_mL: float = 2.0,  # synovial fluid volume (mL)
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> JointPKResult:
    """Simulate plasma and synovial fluid PK.

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    dose_mg : float
        Administered dose (mg).
    cl_L_per_h : float
        Total plasma clearance (L/h).
    vd_L : float
        Volume of distribution of plasma compartment (L).
    route : str
        ``"oral"`` (default) or ``"iv"``.
    ka_per_h : float
        First-order absorption rate constant for oral route (1/h).
    f_oral : float
        Oral bioavailability fraction (0–1).
    k_sf_in_per_h : float
        First-order transfer rate constant from plasma into synovial fluid (1/h).
    k_sf_out_per_h : float
        First-order drainage rate constant from synovial fluid back to plasma (1/h).
    v_synovial_mL : float
        Volume of synovial fluid (mL).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Forward-Euler time step (h).

    Returns
    -------
    JointPKResult
    """
    # ── Validation ──────────────────────────────────────────────────────────
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if v_synovial_mL <= 0:
        raise ValueError(f"v_synovial_mL must be > 0, got {v_synovial_mL}")
    route = route.lower()
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got {route!r}")
    f_oral = float(np.clip(f_oral, 0.0, 1.0))
    k_sf_in_per_h = max(k_sf_in_per_h, 0.0)
    k_sf_out_per_h = max(k_sf_out_per_h, 0.0)

    # ── Setup ────────────────────────────────────────────────────────────────
    v_synovial_L = v_synovial_mL / 1000.0  # convert mL → L
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    cp = np.zeros(n_steps + 1)  # plasma concentration (mg/L)
    csf = np.zeros(n_steps + 1)  # synovial concentration (mg/L)

    # Initial amounts (mg)
    if route == "iv":
        a_plasma = dose_mg  # IV bolus → all into plasma instantly
        a_gut = 0.0
    else:
        a_plasma = 0.0
        a_gut = dose_mg * f_oral  # oral: effective dose into gut depot

    a_synovial = 0.0  # synovial amount (mg)

    # Set t=0 concentrations
    cp[0] = a_plasma / vd_L
    csf[0] = a_synovial / v_synovial_L if v_synovial_L > 0 else 0.0

    # ── Forward Euler ODE ────────────────────────────────────────────────────
    for i in range(n_steps):
        # Absorption from gut (oral only)
        if route == "oral" and ka_per_h > 0:
            d_gut = ka_per_h * a_gut * dt_h
            a_gut = max(a_gut - d_gut, 0.0)
            a_plasma += d_gut

        # Concentration-gradient-driven transfer between compartments.
        # Transfer in (plasma → synovial): proportional to plasma concentration
        #   dA_sf/dt|in  = k_sf_in * V_sf * C_plasma
        # Transfer out (synovial → plasma, lymphatic drainage):
        #   dA_sf/dt|out = k_sf_out * A_sf
        # At steady state: C_sf = C_plasma * k_sf_in / k_sf_out  (<1 for k_in<k_out)
        c_plasma = a_plasma / vd_L
        d_in = k_sf_in_per_h * v_synovial_L * c_plasma * dt_h
        d_out = k_sf_out_per_h * a_synovial * dt_h
        # Plasma elimination
        d_elim = ke * a_plasma * dt_h

        a_plasma = max(a_plasma - d_elim - d_in + d_out, 0.0)
        a_synovial = max(a_synovial + d_in - d_out, 0.0)

        cp[i + 1] = a_plasma / vd_L
        csf[i + 1] = a_synovial / v_synovial_L

    # ── PK metrics ───────────────────────────────────────────────────────────
    cmax_plasma = float(np.max(cp))
    cmax_synovial = float(np.max(csf))
    auc_plasma = float(np_trapz(cp, times))
    auc_synovial = float(np_trapz(csf, times))

    # Synovial:plasma ratio from AUC (or from peak if both are non-zero)
    if auc_plasma > 0:
        sf_ratio = auc_synovial / auc_plasma
    else:
        sf_ratio = 0.0

    # Effective half-life in synovial: t½ ≈ ln2 / k_sf_out
    # (drainage dominates clearance from the compartment)
    if k_sf_out_per_h > 0:
        t_half_synovial_h = float(np.log(2) / k_sf_out_per_h)
    else:
        t_half_synovial_h = float("inf")

    notes = (
        f"Forward Euler dt={dt_h} h. "
        f"Synovial vol={v_synovial_mL} mL. "
        f"k_in={k_sf_in_per_h} 1/h, k_out={k_sf_out_per_h} 1/h."
    )

    return JointPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times.tolist(),
        c_plasma_mg_L=cp.tolist(),
        c_synovial_mg_L=csf.tolist(),
        cmax_plasma=cmax_plasma,
        cmax_synovial=cmax_synovial,
        auc_plasma=auc_plasma,
        auc_synovial=auc_synovial,
        synovial_plasma_ratio=sf_ratio,
        t_half_synovial_h=t_half_synovial_h,
        notes=notes,
    )
