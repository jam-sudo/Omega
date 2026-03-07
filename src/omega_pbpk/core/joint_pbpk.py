"""Cartilage/joint drug distribution PBPK model.

Models intra-articular (IA) drug injection and synovial fluid / cartilage
drug distribution following joint injection.

Model structure
---------------
Four compartments (amounts in mg):

  1. Synovial fluid (injection site, default ~3 mL for knee)
  2. Cartilage (slow uptake, avascular, ~50 mL volume)
  3. Systemic plasma (drains from both synovial fluid and cartilage release)

Drug deposited entirely in synovial fluid at time 0.

ODEs
----
  dA_sf/dt   = -(k_sf_turnover + k_cart_uptake) * A_sf
  dA_cart/dt =  k_cart_uptake * A_sf - k_cart_release * A_cart
  dA_plasma/dt = k_sf_turnover * A_sf + k_cart_release * A_cart
                 - ke_sys * A_plasma

Concentrations
--------------
  C_synovial  = A_sf    / v_synovial_L
  C_cartilage = A_cart  / 0.05 L  (cartilage pseudo-volume)
  C_plasma    = A_plasma / vd_sys_L

References
----------
- Baber N et al., Br J Clin Pharmacol. 1986;22(4):385-93 (IA pharmacokinetics)
- Biran S et al., Ann Rheum Dis. 1988;47(4):304-6 (cartilage PK)
- Edwards JCW, Ann Rheum Dis. 1994;53(4):213-22 (synovial fluid turnover)
- Chevalier X et al., Ann Rheum Dis. 2010;69(6):1028-33 (IA therapy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_LN2 = math.log(2.0)

# Cartilage pseudo-volume (L) — avascular, ~50 mL
_V_CARTILAGE_L = 0.05


@dataclass(frozen=True)
class JointPKResult:
    """Results from intra-articular joint PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Administered dose (mg).
    times_h : Simulation time points (h).
    c_synovial_mg_L : Synovial fluid concentration (mg/L).
    c_cartilage_mg_L : Cartilage concentration (mg/L).
    c_plasma_mg_L : Systemic plasma concentration (mg/L).
    cmax_synovial_mg_L : Peak synovial fluid concentration (mg/L).
    tmax_synovial_h : Time of peak synovial concentration (h).
    auc_synovial : AUC in synovial fluid (mg*h/L).
    cmax_cartilage_mg_L : Peak cartilage concentration (mg/L).
    auc_cartilage : AUC in cartilage (mg*h/L).
    synovial_t_half_h : Half-life of drug in synovial fluid (h).
    cartilage_auc_ratio : AUC_cartilage / AUC_synovial (penetration index).
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_synovial_mg_L: list[float]
    c_cartilage_mg_L: list[float]
    c_plasma_mg_L: list[float]
    cmax_synovial_mg_L: float
    tmax_synovial_h: float
    auc_synovial: float
    cmax_cartilage_mg_L: float
    auc_cartilage: float
    synovial_t_half_h: float
    cartilage_auc_ratio: float
    notes: str


def _trapz(y: list[float], x: list[float]) -> float:
    """Trapezoidal numerical integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
    return total


def simulate_joint_pk(
    drug_name: str,
    dose_mg: float,
    v_synovial_mL: float = 3.0,
    k_sf_turnover_per_h: float = 0.2,
    k_cart_uptake_per_h: float = 0.05,
    k_cart_release_per_h: float = 0.01,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 20.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> JointPKResult:
    """Simulate intra-articular drug pharmacokinetics.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose injected into the joint (mg).
    v_synovial_mL : Synovial fluid volume (mL, default 3.0 for normal knee).
    k_sf_turnover_per_h : Synovial fluid turnover rate — drainage to systemic (h^-1).
    k_cart_uptake_per_h : Cartilage drug uptake rate constant (h^-1).
    k_cart_release_per_h : Cartilage drug release rate constant back to systemic (h^-1).
    cl_sys_L_per_h : Systemic plasma clearance (L/h).
    vd_sys_L : Systemic volume of distribution (L).
    t_end_h : Simulation duration (h).
    dt_h : Forward Euler time step (h).

    Returns
    -------
    JointPKResult with full time-course and PK summary.

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if v_synovial_mL <= 0:
        raise ValueError("v_synovial_mL must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")

    v_synovial_L = v_synovial_mL / 1000.0
    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Analytical half-life in synovial fluid
    synovial_t_half = _LN2 / (k_sf_turnover_per_h + k_cart_uptake_per_h)

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    actual_dt = t_end_h / n_steps

    times: list[float] = []
    c_synovial: list[float] = []
    c_cartilage: list[float] = []
    c_plasma: list[float] = []

    # Initial amounts (mg)
    a_sf = dose_mg
    a_cart = 0.0
    a_plasma = 0.0

    for i in range(n_steps + 1):
        t_i = i * actual_dt
        times.append(t_i)
        c_synovial.append(a_sf / v_synovial_L)
        c_cartilage.append(a_cart / _V_CARTILAGE_L)
        c_plasma.append(a_plasma / vd_sys_L)

        if i < n_steps:
            # ODEs (Forward Euler on amounts)
            da_sf = -(k_sf_turnover_per_h + k_cart_uptake_per_h) * a_sf
            da_cart = k_cart_uptake_per_h * a_sf - k_cart_release_per_h * a_cart
            da_plasma = (
                k_sf_turnover_per_h * a_sf + k_cart_release_per_h * a_cart - ke_sys * a_plasma
            )

            a_sf = max(0.0, a_sf + da_sf * actual_dt)
            a_cart = max(0.0, a_cart + da_cart * actual_dt)
            a_plasma = max(0.0, a_plasma + da_plasma * actual_dt)

    # PK metrics
    cmax_syn = max(c_synovial)
    tmax_syn_idx = c_synovial.index(cmax_syn)
    tmax_syn = times[tmax_syn_idx]

    auc_syn = _trapz(c_synovial, times)
    cmax_cart = max(c_cartilage)
    auc_cart = _trapz(c_cartilage, times)

    cart_auc_ratio = auc_cart / auc_syn if auc_syn > 0.0 else 0.0

    # Build notes
    if cart_auc_ratio > 0.5:
        notes = "Good cartilage penetration — suitable for OA therapy"
    elif cart_auc_ratio > 0.1:
        notes = "Moderate cartilage penetration"
    else:
        notes = "Poor cartilage penetration"

    return JointPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_synovial_mg_L=c_synovial,
        c_cartilage_mg_L=c_cartilage,
        c_plasma_mg_L=c_plasma,
        cmax_synovial_mg_L=cmax_syn,
        tmax_synovial_h=tmax_syn,
        auc_synovial=auc_syn,
        cmax_cartilage_mg_L=cmax_cart,
        auc_cartilage=auc_cart,
        synovial_t_half_h=synovial_t_half,
        cartilage_auc_ratio=cart_auc_ratio,
        notes=notes,
    )


def compare_injection_volumes(
    drug_name: str,
    dose_mg: float,
    volumes_mL: list[float] | None = None,
    **kwargs,
) -> list[JointPKResult]:
    """Compare different synovial fluid volumes (injection dilution effect).

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg).
    volumes_mL : List of synovial fluid volumes (mL) to compare.
        Default: [1.0, 3.0, 5.0].
    **kwargs : Additional parameters forwarded to simulate_joint_pk.

    Returns
    -------
    List of JointPKResult objects sorted by auc_cartilage descending.
    """
    if volumes_mL is None:
        volumes_mL = [1.0, 3.0, 5.0]

    results = [
        simulate_joint_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            v_synovial_mL=v,
            **kwargs,
        )
        for v in volumes_mL
    ]
    results.sort(key=lambda r: r.auc_cartilage, reverse=True)
    return results


__all__ = [
    "JointPKResult",
    "simulate_joint_pk",
    "compare_injection_volumes",
]
