"""Cerebral blood flow PBPK model.

Models cerebral blood flow changes and their effect on CNS drug delivery
using a three-compartment approach: plasma, brain vascular, and brain
parenchyma (tissue).

Brain uptake rate scales linearly with CBF factor:
    k_brain_in = cbf_factor * base_k_brain

Forward-Euler ODE integration is used throughout.

References
----------
- Hoshi Y et al., J Cereb Blood Flow Metab. 2013;33:1600–1606
- Loryan I et al., Fluids Barriers CNS. 2014;11:12
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "CBFPKResult",
    "simulate_cbf_pk",
    "compare_cbf_conditions",
]


@dataclass(frozen=True)
class CBFPKResult:
    """Result of cerebral blood flow PK simulation."""

    drug_name: str
    dose_mg: float
    cbf_factor: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_brain_vasc_mg_L: list[float]
    c_brain_tissue_mg_L: list[float]
    cmax_plasma: float
    cmax_brain_vasc: float
    cmax_brain_tissue: float
    auc_plasma: float
    auc_brain_tissue: float
    brain_to_plasma_ratio: float
    notes: str


def simulate_cbf_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    cbf_factor: float = 1.0,
    base_k_brain: float = 0.3,
    k_brain_out: float = 0.15,
    k_tissue_in: float = 0.1,
    k_tissue_out: float = 0.05,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CBFPKResult:
    """Simulate cerebral blood flow PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    cbf_factor : float
        Relative cerebral blood flow (1.0 = normal).
    base_k_brain : float
        Base rate constant for plasma-to-brain-vascular transfer (1/h).
    k_brain_out : float
        Rate constant for brain-vascular-to-plasma transfer (1/h).
    k_tissue_in : float
        Rate constant for brain-vascular-to-tissue transfer (1/h).
    k_tissue_out : float
        Rate constant for tissue-to-brain-vascular transfer (1/h).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    CBFPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if cbf_factor <= 0:
        raise ValueError("cbf_factor must be positive")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be positive")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")

    k_brain_in = cbf_factor * base_k_brain
    ke = cl_L_per_h / vd_L  # elimination rate constant

    n_steps = int(math.ceil(t_end_h / dt_h))
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    c_plasma = np.zeros(n_steps + 1)
    c_brain_vasc = np.zeros(n_steps + 1)
    c_brain_tissue = np.zeros(n_steps + 1)

    # Initial condition: IV bolus into plasma
    c_plasma[0] = dose_mg / vd_L

    for i in range(n_steps):
        dt = times[i + 1] - times[i]
        cp = c_plasma[i]
        cbv = c_brain_vasc[i]
        cbt = c_brain_tissue[i]

        # Plasma: elimination + exchange with brain vascular
        dc_plasma = -ke * cp - k_brain_in * cp + k_brain_out * cbv
        # Brain vascular: uptake from plasma, back to plasma, exchange with tissue
        dc_bv = k_brain_in * cp - k_brain_out * cbv - k_tissue_in * cbv + k_tissue_out * cbt
        # Brain tissue: uptake from vascular, back to vascular
        dc_bt = k_tissue_in * cbv - k_tissue_out * cbt

        c_plasma[i + 1] = max(0.0, cp + dt * dc_plasma)
        c_brain_vasc[i + 1] = max(0.0, cbv + dt * dc_bv)
        c_brain_tissue[i + 1] = max(0.0, cbt + dt * dc_bt)

    cmax_plasma = float(np.max(c_plasma))
    cmax_brain_vasc = float(np.max(c_brain_vasc))
    cmax_brain_tissue = float(np.max(c_brain_tissue))
    auc_plasma = float(np_trapz(c_plasma, times))
    auc_brain_tissue = float(np_trapz(c_brain_tissue, times))
    brain_to_plasma_ratio = cmax_brain_tissue / cmax_plasma if cmax_plasma > 0 else 0.0

    notes_parts: list[str] = []
    if cbf_factor < 1.0:
        notes_parts.append(f"Reduced CBF ({cbf_factor:.2f}x normal)")
    elif cbf_factor > 1.0:
        notes_parts.append(f"Increased CBF ({cbf_factor:.2f}x normal)")
    else:
        notes_parts.append("Normal CBF")
    notes_parts.append(f"Brain/plasma Cmax ratio: {brain_to_plasma_ratio:.4f}")

    return CBFPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cbf_factor=cbf_factor,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        c_brain_vasc_mg_L=c_brain_vasc.tolist(),
        c_brain_tissue_mg_L=c_brain_tissue.tolist(),
        cmax_plasma=cmax_plasma,
        cmax_brain_vasc=cmax_brain_vasc,
        cmax_brain_tissue=cmax_brain_tissue,
        auc_plasma=auc_plasma,
        auc_brain_tissue=auc_brain_tissue,
        brain_to_plasma_ratio=brain_to_plasma_ratio,
        notes="; ".join(notes_parts),
    )


def compare_cbf_conditions(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    cbf_factors: list[float],
    **kwargs,
) -> list[CBFPKResult]:
    """Run simulations across multiple CBF conditions.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    cbf_factors : list[float]
        List of CBF factors to simulate.
    **kwargs
        Additional keyword arguments passed to simulate_cbf_pk.

    Returns
    -------
    list[CBFPKResult]
    """
    if not cbf_factors:
        raise ValueError("cbf_factors must not be empty")
    return [
        simulate_cbf_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            cbf_factor=f,
            **kwargs,
        )
        for f in cbf_factors
    ]
