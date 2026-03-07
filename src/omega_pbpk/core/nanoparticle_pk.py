"""Nanoparticle Drug Delivery Pharmacokinetics.

Models PK of nanoparticle-encapsulated drugs (liposomes, polymeric NPs, solid lipid NPs).
Drug exists in two forms: encapsulated (NP) and released free drug.
NPs have slower clearance and accumulate in tumors via the EPR effect.

Three-compartment IV model:
  - NP in blood (encapsulated drug)
  - Free drug in plasma (released from NP)
  - Tumor compartment (EPR accumulation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# NP type default release rate constants (per hour)
_NP_RELEASE_RATES: dict[str, float] = {
    "liposome": 0.1,
    "polymeric": 0.03,
    "solid_lipid": 0.05,
    "custom": 0.1,  # placeholder; overridden by user k_release
}

_VALID_NP_TYPES = frozenset(_NP_RELEASE_RATES.keys())

# Slow tumor efflux rate (1/h)
_K_TUMOR_OUT_PER_H: float = 0.05


@dataclass(frozen=True)
class NanoparticlePKResult:
    """Result from nanoparticle PK simulation.

    Attributes:
        drug_name: Name of the drug.
        dose_mg: Administered dose (mg).
        np_type: Nanoparticle formulation type.
        k_release_per_h: First-order drug release rate from NP (1/h).
        epr_factor: EPR accumulation multiplier for tumor (>=1).
        times_h: Simulation time points (h).
        c_np: NP blood concentration (mg/L).
        c_free: Free drug plasma concentration (mg/L).
        c_tumor: Tumor drug concentration (mg/L).
        cmax_np: Peak NP blood concentration (mg/L).
        cmax_free: Peak free drug plasma concentration (mg/L).
        cmax_tumor: Peak tumor drug concentration (mg/L).
        auc_np: AUC of NP blood concentration (mg·h/L).
        auc_free: AUC of free drug plasma concentration (mg·h/L).
        auc_tumor: AUC of tumor drug concentration (mg·h/L).
        tumor_plasma_ratio: Ratio of tumor AUC to free plasma AUC.
        notes: Descriptive notes about the simulation.
    """

    drug_name: str
    dose_mg: float
    np_type: str
    k_release_per_h: float
    epr_factor: float
    times_h: list[float]
    c_np: list[float]
    c_free: list[float]
    c_tumor: list[float]
    cmax_np: float
    cmax_free: float
    cmax_tumor: float
    auc_np: float
    auc_free: float
    auc_tumor: float
    tumor_plasma_ratio: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    k_release_per_h: float,
    cl_np_L_per_h: float,
    vd_np_L: float,
    cl_free_L_per_h: float,
    vd_free_L: float,
    epr_factor: float,
    np_type: str,
) -> None:
    """Validate simulation inputs, raising ValueError on invalid values."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if k_release_per_h <= 0:
        raise ValueError(f"k_release_per_h must be > 0, got {k_release_per_h}")
    if cl_np_L_per_h <= 0:
        raise ValueError(f"cl_np_L_per_h must be > 0, got {cl_np_L_per_h}")
    if vd_np_L <= 0:
        raise ValueError(f"vd_np_L must be > 0, got {vd_np_L}")
    if cl_free_L_per_h <= 0:
        raise ValueError(f"cl_free_L_per_h must be > 0, got {cl_free_L_per_h}")
    if vd_free_L <= 0:
        raise ValueError(f"vd_free_L must be > 0, got {vd_free_L}")
    if epr_factor < 1.0:
        raise ValueError(f"epr_factor must be >= 1.0, got {epr_factor}")
    if np_type not in _VALID_NP_TYPES:
        raise ValueError(f"np_type must be one of {sorted(_VALID_NP_TYPES)}, got '{np_type}'")


def _trapz_auc(times: NDArray[np.float64], conc: NDArray[np.float64]) -> float:
    """Compute AUC using the trapezoidal rule."""
    return float(np.trapz(conc, times))


def simulate_nanoparticle_pk(
    drug_name: str,
    dose_mg: float,
    np_type: str = "liposome",
    k_release_per_h: float = 0.1,
    cl_np_L_per_h: float = 0.5,
    vd_np_L: float = 5.0,
    cl_free_L_per_h: float = 10.0,
    vd_free_L: float = 50.0,
    epr_factor: float = 2.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> NanoparticlePKResult:
    """Simulate nanoparticle drug delivery pharmacokinetics.

    Three-compartment IV model:
      - Compartment 1: NP blood (encapsulated drug)
      - Compartment 2: Free drug plasma (released from NP)
      - Compartment 3: Tumor (EPR accumulation)

    ODEs (Forward Euler):
      dC_np/dt = -(cl_np/vd_np + k_release) * C_np
      dC_free/dt = k_release * C_np * vd_np/vd_free - (cl_free/vd_free) * C_free
      dC_tumor/dt = epr_factor * k_release * C_np - k_tumor_out * C_tumor

    For np_type != "custom", the provided k_release_per_h is used as-is.
    NP type presets define default release rates but the explicit k_release_per_h
    argument always takes precedence.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV dose administered (mg).
        np_type: Nanoparticle type ("liposome", "polymeric", "solid_lipid", "custom").
        k_release_per_h: First-order drug release rate from NP (1/h).
        cl_np_L_per_h: NP blood clearance (L/h).
        vd_np_L: NP volume of distribution (L).
        cl_free_L_per_h: Free drug clearance (L/h).
        vd_free_L: Free drug volume of distribution (L).
        epr_factor: EPR tumor accumulation multiplier (>=1).
        t_end_h: Simulation duration (h).
        dt_h: Time step for Forward Euler integration (h).

    Returns:
        NanoparticlePKResult with full time-course and summary PK metrics.
    """
    _validate_inputs(
        dose_mg,
        k_release_per_h,
        cl_np_L_per_h,
        vd_np_L,
        cl_free_L_per_h,
        vd_free_L,
        epr_factor,
        np_type,
    )

    # Initial conditions: all drug in NP compartment
    c_np = dose_mg / vd_np_L  # mg/L
    c_free = 0.0
    c_tumor = 0.0

    # Rate constants
    k_np_elim = cl_np_L_per_h / vd_np_L  # elimination rate of NP
    k_free_elim = cl_free_L_per_h / vd_free_L  # elimination rate of free drug
    k_transfer = k_release_per_h * vd_np_L / vd_free_L  # transfer to free compartment

    n_steps = max(1, round(t_end_h / dt_h))
    times = np.zeros(n_steps + 1)
    arr_np = np.zeros(n_steps + 1)
    arr_free = np.zeros(n_steps + 1)
    arr_tumor = np.zeros(n_steps + 1)

    arr_np[0] = c_np
    arr_free[0] = c_free
    arr_tumor[0] = c_tumor

    for i in range(n_steps):
        times[i + 1] = (i + 1) * dt_h
        # Forward Euler
        dc_np = -(k_np_elim + k_release_per_h) * c_np
        dc_free = k_transfer * c_np - k_free_elim * c_free
        dc_tumor = epr_factor * k_release_per_h * c_np - _K_TUMOR_OUT_PER_H * c_tumor

        c_np = max(0.0, c_np + dc_np * dt_h)
        c_free = max(0.0, c_free + dc_free * dt_h)
        c_tumor = max(0.0, c_tumor + dc_tumor * dt_h)

        arr_np[i + 1] = c_np
        arr_free[i + 1] = c_free
        arr_tumor[i + 1] = c_tumor

    auc_np = _trapz_auc(times, arr_np)
    auc_free = _trapz_auc(times, arr_free)
    auc_tumor = _trapz_auc(times, arr_tumor)
    tumor_plasma_ratio = auc_tumor / auc_free if auc_free > 0 else 0.0

    notes = (
        f"{np_type} NP: k_release={k_release_per_h:.3f}/h, "
        f"EPR={epr_factor:.1f}x, tumor/plasma AUC={tumor_plasma_ratio:.2f}"
    )

    return NanoparticlePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        np_type=np_type,
        k_release_per_h=k_release_per_h,
        epr_factor=epr_factor,
        times_h=times.tolist(),
        c_np=arr_np.tolist(),
        c_free=arr_free.tolist(),
        c_tumor=arr_tumor.tolist(),
        cmax_np=float(np.max(arr_np)),
        cmax_free=float(np.max(arr_free)),
        cmax_tumor=float(np.max(arr_tumor)),
        auc_np=auc_np,
        auc_free=auc_free,
        auc_tumor=auc_tumor,
        tumor_plasma_ratio=tumor_plasma_ratio,
        notes=notes,
    )


def compare_np_formulations(
    drug_name: str,
    dose_mg: float,
    cl_free: float,
    vd_free: float,
) -> dict[str, dict[str, float]]:
    """Compare free drug vs nanoparticle formulations.

    Simulates free drug (instantaneous release approximation) and three NP types,
    reporting AUC ratios relative to free drug and tumor/plasma AUC ratios.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV dose administered (mg).
        cl_free: Free drug clearance (L/h).
        vd_free: Free drug volume of distribution (L).

    Returns:
        Dict mapping formulation name to dict with keys:
          - "auc_free": Free drug AUC (mg·h/L)
          - "auc_np": NP AUC (mg·h/L), 0 for free drug
          - "auc_tumor": Tumor AUC (mg·h/L)
          - "auc_ratio_vs_free": Free drug AUC relative to free-drug baseline
          - "tumor_plasma_ratio": Tumor AUC / free plasma AUC
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_free <= 0:
        raise ValueError(f"cl_free must be > 0, got {cl_free}")
    if vd_free <= 0:
        raise ValueError(f"vd_free must be > 0, got {vd_free}")

    # Free drug: very fast release (large k_release), effectively free from start
    # Model as NP with negligible NP compartment: k_release very high → most drug
    # transfers immediately. We simulate with high k_release and small vd_np.
    free_result = simulate_nanoparticle_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        np_type="custom",
        k_release_per_h=100.0,  # effectively immediate release
        cl_np_L_per_h=cl_free,
        vd_np_L=vd_free,
        cl_free_L_per_h=cl_free,
        vd_free_L=vd_free,
        epr_factor=1.0,  # no EPR for free drug
        t_end_h=72.0,
    )
    baseline_auc_free = free_result.auc_free + free_result.auc_np  # total systemic exposure

    results: dict[str, dict[str, float]] = {}

    # Free drug baseline
    results["free_drug"] = {
        "auc_free": free_result.auc_free + free_result.auc_np,
        "auc_np": 0.0,
        "auc_tumor": free_result.auc_tumor,
        "auc_ratio_vs_free": 1.0,
        "tumor_plasma_ratio": free_result.tumor_plasma_ratio,
    }

    # NP formulations
    np_configs = [
        ("liposome", _NP_RELEASE_RATES["liposome"]),
        ("polymeric", _NP_RELEASE_RATES["polymeric"]),
        ("solid_lipid", _NP_RELEASE_RATES["solid_lipid"]),
    ]
    for np_type, k_release in np_configs:
        res = simulate_nanoparticle_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            np_type=np_type,
            k_release_per_h=k_release,
            cl_np_L_per_h=0.5,
            vd_np_L=5.0,
            cl_free_L_per_h=cl_free,
            vd_free_L=vd_free,
            epr_factor=2.0,
            t_end_h=72.0,
        )
        total_auc = res.auc_free + res.auc_np
        auc_ratio = total_auc / baseline_auc_free if baseline_auc_free > 0 else 0.0
        results[np_type] = {
            "auc_free": res.auc_free,
            "auc_np": res.auc_np,
            "auc_tumor": res.auc_tumor,
            "auc_ratio_vs_free": auc_ratio,
            "tumor_plasma_ratio": res.tumor_plasma_ratio,
        }

    return results
