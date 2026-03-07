"""Two-pore model for large molecule (biologic) vascular permeability (Phase 402).

The two-pore formalism (Rippe & Haraldsson 1994) describes transcapillary
exchange via small pores (size-selective, charge-selective) and large pores
(mainly convective). For macromolecules like IgG (~150 kDa), small pores
dominate diffusion while large pores allow bulk-flow convection.

References
----------
- Rippe B, Haraldsson B. Physiol Rev 1994;74:163-219.
- Baxter LT et al. Microvasc Res 1994;48:31-57.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "TwoPorePKResult",
    "simulate_two_pore_pk",
    "mw_effect_on_pk",
]


@dataclass
class TwoPorePKResult:
    """Results from two-pore PK simulation.

    Attributes
    ----------
    drug_name : Name of the biologic drug.
    mw_kDa : Molecular weight (kDa).
    dose_mg : Administered IV bolus dose (mg).
    sigma_small : Reflection coefficient for small pores (0–1).
    sigma_large : Reflection coefficient for large pores (0–1).
    times_days : Simulation time points (days).
    c_central_mg_L : Plasma (central compartment) concentration (mg/L).
    c_tissue_mg_L : Tissue (interstitial) concentration (mg/L).
    cmax : Peak plasma concentration (mg/L).
    t_half_days : Plasma elimination half-life (days).
    auc_mg_day_per_L : AUC from 0 to t_end (mg*day/L), trapezoidal.
    notes : Simulation notes.
    """

    drug_name: str
    mw_kDa: float
    dose_mg: float
    sigma_small: float
    sigma_large: float
    times_days: list[float] = field(default_factory=list)
    c_central_mg_L: list[float] = field(default_factory=list)
    c_tissue_mg_L: list[float] = field(default_factory=list)
    cmax: float = 0.0
    t_half_days: float = 0.0
    auc_mg_day_per_L: float = 0.0
    notes: str = ""


def _compute_auc_trap(times: list[float], concentrations: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i] + concentrations[i - 1]) * dt
    return auc


def _estimate_half_life(times: list[float], concentrations: list[float], cmax: float) -> float:
    """Estimate terminal half-life from the log-linear tail.

    Finds the point where concentration drops below cmax/2 and uses
    the last 30% of time points for terminal slope estimation.
    """
    n = len(times)
    if n < 4:
        return float("nan")

    # Use last 30% of points for terminal slope (at least 3 points)
    n_term = max(3, n // 3)
    t_term = times[-n_term:]
    c_term = concentrations[-n_term:]

    # Filter out zero or negative concentrations
    valid = [(t, c) for t, c in zip(t_term, c_term, strict=True) if c > 1e-15]
    if len(valid) < 2:
        return float("nan")

    t_v, c_v = zip(*valid, strict=False)  # noqa: B905  – unpacking pairs
    # Log-linear regression: log(C) = log(C0) - ke*t
    n_v = len(t_v)
    mean_t = sum(t_v) / n_v
    mean_lnc = sum(math.log(c) for c in c_v) / n_v

    stt = sum((t - mean_t) ** 2 for t in t_v)
    stlnc = sum((t - mean_t) * (math.log(c) - mean_lnc) for t, c in zip(t_v, c_v, strict=True))

    if stt < 1e-15:
        return float("nan")

    ke = -stlnc / stt
    if ke <= 0:
        return float("nan")

    return math.log(2.0) / ke


def simulate_two_pore_pk(
    drug_name: str,
    mw_kDa: float,
    dose_mg: float,
    cl_L_per_day: float,
    vd_central_L: float,
    sigma_small: float,
    sigma_large: float,
    lymph_flow_L_per_day: float,
    t_end_days: float,
    dt_days: float = 0.1,
) -> TwoPorePKResult:
    """Simulate two-pore PK for a biologic drug (IV bolus).

    Uses a 2-compartment model (central plasma + interstitial tissue) with
    transcapillary exchange governed by the two-pore theory.

    Transport from central to tissue:
        flux_out = lymph_flow * ((1-sigma_small)*0.95 + (1-sigma_large)*0.05) * C_central
    Lymph return from tissue to central:
        flux_in = lymph_flow * C_tissue
    Elimination from central:
        dC_central/dt -= ke * C_central  where ke = cl_L_per_day / vd_central_L

    Volume of tissue interstitial is estimated as 2 * vd_central_L.

    Parameters
    ----------
    drug_name : Name of the biologic.
    mw_kDa : Molecular weight in kDa (must be >0).
    dose_mg : IV bolus dose in mg (must be >0).
    cl_L_per_day : Systemic clearance (L/day, must be >0).
    vd_central_L : Volume of central compartment (L, must be >0).
    sigma_small : Reflection coefficient for small pores (0–1).
    sigma_large : Reflection coefficient for large pores (0–1).
    lymph_flow_L_per_day : Lymph flow rate (L/day, must be >0).
    t_end_days : Simulation end time (days, must be >0).
    dt_days : Forward Euler step size (days, default 0.1).

    Returns
    -------
    TwoPorePKResult with full time-course and summary PK parameters.

    Raises
    ------
    ValueError : If any input is out of valid range.
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string.")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be positive.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if cl_L_per_day <= 0:
        raise ValueError("cl_L_per_day must be positive.")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be positive.")
    if not (0.0 <= sigma_small <= 1.0):
        raise ValueError("sigma_small must be between 0 and 1.")
    if not (0.0 <= sigma_large <= 1.0):
        raise ValueError("sigma_large must be between 0 and 1.")
    if lymph_flow_L_per_day <= 0:
        raise ValueError("lymph_flow_L_per_day must be positive.")
    if t_end_days <= 0:
        raise ValueError("t_end_days must be positive.")
    if dt_days <= 0:
        raise ValueError("dt_days must be positive.")

    # Derived
    ke = cl_L_per_day / vd_central_L  # elimination rate constant (1/day)
    vd_tissue_L = 2.0 * vd_central_L  # tissue interstitial volume

    # Transcapillary permeability factor (two-pore weighted combination)
    # Small pores carry 95% of lymph, large pores 5%
    perm_factor = (1.0 - sigma_small) * 0.95 + (1.0 - sigma_large) * 0.05

    # Initial conditions: IV bolus into central compartment
    c_central = dose_mg / vd_central_L
    c_tissue = 0.0

    n_steps = max(1, int(round(t_end_days / dt_days)))
    dt_actual = t_end_days / n_steps

    times: list[float] = [0.0]
    c_central_list: list[float] = [c_central]
    c_tissue_list: list[float] = [c_tissue]

    for _ in range(n_steps):
        # Transcapillary fluxes (per unit time, L/day * mg/L = mg/day)
        flux_out = lymph_flow_L_per_day * perm_factor * c_central  # central -> tissue
        flux_in = lymph_flow_L_per_day * c_tissue  # tissue -> central (lymph return)

        dc_central = -ke * c_central - flux_out / vd_central_L + flux_in / vd_central_L
        dc_tissue = flux_out / vd_tissue_L - flux_in / vd_tissue_L

        c_central = max(0.0, c_central + dc_central * dt_actual)
        c_tissue = max(0.0, c_tissue + dc_tissue * dt_actual)

        times.append(times[-1] + dt_actual)
        c_central_list.append(c_central)
        c_tissue_list.append(c_tissue)

    cmax = max(c_central_list)
    auc = _compute_auc_trap(times, c_central_list)
    t_half = _estimate_half_life(times, c_central_list, cmax)

    notes = f"Two-pore IV bolus; perm_factor={perm_factor:.4f}; ke={ke:.4f}/day; n_steps={n_steps}"

    return TwoPorePKResult(
        drug_name=drug_name,
        mw_kDa=mw_kDa,
        dose_mg=dose_mg,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        times_days=times,
        c_central_mg_L=c_central_list,
        c_tissue_mg_L=c_tissue_list,
        cmax=cmax,
        t_half_days=t_half if not math.isnan(t_half) else 0.0,
        auc_mg_day_per_L=auc,
        notes=notes,
    )


def mw_effect_on_pk(
    drug_name: str,
    dose_mg: float,
    mw_values_kDa: list[float],
    cl_per_mw_factor: float = 0.005,
) -> list[dict[str, object]]:
    """Evaluate MW effect on PK using the two-pore model.

    For each MW value, sigma_small is estimated as:
        sigma_small = clamp(0.6 + 0.004 * mw_kDa, 0.90, 0.99)

    Clearance scales with MW: cl = cl_per_mw_factor * mw_kDa (L/day).

    Parameters
    ----------
    drug_name : Name of the drug.
    dose_mg : IV bolus dose (mg, must be >0).
    mw_values_kDa : List of MW values to test (kDa, all must be positive).
    cl_per_mw_factor : Scaling factor for CL vs MW (L/day per kDa, must be >0).

    Returns
    -------
    List of dicts, one per MW, each containing:
      mw_kDa, sigma_small, cl_L_per_day, auc_mg_day_per_L, t_half_days, cmax.

    Raises
    ------
    ValueError : If inputs are invalid.
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")
    if not mw_values_kDa:
        raise ValueError("mw_values_kDa must be a non-empty list.")
    if any(mw <= 0 for mw in mw_values_kDa):
        raise ValueError("All mw_values_kDa must be positive.")
    if cl_per_mw_factor <= 0:
        raise ValueError("cl_per_mw_factor must be positive.")

    results: list[dict[str, object]] = []

    for mw in mw_values_kDa:
        sigma_small = min(0.99, max(0.90, 0.6 + 0.004 * mw))
        sigma_large = 0.1
        cl = cl_per_mw_factor * mw
        # Scale Vd with MW (rough allometric: larger molecules have smaller Vd)
        vd_central = max(3.0, 10.0 - 0.05 * mw)  # L, floor at 3 L
        lymph_flow = 2.0  # L/day typical

        pk = simulate_two_pore_pk(
            drug_name=drug_name,
            mw_kDa=mw,
            dose_mg=dose_mg,
            cl_L_per_day=cl,
            vd_central_L=vd_central,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            lymph_flow_L_per_day=lymph_flow,
            t_end_days=60.0,
            dt_days=0.5,
        )

        results.append(
            {
                "mw_kDa": mw,
                "sigma_small": sigma_small,
                "cl_L_per_day": cl,
                "auc_mg_day_per_L": pk.auc_mg_day_per_L,
                "t_half_days": pk.t_half_days,
                "cmax": pk.cmax,
            }
        )

    return results
