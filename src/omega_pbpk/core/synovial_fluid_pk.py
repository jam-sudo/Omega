"""Synovial fluid PK model for joint-targeting drugs (NSAIDs, DMARDs, corticosteroids).

Two-compartment model: plasma + synovial fluid (SF).
SF equilibrates with plasma via synovium permeability with delayed kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SynovialFluidPKResult:
    """PK result for synovial fluid compartment simulation."""

    # Inputs
    drug_name: str
    dose_mg: float
    logp: float
    mw_da: float

    # Trajectories
    times_h: list
    c_plasma_mg_L: list
    c_sf_mg_L: list

    # Summary metrics
    cmax_plasma: float
    cmax_sf: float
    tmax_plasma_h: float
    tmax_sf_h: float
    auc_plasma: float
    auc_sf: float
    sf_to_plasma_auc_ratio: float
    kp_sf: float
    sf_penetration_index: float  # sf_to_plasma_auc_ratio × 100 (%)
    therapeutic_adequacy: str  # "poor" / "moderate" / "good"
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_kp_sf(logp: float, mw_da: float) -> float:
    """Compute SF-to-plasma partition coefficient."""
    base = 0.7
    logp_factor = max(0.1, 1.0 - (logp - 2.0) * 0.05)
    mw_factor = max(0.1, 1.0 - (mw_da - 400.0) / 2000.0)
    return base * logp_factor * mw_factor


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC by trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


def _therapeutic_adequacy(penetration_pct: float) -> str:
    if penetration_pct < 30.0:
        return "poor"
    elif penetration_pct <= 70.0:
        return "moderate"
    else:
        return "good"


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_synovial_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    mw_da: float = 300.0,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 10.0,
    ps_syn_L_per_h: float = 0.05,
    v_sf_L: float = 0.002,
    ke_sf_per_h: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SynovialFluidPKResult:
    """Simulate plasma and synovial fluid PK after an IV bolus dose.

    Two-compartment forward-Euler model:
        dC_plasma/dt = -ke * C_plasma - PS_syn*(C_plasma - C_sf/Kp_sf) / Vd_plasma
        dC_sf/dt    =  PS_syn*(C_plasma - C_sf/Kp_sf) / V_sf - ke_sf * C_sf

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose (mg).
    logp:
        Octanol-water log partition coefficient.
    mw_da:
        Molecular weight (Da).
    cl_L_per_h:
        Plasma clearance (L/h).
    vd_plasma_L:
        Plasma (central) volume of distribution (L).
    ps_syn_L_per_h:
        Synovium permeability-surface area product (L/h).
    v_sf_L:
        Synovial fluid volume per joint (L).
    ke_sf_per_h:
        First-order elimination rate from SF (1/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step (h).

    Returns
    -------
    SynovialFluidPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if ps_syn_L_per_h <= 0:
        raise ValueError("ps_syn_L_per_h must be positive")

    ke = cl_L_per_h / vd_plasma_L
    kp_sf = _compute_kp_sf(logp, mw_da)

    # Initial conditions: IV bolus at t=0
    c_plasma = dose_mg / vd_plasma_L
    c_sf = 0.0

    n_steps = max(1, int(round(t_end_h / dt_h)))

    times_h: list[float] = []
    c_plasma_list: list[float] = []
    c_sf_list: list[float] = []

    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_sf_list.append(c_sf)

        if _ == n_steps:
            break

        # Driving force for SF exchange
        driving_force = c_plasma - c_sf / kp_sf  # mg/L equivalent

        dc_plasma = -ke * c_plasma - ps_syn_L_per_h * driving_force / vd_plasma_L
        dc_sf = ps_syn_L_per_h * driving_force / v_sf_L - ke_sf_per_h * c_sf

        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_sf = max(0.0, c_sf + dc_sf * dt_h)
        t += dt_h

    # Summary metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_sf = max(c_sf_list) if c_sf_list else 0.0

    tmax_plasma_h = times_h[c_plasma_list.index(cmax_plasma)] if cmax_plasma > 0 else 0.0
    tmax_sf_h = times_h[c_sf_list.index(cmax_sf)] if cmax_sf > 0 else 0.0

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_sf = _trapezoidal_auc(times_h, c_sf_list)

    sf_to_plasma_auc_ratio = auc_sf / auc_plasma if auc_plasma > 0 else 0.0
    sf_penetration_index = sf_to_plasma_auc_ratio * 100.0
    adequacy = _therapeutic_adequacy(sf_penetration_index)

    notes = (
        f"Synovial fluid PK: Kp_sf={kp_sf:.3f}, PS_syn={ps_syn_L_per_h:.3f} L/h, "
        f"SF penetration {sf_penetration_index:.1f}% ({adequacy}); "
        f"tmax_sf={tmax_sf_h:.2f}h > tmax_plasma={tmax_plasma_h:.2f}h"
    )

    return SynovialFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        mw_da=mw_da,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_sf_mg_L=c_sf_list,
        cmax_plasma=cmax_plasma,
        cmax_sf=cmax_sf,
        tmax_plasma_h=tmax_plasma_h,
        tmax_sf_h=tmax_sf_h,
        auc_plasma=auc_plasma,
        auc_sf=auc_sf,
        sf_to_plasma_auc_ratio=sf_to_plasma_auc_ratio,
        kp_sf=kp_sf,
        sf_penetration_index=sf_penetration_index,
        therapeutic_adequacy=adequacy,
        notes=notes,
    )


def compare_antiarthritic_drugs(
    drugs_params: list[dict],
) -> list[SynovialFluidPKResult]:
    """Simulate and rank multiple drugs by SF penetration.

    Parameters
    ----------
    drugs_params:
        List of dicts; each dict is keyword arguments for
        ``simulate_synovial_fluid_pk``.  The ``drug_name`` and ``dose_mg``
        keys are required; all others are optional.

    Returns
    -------
    List of SynovialFluidPKResult sorted by sf_to_plasma_auc_ratio descending.
    """
    results = []
    for params in drugs_params:
        result = simulate_synovial_fluid_pk(**params)
        results.append(result)
    results.sort(key=lambda r: r.sf_to_plasma_auc_ratio, reverse=True)
    return results
