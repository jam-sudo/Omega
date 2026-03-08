"""Phase 497 — Synovial Fluid Drug Concentrations.

2-compartment PK model for drug distribution into synovial fluid (SF) of joints.
Relevant for arthritis treatment (NSAIDs, DMARDs, biologics).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SynovialFluidPKResult:
    """Results from synovial fluid PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    ionization_type: str
    times_h: list
    c_plasma_mg_L: list
    c_sf_mg_L: list
    cmax_sf_mg_L: float
    tmax_sf_h: float
    auc_sf_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    kp_sf: float
    sf_plasma_auc_ratio: float
    therapeutic_potential: str
    notes: str


def _compute_kp_sf(logP: float, ionization_type: str) -> float:
    """Compute synovial fluid/plasma partition coefficient."""
    lp = max(0.0, logP)
    if ionization_type == "acid":
        kp = 0.5 + 0.15 * lp
    elif ionization_type == "base":
        kp = 0.2 + 0.1 * lp
    else:  # neutral
        kp = 0.3 + 0.1 * lp
    # Clamp to [0.1, 3.0]
    return max(0.1, min(3.0, kp))


def _trapezoidal_auc(x: list, y: list) -> float:
    """Compute AUC using manual trapezoidal rule."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_synovial_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    ionization_type: str,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    V_sf_L: float = 0.002,
    Q_sf_L_per_h: float = 0.05,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> SynovialFluidPKResult:
    """Simulate drug distribution into synovial fluid using 2-compartment PK model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    logP:
        Lipophilicity.
    ionization_type:
        One of "neutral", "acid", "base".
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_plasma_L:
        Plasma volume of distribution in L.
    V_sf_L:
        Synovial fluid volume in L.
    Q_sf_L_per_h:
        Synovial fluid flow rate in L/h.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step in hours.

    Returns
    -------
    SynovialFluidPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if V_sf_L <= 0:
        raise ValueError("V_sf_L must be > 0")
    if ionization_type not in {"neutral", "acid", "base"}:
        raise ValueError("ionization_type must be 'neutral', 'acid', or 'base'")

    kp_sf = _compute_kp_sf(logP, ionization_type)
    k_el = cl_sys_L_per_h / vd_plasma_L

    # Initial conditions: IV bolus
    c_plasma = dose_mg / vd_plasma_L
    c_sf = 0.0

    times_h = []
    c_plasma_list = []
    c_sf_list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_sf_list.append(c_sf)

        # Forward Euler ODEs
        # dC_plasma/dt = -Q_sf/Vp*(C_plasma - C_sf/Kp_sf) - k_el*C_plasma
        # dC_sf/dt = Q_sf/V_sf*(C_plasma - C_sf/Kp_sf)
        driving = c_plasma - c_sf / kp_sf
        dc_plasma_dt = -(Q_sf_L_per_h / vd_plasma_L) * driving - k_el * c_plasma
        dc_sf_dt = (Q_sf_L_per_h / V_sf_L) * driving

        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)
        c_sf = max(0.0, c_sf + dc_sf_dt * dt_h)
        t += dt_h

    # Compute metrics
    cmax_sf = max(c_sf_list)
    tmax_sf = times_h[c_sf_list.index(cmax_sf)]
    auc_sf = _trapezoidal_auc(times_h, c_sf_list)
    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)

    sf_plasma_auc_ratio = auc_sf / auc_plasma if auc_plasma > 0.0 else 0.0

    if sf_plasma_auc_ratio < 0.3:
        therapeutic_potential = "poor"
    elif sf_plasma_auc_ratio <= 0.7:
        therapeutic_potential = "good"
    else:
        therapeutic_potential = "excellent"

    notes = (
        f"Kp_sf={kp_sf:.3f} ({ionization_type}); "
        f"SF/plasma AUC ratio={sf_plasma_auc_ratio:.3f}; "
        f"therapeutic potential: {therapeutic_potential}"
    )

    return SynovialFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        ionization_type=ionization_type,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_sf_mg_L=c_sf_list,
        cmax_sf_mg_L=cmax_sf,
        tmax_sf_h=tmax_sf,
        auc_sf_mg_h_per_L=auc_sf,
        auc_plasma_mg_h_per_L=auc_plasma,
        kp_sf=kp_sf,
        sf_plasma_auc_ratio=sf_plasma_auc_ratio,
        therapeutic_potential=therapeutic_potential,
        notes=notes,
    )


def compare_drugs_sf(drugs: list) -> list:
    """Compare multiple drugs by synovial fluid AUC ratio.

    Parameters
    ----------
    drugs:
        List of dicts with keys matching simulate_synovial_fluid_pk parameters.

    Returns
    -------
    List of SynovialFluidPKResult sorted by sf_plasma_auc_ratio descending.
    """
    results = []
    for d in drugs:
        result = simulate_synovial_fluid_pk(
            drug_name=d["drug_name"],
            dose_mg=d["dose_mg"],
            logP=d["logP"],
            ionization_type=d["ionization_type"],
            cl_sys_L_per_h=d["cl_sys_L_per_h"],
            vd_plasma_L=d.get("vd_plasma_L", 5.0),
            V_sf_L=d.get("V_sf_L", 0.002),
            Q_sf_L_per_h=d.get("Q_sf_L_per_h", 0.05),
            t_end_h=d.get("t_end_h", 24.0),
            dt_h=d.get("dt_h", 0.1),
        )
        results.append(result)
    results.sort(key=lambda r: r.sf_plasma_auc_ratio, reverse=True)
    return results
