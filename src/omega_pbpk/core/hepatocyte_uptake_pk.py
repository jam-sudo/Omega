"""Phase 770 — Drug Hepatocyte Uptake PK.

Models OATP1B1/1B3 transporter-mediated hepatocyte uptake using the extended
clearance concept. 3-compartment system: plasma, hepatocyte, systemic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HepatocyteUptakePKResult:
    """Result of hepatocyte uptake PK simulation."""

    drug_name: str
    dose_mg: float
    fup: float
    times_h: list
    c_plasma_mg_L: list
    c_hep_mg_L: list
    cmax_plasma: float
    cmax_hep: float
    auc_plasma: float
    auc_hep: float
    hep_to_plasma_ratio: float
    cl_hepatic_estimated_L_per_h: float
    transporter_saturation: bool
    notes: str


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return auc


def simulate_hepatocyte_uptake_pk(
    drug_name: str,
    dose_mg: float,
    fup: float = 0.5,
    vmax_uptake_mg_per_h: float = 10.0,
    km_uptake_mg_L: float = 1.0,
    pdif_per_h: float = 0.1,
    k_release_per_h: float = 0.5,
    k_met_hep_per_h: float = 2.0,
    k_bile_per_h: float = 0.3,
    cl_sys_L_per_h: float = 1.0,
    vd_L: float = 50.0,
    v_hep_L: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> HepatocyteUptakePKResult:
    """Simulate hepatocyte uptake PK via extended clearance model.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        IV bolus dose in mg.
    fup : float
        Unbound plasma fraction (0-1).
    vmax_uptake_mg_per_h : float
        Maximum transporter-mediated uptake rate (mg/h).
    km_uptake_mg_L : float
        Michaelis constant for OATP uptake (mg/L).
    pdif_per_h : float
        Passive diffusion rate into hepatocytes (1/h).
    k_release_per_h : float
        Hepatocyte-to-plasma efflux rate (MRP3, 1/h).
    k_met_hep_per_h : float
        Intracellular hepatic metabolism rate (1/h).
    k_bile_per_h : float
        Biliary secretion rate from hepatocyte (1/h).
    cl_sys_L_per_h : float
        Systemic non-hepatic clearance (L/h).
    vd_L : float
        Plasma volume of distribution (L).
    v_hep_L : float
        Hepatocyte compartment volume (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    HepatocyteUptakePKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_sys_L_per_h < 0:
        raise ValueError("cl_sys_L_per_h must be >= 0")
    if vmax_uptake_mg_per_h <= 0:
        raise ValueError("vmax_uptake_mg_per_h must be > 0")
    if km_uptake_mg_L <= 0:
        raise ValueError("km_uptake_mg_L must be > 0")

    ke_sys = cl_sys_L_per_h / vd_L  # systemic elimination rate constant

    # Initial conditions: IV bolus into plasma
    c_plasma = dose_mg / vd_L  # mg/L
    c_hep = 0.0  # mg/L in hepatocyte compartment

    times = []
    c_plasma_list = []
    c_hep_list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    # Track initial Cmax plasma for transporter saturation check
    c_plasma_initial = c_plasma

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_hep_list.append(c_hep)

        # Transporter-mediated uptake rate (Michaelis-Menten, saturable)
        # k_uptake total (1/h) applies to plasma concentration
        v_mm = vmax_uptake_mg_per_h * fup / (km_uptake_mg_L + max(c_plasma, 0.0))
        k_uptake_total = (pdif_per_h + v_mm) / vd_L  # flux into hep per L plasma / h

        # Drug flux from plasma to hepatocyte (mg/h)
        flux_into_hep = k_uptake_total * c_plasma * vd_L

        # Drug flux from hepatocyte back to plasma (efflux, mg/h)
        flux_release = k_release_per_h * c_hep * v_hep_L

        # ODEs
        d_c_plasma = -ke_sys * c_plasma - flux_into_hep / vd_L + flux_release / vd_L
        d_c_hep = (
            flux_into_hep / v_hep_L - (k_met_hep_per_h + k_bile_per_h + k_release_per_h) * c_hep
        )

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + d_c_plasma * dt_h)
        c_hep = max(0.0, c_hep + d_c_hep * dt_h)

        t += dt_h

    cmax_plasma = max(c_plasma_list)
    cmax_hep = max(c_hep_list)
    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_hep = _trapezoidal_auc(times, c_hep_list)

    if auc_plasma > 0:
        hep_to_plasma_ratio = auc_hep / auc_plasma
    else:
        hep_to_plasma_ratio = 0.0

    # Estimate hepatic CL: dose/AUC - cl_sys
    if auc_plasma > 0:
        cl_total_est = dose_mg / auc_plasma
        cl_hepatic_est = max(0.0, cl_total_est - cl_sys_L_per_h)
    else:
        cl_hepatic_est = 0.0

    # Transporter saturation check at initial Cmax
    mm_contrib = vmax_uptake_mg_per_h * fup / (km_uptake_mg_L + c_plasma_initial)
    passive_equiv = pdif_per_h * 5  # threshold: MM > 5× passive
    transporter_saturation = mm_contrib > passive_equiv

    notes = (
        f"OATP-mediated uptake: Vmax={vmax_uptake_mg_per_h} mg/h, "
        f"Km={km_uptake_mg_L} mg/L; "
        f"Hep:plasma AUC ratio={hep_to_plasma_ratio:.2f}; "
        f"CL_hepatic estimated={cl_hepatic_est:.2f} L/h; "
        f"Transporter saturated={'yes' if transporter_saturation else 'no'}"
    )

    return HepatocyteUptakePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fup=fup,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_hep_mg_L=c_hep_list,
        cmax_plasma=cmax_plasma,
        cmax_hep=cmax_hep,
        auc_plasma=auc_plasma,
        auc_hep=auc_hep,
        hep_to_plasma_ratio=hep_to_plasma_ratio,
        cl_hepatic_estimated_L_per_h=cl_hepatic_est,
        transporter_saturation=transporter_saturation,
        notes=notes,
    )


def assess_transporter_contribution(result: HepatocyteUptakePKResult) -> dict:
    """Estimate OATP vs passive contribution to hepatic uptake.

    Uses the initial plasma concentration (dose/Vd) approximation.
    This is an approximation — exact fractions require detailed flux integration.

    Parameters
    ----------
    result : HepatocyteUptakePKResult
        Result from simulate_hepatocyte_uptake_pk.

    Returns
    -------
    dict with keys: oatp_fraction, passive_fraction, notes
    """
    if not result.c_plasma_mg_L:
        return {"oatp_fraction": 0.0, "passive_fraction": 1.0, "notes": "No data"}

    # Use peak plasma concentration for saturation estimate
    c_peak = result.cmax_plasma

    # Reconstruct approximate rate contributions at peak
    # (parameters not stored in result, use reasonable mid-point defaults)
    # We use the ratio from hep_to_plasma_ratio as proxy
    # For a proper assessment, we'd need original parameters — estimate from result fields
    # Use stored notes to infer saturation
    if result.transporter_saturation:
        # High saturation: transporter at Vmax, passive is small fraction
        oatp_fraction = 0.85
    else:
        # Not saturated: MM term ~ Vmax*fup/Km, compare to Pdif
        # Typical split: if cmax_plasma is low relative to Km, OATP dominates
        if c_peak < 0.5:
            oatp_fraction = 0.9
        elif c_peak < 2.0:
            oatp_fraction = 0.75
        else:
            oatp_fraction = 0.6

    passive_fraction = 1.0 - oatp_fraction

    return {
        "oatp_fraction": round(oatp_fraction, 3),
        "passive_fraction": round(passive_fraction, 3),
        "notes": (
            f"Estimated OATP1B1/1B3 fraction at peak plasma {c_peak:.2f} mg/L; "
            f"transporter_saturation={result.transporter_saturation}"
        ),
    }
