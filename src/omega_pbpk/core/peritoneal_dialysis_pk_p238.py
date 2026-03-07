"""Peritoneal dialysis PK model (Phase 238).

Models drug PK during peritoneal dialysis with dwell time simulation,
using a two-compartment ODE system (plasma + dialysate) integrated by
forward Euler.
"""

from __future__ import annotations

from dataclasses import dataclass


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


@dataclass
class PeritonealDialysisPKResult:
    """Result of peritoneal dialysis PK simulation (Phase 238)."""

    drug_name: str
    dose_mg: float
    dwell_time_h: float
    times_h: list
    c_plasma_mg_L: list
    c_dialysate_mg_L: list
    cmax_plasma_mg_L: float
    trough_plasma_mg_L: float
    drug_removed_mg: float
    clearance_pd_L_per_h: float
    auc_plasma_mg_h_per_L: float
    notes: str


def simulate_peritoneal_dialysis_pk(
    drug_name: str,
    dose_mg: float,
    dwell_time_h: float,
    cl_sys_L_per_h: float,
    vp_L: float = 3.0,
    pd_mass_transfer_coeff_L_per_h: float = 1.5,
    dialysate_volume_L: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PeritonealDialysisPKResult:
    """Simulate drug PK during peritoneal dialysis.

    Uses a 2-compartment ODE model (plasma + dialysate) with forward Euler:
      dC_plasma/dt   = -ke * C_plasma - k_pd * C_plasma
      dC_dialysate/dt = k_pd * C_plasma * Vp / Vd - k_drain * C_dialysate

    During dwell (0 to dwell_time_h): k_drain = 0 (closed system).
    After dwell: k_drain = 1.0 (rapid drain).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg (absorbed instantaneously at t=0).
    dwell_time_h:
        Dwell time in hours (dialysate closed during this period).
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vp_L:
        Plasma/central volume in L.
    pd_mass_transfer_coeff_L_per_h:
        Peritoneal mass transfer coefficient in L/h.
    dialysate_volume_L:
        Volume of dialysate fill in L.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    PeritonealDialysisPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vp_L <= 0:
        raise ValueError(f"vp_L must be > 0, got {vp_L}")
    if dwell_time_h <= 0:
        raise ValueError(f"dwell_time_h must be > 0, got {dwell_time_h}")

    ke = cl_sys_L_per_h / vp_L
    k_pd = pd_mass_transfer_coeff_L_per_h / vp_L
    vd = dialysate_volume_L

    # IV bolus: initial plasma concentration
    c_plasma = dose_mg / vp_L
    c_dialysate = 0.0

    times: list = []
    c_plasma_list: list = []
    c_dialysate_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_dialysate_list.append(c_dialysate)

        # k_drain depends on whether we are in dwell phase
        k_drain = 0.0 if t < dwell_time_h else 1.0

        # ODEs (forward Euler)
        dcp = -ke * c_plasma - k_pd * c_plasma
        dcd = k_pd * c_plasma * vp_L / vd - k_drain * c_dialysate

        c_plasma = max(0.0, c_plasma + dcp * dt_h)
        c_dialysate = max(0.0, c_dialysate + dcd * dt_h)

        t = round(t + dt_h, 10)

    auc_plasma = _trapz(times, c_plasma_list)
    cmax_plasma = max(c_plasma_list)
    trough_plasma = c_plasma_list[-1]

    # Drug removed into dialysate at end of simulation
    drug_removed_mg = c_dialysate_list[-1] * vd

    # PD clearance: drug removed relative to plasma AUC
    if auc_plasma > 0.0:
        clearance_pd = drug_removed_mg / auc_plasma
    else:
        clearance_pd = 0.0

    notes = (
        f"ke={ke:.4f}/h, k_pd={k_pd:.4f}/h; "
        f"dwell_time={dwell_time_h}h; "
        f"AUC_plasma={auc_plasma:.3f} mg*h/L; "
        f"drug_removed={drug_removed_mg:.3f} mg"
    )

    return PeritonealDialysisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dwell_time_h=dwell_time_h,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_dialysate_mg_L=c_dialysate_list,
        cmax_plasma_mg_L=cmax_plasma,
        trough_plasma_mg_L=trough_plasma,
        drug_removed_mg=drug_removed_mg,
        clearance_pd_L_per_h=clearance_pd,
        auc_plasma_mg_h_per_L=auc_plasma,
        notes=notes,
    )


def compare_dwell_times(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vp_L: float = 3.0,
    dwell_times_h: list = None,
) -> list:
    """Compare drug removal across multiple dwell times.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vp_L:
        Plasma volume in L.
    dwell_times_h:
        List of dwell times to compare. Defaults to [2.0, 4.0, 6.0, 8.0].

    Returns
    -------
    list of PeritonealDialysisPKResult sorted by drug_removed_mg descending.
    """
    if dwell_times_h is None:
        dwell_times_h = [2.0, 4.0, 6.0, 8.0]

    results = []
    for dwell in dwell_times_h:
        result = simulate_peritoneal_dialysis_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            dwell_time_h=dwell,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vp_L=vp_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.drug_removed_mg, reverse=True)
    return results
