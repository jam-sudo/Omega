"""Drug efflux at the blood-brain barrier — Phase 171.

Two-compartment model (plasma + brain) with passive influx/efflux
and active P-gp-mediated efflux across the BBB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class BBBEffluxResult:
    """Result of BBB efflux simulation."""

    drug_name: str
    dose_mg: float
    pgp_inhibition_fraction: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_brain_mg_L: list[float]
    cmax_plasma: float
    cmax_brain: float
    auc_plasma: float
    auc_brain: float
    kp_brain: float
    pgp_contribution: float
    notes: list[str]


def simulate_bbb_efflux(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    k_bbb_in_per_h: float = 0.3,
    k_bbb_out_per_h: float = 0.1,
    k_pgp_efflux_per_h: float = 0.5,
    pgp_inhibition_fraction: float = 0.0,
    v_brain_L: float = 1.4,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> BBBEffluxResult:
    """Simulate drug transport across the BBB with active efflux.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg). Must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_plasma_L : float
        Volume of distribution for plasma (L). Must be > 0.
    k_bbb_in_per_h : float
        Passive BBB influx rate constant (1/h).
    k_bbb_out_per_h : float
        Passive BBB efflux rate constant (1/h).
    k_pgp_efflux_per_h : float
        P-gp active efflux rate constant (1/h).
    pgp_inhibition_fraction : float
        Fraction of P-gp inhibition (0-1).
    v_brain_L : float
        Brain volume (L).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    BBBEffluxResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if not 0.0 <= pgp_inhibition_fraction <= 1.0:
        raise ValueError("pgp_inhibition_fraction must be between 0 and 1")
    if v_brain_L <= 0:
        raise ValueError("v_brain_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")

    ke = cl_L_per_h / vd_plasma_L
    effective_pgp = k_pgp_efflux_per_h * (1.0 - pgp_inhibition_fraction)

    n_steps = int(math.ceil(t_end_h / dt_h))
    times: list[float] = []
    c_plasma: list[float] = []
    c_brain: list[float] = []

    cp = dose_mg / vd_plasma_L
    cb = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(round(t, 6))
        c_plasma.append(cp)
        c_brain.append(cb)

        influx = k_bbb_in_per_h * cp
        efflux = (k_bbb_out_per_h + effective_pgp) * cb

        dcp = (-ke * cp - k_bbb_in_per_h * cp * (v_brain_L / vd_plasma_L)
               + efflux * (v_brain_L / vd_plasma_L)) * dt_h
        dcb = (influx - efflux) * dt_h

        cp = max(cp + dcp, 0.0)
        cb = max(cb + dcb, 0.0)

    cmax_plasma = max(c_plasma)
    cmax_brain = max(c_brain)
    auc_plasma = float(np_trapz(c_plasma, times))
    auc_brain = float(np_trapz(c_brain, times))
    kp_brain = auc_brain / auc_plasma if auc_plasma > 0 else 0.0

    # Compute pgp_contribution: run without P-gp to get reference brain AUC
    cp_ref = dose_mg / vd_plasma_L
    cb_ref = 0.0
    c_brain_nopgp: list[float] = []
    times_ref: list[float] = []
    for i in range(n_steps + 1):
        t = i * dt_h
        times_ref.append(round(t, 6))
        c_brain_nopgp.append(cb_ref)

        influx_ref = k_bbb_in_per_h * cp_ref
        efflux_ref = k_bbb_out_per_h * cb_ref

        dcp_ref = (-ke * cp_ref - k_bbb_in_per_h * cp_ref * (v_brain_L / vd_plasma_L)
                   + efflux_ref * (v_brain_L / vd_plasma_L)) * dt_h
        dcb_ref = (influx_ref - efflux_ref) * dt_h

        cp_ref = max(cp_ref + dcp_ref, 0.0)
        cb_ref = max(cb_ref + dcb_ref, 0.0)

    auc_brain_nopgp = float(np_trapz(c_brain_nopgp, times_ref))
    pgp_contribution = (1.0 - auc_brain / auc_brain_nopgp) if auc_brain_nopgp > 0 else 0.0

    notes: list[str] = []
    if pgp_inhibition_fraction > 0:
        notes.append(f"P-gp inhibited by {pgp_inhibition_fraction * 100:.0f}%")
    if kp_brain < 0.1:
        notes.append("Low brain penetration (Kp,brain < 0.1)")
    if pgp_contribution > 0.5:
        notes.append("P-gp is a major barrier to brain entry")

    return BBBEffluxResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        pgp_inhibition_fraction=pgp_inhibition_fraction,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_brain_mg_L=c_brain,
        cmax_plasma=cmax_plasma,
        cmax_brain=cmax_brain,
        auc_plasma=auc_plasma,
        auc_brain=auc_brain,
        kp_brain=kp_brain,
        pgp_contribution=pgp_contribution,
        notes=notes,
    )


def compare_efflux_inhibition(
    drug_name: str,
    dose_mg: float,
    inhibition_fractions: list[float],
    cl_L_per_h: float,
    vd_plasma_L: float,
    k_bbb_in_per_h: float = 0.3,
    k_bbb_out_per_h: float = 0.1,
    k_pgp_efflux_per_h: float = 0.5,
    v_brain_L: float = 1.4,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list[BBBEffluxResult]:
    """Compare BBB efflux at multiple P-gp inhibition levels.

    Parameters
    ----------
    inhibition_fractions : list[float]
        List of pgp_inhibition_fraction values to compare.

    Returns
    -------
    list[BBBEffluxResult]
    """
    if not inhibition_fractions:
        raise ValueError("inhibition_fractions must not be empty")
    results: list[BBBEffluxResult] = []
    for frac in inhibition_fractions:
        r = simulate_bbb_efflux(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_plasma_L=vd_plasma_L,
            k_bbb_in_per_h=k_bbb_in_per_h,
            k_bbb_out_per_h=k_bbb_out_per_h,
            k_pgp_efflux_per_h=k_pgp_efflux_per_h,
            pgp_inhibition_fraction=frac,
            v_brain_L=v_brain_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)
    return results
