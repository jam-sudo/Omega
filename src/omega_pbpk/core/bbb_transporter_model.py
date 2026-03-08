"""Phase 353 — Drug Permeation Through Blood-Brain Barrier Transporter Model.

Models passive + transporter-mediated drug transport across the BBB using
a 2-compartment (plasma + brain) Forward Euler ODE system.
"""

from __future__ import annotations

from dataclasses import dataclass

V_BRAIN_L = 1.4  # human brain volume, L
BRAIN_SURFACE_CM2 = 150.0  # BBB surface area, cm^2


def _passive_cl(logP: float, psa: float, mw: float) -> float:
    """Estimate passive BBB clearance (L/h) via Clark 2003."""
    log_ppas = 0.258 * logP - 0.011 * psa - 0.00044 * mw - 2.5
    ppas_cm_per_s = 10.0**log_ppas
    # Convert cm/s * cm^2 -> L/h: (cm^3/s) * 3600 / 1000
    cl = ppas_cm_per_s * BRAIN_SURFACE_CM2 * 3600.0 / 1000.0
    return cl


def _auc_trap(xs: list[float], ys: list[float]) -> float:
    return sum(0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1]) for i in range(1, len(xs)))


@dataclass
class BBBTransporterResult:
    """Result of BBB transporter simulation."""

    times_h: list
    c_plasma_mg_L: list
    c_brain_mg_L: list
    auc_plasma_mg_h_per_L: float
    auc_brain_mg_h_per_L: float
    Kp_brain: float
    cmax_brain_mg_L: float
    tmax_brain_h: float
    passive_cl_L_per_h: float
    net_bbb_in_cl_L_per_h: float
    bbb_penetration_class: str
    transporter_impact: str
    notes: str


def simulate_bbb_transport(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    psa_angstrom2: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    pgp_substrate: bool = False,
    bcrp_substrate: bool = False,
    oatp1a4_substrate: bool = False,
    pgp_efflux_cl_L_per_h: float = 2.0,
    bcrp_efflux_cl_L_per_h: float = 1.0,
    oatp_influx_cl_L_per_h: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> BBBTransporterResult:
    """Simulate BBB drug transport with passive + transporter-mediated clearances.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose in mg.
    logP:
        Lipophilicity (log octanol-water partition coefficient).
    mw_Da:
        Molecular weight in Da.
    psa_angstrom2:
        Polar surface area in Angstrom^2.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    pgp_substrate:
        Whether drug is a P-gp efflux substrate.
    bcrp_substrate:
        Whether drug is a BCRP efflux substrate.
    oatp1a4_substrate:
        Whether drug is an OATP1A4 influx substrate.
    pgp_efflux_cl_L_per_h:
        P-gp efflux clearance (L/h), default 2.0.
    bcrp_efflux_cl_L_per_h:
        BCRP efflux clearance (L/h), default 1.0.
    oatp_influx_cl_L_per_h:
        OATP1A4 influx clearance (L/h), default 0.5.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    BBBTransporterResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # --- Clearance calculations ---
    cl_passive = _passive_cl(logP, psa_angstrom2, mw_Da)
    cl_efflux = (pgp_efflux_cl_L_per_h if pgp_substrate else 0.0) + (
        bcrp_efflux_cl_L_per_h if bcrp_substrate else 0.0
    )
    cl_influx = oatp_influx_cl_L_per_h if oatp1a4_substrate else 0.0
    cl_bbb_in = max(cl_passive + cl_influx - cl_efflux, 0.001)
    cl_bbb_out = cl_passive  # passive-only outward

    # --- Forward Euler simulation ---
    n_steps = max(int(t_end_h / dt_h), 1)
    times_h: list[float] = []
    c_plasma: list[float] = []
    c_brain: list[float] = []

    cp = dose_mg / vd_sys_L
    cb = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_plasma.append(cp)
        c_brain.append(cb)

        if i < n_steps:
            dcp = (
                -(cl_sys_L_per_h / vd_sys_L + cl_bbb_in / vd_sys_L) * cp
                + (cl_bbb_out / vd_sys_L) * cb
            )
            dcb = (cl_bbb_in / V_BRAIN_L) * cp - (cl_bbb_out / V_BRAIN_L) * cb
            cp = max(cp + dcp * dt_h, 0.0)
            cb = max(cb + dcb * dt_h, 0.0)

    # --- Derived metrics ---
    auc_plasma = _auc_trap(times_h, c_plasma)
    auc_brain = _auc_trap(times_h, c_brain)

    cmax_brain = max(c_brain)
    tmax_brain = times_h[c_brain.index(cmax_brain)]

    # Kp_brain: average ratio over last 10% of simulation
    n_tail = max(int(0.1 * len(times_h)), 1)
    tail_plasma = c_plasma[-n_tail:]
    tail_brain = c_brain[-n_tail:]
    kp_vals = [b / p for b, p in zip(tail_brain, tail_plasma) if p > 1e-12]
    kp_brain = sum(kp_vals) / len(kp_vals) if kp_vals else 0.0

    # Classifications
    if kp_brain > 0.3:
        penetration_class = "good"
    elif kp_brain >= 0.1:
        penetration_class = "moderate"
    else:
        penetration_class = "poor"

    if cl_efflux > cl_passive:
        transporter_impact = "efflux_limited"
    elif cl_influx > 0.0:
        transporter_impact = "influx_enhanced"
    else:
        transporter_impact = "passive"

    notes_parts: list[str] = [
        f"Drug: {drug_name}",
        f"Passive CL: {cl_passive:.4f} L/h",
        f"Efflux CL: {cl_efflux:.2f} L/h",
        f"Influx CL: {cl_influx:.2f} L/h",
        f"Net BBB-in CL: {cl_bbb_in:.4f} L/h",
    ]
    notes = "; ".join(notes_parts)

    return BBBTransporterResult(
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        c_brain_mg_L=c_brain,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_brain_mg_h_per_L=auc_brain,
        Kp_brain=kp_brain,
        cmax_brain_mg_L=cmax_brain,
        tmax_brain_h=tmax_brain,
        passive_cl_L_per_h=cl_passive,
        net_bbb_in_cl_L_per_h=cl_bbb_in,
        bbb_penetration_class=penetration_class,
        transporter_impact=transporter_impact,
        notes=notes,
    )
