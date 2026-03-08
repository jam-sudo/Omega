"""Phase 516 — Cerebrospinal Fluid Drug Dynamics (Lumbar).

Model drug dynamics in the lumbar cistern with CSF bulk flow and
choroid plexus secretion.  2-compartment model: plasma and lumbar CSF.
Forward Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CSFLumbarDynamicsResult:
    """Result of lumbar CSF drug dynamics simulation."""

    drug_name: str
    dose_mg: float
    ps_csf_L_per_h: float

    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_csf_mg_L: list = field(default_factory=list)

    cmax_plasma: float = 0.0
    tmax_plasma_h: float = 0.0
    auc_plasma: float = 0.0

    cmax_csf: float = 0.0
    tmax_csf_h: float = 0.0
    auc_csf: float = 0.0

    csf_plasma_auc_ratio: float = 0.0
    csf_penetration_pct: float = 0.0

    t_half_plasma_h: float = 0.0
    t_half_csf_h: float = 0.0

    notes: str = ""


# ---------------------------------------------------------------------------
# Helper: trapezoidal AUC
# ---------------------------------------------------------------------------


def _trapz(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_csf_lumbar_dynamics(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    ps_csf_L_per_h: float = 0.05,
    v_csf_L: float = 0.03,
    qcsf_L_per_h: float = 0.021,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> CSFLumbarDynamicsResult:
    """Simulate lumbar CSF drug dynamics.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Plasma clearance (L/h).
    vd_plasma_L:
        Plasma volume of distribution (L).
    ps_csf_L_per_h:
        Permeability-surface area product for CSF (L/h).
    v_csf_L:
        Lumbar CSF volume (L); default 0.03 L = 30 mL.
    qcsf_L_per_h:
        CSF bulk flow rate (L/h); default 0.021 L/h = 21 mL/h = 0.35 mL/min.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    CSFLumbarDynamicsResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be positive")
    if ps_csf_L_per_h <= 0:
        raise ValueError("ps_csf_L_per_h must be positive")

    # Rate constants — work in amounts (mg) to avoid unit inconsistencies.
    # Plasma: A_plasma (mg),  C_plasma = A_plasma / Vd_plasma
    # CSF:    A_csf   (mg),  C_csf   = A_csf   / V_csf
    #
    # Fluxes (mg/h):
    #   Plasma → CSF (passive): PS * C_plasma
    #   CSF → plasma (back):    PS * C_csf
    #   CSF → drain  (flow):    Qcsf * C_csf
    #   Plasma elimination:     CL * C_plasma
    #
    # Amount ODEs:
    #   dA_plasma/dt = -CL*C_plasma - PS*C_plasma + PS*C_csf
    #   dA_csf/dt   =  PS*C_plasma  - PS*C_csf    - Qcsf*C_csf

    # Initial conditions (amounts, mg)
    a_plasma = dose_mg
    a_csf = 0.0

    n_steps = int(round(t_end_h / dt_h)) + 1
    times: list = []
    cp_ts: list = []
    cc_ts: list = []

    t = 0.0
    for _ in range(n_steps):
        c_p = a_plasma / vd_plasma_L
        c_c = a_csf / v_csf_L
        times.append(t)
        cp_ts.append(c_p)
        cc_ts.append(c_c)

        # Fluxes (mg/h)
        elim_flux = cl_L_per_h * c_p
        plasma_to_csf = ps_csf_L_per_h * c_p
        csf_to_plasma = ps_csf_L_per_h * c_c
        csf_drain = qcsf_L_per_h * c_c

        # Forward Euler on amounts
        da_plasma = -elim_flux - plasma_to_csf + csf_to_plasma
        da_csf = plasma_to_csf - csf_to_plasma - csf_drain

        a_plasma = max(0.0, a_plasma + da_plasma * dt_h)
        a_csf = max(0.0, a_csf + da_csf * dt_h)

        t += dt_h

    # PK metrics
    cmax_plasma = max(cp_ts)
    tmax_plasma_h = times[cp_ts.index(cmax_plasma)]
    auc_plasma = _trapz(times, cp_ts)

    cmax_csf = max(cc_ts)
    tmax_csf_h = times[cc_ts.index(cmax_csf)]
    auc_csf = _trapz(times, cc_ts)

    csf_plasma_auc_ratio = auc_csf / auc_plasma if auc_plasma > 0 else 0.0
    csf_penetration_pct = (cmax_csf / cmax_plasma * 100.0) if cmax_plasma > 0 else 0.0

    # Plasma t½: time for plasma to drop to 50% of its Cmax after peak
    t_half_plasma_h = 0.0
    idx_plasma_peak = cp_ts.index(cmax_plasma)
    half_plasma = cmax_plasma * 0.5
    for i in range(idx_plasma_peak + 1, len(cp_ts)):
        if cp_ts[i] <= half_plasma:
            frac = (cp_ts[i - 1] - half_plasma) / (cp_ts[i - 1] - cp_ts[i] + 1e-30)
            t_half_plasma_h = times[i - 1] + frac * (times[i] - times[i - 1]) - tmax_plasma_h
            break

    # CSF t½: time for CSF to drop to 50% of its own Cmax after CSF peak
    t_half_csf_h = 0.0
    if cmax_csf > 0:
        idx_csf_peak = cc_ts.index(cmax_csf)
        half_csf = cmax_csf * 0.5
        for i in range(idx_csf_peak + 1, len(cc_ts)):
            if cc_ts[i] <= half_csf:
                frac = (cc_ts[i - 1] - half_csf) / (cc_ts[i - 1] - cc_ts[i] + 1e-30)
                t_half_csf_h = times[i - 1] + frac * (times[i] - times[i - 1]) - tmax_csf_h
                break

    notes = (
        f"ps_csf={ps_csf_L_per_h:.4f} L/h; "
        f"csf_penetration={csf_penetration_pct:.1f}%; "
        f"auc_ratio={csf_plasma_auc_ratio:.4f}"
    )

    return CSFLumbarDynamicsResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ps_csf_L_per_h=ps_csf_L_per_h,
        times_h=times,
        c_plasma_mg_L=cp_ts,
        c_csf_mg_L=cc_ts,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_csf=cmax_csf,
        tmax_csf_h=tmax_csf_h,
        auc_csf=auc_csf,
        csf_plasma_auc_ratio=csf_plasma_auc_ratio,
        csf_penetration_pct=csf_penetration_pct,
        t_half_plasma_h=t_half_plasma_h,
        t_half_csf_h=t_half_csf_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_csf_penetration(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    ps_values_L_per_h: list,
) -> list:
    """Compare CSF penetration across different PS values.

    Returns a list of :class:`CSFLumbarDynamicsResult` sorted by
    *csf_plasma_auc_ratio* in descending order.
    """
    results = [
        simulate_csf_lumbar_dynamics(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_plasma_L=vd_plasma_L,
            ps_csf_L_per_h=ps,
        )
        for ps in ps_values_L_per_h
    ]
    results.sort(key=lambda r: r.csf_plasma_auc_ratio, reverse=True)
    return results
