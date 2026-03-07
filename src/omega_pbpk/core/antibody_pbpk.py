"""Biologic/mAb PBPK model for antibody pharmacokinetics.

Two-compartment model (central plasma + peripheral tissue) with FcRn
neonatal Fc receptor recycling and optional target-mediated drug
disposition (TMDD) for monoclonal antibody therapeutics.

Model structure
---------------
1. Central (plasma) compartment — CL, FcRn recycling
2. Peripheral (tissue/interstitial) compartment — distribution
3. SC depot absorption (optional, for subcutaneous route)
4. TMDD target-receptor binding in central (optional)

References
----------
- Dua P et al., J Pharmacokinet Pharmacodyn. 2015;42:363–82
- Ng CM et al., J Pharmacokinet Pharmacodyn. 2006;33:627–44
- Gibiansky L et al., J Pharmacokinet Pharmacodyn. 2012;39:5–16
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class AntibodyPKResult:
    """Results from antibody/mAb PBPK simulation.

    Attributes
    ----------
    drug_name : Name of the antibody drug.
    route : Administration route ('iv' or 'sc').
    dose_mg : Dose administered (mg).
    mw_kDa : Molecular weight (kDa).
    times_day : Simulation time points (days).
    cc_mg_L : Central/plasma concentration (mg/L).
    ct_mg_L : Tissue concentration (mg/L).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_day : Time of peak plasma concentration (days).
    auc_0_t : AUC from 0 to t_end (mg*day/L).
    t_half_day : Estimated terminal half-life (days).
    sc_bioavailability : SC bioavailability fraction (1.0 for IV).
    tmdd_enabled : Whether TMDD was enabled.
    fcrn_recycling : FcRn recycling fraction used.
    notes : Summary text.
    """

    drug_name: str
    route: str
    dose_mg: float
    mw_kDa: float
    times_day: list[float]
    cc_mg_L: list[float]
    ct_mg_L: list[float]
    cmax_mg_L: float
    tmax_day: float
    auc_0_t: float
    t_half_day: float
    sc_bioavailability: float
    tmdd_enabled: bool
    fcrn_recycling: float
    notes: str


def _estimate_half_life(times: np.ndarray, conc: np.ndarray) -> float:
    """Estimate terminal half-life from last 30% of the concentration profile."""
    n = len(times)
    start = int(n * 0.7)
    t_tail = times[start:]
    c_tail = conc[start:]

    # Filter positive concentrations for log fitting
    mask = c_tail > 1e-15
    if np.sum(mask) < 2:
        return float("nan")

    t_fit = t_tail[mask]
    ln_c = np.log(c_tail[mask])

    # Linear regression: ln(C) = -ke*t + b
    n_pts = len(t_fit)
    sum_t = np.sum(t_fit)
    sum_lnc = np.sum(ln_c)
    sum_t2 = np.sum(t_fit**2)
    sum_t_lnc = np.sum(t_fit * ln_c)

    denom = n_pts * sum_t2 - sum_t**2
    if abs(denom) < 1e-30:
        return float("nan")

    slope = (n_pts * sum_t_lnc - sum_t * sum_lnc) / denom
    if slope >= 0:
        return float("nan")

    ke_app = -slope
    return float(np.log(2) / ke_app)


def simulate_antibody_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "iv",
    mw_kDa: float = 150.0,
    cl_L_per_day: float = 0.2,
    vd_central_L: float = 3.0,
    vd_tissue_L: float = 5.0,
    k12_per_day: float = 0.5,
    k21_per_day: float = 0.3,
    ka_sc_per_day: float = 0.5,
    f_sc: float = 0.7,
    fcrn_recycling: float = 0.7,
    tmdd: bool = False,
    r0_nM: float = 10.0,
    kon_per_nM_per_day: float = 0.1,
    koff_per_day: float = 1.0,
    kint_per_day: float = 0.1,
    t_end_day: float = 28.0,
    dt_day: float = 0.05,
) -> AntibodyPKResult:
    """Simulate antibody/mAb PK with 2-compartment model, FcRn recycling, and optional TMDD.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg).
    route : 'iv' (bolus) or 'sc' (subcutaneous depot).
    mw_kDa : Molecular weight in kDa (typical IgG1 ~150 kDa).
    cl_L_per_day : Linear clearance (L/day).
    vd_central_L : Central compartment volume (L).
    vd_tissue_L : Peripheral tissue volume (L).
    k12_per_day : Distribution rate central->tissue (1/day).
    k21_per_day : Redistribution rate tissue->central (1/day).
    ka_sc_per_day : SC depot absorption rate (1/day).
    f_sc : SC bioavailability fraction.
    fcrn_recycling : FcRn recycling fraction (0-1). Reduces effective CL.
    tmdd : Enable target-mediated drug disposition.
    r0_nM : Baseline target receptor concentration (nM).
    kon_per_nM_per_day : Association rate constant (1/(nM*day)).
    koff_per_day : Dissociation rate constant (1/day).
    kint_per_day : Drug-target complex internalization rate (1/day).
    t_end_day : Simulation duration (days).
    dt_day : Time step (days).

    Returns
    -------
    AntibodyPKResult

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_day <= 0:
        raise ValueError("cl_L_per_day must be > 0")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be > 0")
    if vd_tissue_L <= 0:
        raise ValueError("vd_tissue_L must be > 0")
    route_lower = route.lower()
    if route_lower not in ("iv", "sc"):
        raise ValueError(f"route must be 'iv' or 'sc', got '{route}'")

    # Effective clearance reduced by FcRn recycling
    cl_eff = cl_L_per_day * (1.0 - 0.5 * fcrn_recycling)

    n_steps = max(int(t_end_day / dt_day), 1)
    t = np.linspace(0.0, t_end_day, n_steps + 1)
    dt = t[1] - t[0] if len(t) > 1 else dt_day

    vc = vd_central_L
    vt = vd_tissue_L

    # State arrays
    cc = np.zeros(n_steps + 1)  # central conc (mg/L)
    ct = np.zeros(n_steps + 1)  # tissue conc (mg/L)
    a_sc = np.zeros(n_steps + 1)  # SC depot amount (mg)

    # TMDD state
    r_nM = np.zeros(n_steps + 1)  # free receptor (nM)
    rc_nM = np.zeros(n_steps + 1)  # drug-receptor complex (nM)

    # Initial conditions
    if route_lower == "iv":
        cc[0] = dose_mg / vc  # IV bolus at t=0
    else:
        a_sc[0] = dose_mg  # SC depot

    if tmdd:
        kdeg = 0.1  # baseline receptor degradation rate (1/day)
        ksyn = r0_nM * kdeg  # synthesis rate to maintain steady-state
        r_nM[0] = r0_nM

    # Forward Euler integration
    for i in range(n_steps):
        # SC depot absorption
        sc_input = 0.0
        if route_lower == "sc":
            sc_abs = ka_sc_per_day * a_sc[i]
            sc_input = sc_abs * f_sc
            a_sc[i + 1] = max(0.0, a_sc[i] - sc_abs * dt)

        # TMDD contribution
        tmdd_loss_mg_per_day = 0.0
        if tmdd:
            # Convert central conc to nM for TMDD math
            cc_nM = cc[i] * 1e6 / mw_kDa

            dr = (
                ksyn
                - kdeg * r_nM[i]
                - kon_per_nM_per_day * cc_nM * r_nM[i]
                + koff_per_day * rc_nM[i]
            )
            drc = kon_per_nM_per_day * cc_nM * r_nM[i] - (koff_per_day + kint_per_day) * rc_nM[i]

            r_nM[i + 1] = max(0.0, r_nM[i] + dr * dt)
            rc_nM[i + 1] = max(0.0, rc_nM[i] + drc * dt)

            # Drug loss from complex internalization (nM -> mg)
            tmdd_loss_mg_per_day = kint_per_day * rc_nM[i] * mw_kDa / 1e6 * vc

        # Central compartment ODE
        dcc = (
            sc_input
            - cl_eff * cc[i]
            - k12_per_day * vc * cc[i]
            + k21_per_day * vt * ct[i]
            - tmdd_loss_mg_per_day
        ) / vc
        cc[i + 1] = max(0.0, cc[i] + dcc * dt)

        # Tissue compartment ODE
        dct = (k12_per_day * vc * cc[i] - k21_per_day * vt * ct[i]) / vt
        ct[i + 1] = max(0.0, ct[i] + dct * dt)

    # PK metrics
    cmax = float(np.max(cc))
    tmax = float(t[int(np.argmax(cc))])
    auc = float(np_trapz(cc, t))
    t_half = _estimate_half_life(t, cc)
    sc_bio = f_sc if route_lower == "sc" else 1.0

    notes = (
        f"Antibody PBPK: {drug_name}, {dose_mg}mg {route_lower.upper()}. "
        f"FcRn recycling={fcrn_recycling:.0%}, TMDD={'on' if tmdd else 'off'}. "
        f"Cmax={cmax:.4f} mg/L at day {tmax:.2f}. "
        f"t½={t_half:.1f} days. AUC={auc:.3f} mg·day/L."
    )

    return AntibodyPKResult(
        drug_name=drug_name,
        route=route_lower,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        times_day=t.tolist(),
        cc_mg_L=cc.tolist(),
        ct_mg_L=ct.tolist(),
        cmax_mg_L=cmax,
        tmax_day=tmax,
        auc_0_t=auc,
        t_half_day=t_half,
        sc_bioavailability=sc_bio,
        tmdd_enabled=tmdd,
        fcrn_recycling=fcrn_recycling,
        notes=notes,
    )


def compare_sc_iv(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> tuple[AntibodyPKResult, AntibodyPKResult]:
    """Compare SC vs IV administration for the same antibody.

    Returns
    -------
    Tuple of (SC result, IV result).
    """
    sc_result = simulate_antibody_pk(drug_name, dose_mg, route="sc", **kwargs)
    iv_result = simulate_antibody_pk(drug_name, dose_mg, route="iv", **kwargs)
    return (sc_result, iv_result)


__all__ = [
    "AntibodyPKResult",
    "simulate_antibody_pk",
    "compare_sc_iv",
]
