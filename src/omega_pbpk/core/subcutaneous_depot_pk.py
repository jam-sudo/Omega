"""Subcutaneous depot injection PK model (Phase 776).

Models the SC injection site depot with 3-compartment PK:
  1. SC depot   — first-order absorption (ka_sc)
  2. Central (plasma) — CL + inter-compartmental flow Q12
  3. Peripheral (tissue) — Q12 transfer from/to central

Typical use case: biologics, peptides, long-acting formulations
with slow SC absorption (ka 0.01–0.1/h).

References
----------
- Gibiansky L & Gibiansky E, AAPS J. 2009;11(4):553-7 (SC mAb PK)
- Zhao L et al., Clin Pharmacokinet. 2013;52(12):1039-49 (SC biologic PK)
- Dirks NL & Meibohm B, Clin Pharmacokinet. 2010;49(10):633-59 (biologic PK)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SCDepotPKResult:
    """Results from subcutaneous depot PK simulation.

    Attributes
    ----------
    drug_name : str — drug name.
    dose_mg : float — SC dose (mg).
    ka_sc_per_h : float — SC absorption rate constant (h^-1).
    f_sc : float — SC bioavailability fraction.
    times_h : list — simulation time points (h).
    a_depot_mg : list — SC depot drug amount (mg).
    c_central_mg_L : list — central compartment concentration (mg/L).
    c_peripheral_mg_L : list — peripheral compartment concentration (mg/L).
    cmax_central : float — peak central concentration (mg/L).
    tmax_central_h : float — time of Cmax in central (h).
    auc_central : float — AUC in central compartment (mg*h/L).
    t_half_terminal_h : float — terminal half-life from log-linear regression (h).
    accumulation_ratio : float — steady-state accumulation ratio estimate.
    notes : str — summary.
    """

    drug_name: str
    dose_mg: float
    ka_sc_per_h: float
    f_sc: float
    times_h: list = field(default_factory=list)
    a_depot_mg: list = field(default_factory=list)
    c_central_mg_L: list = field(default_factory=list)
    c_peripheral_mg_L: list = field(default_factory=list)
    cmax_central: float = 0.0
    tmax_central_h: float = 0.0
    auc_central: float = 0.0
    t_half_terminal_h: float = 0.0
    accumulation_ratio: float = 1.0
    notes: str = ""


def _trap_auc(times: list, conc: list) -> float:
    """Trapezoidal AUC from lists."""
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (conc[i] + conc[i + 1]) * (times[i + 1] - times[i])
    return auc


def _log_linear_half_life(times: list, conc: list, tail_fraction: float = 0.30) -> float:
    """Estimate terminal half-life by log-linear regression on last tail_fraction of profile.

    Uses manual least-squares on ln(C) vs t for points where C > 0.
    Returns nan if insufficient data.
    """
    n = len(times)
    start = int(n * (1.0 - tail_fraction))
    t_tail = times[start:]
    c_tail = conc[start:]

    # Filter positive concentrations
    pairs = [(t, c) for t, c in zip(t_tail, c_tail) if c > 1e-15]
    if len(pairs) < 2:
        return float("nan")

    ln_c = [math.log(c) for _, c in pairs]
    t_vals = [t for t, _ in pairs]

    n_pts = len(t_vals)
    sum_t = sum(t_vals)
    sum_ln = sum(ln_c)
    sum_t2 = sum(t * t for t in t_vals)
    sum_t_ln = sum(t * lc for t, lc in zip(t_vals, ln_c))

    denom = n_pts * sum_t2 - sum_t * sum_t
    if abs(denom) < 1e-30:
        return float("nan")

    slope = (n_pts * sum_t_ln - sum_t * sum_ln) / denom
    if slope >= 0:
        # No terminal decline observed — use a fallback
        return float("nan")

    return math.log(2.0) / (-slope)


def simulate_sc_depot_pk(
    drug_name: str,
    dose_mg: float,
    ka_sc_per_h: float = 0.05,
    cl_L_per_h: float = 0.5,
    vd_central_L: float = 3.0,
    vd_peripheral_L: float = 5.0,
    q12_L_per_h: float = 0.2,
    f_sc: float = 0.75,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> SCDepotPKResult:
    """Simulate subcutaneous depot injection PK.

    3-compartment model: SC depot -> central plasma -> peripheral tissue.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : SC dose (mg).
    ka_sc_per_h : SC absorption rate constant (h^-1).
    cl_L_per_h : Central clearance (L/h).
    vd_central_L : Central volume of distribution (L).
    vd_peripheral_L : Peripheral volume of distribution (L).
    q12_L_per_h : Inter-compartmental clearance central<->peripheral (L/h).
    f_sc : SC bioavailability fraction (0–1).
    t_end_h : Simulation duration (h).
    dt_h : Forward Euler time step (h).

    Returns
    -------
    SCDepotPKResult

    Raises
    ------
    ValueError : On invalid parameters.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_sc_per_h <= 0:
        raise ValueError(f"ka_sc_per_h must be > 0, got {ka_sc_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_central_L <= 0:
        raise ValueError(f"vd_central_L must be > 0, got {vd_central_L}")
    if vd_peripheral_L <= 0:
        raise ValueError(f"vd_peripheral_L must be > 0, got {vd_peripheral_L}")
    if not 0.0 < f_sc <= 1.0:
        raise ValueError(f"f_sc must be in (0, 1], got {f_sc}")

    n_steps = max(int(t_end_h / dt_h), 1)

    times: list = [0.0] * (n_steps + 1)
    a_depot: list = [0.0] * (n_steps + 1)
    c_central: list = [0.0] * (n_steps + 1)
    c_peri: list = [0.0] * (n_steps + 1)

    # Initial depot amount (only f_sc fraction is bioavailable)
    a_depot[0] = dose_mg  # total dose in depot; f_sc applied at absorption step
    cc = 0.0  # central amount (mg)
    cp = 0.0  # peripheral amount (mg)

    for i in range(n_steps):
        times[i + 1] = (i + 1) * dt_h

        # Concentrations for transfer
        cc_conc = cc / vd_central_L
        cp_conc = cp / vd_peripheral_L

        # SC depot absorption (first-order, f_sc applied to absorbed fraction)
        abs_rate = ka_sc_per_h * a_depot[i] * f_sc

        # Transfer between central and peripheral (concentration-driven)
        q_cp = q12_L_per_h * cc_conc  # central -> peripheral flux (mg/h)
        q_pc = q12_L_per_h * cp_conc  # peripheral -> central flux (mg/h)

        # Elimination from central
        elim = cl_L_per_h * cc_conc  # mg/h

        # ODEs
        da_depot = -ka_sc_per_h * a_depot[i]
        d_cc = abs_rate - elim - q_cp + q_pc
        d_cp = q_cp - q_pc

        a_depot[i + 1] = max(0.0, a_depot[i] + da_depot * dt_h)
        cc = max(0.0, cc + d_cc * dt_h)
        cp = max(0.0, cp + d_cp * dt_h)

        c_central[i + 1] = cc / vd_central_L
        c_peri[i + 1] = cp / vd_peripheral_L

    # PK metrics
    cmax = max(c_central)
    tmax = times[c_central.index(cmax)]
    auc = _trap_auc(times, c_central)
    t_half = _log_linear_half_life(times, c_central, tail_fraction=0.30)
    if math.isnan(t_half) or t_half <= 0:
        # Fallback: use ke = CL/Vd approximation
        ke_approx = cl_L_per_h / vd_central_L
        t_half = math.log(2.0) / ke_approx

    # Accumulation ratio estimate for repeat dosing at tau = t_half
    tau_default = t_half if t_half > 0 else 1.0
    accum_ratio = (
        1.0 / (1.0 - math.exp(-math.log(2.0) * tau_default / t_half)) if t_half > 0 else 1.0
    )
    accum_ratio = min(max(accum_ratio, 1.0), 10.0)

    notes = (
        f"SC depot PK: {drug_name}, dose={dose_mg} mg. "
        f"ka_sc={ka_sc_per_h:.4f}/h, CL={cl_L_per_h:.2f} L/h, "
        f"Vc={vd_central_L:.1f} L, Vp={vd_peripheral_L:.1f} L, Q12={q12_L_per_h:.2f} L/h. "
        f"F_sc={f_sc:.2f}. "
        f"Cmax={cmax:.4f} mg/L at Tmax={tmax:.1f} h. AUC={auc:.3f} mg*h/L. "
        f"t1/2={t_half:.1f} h, accumulation ratio={accum_ratio:.2f}."
    )

    return SCDepotPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_sc_per_h=ka_sc_per_h,
        f_sc=f_sc,
        times_h=times,
        a_depot_mg=a_depot,
        c_central_mg_L=c_central,
        c_peripheral_mg_L=c_peri,
        cmax_central=cmax,
        tmax_central_h=tmax,
        auc_central=auc,
        t_half_terminal_h=t_half,
        accumulation_ratio=accum_ratio,
        notes=notes,
    )


def estimate_repeat_dose_accumulation(
    drug_name: str,
    dose_mg: float,
    tau_h: float,
    t_half_h: float,
) -> dict:
    """Estimate accumulation at steady state for repeat dosing.

    accumulation_ratio = 1 / (1 - exp(-ln(2) * tau_h / t_half_h))
    Clamped to [1, 10].

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Single dose (mg).
    tau_h : Dosing interval (h).
    t_half_h : Terminal half-life (h).

    Returns
    -------
    dict with keys: drug_name, dose_mg, tau_h, t_half_h,
    accumulation_ratio, css_max_estimate_mg_L, notes.

    Raises
    ------
    ValueError : If tau_h or t_half_h <= 0.
    """
    if tau_h <= 0:
        raise ValueError(f"tau_h must be > 0, got {tau_h}")
    if t_half_h <= 0:
        raise ValueError(f"t_half_h must be > 0, got {t_half_h}")

    exponent = math.log(2.0) * tau_h / t_half_h
    raw_ratio = 1.0 / (1.0 - math.exp(-exponent))
    accumulation_ratio = min(max(raw_ratio, 1.0), 10.0)

    # Rough Css_max estimate (informational only; actual depends on PK)
    css_max_estimate = accumulation_ratio * dose_mg

    notes = (
        f"Repeat dose accumulation: {drug_name}, dose={dose_mg} mg, "
        f"tau={tau_h} h, t1/2={t_half_h} h. "
        f"Accumulation ratio={accumulation_ratio:.3f} "
        f"(clamped to [1, 10])."
    )

    return {
        "drug_name": drug_name,
        "dose_mg": dose_mg,
        "tau_h": tau_h,
        "t_half_h": t_half_h,
        "accumulation_ratio": accumulation_ratio,
        "css_max_estimate_mg_L": css_max_estimate,
        "notes": notes,
    }


__all__ = [
    "SCDepotPKResult",
    "simulate_sc_depot_pk",
    "estimate_repeat_dose_accumulation",
]
