"""Rectal drug absorption pharmacokinetics (Phase 663).

Models rectal drug absorption via suppositories or enemas. The rectal route
partially bypasses hepatic first-pass metabolism because the inferior and middle
hemorrhoidal veins drain directly into the systemic circulation (~50% bypass),
while the superior hemorrhoidal vein drains into the portal circulation and is
subject to hepatic first-pass.

Model structure
---------------
Two-path absorption (Forward Euler ODE):
  1. Rectal compartment -> systemic (inferior/middle hemorrhoidal, bypass fraction)
  2. Rectal compartment -> portal -> liver first-pass -> systemic

ODEs:
  dA_rectal/dt = -ka * A_rectal
  dC_plasma/dt = ka * A_rectal * (f_bypass + (1-f_bypass)*fh) / Vd - ke * C_plasma

where:
  f_bypass  = fraction bypassing hepatic first-pass (default 0.5)
  fh        = hepatic bioavailability (1 - extraction ratio)
  ke        = CL / Vd

References
----------
- de Boer AG et al., Clin Pharmacokinet. 1982;7(4):285-311 (rectal drug delivery)
- van Hoogdalem E et al., Clin Pharmacokinet. 1991;21(1):11-26 (rectal absorption)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RectalAbsorptionResult:
    """Result of a rectal absorption PK simulation."""

    drug_name: str
    dose_mg: float
    f_bypass: float
    times_h: list
    a_rectal_mg: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    f_systemic: float
    f_hepatic_bypass: float
    vs_oral_auc_ratio: float
    notes: str


def _validate_simulate_inputs(
    dose_mg: float,
    f_bypass: float,
    fh: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_rectal_per_h: float,
) -> None:
    """Validate inputs for simulate_rectal_pk."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < f_bypass < 1):
        raise ValueError(f"f_bypass must be in (0, 1), got {f_bypass}")
    if not (0 < fh <= 1):
        raise ValueError(f"fh must be in (0, 1], got {fh}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if ka_rectal_per_h <= 0:
        raise ValueError(f"ka_rectal_per_h must be > 0, got {ka_rectal_per_h}")


def _simulate_oral_auc(
    dose_mg: float,
    ka_oral_per_h: float,
    f_oral: float,
    fh: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> float:
    """Simulate a simple oral 1-cpt PK and return AUC."""
    ke = cl_L_per_h / vd_L
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    # Oral: dC/dt = ka * f * dose_mg * exp(-ka*t) / Vd - ke * C
    # A_gut(0) = dose_mg * f_oral * fh  (available after first-pass)
    a_gut = dose_mg * f_oral * fh
    c = 0.0
    times = [0.0]
    concs = [0.0]

    for _ in range(n_steps):
        dc = ka_oral_per_h * a_gut / vd_L - ke * c
        da = -ka_oral_per_h * a_gut
        c = max(c + dc * dt_h, 0.0)
        a_gut = max(a_gut + da * dt_h, 0.0)
        times.append(times[-1] + dt_h)
        concs.append(c)

    # Trapezoidal AUC
    auc = sum(
        (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i]) for i in range(len(concs) - 1)
    )
    return auc


def simulate_rectal_pk(
    drug_name: str,
    dose_mg: float,
    ka_rectal_per_h: float = 1.5,
    f_bypass: float = 0.5,
    fh: float = 0.7,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> RectalAbsorptionResult:
    """Simulate rectal drug absorption pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Rectal dose (mg).
    ka_rectal_per_h:
        First-order rectal absorption rate constant (1/h). Rectal solutions
        absorb faster than oral (~1.5/h vs ~1.0/h typical).
    f_bypass:
        Fraction of absorbed drug bypassing hepatic first-pass via inferior/middle
        hemorrhoidal veins (default 0.5, i.e., 50%).
    fh:
        Hepatic bioavailability for the portal-drained fraction (1 - extraction
        ratio). Default 0.7.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler integration step size (h).

    Returns
    -------
    RectalAbsorptionResult

    Raises
    ------
    ValueError
        If any parameter is outside valid range.
    """
    _validate_simulate_inputs(dose_mg, f_bypass, fh, cl_L_per_h, vd_L, ka_rectal_per_h)

    ke = cl_L_per_h / vd_L

    # Effective fraction reaching systemic circulation
    f_systemic = f_bypass + (1.0 - f_bypass) * fh

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    # State variables
    a_rectal = dose_mg  # mg remaining in rectal compartment
    c_plasma = 0.0  # mg/L plasma concentration

    times_h: list[float] = [0.0]
    a_rectal_list: list[float] = [a_rectal]
    c_plasma_list: list[float] = [c_plasma]

    for _ in range(n_steps):
        # Rate of absorption into systemic circulation (combined bypass + portal paths)
        abs_rate_mg_per_h = ka_rectal_per_h * a_rectal * f_systemic

        # ODE: dC/dt = absorption_rate / Vd - ke * C
        dc = abs_rate_mg_per_h / vd_L - ke * c_plasma
        # ODE: dA_rectal/dt = -ka * A_rectal
        da = -ka_rectal_per_h * a_rectal

        c_plasma = max(c_plasma + dc * dt_h, 0.0)
        a_rectal = max(a_rectal + da * dt_h, 0.0)

        times_h.append(times_h[-1] + dt_h)
        a_rectal_list.append(a_rectal)
        c_plasma_list.append(c_plasma)

    # PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]

    # Trapezoidal AUC
    auc = sum(
        (c_plasma_list[i] + c_plasma_list[i + 1]) / 2.0 * (times_h[i + 1] - times_h[i])
        for i in range(len(c_plasma_list) - 1)
    )

    # Terminal half-life from ke
    t_half_h = math.log(2.0) / ke

    # Compute oral AUC for comparison (same dose, ka_rectal as proxy oral ka, f_oral=1, fh)
    oral_auc = _simulate_oral_auc(
        dose_mg=dose_mg,
        ka_oral_per_h=ka_rectal_per_h,
        f_oral=1.0,
        fh=fh,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    vs_oral_auc_ratio = auc / oral_auc if oral_auc > 0.0 else float("inf")

    # Notes
    bypass_pct = f_bypass * 100.0
    hepatic_extraction = (1.0 - fh) * 100.0
    notes = (
        f"Rectal PK: {bypass_pct:.0f}% of absorbed drug bypasses hepatic first-pass. "
        f"Hepatic extraction = {hepatic_extraction:.0f}% (fh={fh:.2f}). "
        f"Effective systemic bioavailability = {f_systemic * 100:.1f}%. "
        f"Rectal/oral AUC ratio = {vs_oral_auc_ratio:.2f}."
    )

    return RectalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_bypass=f_bypass,
        times_h=times_h,
        a_rectal_mg=a_rectal_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_h=t_half_h,
        f_systemic=f_systemic,
        f_hepatic_bypass=f_bypass,
        vs_oral_auc_ratio=vs_oral_auc_ratio,
        notes=notes,
    )


def compare_rectal_vs_oral(
    drug_name: str,
    dose_mg: float,
    ka_oral_per_h: float = 1.0,
    ka_rectal_per_h: float = 1.5,
    f_oral: float = 0.8,
    f_bypass: float = 0.5,
    fh: float = 0.7,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> dict:
    """Compare rectal vs oral pharmacokinetics for a drug at the same dose.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose in mg (same for both routes).
    ka_oral_per_h:
        Oral absorption rate constant (1/h).
    ka_rectal_per_h:
        Rectal absorption rate constant (1/h).
    f_oral:
        Oral fraction absorbed (pre-hepatic bioavailability).
    f_bypass:
        Rectal bypass fraction (fraction avoiding hepatic first-pass).
    fh:
        Hepatic bioavailability (1 - hepatic extraction ratio).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).

    Returns
    -------
    dict with keys:
        rectal_result : RectalAbsorptionResult
        oral_result   : RectalAbsorptionResult (simulated with oral parameters)
        auc_ratio     : float (rectal AUC / oral AUC)
        cmax_ratio    : float (rectal Cmax / oral Cmax)
        notes         : str
    """
    t_end_h = 24.0
    dt_h = 0.05

    # Rectal simulation
    rectal = simulate_rectal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_rectal_per_h=ka_rectal_per_h,
        f_bypass=f_bypass,
        fh=fh,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Oral simulation: same ODE structure, f_bypass=0 (all portal), use f_oral*fh effective
    # We use f_bypass approaching 0 and set effective fh = f_oral * fh
    # For simplicity, simulate oral as 1-cpt with effective F = f_oral * fh
    ke = cl_L_per_h / vd_L
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    a_gut = dose_mg * f_oral * fh  # effective oral drug available
    c = 0.0
    oral_times: list[float] = [0.0]
    oral_concs: list[float] = [0.0]
    oral_a_gut: list[float] = [a_gut]

    for _ in range(n_steps):
        dc = ka_oral_per_h * a_gut / vd_L - ke * c
        da = -ka_oral_per_h * a_gut
        c = max(c + dc * dt_h, 0.0)
        a_gut = max(a_gut + da * dt_h, 0.0)
        oral_times.append(oral_times[-1] + dt_h)
        oral_concs.append(c)
        oral_a_gut.append(a_gut)

    oral_cmax = max(oral_concs)
    oral_tmax_idx = oral_concs.index(oral_cmax)
    oral_tmax = oral_times[oral_tmax_idx]
    oral_auc = sum(
        (oral_concs[i] + oral_concs[i + 1]) / 2.0 * (oral_times[i + 1] - oral_times[i])
        for i in range(len(oral_concs) - 1)
    )
    oral_t_half = math.log(2.0) / ke
    oral_f_systemic = f_oral * fh

    oral_notes = (
        f"Oral PK: f_oral={f_oral}, fh={fh}, F={oral_f_systemic:.2f}. "
        f"Hepatic first-pass extraction={(1 - fh) * 100:.0f}%."
    )

    oral_result = RectalAbsorptionResult(
        drug_name=f"{drug_name} (oral)",
        dose_mg=dose_mg,
        f_bypass=0.0,
        times_h=oral_times,
        a_rectal_mg=oral_a_gut,
        c_plasma_mg_L=oral_concs,
        cmax_mg_L=oral_cmax,
        tmax_h=oral_tmax,
        auc_mg_h_per_L=oral_auc,
        t_half_h=oral_t_half,
        f_systemic=oral_f_systemic,
        f_hepatic_bypass=0.0,
        vs_oral_auc_ratio=1.0,
        notes=oral_notes,
    )

    auc_ratio = rectal.auc_mg_h_per_L / oral_auc if oral_auc > 0.0 else float("inf")
    cmax_ratio = rectal.cmax_mg_L / oral_cmax if oral_cmax > 0.0 else float("inf")

    advantage = "higher" if auc_ratio > 1.0 else "lower"
    notes = (
        f"Rectal route gives {advantage} AUC vs oral (ratio={auc_ratio:.2f}). "
        f"Rectal F={rectal.f_systemic * 100:.1f}% vs oral F={oral_f_systemic * 100:.1f}%. "
        f"Benefit is greatest for high hepatic extraction drugs."
    )

    return {
        "rectal_result": rectal,
        "oral_result": oral_result,
        "auc_ratio": auc_ratio,
        "cmax_ratio": cmax_ratio,
        "notes": notes,
    }


__all__ = [
    "RectalAbsorptionResult",
    "simulate_rectal_pk",
    "compare_rectal_vs_oral",
]
