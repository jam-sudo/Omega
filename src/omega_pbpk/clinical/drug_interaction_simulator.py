"""Drug interaction simulator — time-course simulation of perpetrator + victim PK.

Simulates reversible and mechanism-based inhibition (MBI) effects on victim drug
clearance over multiple days, returning dynamic PK profiles and AUCR estimates.

Reference: Rowland & Tozer, Clinical Pharmacokinetics; FDA DDI Guidance 2020.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CYP3A4 enzyme degradation rate (kdeg) — half-life ~9 h
_KDEG_CYP3A4_PER_H: float = 0.00193 * 24.0  # ≈ 0.0463/h (9 h half-life in vivo)
_KINACT_DEFAULT_PER_H: float = 0.01  # simplified kinact for MBI

# 1-compartment oral absorption rate constant (h⁻¹)
_KA_DEFAULT_PER_H: float = 1.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DrugInteractionSimResult:
    """Result of a dynamic drug-drug interaction simulation."""

    victim_name: str
    perpetrator_name: str
    inhibition_type: str  # "reversible" or "mechanism_based"
    ki_mg_L: float
    fm_victim: float
    times_h: list[float]
    c_victim_mg_L: list[float]
    c_perpetrator_mg_L: list[float]
    cl_victim_dynamic: list[float]
    auc_day1: float
    auc_last_day: float
    aucr: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration over paired (time, value) lists."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return auc


def _simulate_1cpt_oral(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    t_start: float,
    t_end: float,
    dt_h: float,
    c0: float = 0.0,
    depot0: float = 0.0,
) -> tuple[list[float], list[float], list[float]]:
    """Forward Euler 1-compartment oral absorption.

    Returns (times, concentrations, depot_amounts).
    Dose is placed into depot at t_start.
    """
    n_steps = max(1, int(round((t_end - t_start) / dt_h)))
    times = [t_start + i * dt_h for i in range(n_steps + 1)]
    c = [0.0] * len(times)
    depot = [0.0] * len(times)

    c[0] = c0
    depot[0] = depot0 + dose_mg  # dose into gut depot

    ke = cl_L_per_h / vd_L  # elimination rate constant

    for i in range(n_steps):
        dc = ka_per_h * depot[i] / vd_L - ke * c[i]
        d_depot = -ka_per_h * depot[i]
        c[i + 1] = max(0.0, c[i] + dc * dt_h)
        depot[i + 1] = max(0.0, depot[i] + d_depot * dt_h)

    return times, c, depot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_drug_interaction(  # noqa: PLR0913
    victim_name: str,
    perpetrator_name: str,
    victim_dose_mg: float,
    perpetrator_dose_mg: float,
    victim_cl_L_per_h: float,
    victim_vd_L: float,
    perpetrator_cl_L_per_h: float,
    perpetrator_vd_L: float,
    ki_mg_L: float,
    fm_victim: float,
    inhibition_type: str = "reversible",
    n_days: int = 7,
    dt_h: float = 0.1,
) -> DrugInteractionSimResult:
    """Simulate the time-course of a drug-drug interaction.

    Both drugs are dosed once daily (oral, 1-compartment).  The perpetrator
    reaches a concentration that dynamically modulates victim CL according to
    either reversible competitive inhibition or mechanism-based inhibition (MBI).

    Parameters
    ----------
    victim_name, perpetrator_name : str
        Drug names (used for labelling only).
    victim_dose_mg, perpetrator_dose_mg : float
        Single oral doses in mg.
    victim_cl_L_per_h, victim_vd_L : float
        Baseline CL (L/h) and volume of distribution (L) for victim.
    perpetrator_cl_L_per_h, perpetrator_vd_L : float
        CL (L/h) and Vd (L) for perpetrator.
    ki_mg_L : float
        Inhibition constant in mg/L (or IC50 for reversible).
    fm_victim : float
        Fraction of victim CL mediated by the inhibited enzyme (0–1).
    inhibition_type : str
        "reversible" (competitive) or "mechanism_based" (MBI).
    n_days : int
        Number of simulation days (default 7 to capture steady-state DDI).
    dt_h : float
        Integration time step in hours.

    Returns
    -------
    DrugInteractionSimResult
        Full time-course arrays and summary AUC/AUCR metrics.
    """
    # --- Input validation ---
    if not victim_name or not perpetrator_name:
        raise ValueError("victim_name and perpetrator_name must be non-empty strings")
    if victim_dose_mg <= 0:
        raise ValueError("victim_dose_mg must be > 0")
    if perpetrator_dose_mg <= 0:
        raise ValueError("perpetrator_dose_mg must be > 0")
    if victim_cl_L_per_h <= 0:
        raise ValueError("victim_cl_L_per_h must be > 0")
    if victim_vd_L <= 0:
        raise ValueError("victim_vd_L must be > 0")
    if perpetrator_cl_L_per_h <= 0:
        raise ValueError("perpetrator_cl_L_per_h must be > 0")
    if perpetrator_vd_L <= 0:
        raise ValueError("perpetrator_vd_L must be > 0")
    if ki_mg_L <= 0:
        raise ValueError("ki_mg_L must be > 0")
    if not 0.0 <= fm_victim <= 1.0:
        raise ValueError("fm_victim must be in [0, 1]")
    if inhibition_type not in ("reversible", "mechanism_based"):
        raise ValueError("inhibition_type must be 'reversible' or 'mechanism_based'")
    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # --- Setup ---
    total_h = n_days * 24.0
    n_total = max(1, int(round(total_h / dt_h)))
    all_times = [i * dt_h for i in range(n_total + 1)]

    # Pre-allocate output arrays
    c_victim = [0.0] * len(all_times)
    c_perp = [0.0] * len(all_times)
    cl_dynamic = [0.0] * len(all_times)

    # State variables
    c_v = 0.0  # victim plasma concentration (mg/L)
    c_p = 0.0  # perpetrator plasma concentration (mg/L)
    depot_v = 0.0  # victim gut depot (mg)
    depot_p = 0.0  # perpetrator gut depot (mg)
    f_inhib = 0.0  # MBI: fraction of enzyme inhibited (0–1)

    ke_p = perpetrator_cl_L_per_h / perpetrator_vd_L
    kdeg = _KDEG_CYP3A4_PER_H
    kinact = _KINACT_DEFAULT_PER_H

    # Dose index — administer once every 24 h (at t=0, 24, 48, …)
    # Pre-build set of dose step indices (nearest step to each dose time)
    dose_steps: set[int] = set()
    for day in range(n_days):
        t_dose = day * 24.0
        idx = int(round(t_dose / dt_h))
        dose_steps.add(idx)

    # Initial dose
    depot_v += victim_dose_mg
    depot_p += perpetrator_dose_mg

    c_victim[0] = c_v
    c_perp[0] = c_p

    # Compute initial dynamic CL
    if inhibition_type == "reversible":
        if c_p > 0:
            cl_t = victim_cl_L_per_h * (1.0 - fm_victim * c_p / (c_p + ki_mg_L))
        else:
            cl_t = victim_cl_L_per_h
    else:
        cl_t = victim_cl_L_per_h * (1.0 - fm_victim * f_inhib)
    cl_dynamic[0] = cl_t

    # --- Forward Euler integration ---
    for i in range(n_total):
        # Compute current dynamic CL for victim
        if inhibition_type == "reversible":
            if c_p > 0:
                inh_fraction = fm_victim * c_p / (c_p + ki_mg_L)
            else:
                inh_fraction = 0.0
            cl_t = victim_cl_L_per_h * (1.0 - inh_fraction)
        else:
            # MBI: enzyme inhibition state
            # d(f_inhib)/dt = kinact * C_perp / (Ki + C_perp) - kdeg * f_inhib
            kinact_eff = kinact * c_p / (ki_mg_L + c_p) if (ki_mg_L + c_p) > 0 else 0.0
            df_inhib = kinact_eff - kdeg * f_inhib
            f_inhib = max(0.0, min(1.0, f_inhib + df_inhib * dt_h))
            cl_t = victim_cl_L_per_h * (1.0 - fm_victim * f_inhib)

        cl_t = max(cl_t, victim_cl_L_per_h * 0.001)  # floor at 0.1% of baseline
        ke_v_t = cl_t / victim_vd_L

        # Victim ODE
        dc_v = _KA_DEFAULT_PER_H * depot_v / victim_vd_L - ke_v_t * c_v
        d_depot_v = -_KA_DEFAULT_PER_H * depot_v

        # Perpetrator ODE
        dc_p = _KA_DEFAULT_PER_H * depot_p / perpetrator_vd_L - ke_p * c_p
        d_depot_p = -_KA_DEFAULT_PER_H * depot_p

        # Update states
        c_v = max(0.0, c_v + dc_v * dt_h)
        c_p = max(0.0, c_p + dc_p * dt_h)
        depot_v = max(0.0, depot_v + d_depot_v * dt_h)
        depot_p = max(0.0, depot_p + d_depot_p * dt_h)

        # Store
        c_victim[i + 1] = c_v
        c_perp[i + 1] = c_p
        cl_dynamic[i + 1] = cl_t

        # Administer next dose at the START of the next step (if it is a dose step)
        next_idx = i + 1
        if next_idx in dose_steps and next_idx > 0:
            depot_v += victim_dose_mg
            depot_p += perpetrator_dose_mg

    # --- AUC calculation ---
    steps_per_day = max(1, int(round(24.0 / dt_h)))

    # Day 1: t=0 to t=24h
    day1_end = min(steps_per_day + 1, len(all_times))
    auc_day1 = _trapz(all_times[:day1_end], c_victim[:day1_end])

    # Last day: t=(n_days-1)*24 to t=n_days*24
    last_day_start = (n_days - 1) * steps_per_day
    last_day_end = min(last_day_start + steps_per_day + 1, len(all_times))
    auc_last_day = _trapz(
        all_times[last_day_start:last_day_end],
        c_victim[last_day_start:last_day_end],
    )

    # AUCR
    aucr = auc_last_day / auc_day1 if auc_day1 > 0 else 1.0

    # --- Notes ---
    notes: list[str] = []
    if inhibition_type == "mechanism_based":
        notes.append(f"MBI: kdeg={kdeg:.4f}/h, kinact={kinact:.4f}/h")
    if aucr >= 5.0:
        notes.append("Strong DDI: AUCR >= 5 (FDA strong inhibitor threshold)")
    elif aucr >= 2.0:
        notes.append("Moderate DDI: 2 <= AUCR < 5")
    elif aucr >= 1.25:
        notes.append("Weak DDI: 1.25 <= AUCR < 2")
    else:
        notes.append("No clinically significant DDI: AUCR < 1.25")

    return DrugInteractionSimResult(
        victim_name=victim_name,
        perpetrator_name=perpetrator_name,
        inhibition_type=inhibition_type,
        ki_mg_L=ki_mg_L,
        fm_victim=fm_victim,
        times_h=all_times,
        c_victim_mg_L=c_victim,
        c_perpetrator_mg_L=c_perp,
        cl_victim_dynamic=cl_dynamic,
        auc_day1=auc_day1,
        auc_last_day=auc_last_day,
        aucr=aucr,
        notes=notes,
    )


def onset_offset_analysis(
    victim_name: str,
    perpetrator_name: str,
    victim_cl_L_per_h: float,
    fm_victim: float,
    ki_mg_L: float,
    kdeg_per_h: float,
    kinact_per_h: float,
    c_perp_steady_mg_L: float,
) -> dict:
    """Estimate onset and offset timing for an MBI drug interaction.

    Uses analytical approximations:
    - Onset (t_90pct): time to reach 90% of maximum inhibition
      t_90 = 2.3 / (kinact_eff + kdeg)
    - Offset (t_recovery): time to recover enzyme activity after withdrawal
      t_recovery = 2.3 / kdeg  (two half-lives of enzyme regeneration)

    Parameters
    ----------
    victim_name, perpetrator_name : str
        Drug names for identification.
    victim_cl_L_per_h : float
        Victim baseline clearance (L/h).
    fm_victim : float
        Fraction of victim CL via inhibited enzyme (0–1).
    ki_mg_L : float
        Kinact half-saturation constant (mg/L).
    kdeg_per_h : float
        Enzyme degradation rate constant (1/h).
    kinact_per_h : float
        Maximum inactivation rate constant (1/h).
    c_perp_steady_mg_L : float
        Steady-state perpetrator plasma concentration (mg/L).

    Returns
    -------
    dict with keys: t_onset_h, t_recovery_h, max_aucr_estimate,
                    victim_name, perpetrator_name, fm_victim, ki_mg_L
    """
    # --- Input validation ---
    if not victim_name or not perpetrator_name:
        raise ValueError("victim_name and perpetrator_name must be non-empty strings")
    if victim_cl_L_per_h <= 0:
        raise ValueError("victim_cl_L_per_h must be > 0")
    if not 0.0 <= fm_victim <= 1.0:
        raise ValueError("fm_victim must be in [0, 1]")
    if ki_mg_L <= 0:
        raise ValueError("ki_mg_L must be > 0")
    if kdeg_per_h <= 0:
        raise ValueError("kdeg_per_h must be > 0")
    if kinact_per_h <= 0:
        raise ValueError("kinact_per_h must be > 0")
    if c_perp_steady_mg_L < 0:
        raise ValueError("c_perp_steady_mg_L must be >= 0")

    # Effective inactivation rate at steady-state perpetrator concentration
    if c_perp_steady_mg_L > 0:
        kinact_eff = kinact_per_h * c_perp_steady_mg_L / (ki_mg_L + c_perp_steady_mg_L)
    else:
        kinact_eff = 0.0

    # Onset: t_90% ≈ 2.3 / (kinact_eff + kdeg)
    t_onset_h = 2.3 / (kinact_eff + kdeg_per_h)

    # Recovery: t_90% ≈ 2.3 / kdeg (enzyme resynthesis dominates after withdrawal)
    t_recovery_h = 2.3 / kdeg_per_h

    # Maximum steady-state inhibited fraction
    if (kinact_eff + kdeg_per_h) > 0:
        f_inhib_max = kinact_eff / (kinact_eff + kdeg_per_h)
    else:
        f_inhib_max = 0.0

    # AUCR estimate at maximum inhibition
    # CL_reduced = CL_victim * (1 - fm * f_inhib_max)
    # AUCR ≈ CL_victim / CL_reduced = 1 / (1 - fm * f_inhib_max)
    cl_reduced_fraction = 1.0 - fm_victim * f_inhib_max
    cl_reduced_fraction = max(cl_reduced_fraction, 0.001)
    max_aucr_estimate = 1.0 / cl_reduced_fraction

    return {
        "victim_name": victim_name,
        "perpetrator_name": perpetrator_name,
        "t_onset_h": t_onset_h,
        "t_recovery_h": t_recovery_h,
        "max_aucr_estimate": max_aucr_estimate,
        "fm_victim": fm_victim,
        "ki_mg_L": ki_mg_L,
        "f_inhib_max": f_inhib_max,
        "kinact_eff_per_h": kinact_eff,
        "kdeg_per_h": kdeg_per_h,
    }


__all__ = [
    "DrugInteractionSimResult",
    "simulate_drug_interaction",
    "onset_offset_analysis",
]
