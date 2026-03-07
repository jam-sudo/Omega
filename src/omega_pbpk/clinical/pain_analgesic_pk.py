"""PK/PD modelling of analgesic drugs (opioids and NSAIDs).

A 1-compartment oral model drives an effect compartment via a first-order
ke0 rate constant.  Analgesia is predicted using a simple Hill (Emax) model.

Model equations (forward Euler)
--------------------------------
dA_gut/dt  = -ka * A_gut                  (absorption depot, mg)
dCp/dt     = ka * A_gut / Vd - (CL/Vd) * Cp   (plasma, mg/L)
dCe/dt     = ke0 * (Cp - Ce)              (effect compartment, mg/L)
effect(t)  = Emax * Ce / (EC50 + Ce)      (Hill n=1, as analgesia %)

Dosing
------
Opioids : ka = 1.5 /h, F = 0.70
NSAIDs  : ka = 2.0 /h, F = 0.90

References
----------
Sheiner LB et al. Simultaneous modelling of pharmacokinetics and
pharmacodynamics: application to d-tubocurarine. Clin Pharmacol Ther.
1979;25(3):358-71.

Mather LE. Clinical pharmacokinetics of fentanyl and its newer derivatives.
Clin Pharmacokinet. 1983;8(5):422-46.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPIOID_KA: float = 1.5  # /h
_OPIOID_F: float = 0.70

_NSAID_KA: float = 2.0  # /h
_NSAID_F: float = 0.90

_VALID_DRUG_CLASSES = {"opioid", "nsaid"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnalgesicPKResult:
    """Result of an analgesic PK/PD simulation."""

    drug_name: str
    drug_class: str
    dose_mg: float
    interval_h: float
    n_doses: int
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_effect_mg_L: list[float] = field(default_factory=list)
    analgesia_pct: list[float] = field(default_factory=list)
    cmax: float = 0.0
    auc: float = 0.0
    t_half_h: float = 0.0
    avg_analgesia_pct: float = 0.0
    analgesic_duration_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


def _simulate_single_compartment(
    dose_mg: float,
    ka: float,
    F: float,
    cl_L_per_h: float,
    vd_L: float,
    ke0_per_h: float,
    ec50_mg_L: float,
    emax_analgesia: float,
    t_start: float,
    t_end: float,
    dt_h: float,
    cp0: float,
    ce0: float,
    a_gut0: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Forward Euler 1-cpt + effect compartment simulation for one dosing interval.

    Returns (times, cp_list, ce_list, effect_list).
    """
    ke = cl_L_per_h / vd_L  # elimination rate constant (/h)
    times: list[float] = []
    cp_list: list[float] = []
    ce_list: list[float] = []
    effect_list: list[float] = []

    t = t_start
    a_gut = a_gut0  # amount in gut depot, mg
    cp = cp0  # plasma concentration, mg/L
    ce = ce0  # effect compartment concentration, mg/L

    while t <= t_end + dt_h * 0.5:
        # Record state
        times.append(round(t, 6))
        cp_list.append(cp)
        ce_list.append(ce)
        effect = emax_analgesia * ce / (ec50_mg_L + ce) if (ec50_mg_L + ce) > 0 else 0.0
        effect_list.append(effect)

        # Forward Euler updates
        dAg = -ka * a_gut
        dCp = (ka * a_gut) / vd_L - ke * cp
        dCe = ke0_per_h * (cp - ce)

        a_gut += dAg * dt_h
        if a_gut < 0.0:
            a_gut = 0.0
        cp += dCp * dt_h
        if cp < 0.0:
            cp = 0.0
        ce += dCe * dt_h
        if ce < 0.0:
            ce = 0.0

        t += dt_h

    return times, cp_list, ce_list, effect_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_analgesic_duration(
    effect_times: list[float],
    effect_values: list[float],
    threshold: float,
) -> float:
    """Compute duration (h) over which analgesic effect exceeds a threshold.

    Uses trapezoidal integration of the indicator function (1 if above
    threshold, 0 otherwise), which equals the total time above threshold.

    Parameters
    ----------
    effect_times:
        Time points (h).
    effect_values:
        Corresponding analgesic effect values (%).
    threshold:
        Effect threshold (%, e.g. 50.0).

    Returns
    -------
    float
        Total duration in hours where effect > threshold.
    """
    if not effect_times:
        raise ValueError("effect_times must be a non-empty list.")
    if len(effect_times) != len(effect_values):
        raise ValueError("effect_times and effect_values must have the same length.")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a non-negative finite float, got {threshold!r}")

    duration = 0.0
    for i in range(1, len(effect_times)):
        dt = effect_times[i] - effect_times[i - 1]
        above_prev = effect_values[i - 1] > threshold
        above_curr = effect_values[i] > threshold
        if above_prev and above_curr:
            duration += dt
        elif above_prev or above_curr:
            # Linear interpolation to find crossing time
            e0, e1 = effect_values[i - 1], effect_values[i]
            if abs(e1 - e0) > 1e-12:
                t_cross = (threshold - e0) / (e1 - e0) * dt
                if above_prev:
                    duration += t_cross
                else:
                    duration += dt - t_cross
    return duration


def simulate_analgesic_pk(
    drug_name: str,
    drug_class: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ke0_per_h: float,
    ec50_mg_L: float,
    emax_analgesia: float,
    interval_h: float,
    n_doses: int,
    dt_h: float = 0.05,
) -> AnalgesicPKResult:
    """Simulate analgesic PK/PD using a 1-compartment oral model with effect compartment.

    Parameters
    ----------
    drug_name:
        Name of the drug (informational).
    drug_class:
        ``"opioid"`` or ``"nsaid"``.
    dose_mg:
        Dose per administration (mg).
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    ke0_per_h:
        Effect compartment equilibration rate constant (/h).
    ec50_mg_L:
        Effect compartment concentration producing 50% of Emax (mg/L).
    emax_analgesia:
        Maximum possible analgesia (%).  Typically 100.0.
    interval_h:
        Dosing interval (h).
    n_doses:
        Number of doses to simulate.
    dt_h:
        Integration time step (h).  Default 0.05 h (3 min).

    Returns
    -------
    AnalgesicPKResult
    """
    # --- Input validation ---
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string.")
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(
            f"drug_class must be one of {sorted(_VALID_DRUG_CLASSES)}, got {drug_class!r}"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if ke0_per_h <= 0:
        raise ValueError(f"ke0_per_h must be positive, got {ke0_per_h}")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be positive, got {ec50_mg_L}")
    if emax_analgesia <= 0:
        raise ValueError(f"emax_analgesia must be positive, got {emax_analgesia}")
    if interval_h <= 0:
        raise ValueError(f"interval_h must be positive, got {interval_h}")
    if n_doses < 1:
        raise ValueError(f"n_doses must be >= 1, got {n_doses}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")

    # Drug class specific PK parameters
    if drug_class == "opioid":
        ka = _OPIOID_KA
        F = _OPIOID_F
    else:
        ka = _NSAID_KA
        F = _NSAID_F

    # Simulation state
    all_times: list[float] = []
    all_cp: list[float] = []
    all_ce: list[float] = []
    all_effect: list[float] = []

    cp = 0.0
    ce = 0.0
    a_gut = 0.0
    t_offset = 0.0

    for dose_num in range(n_doses):
        # Add dose to gut depot
        a_gut += F * dose_mg

        t_start = t_offset
        t_end = t_offset + interval_h

        times, cp_list, ce_list, effect_list = _simulate_single_compartment(
            dose_mg=dose_mg,
            ka=ka,
            F=F,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            ke0_per_h=ke0_per_h,
            ec50_mg_L=ec50_mg_L,
            emax_analgesia=emax_analgesia,
            t_start=t_start,
            t_end=t_end,
            dt_h=dt_h,
            cp0=cp,
            ce0=ce,
            a_gut0=a_gut,
        )

        # For next interval, use last state from this interval
        # We need to recalculate the final a_gut state
        n_steps = len(times)
        a_gut_temp = F * dose_mg if dose_num == 0 else a_gut
        for _ in range(n_steps - 1):
            a_gut_temp = max(0.0, a_gut_temp - ka * a_gut_temp * dt_h)
        a_gut = a_gut_temp

        cp = cp_list[-1]
        ce = ce_list[-1]

        # Skip first point for continuation (already recorded at end of previous interval)
        if dose_num == 0:
            all_times.extend(times)
            all_cp.extend(cp_list)
            all_ce.extend(ce_list)
            all_effect.extend(effect_list)
        else:
            all_times.extend(times[1:])
            all_cp.extend(cp_list[1:])
            all_ce.extend(ce_list[1:])
            all_effect.extend(effect_list[1:])

        t_offset += interval_h

    # --- Derived PK metrics ---
    cmax = max(all_cp) if all_cp else 0.0
    auc = _trapezoidal_auc(all_times, all_cp)
    ke = cl_L_per_h / vd_L
    t_half = math.log(2.0) / ke if ke > 0 else 0.0

    avg_effect = sum(all_effect) / len(all_effect) if all_effect else 0.0
    dur = compute_analgesic_duration(all_times, all_effect, threshold=50.0)

    notes_parts: list[str] = [
        f"{drug_class.capitalize()}: ka={ka:.1f}/h, F={F:.0%}.",
        f"ke0={ke0_per_h:.2f}/h, EC50={ec50_mg_L:.3f} mg/L.",
    ]
    if avg_effect < 30.0:
        notes_parts.append("Average analgesia below 30%; consider dose increase.")
    notes = " ".join(notes_parts)

    return AnalgesicPKResult(
        drug_name=drug_name,
        drug_class=drug_class,
        dose_mg=dose_mg,
        interval_h=interval_h,
        n_doses=n_doses,
        times_h=all_times,
        c_plasma_mg_L=all_cp,
        c_effect_mg_L=all_ce,
        analgesia_pct=all_effect,
        cmax=round(cmax, 6),
        auc=round(auc, 6),
        t_half_h=round(t_half, 4),
        avg_analgesia_pct=round(avg_effect, 4),
        analgesic_duration_h=round(dur, 4),
        notes=notes,
    )
