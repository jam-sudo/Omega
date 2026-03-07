"""Phase 309 — Drug Accumulation in Adipose Tissue.

Models long-term drug accumulation in adipose tissue during chronic dosing,
relevant for lipophilic drugs with large Vd. Uses a two-compartment ODE
(plasma + adipose) with forward Euler integration.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AdiposeAccumulationResult:
    """Result from adipose accumulation or washout simulation."""

    drug_name: str
    dose_mg: float
    kp_adipose: float
    logp: float

    times_h: list
    c_plasma_mg_L: list
    c_adipose_mg_L: list

    auc_plasma_mg_h_per_L: float
    cmax_plasma_mg_L: float

    cmax_adipose_mg_L: float
    tmax_adipose_h: float

    steady_state_adipose_mg_L: float
    accumulation_ratio: float  # >= 1 for multi-dose; -1.0 sentinel for washout
    washout_t50_h: float  # -1.0 if not applicable (accumulation run)
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_FIELDS = frozenset(
    [
        "drug_name",
        "dose_mg",
        "cl_sys_L_per_h",
        "vd_plasma_L",
        "kp_adipose",
        "adipose_volume_L",
        "adipose_blood_flow_L_per_h",
        "n_doses",
        "dosing_interval_h",
        "logp",
    ]
)


def _validate_common(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
    kp_adipose: float,
    adipose_volume_L: float,
    adipose_blood_flow_L_per_h: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if kp_adipose <= 0:
        raise ValueError("kp_adipose must be > 0")
    if adipose_volume_L <= 0:
        raise ValueError("adipose_volume_L must be > 0")
    if adipose_blood_flow_L_per_h <= 0:
        raise ValueError("adipose_blood_flow_L_per_h must be > 0")


def _trapezoidal_auc(times: list, concs: list) -> float:
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _run_ode(
    a_plasma_0: float,
    a_adipose_0: float,
    dose_schedule: list,  # list of (time_h, dose_mg) tuples
    t_end_h: float,
    dt_h: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
    kp_adipose: float,
    adipose_volume_L: float,
    adipose_blood_flow_L_per_h: float,
) -> tuple:
    """Run forward Euler ODE; return (times, c_plasma, c_adipose)."""
    k_elim = cl_sys_L_per_h / vd_plasma_L
    k_in = adipose_blood_flow_L_per_h / vd_plasma_L
    k_out = adipose_blood_flow_L_per_h / (kp_adipose * adipose_volume_L)

    # Convert schedule to a dict for quick lookup (round to dt precision)
    dose_map: dict = {}
    for t_dose, d_mg in dose_schedule:
        # Find the nearest grid point
        step = round(t_dose / dt_h)
        dose_map[step] = dose_map.get(step, 0.0) + d_mg

    n_steps = max(1, round(t_end_h / dt_h))
    times: list = []
    c_plasma: list = []
    c_adipose: list = []

    a_p = a_plasma_0
    a_a = a_adipose_0
    t = 0.0

    for step in range(n_steps + 1):
        # Apply bolus dose at this step
        if step in dose_map:
            a_p += dose_map[step]

        cp = a_p / vd_plasma_L
        ca = a_a / adipose_volume_L

        times.append(t)
        c_plasma.append(max(cp, 0.0))
        c_adipose.append(max(ca, 0.0))

        if step == n_steps:
            break

        # dA_plasma/dt = -k_elim*A_plasma - k_in*A_plasma + k_out*A_adipose
        da_p = (-k_elim - k_in) * a_p + k_out * a_a
        # dA_adipose/dt = k_in*A_plasma - k_out*A_adipose
        da_a = k_in * a_p - k_out * a_a

        a_p = max(a_p + da_p * dt_h, 0.0)
        a_a = max(a_a + da_a * dt_h, 0.0)
        t += dt_h

    return times, c_plasma, c_adipose


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def simulate_adipose_accumulation(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    n_doses: int,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
    kp_adipose: float,
    adipose_volume_L: float,
    adipose_blood_flow_L_per_h: float,
    logp: float,
    dt_h: float = 0.1,
) -> AdiposeAccumulationResult:
    """Simulate multi-dose drug accumulation in adipose tissue.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose administered at each dosing occasion (mg).
    dosing_interval_h:
        Time between doses (h).
    n_doses:
        Number of doses to administer (>= 1).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Plasma / central volume of distribution (L).
    kp_adipose:
        Tissue-to-plasma partition coefficient for adipose.
    adipose_volume_L:
        Volume of adipose tissue compartment (L).
    adipose_blood_flow_L_per_h:
        Blood flow to adipose tissue (L/h).
    logp:
        Logarithm of octanol-water partition coefficient.
    dt_h:
        Integration step size (h).

    Returns
    -------
    AdiposeAccumulationResult
    """
    _validate_common(
        dose_mg,
        cl_sys_L_per_h,
        vd_plasma_L,
        kp_adipose,
        adipose_volume_L,
        adipose_blood_flow_L_per_h,
    )
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")

    # Build dose schedule
    dose_schedule = [(i * dosing_interval_h, dose_mg) for i in range(n_doses)]
    t_end_h = (n_doses - 1) * dosing_interval_h + dosing_interval_h

    times, c_plasma, c_adipose = _run_ode(
        a_plasma_0=0.0,
        a_adipose_0=0.0,
        dose_schedule=dose_schedule,
        t_end_h=t_end_h,
        dt_h=dt_h,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_plasma_L=vd_plasma_L,
        kp_adipose=kp_adipose,
        adipose_volume_L=adipose_volume_L,
        adipose_blood_flow_L_per_h=adipose_blood_flow_L_per_h,
    )

    auc_plasma = _trapezoidal_auc(times, c_plasma)
    cmax_plasma = max(c_plasma) if c_plasma else 0.0

    cmax_adipose = max(c_adipose) if c_adipose else 0.0
    tmax_adipose = times[c_adipose.index(cmax_adipose)] if c_adipose else 0.0
    steady_state_adipose = c_adipose[-1] if c_adipose else 0.0

    # First-dose adipose peak: run single-dose simulation
    times_1, _, c_adipose_1 = _run_ode(
        a_plasma_0=0.0,
        a_adipose_0=0.0,
        dose_schedule=[(0.0, dose_mg)],
        t_end_h=dosing_interval_h,
        dt_h=dt_h,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_plasma_L=vd_plasma_L,
        kp_adipose=kp_adipose,
        adipose_volume_L=adipose_volume_L,
        adipose_blood_flow_L_per_h=adipose_blood_flow_L_per_h,
    )
    first_dose_cmax_adipose = max(c_adipose_1) if c_adipose_1 else 0.0

    if first_dose_cmax_adipose > 0.0:
        accumulation_ratio = cmax_adipose / first_dose_cmax_adipose
    else:
        accumulation_ratio = 1.0

    # Clip to >= 1.0 (numerical noise guard)
    accumulation_ratio = max(accumulation_ratio, 1.0)

    notes = (
        f"logP={logp:.2f}; {n_doses} doses x {dose_mg} mg every {dosing_interval_h} h; "
        f"kp_adipose={kp_adipose:.2f}; adipose_vol={adipose_volume_L:.1f} L"
    )

    return AdiposeAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        kp_adipose=kp_adipose,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_adipose_mg_L=c_adipose,
        auc_plasma_mg_h_per_L=auc_plasma,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_adipose_mg_L=cmax_adipose,
        tmax_adipose_h=tmax_adipose,
        steady_state_adipose_mg_L=steady_state_adipose,
        accumulation_ratio=accumulation_ratio,
        washout_t50_h=-1.0,
        notes=notes,
    )


def predict_washout(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float,
    kp_adipose: float,
    adipose_volume_L: float,
    adipose_blood_flow_L_per_h: float,
    initial_adipose_conc_mg_L: float,
    t_end_h: float,
    dt_h: float = 0.1,
    logp: float = 0.0,
) -> AdiposeAccumulationResult:
    """Simulate adipose washout after cessation of dosing.

    The adipose compartment starts at `initial_adipose_conc_mg_L`; plasma
    starts at zero. No new doses are added.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Reference dose (mg); used for bookkeeping only.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_plasma_L:
        Plasma volume (L).
    kp_adipose:
        Tissue-to-plasma partition coefficient for adipose.
    adipose_volume_L:
        Volume of adipose compartment (L).
    adipose_blood_flow_L_per_h:
        Blood flow to adipose tissue (L/h).
    initial_adipose_conc_mg_L:
        Starting concentration in adipose (mg/L).
    t_end_h:
        Duration of washout simulation (h).
    dt_h:
        Integration step size (h).
    logp:
        logP value for bookkeeping.

    Returns
    -------
    AdiposeAccumulationResult
        `washout_t50_h` is the time for adipose concentration to fall to 50 %
        of its initial value; `accumulation_ratio` is set to -1.0 (sentinel).
    """
    _validate_common(
        dose_mg,
        cl_sys_L_per_h,
        vd_plasma_L,
        kp_adipose,
        adipose_volume_L,
        adipose_blood_flow_L_per_h,
    )
    if initial_adipose_conc_mg_L < 0:
        raise ValueError("initial_adipose_conc_mg_L must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")

    initial_a_adipose = initial_adipose_conc_mg_L * adipose_volume_L

    times, c_plasma, c_adipose = _run_ode(
        a_plasma_0=0.0,
        a_adipose_0=initial_a_adipose,
        dose_schedule=[],
        t_end_h=t_end_h,
        dt_h=dt_h,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_plasma_L=vd_plasma_L,
        kp_adipose=kp_adipose,
        adipose_volume_L=adipose_volume_L,
        adipose_blood_flow_L_per_h=adipose_blood_flow_L_per_h,
    )

    auc_plasma = _trapezoidal_auc(times, c_plasma)
    cmax_plasma = max(c_plasma) if c_plasma else 0.0
    cmax_adipose = max(c_adipose) if c_adipose else 0.0
    tmax_adipose = times[c_adipose.index(cmax_adipose)] if c_adipose else 0.0
    steady_state_adipose = c_adipose[-1] if c_adipose else 0.0

    # Compute washout t50
    half = initial_adipose_conc_mg_L * 0.5
    washout_t50_h = -1.0
    for i in range(1, len(c_adipose)):
        if c_adipose[i] <= half:
            # Linear interpolation
            dt = times[i] - times[i - 1]
            dc = c_adipose[i] - c_adipose[i - 1]
            if abs(dc) > 1e-12:
                frac = (half - c_adipose[i - 1]) / dc
                washout_t50_h = times[i - 1] + frac * dt
            else:
                washout_t50_h = times[i]
            break

    notes = (
        f"Washout: initial_adipose_conc={initial_adipose_conc_mg_L:.4f} mg/L; "
        f"kp_adipose={kp_adipose:.2f}; t50={washout_t50_h:.2f} h"
    )

    return AdiposeAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        kp_adipose=kp_adipose,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_adipose_mg_L=c_adipose,
        auc_plasma_mg_h_per_L=auc_plasma,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_adipose_mg_L=cmax_adipose,
        tmax_adipose_h=tmax_adipose,
        steady_state_adipose_mg_L=steady_state_adipose,
        accumulation_ratio=-1.0,
        washout_t50_h=washout_t50_h,
        notes=notes,
    )
