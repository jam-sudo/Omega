"""Cancer drug PK — cytotoxic and targeted therapy simulation.

Models multi-cycle 1-compartment PK for anticancer drugs with
myelosuppression (nadir) prediction via an Emax model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "CancerDrugPKResult",
    "simulate_cancer_drug_pk",
    "nadir_prediction",
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

VALID_DRUG_CLASSES = ("cytotoxic", "targeted")
VALID_ROUTES = ("iv", "oral")


@dataclass
class CancerDrugPKResult:
    """Result of a multi-cycle cancer drug PK simulation."""

    drug_name: str
    drug_class: str
    dose_mg: float
    bsa_m2: float
    interval_h: float
    n_cycles: int
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax_per_cycle: list[float] = field(default_factory=list)
    auc_per_cycle: list[float] = field(default_factory=list)
    trough_per_cycle: list[float] = field(default_factory=list)
    cumulative_auc: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list[float], conc: list[float]) -> float:
    """Compute AUC using the trapezoidal rule (manual implementation)."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (conc[i - 1] + conc[i]) * dt
    return auc


def _run_1cpt_forward_euler(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    interval_h: float,
    n_cycles: int,
    dt_h: float,
    route: str,
    ka_per_h: float,
) -> tuple[list[float], list[float]]:
    """Run forward Euler 1-compartment PK for n_cycles doses.

    Returns (times_h, c_plasma_mg_L) over the full simulation.
    """
    kel = cl_L_per_h / vd_L
    t_total = n_cycles * interval_h
    n_steps = max(int(round(t_total / dt_h)), 1)

    times: list[float] = []
    concs: list[float] = []

    # State variables
    c_central = 0.0  # mg/L in central compartment
    c_gut = 0.0      # mg in gut compartment (oral only)

    # Dose times (one per cycle)
    dose_schedule = [i * interval_h for i in range(n_cycles)]
    dose_idx = 0

    t = 0.0
    for step in range(n_steps + 1):
        t = step * dt_h

        # Administer dose at the start of each cycle
        while dose_idx < len(dose_schedule) and dose_schedule[dose_idx] <= t + 1e-9:
            dose_t = dose_schedule[dose_idx]
            # Only dose if we're within dt_h of the scheduled time
            if abs(t - dose_t) <= dt_h / 2.0 or (step == 0 and dose_t == 0.0):
                if route == "iv":
                    c_central += dose_mg / vd_L
                else:
                    c_gut += dose_mg
            dose_idx += 1

        times.append(t)
        concs.append(max(c_central, 0.0))

        if step < n_steps:
            if route == "iv":
                dc_dt = -kel * c_central
                c_central = max(c_central + dc_dt * dt_h, 0.0)
            else:
                # 1-cpt with first-order absorption
                absorption_rate = ka_per_h * c_gut
                dc_central_dt = (absorption_rate / vd_L) - kel * c_central
                dc_gut_dt = -ka_per_h * c_gut
                c_central = max(c_central + dc_central_dt * dt_h, 0.0)
                c_gut = max(c_gut + dc_gut_dt * dt_h, 0.0)

    return times, concs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_cancer_drug_pk(
    drug_name: str,
    drug_class: str,
    dose_mg_m2: float,
    bsa_m2: float,
    cl_L_per_h: float,
    vd_L: float,
    interval_h: float = 504.0,
    n_cycles: int = 6,
    dt_h: float = 0.1,
    route: str | None = None,
    ka_per_h: float = 1.0,
) -> CancerDrugPKResult:
    """Simulate multi-cycle cancer drug PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_class:
        ``"cytotoxic"`` (IV bolus, dose in mg/m2) or ``"targeted"`` (oral,
        dose in fixed mg — bsa_m2 ignored).
    dose_mg_m2:
        For cytotoxic: dose in mg/m2. For targeted: total fixed dose in mg.
    bsa_m2:
        Body surface area (m2); used only for cytotoxic class.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    interval_h:
        Dosing interval (hours). Default 504 h = 21 days (3-week cycle).
    n_cycles:
        Number of treatment cycles.
    dt_h:
        Forward Euler time step (h).
    route:
        Override route; defaults to ``"iv"`` for cytotoxic, ``"oral"`` for
        targeted.
    ka_per_h:
        Absorption rate constant (h^-1); used for oral route.

    Returns
    -------
    CancerDrugPKResult
    """
    # --- Validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if drug_class not in VALID_DRUG_CLASSES:
        raise ValueError(f"drug_class must be one of {VALID_DRUG_CLASSES}")
    if dose_mg_m2 <= 0:
        raise ValueError("dose_mg_m2 must be > 0")
    if bsa_m2 <= 0:
        raise ValueError("bsa_m2 must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")

    # --- Determine dose and route ---
    if drug_class == "cytotoxic":
        dose_mg = dose_mg_m2 * bsa_m2
        effective_route = route if route in VALID_ROUTES else "iv"
    else:  # targeted
        dose_mg = dose_mg_m2  # treat as fixed mg
        effective_route = route if route in VALID_ROUTES else "oral"

    if effective_route not in VALID_ROUTES:
        raise ValueError(f"route must be one of {VALID_ROUTES}")

    notes: list[str] = []
    notes.append(
        f"Drug class: {drug_class}, route: {effective_route}, "
        f"dose: {dose_mg:.2f} mg, cycles: {n_cycles}"
    )

    # --- Forward Euler simulation ---
    times, concs = _run_1cpt_forward_euler(
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        interval_h=interval_h,
        n_cycles=n_cycles,
        dt_h=dt_h,
        route=effective_route,
        ka_per_h=ka_per_h,
    )

    # --- Per-cycle metrics ---
    cmax_per_cycle: list[float] = []
    auc_per_cycle: list[float] = []
    trough_per_cycle: list[float] = []

    steps_per_cycle = max(int(round(interval_h / dt_h)), 1)

    for cycle in range(n_cycles):
        start_step = cycle * steps_per_cycle
        end_step = min(start_step + steps_per_cycle, len(times) - 1)

        if start_step >= len(times):
            break

        cycle_t = times[start_step : end_step + 1]
        cycle_c = concs[start_step : end_step + 1]

        if not cycle_c:
            cmax_per_cycle.append(0.0)
            auc_per_cycle.append(0.0)
            trough_per_cycle.append(0.0)
            continue

        cmax = max(cycle_c)
        auc = _trapezoidal_auc(cycle_t, cycle_c)
        trough = cycle_c[-1]

        cmax_per_cycle.append(cmax)
        auc_per_cycle.append(auc)
        trough_per_cycle.append(trough)

    cumulative_auc = sum(auc_per_cycle)

    return CancerDrugPKResult(
        drug_name=drug_name,
        drug_class=drug_class,
        dose_mg=dose_mg,
        bsa_m2=bsa_m2,
        interval_h=interval_h,
        n_cycles=n_cycles,
        times_h=times,
        c_plasma_mg_L=concs,
        cmax_per_cycle=cmax_per_cycle,
        auc_per_cycle=auc_per_cycle,
        trough_per_cycle=trough_per_cycle,
        cumulative_auc=cumulative_auc,
        notes=notes,
    )


def nadir_prediction(
    c_plasma_mg_L_list: list[float],
    times_h: list[float],
    myelosuppression_ec50: float,
    emax_myelosuppression: float,
) -> dict:
    """Predict myelosuppression nadir using an Emax model.

    The nadir effect is computed from cumulative AUC:
        nadir_effect = emax * AUC / (ec50 + AUC)

    Parameters
    ----------
    c_plasma_mg_L_list:
        Plasma concentration-time profile (mg/L).
    times_h:
        Corresponding time points (h).
    myelosuppression_ec50:
        EC50 for myelosuppression (mg*h/L, same units as AUC).
    emax_myelosuppression:
        Maximum myelosuppression effect (0–100%).

    Returns
    -------
    dict with keys ``nadir_effect_pct`` and ``risk_category``.
    """
    if not c_plasma_mg_L_list:
        raise ValueError("c_plasma_mg_L_list must be non-empty")
    if not times_h:
        raise ValueError("times_h must be non-empty")
    if len(c_plasma_mg_L_list) != len(times_h):
        raise ValueError("c_plasma_mg_L_list and times_h must have the same length")
    if myelosuppression_ec50 <= 0:
        raise ValueError("myelosuppression_ec50 must be > 0")
    if emax_myelosuppression <= 0:
        raise ValueError("emax_myelosuppression must be > 0")

    cumulative_auc = _trapezoidal_auc(times_h, c_plasma_mg_L_list)

    nadir_effect = emax_myelosuppression * cumulative_auc / (
        myelosuppression_ec50 + cumulative_auc
    )

    if nadir_effect > 80.0:
        risk_category = "severe"
    elif nadir_effect >= 50.0:
        risk_category = "moderate"
    else:
        risk_category = "mild"

    return {
        "nadir_effect_pct": nadir_effect,
        "risk_category": risk_category,
        "cumulative_auc_mg_h_per_L": cumulative_auc,
    }
