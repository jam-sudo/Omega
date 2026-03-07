"""Enterohepatic circulation PK model — Phase 793.

Models hepatic secretion into bile, gallbladder lag, gut reabsorption,
and the resulting secondary plasma peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "EnterohepaticPKResult",
    "simulate_enterohepatic_circulation",
]


@dataclass
class EnterohepaticPKResult:
    """Result of Phase-793 enterohepatic circulation PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_bile_mg: list = field(default_factory=list)
    c_gut_recirculating_mg: list = field(default_factory=list)
    cmax_plasma: float = 0.0
    tmax_primary_h: float = 0.0
    secondary_peak_detected: bool = False
    secondary_peak_time_h: float = 0.0
    secondary_peak_conc_mg_L: float = 0.0
    auc_mg_h_per_L: float = 0.0
    recycling_fraction: float = 0.0
    notes: str = ""


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def _find_primary_peak(c_plasma: list) -> int:
    """Return index of the first local maximum (primary Cmax)."""
    for i in range(1, len(c_plasma) - 1):
        if c_plasma[i] >= c_plasma[i - 1] and c_plasma[i] > c_plasma[i + 1]:
            return i
    # Fallback: global maximum
    return c_plasma.index(max(c_plasma))


def _find_secondary_peak(c_plasma: list, primary_idx: int) -> tuple[bool, int, float]:
    """Detect secondary peak after primary_idx.

    Returns (detected, index, value).
    """
    if primary_idx >= len(c_plasma) - 2:
        return False, 0, 0.0

    cmax = max(c_plasma)
    threshold = 0.03 * cmax  # must be >= 3% of primary Cmax

    for i in range(primary_idx + 2, len(c_plasma) - 1):
        if (
            c_plasma[i] > c_plasma[i - 1]
            and c_plasma[i] > c_plasma[i + 1]
            and c_plasma[i] >= threshold
        ):
            return True, i, c_plasma[i]

    return False, 0, 0.0


def simulate_enterohepatic_circulation(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    cl_total_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    f_biliary: float = 0.3,
    ka_bile_per_h: float = 0.5,
    bile_lag_h: float = 4.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> EnterohepaticPKResult:
    """Simulate enterohepatic circulation pharmacokinetics.

    Compartments
    ------------
    gut_depot      : absorbed oral dose depot (mg)
    plasma         : plasma concentration (mg/L)
    bile_delayed   : bile that has not yet been released to gut (mg); release
                     begins after bile_lag_h using a first-order transfer
    gut_recirculating : recirculating gut compartment fed by bile (mg)

    The rate of biliary secretion from plasma is:
        rate_biliary = f_biliary * CL_total * C_plasma

    After the lag delay, bile transfers to the recirculating gut at ka_bile_per_h.
    Reabsorption from gut_recirculating back to plasma also occurs at ka_bile_per_h.

    Non-biliary elimination:
        rate_elim = (1 - f_biliary) * CL_total * C_plasma
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 <= f_biliary < 1.0):
        raise ValueError(f"f_biliary must be in [0, 1), got {f_biliary}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_total_L_per_h <= 0:
        raise ValueError(f"cl_total_L_per_h must be > 0, got {cl_total_L_per_h}")

    # Derived clearance terms
    cl_biliary = f_biliary * cl_total_L_per_h  # L/h → biliary
    cl_direct = (1.0 - f_biliary) * cl_total_L_per_h  # L/h → direct elimination

    # Rate constant for biliary secretion from plasma (per hour, referred to volume)
    k_bile_sec = cl_biliary / vd_L  # 1/h  (rate = k_bile_sec * C_plasma * Vd gives mg/h)

    # State variables
    gut_depot = dose_mg * f_oral  # mg absorbed from GI
    c_plasma = 0.0  # mg/L
    bile_delayed = 0.0  # mg in bile pre-lag
    gut_recirc = 0.0  # mg in recirculating gut

    # Accumulator to estimate recycling_fraction
    total_bile_secreted = 0.0
    total_reabsorbed = 0.0

    # Lag implementation: track time-since-start for delay
    # We model the bile_lag as a simple threshold: bile_delayed secreted before
    # bile_lag_h remains locked; release begins at t > bile_lag_h.
    # We use a continuous first-order release gated by (t > bile_lag_h).

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_bile_list: list[float] = []
    c_gut_recirc_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h)) + 1
    t = 0.0

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_bile_list.append(bile_delayed)
        c_gut_recirc_list.append(gut_recirc)

        # --- ODE rates ---

        # Gut depot → plasma
        rate_abs = ka_per_h * gut_depot  # mg/h

        # Plasma elimination (direct)
        # rate = (CL/Vd) * C * Vd = CL * C  in mg/h
        # But c_plasma is mg/L and we update as d(C)/dt, so:
        # d(C_plasma)/dt units = mg/L/h
        dc_elim = (cl_direct / vd_L) * c_plasma  # 1/h * mg/L

        # Biliary secretion from plasma
        dc_biliary = k_bile_sec * c_plasma  # mg/L/h (removes from plasma)
        rate_biliary_mg_h = dc_biliary * vd_L  # mg/h entering bile depot

        # Bile release to gut recirculating (gated by lag)
        if t >= bile_lag_h:
            rate_bile_release = ka_bile_per_h * bile_delayed  # mg/h
        else:
            rate_bile_release = 0.0

        # Gut recirculating → plasma reabsorption
        rate_reabs_mg_h = ka_bile_per_h * gut_recirc  # mg/h
        dc_reabs = rate_reabs_mg_h / vd_L  # mg/L/h into plasma

        # --- Update state (forward Euler) ---
        d_gut_depot = -rate_abs
        d_bile_delayed = rate_biliary_mg_h - rate_bile_release
        d_gut_recirc = rate_bile_release - rate_reabs_mg_h
        dc_plasma_dt = (rate_abs / vd_L) - dc_elim - dc_biliary + dc_reabs

        gut_depot = max(0.0, gut_depot + d_gut_depot * dt_h)
        bile_delayed = max(0.0, bile_delayed + d_bile_delayed * dt_h)
        gut_recirc = max(0.0, gut_recirc + d_gut_recirc * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)

        # Accumulators (use rates before update)
        total_bile_secreted += rate_biliary_mg_h * dt_h
        total_reabsorbed += rate_reabs_mg_h * dt_h

        t += dt_h

    # Summary metrics
    cmax = max(c_plasma_list) if c_plasma_list else 0.0
    primary_idx = _find_primary_peak(c_plasma_list)
    tmax_primary = times[primary_idx]

    sec_detected, sec_idx, sec_conc = _find_secondary_peak(c_plasma_list, primary_idx)
    sec_time = times[sec_idx] if sec_detected else 0.0

    auc = _trapezoidal_auc(times, c_plasma_list)

    recycling_fraction = total_reabsorbed / total_bile_secreted if total_bile_secreted > 0 else 0.0
    recycling_fraction = min(recycling_fraction, 0.9999)

    notes = (
        f"EHC simulation: dose={dose_mg}mg, f_biliary={f_biliary:.2f}, "
        f"bile_lag={bile_lag_h}h. "
        f"Secondary peak {'detected' if sec_detected else 'not detected'}."
    )

    return EnterohepaticPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_bile_mg=c_bile_list,
        c_gut_recirculating_mg=c_gut_recirc_list,
        cmax_plasma=cmax,
        tmax_primary_h=tmax_primary,
        secondary_peak_detected=sec_detected,
        secondary_peak_time_h=sec_time,
        secondary_peak_conc_mg_L=sec_conc,
        auc_mg_h_per_L=auc,
        recycling_fraction=recycling_fraction,
        notes=notes,
    )
