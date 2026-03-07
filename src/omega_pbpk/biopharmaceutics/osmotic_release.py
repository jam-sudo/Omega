"""Osmotic drug release modeling — OROS technology (Phase 369).

Models zero-order delivery from osmotic pressure-driven pump systems including
elementary osmotic, push-pull, and mini-osmotic devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

_VALID_DEVICE_TYPES = {"elementary_osmotic", "push_pull", "mini_osmotic"}


@dataclass(frozen=True)
class OsmoticReleaseResult:
    """Result of an osmotic pump drug release and PK simulation."""

    drug_name: str
    device_type: str  # "elementary_osmotic", "push_pull", "mini_osmotic"
    dose_mg: float
    membrane_permeability: float  # Water permeability constant A_w (mL/h/atm/cm²)

    # Release profile
    times_h: list[float]
    release_rate_mg_per_h: list[float]  # Instantaneous release rate
    cumulative_released_mg: list[float]  # Cumulative drug released

    # Summary
    release_duration_h: float  # Time to release 80% of dose
    plateau_rate_mg_per_h: float  # Zero-order plateau delivery rate
    plateau_duration_h: float  # Duration of zero-order plateau
    t_lag_h: float  # Lag before drug release begins
    f_released_at_4h: float  # Fraction released at 4h
    f_released_at_12h: float  # Fraction released at 12h

    # PK integration
    times_pk_h: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    fluctuation_pct: float  # (Cmax-Cmin)/Cavg at steady state

    notes: str


def _device_modifiers(device_type: str) -> tuple[float, float]:
    """Return (efficiency_factor, rate_variability) for each device type.

    efficiency_factor: multiplier on plateau rate (push-pull is most efficient)
    rate_variability: fractional std for noise (elementary can crack)
    """
    if device_type == "elementary_osmotic":
        return 0.80, 0.10
    elif device_type == "push_pull":
        return 0.95, 0.02
    elif device_type == "mini_osmotic":
        return 0.90, 0.03
    else:
        return 0.85, 0.05


def simulate_osmotic_release(
    drug_name: str,
    dose_mg: float,
    device_type: str = "push_pull",
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    membrane_area_cm2: float = 1.0,
    osmotic_pressure_atm: float = 100.0,
    membrane_thickness_um: float = 50.0,
    membrane_permeability: float = 1e-3,
    drug_solubility_mg_mL: float = 100.0,
    t_lag_h: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> OsmoticReleaseResult:
    """Simulate osmotic pump drug release and plasma PK.

    Parameters
    ----------
    drug_name:
        Name of drug being simulated.
    dose_mg:
        Total drug dose loaded into device (mg).
    device_type:
        One of "elementary_osmotic", "push_pull", "mini_osmotic".
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    membrane_area_cm2:
        Semipermeable membrane surface area (cm²).
    osmotic_pressure_atm:
        Osmotic pressure gradient across membrane (atm).
    membrane_thickness_um:
        Membrane thickness in micrometers (used for notes).
    membrane_permeability:
        Water permeability constant A_w [mL/(h·atm·cm²)].
    drug_solubility_mg_mL:
        Drug solubility in the core reservoir (mg/mL).
    t_lag_h:
        Lag time for membrane hydration before release begins (h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step size (h).

    Returns
    -------
    OsmoticReleaseResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if osmotic_pressure_atm <= 0:
        raise ValueError(f"osmotic_pressure_atm must be > 0, got {osmotic_pressure_atm}")
    if membrane_area_cm2 <= 0:
        raise ValueError(f"membrane_area_cm2 must be > 0, got {membrane_area_cm2}")
    if drug_solubility_mg_mL <= 0:
        raise ValueError(f"drug_solubility_mg_mL must be > 0, got {drug_solubility_mg_mL}")
    if device_type not in _VALID_DEVICE_TYPES:
        raise ValueError(f"device_type must be one of {_VALID_DEVICE_TYPES}, got {device_type!r}")

    efficiency, variability = _device_modifiers(device_type)

    # Zero-order plateau rate [mg/h]
    # dimensionally: [mL/(h·atm·cm²)] * [atm] * [cm²] * [mg/mL] = mg/h
    raw_plateau_rate = (
        membrane_permeability * osmotic_pressure_atm * membrane_area_cm2 * drug_solubility_mg_mL
    )
    plateau_rate = raw_plateau_rate * efficiency

    # First-order falloff rate constant for tail phase
    k_falloff = 2.0  # /h — rapid depletion once near-empty

    # Time array
    times = np.arange(0.0, t_end_h + dt_h * 0.5, dt_h)
    n = len(times)

    rng = np.random.default_rng(42)

    release_rates = np.zeros(n)
    cumulative = np.zeros(n)
    m_remaining = dose_mg

    plateau_steps = 0

    for i in range(n):
        t = times[i]
        if t < t_lag_h:
            rate = 0.0
        elif m_remaining > 0.1 * dose_mg:
            # Zero-order plateau with device-specific variability
            noise = 1.0 + variability * rng.standard_normal()
            rate = plateau_rate * max(0.0, noise)
            plateau_steps += 1
        else:
            # First-order tail
            rate = k_falloff * max(0.0, m_remaining)

        # Cap rate so we don't exceed remaining drug
        rate = min(rate, m_remaining / dt_h) if dt_h > 0 else 0.0

        release_rates[i] = rate
        dm = rate * dt_h
        m_remaining = max(0.0, m_remaining - dm)
        cumulative[i] = dose_mg - m_remaining

    plateau_duration_h = float(plateau_steps * dt_h)

    # Release duration: time to 80% released
    target_80 = 0.80 * dose_mg
    release_duration_h = float(t_end_h)
    for i in range(n):
        if cumulative[i] >= target_80:
            release_duration_h = float(times[i])
            break

    # Fractions at specific times
    def _frac_at_time(target_t: float) -> float:
        idx = np.searchsorted(times, target_t)
        idx = min(idx, n - 1)
        return float(cumulative[idx] / dose_mg)

    f_at_4h = _frac_at_time(4.0)
    f_at_12h = _frac_at_time(12.0)

    # --- PK simulation: 1-cpt with release as input ---
    ke = cl_L_per_h / vd_L  # elimination rate constant [/h]

    c_plasma = np.zeros(n)
    for i in range(1, n):
        input_rate = release_rates[i - 1]  # mg/h
        dC = (input_rate / vd_L) - ke * c_plasma[i - 1]
        c_plasma[i] = max(0.0, c_plasma[i - 1] + dC * dt_h)

    cmax = float(np.max(c_plasma))
    tmax_idx = int(np.argmax(c_plasma))
    tmax = float(times[tmax_idx])
    auc = float(np_trapz(c_plasma, times))

    # Fluctuation at plateau pseudo-steady-state
    # Find plateau region (where release_rates are in zero-order phase)
    plateau_mask = (times >= t_lag_h) & (release_rates > plateau_rate * 0.5)
    if np.sum(plateau_mask) >= 3:
        c_plateau = c_plasma[plateau_mask]
        c_max_p = float(np.max(c_plateau))
        c_min_p = float(np.min(c_plateau))
        c_avg_p = float(np.mean(c_plateau))
        fluctuation = (c_max_p - c_min_p) / c_avg_p * 100.0 if c_avg_p > 0 else 0.0
    else:
        fluctuation = 0.0

    notes = (
        f"{device_type} device; membrane thickness {membrane_thickness_um} µm; "
        f"plateau rate {plateau_rate:.3f} mg/h; efficiency {efficiency * 100:.0f}%"
    )

    return OsmoticReleaseResult(
        drug_name=drug_name,
        device_type=device_type,
        dose_mg=dose_mg,
        membrane_permeability=membrane_permeability,
        times_h=times.tolist(),
        release_rate_mg_per_h=release_rates.tolist(),
        cumulative_released_mg=cumulative.tolist(),
        release_duration_h=release_duration_h,
        plateau_rate_mg_per_h=plateau_rate,
        plateau_duration_h=plateau_duration_h,
        t_lag_h=t_lag_h,
        f_released_at_4h=f_at_4h,
        f_released_at_12h=f_at_12h,
        times_pk_h=times.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        fluctuation_pct=fluctuation,
        notes=notes,
    )


def compare_osmotic_devices(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    drug_solubility_mg_mL: float,
) -> dict:
    """Simulate all 3 osmotic device types and compare.

    Parameters
    ----------
    drug_name:
        Name of drug.
    dose_mg:
        Total dose (mg).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    drug_solubility_mg_mL:
        Drug solubility in core (mg/mL).

    Returns
    -------
    dict with keys "elementary_osmotic", "push_pull", "mini_osmotic",
    and "best_zero_order" (device name with lowest fluctuation).
    """
    results: dict = {}
    for device in sorted(_VALID_DEVICE_TYPES):
        results[device] = simulate_osmotic_release(
            drug_name=drug_name,
            dose_mg=dose_mg,
            device_type=device,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            drug_solubility_mg_mL=drug_solubility_mg_mL,
        )

    best = min(results, key=lambda k: results[k].fluctuation_pct)
    results["best_zero_order"] = best
    return results
