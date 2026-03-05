"""Circadian pharmacokinetics — time-varying physiology over 24h cycle."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class CircadianRhythm:
    """Cosine rhythm: value(t) = mesor + amplitude * cos(2π(t - acrophase)/period)."""

    mesor: float  # 24h mean value
    amplitude: float  # peak-to-trough / 2
    acrophase_h: float  # time of peak (hours from midnight)
    period_h: float = 24.0

    def value_at(self, time_h: float) -> float:
        """Return physiological value at given clock time (hours from midnight)."""
        phase = 2.0 * math.pi * (time_h - self.acrophase_h) / self.period_h
        return self.mesor + self.amplitude * math.cos(phase)

    def values_at(self, times_h: np.ndarray) -> np.ndarray:
        """Vectorized evaluation."""
        phase = 2.0 * np.pi * (times_h - self.acrophase_h) / self.period_h
        return self.mesor + self.amplitude * np.cos(phase)


# ── Standard circadian rhythms (literature-based) ───────────────────

# GFR: peaks mid-afternoon (~14:00), ~20% amplitude
GFR_RHYTHM = CircadianRhythm(mesor=120.0, amplitude=24.0, acrophase_h=14.0)

# Hepatic blood flow: peaks early afternoon (~13:00), ~15% amplitude
HEPATIC_FLOW_RHYTHM = CircadianRhythm(mesor=90.0, amplitude=13.5, acrophase_h=13.0)

# CYP3A4 activity: peaks early morning (~08:00), ~30% amplitude
CYP3A4_RHYTHM = CircadianRhythm(mesor=1.0, amplitude=0.3, acrophase_h=8.0)

# Gastric emptying rate: faster in morning, ~25% amplitude
GASTRIC_EMPTYING_RHYTHM = CircadianRhythm(mesor=1.0, amplitude=0.25, acrophase_h=8.0)

# Renal tubular reabsorption: peaks overnight (~02:00), ~15% amplitude
REABSORPTION_RHYTHM = CircadianRhythm(mesor=1.0, amplitude=0.15, acrophase_h=2.0)


@dataclass(frozen=True)
class CircadianPhysiology:
    """Collection of circadian rhythms for a simulation."""

    gfr: CircadianRhythm = field(default_factory=lambda: GFR_RHYTHM)
    hepatic_flow: CircadianRhythm = field(default_factory=lambda: HEPATIC_FLOW_RHYTHM)
    cyp3a4: CircadianRhythm = field(default_factory=lambda: CYP3A4_RHYTHM)
    gastric_emptying: CircadianRhythm = field(default_factory=lambda: GASTRIC_EMPTYING_RHYTHM)
    reabsorption: CircadianRhythm = field(default_factory=lambda: REABSORPTION_RHYTHM)


@dataclass(frozen=True)
class ChronoPKResult:
    """Result of circadian PK simulation."""

    times_h: list[float]
    plasma_conc: list[float]
    auc_0_24: float
    cmax: float
    tmax_h: float
    cmin: float
    gfr_profile: list[float]
    hepatic_flow_profile: list[float]
    cyp3a4_profile: list[float]
    dosing_time_h: float
    cl_effective_L_per_h: float  # time-averaged effective clearance


def simulate_chronopk(
    dose_mg: float,
    cl_base_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 0.0,
    fup: float = 1.0,
    dosing_time_h: float = 8.0,
    route: str = "oral",
    physiology: CircadianPhysiology | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    fraction_renal: float = 0.3,
    fraction_hepatic: float = 0.7,
) -> ChronoPKResult:
    """Simulate 1-compartment PK with circadian-varying clearance.

    Parameters
    ----------
    dose_mg : float
        Dose in mg.
    cl_base_L_per_h : float
        Baseline total clearance (L/h) at mesor physiology.
    vd_L : float
        Volume of distribution (L).
    ka_per_h : float
        Absorption rate constant (1/h). 0 for IV.
    fup : float
        Fraction unbound in plasma.
    dosing_time_h : float
        Clock time of dose administration (hours from midnight).
    route : str
        "oral" or "iv".
    physiology : CircadianPhysiology | None
        Circadian rhythms. None uses defaults.
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).
    fraction_renal : float
        Fraction of CL that is renal (modulated by GFR rhythm).
    fraction_hepatic : float
        Fraction of CL that is hepatic (modulated by hepatic flow + CYP rhythm).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_base_L_per_h <= 0:
        raise ValueError("cl_base_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    if physiology is None:
        physiology = CircadianPhysiology()

    fup = float(np.clip(fup, 1e-6, 1.0))
    is_oral = route.lower() == "oral"

    n_steps = max(int(t_end_h / dt_h), 1)
    t_sim = np.linspace(0.0, t_end_h, n_steps + 1)
    # Clock times (wrap around 24h)
    t_clock = (t_sim + dosing_time_h) % 24.0

    # Circadian physiology profiles
    gfr_profile = physiology.gfr.values_at(t_clock)
    hflow_profile = physiology.hepatic_flow.values_at(t_clock)
    cyp_profile = physiology.cyp3a4.values_at(t_clock)

    # Normalize to mesor (relative scaling factors)
    gfr_factor = gfr_profile / physiology.gfr.mesor
    hflow_factor = hflow_profile / physiology.hepatic_flow.mesor
    cyp_factor = cyp_profile / physiology.cyp3a4.mesor

    # Time-varying clearance
    cl_renal_t = cl_base_L_per_h * fraction_renal * gfr_factor
    cl_hepatic_t = cl_base_L_per_h * fraction_hepatic * hflow_factor * cyp_factor
    cl_total_t = cl_renal_t + cl_hepatic_t

    # Forward Euler integration of 1-cpt model
    # dA_gut/dt = -ka * A_gut  (oral only)
    # dA_central/dt = ka * A_gut - (CL(t)/Vd) * A_central
    cp = np.zeros(n_steps + 1)
    a_gut = dose_mg if is_oral else 0.0
    a_central = 0.0 if is_oral else dose_mg

    for i in range(n_steps):
        ke_t = cl_total_t[i] / vd_L
        if is_oral and ka_per_h > 0:
            absorption = ka_per_h * a_gut * dt_h
            a_gut -= absorption
            a_central += absorption
        elimination = ke_t * a_central * dt_h
        a_central -= elimination
        a_central = max(a_central, 0.0)
        a_gut = max(a_gut, 0.0)
        cp[i + 1] = a_central / vd_L

    # For IV, set t=0 concentration
    if not is_oral:
        cp[0] = dose_mg / vd_L

    # PK metrics
    auc_0_24 = float(np_trapz(cp, t_sim))
    cmax = float(np.max(cp))
    tmax_idx = int(np.argmax(cp))
    tmax_h = float(t_sim[tmax_idx])
    cmin = float(np.min(cp[1:])) if len(cp) > 1 else 0.0

    # Effective clearance (time-averaged)
    cl_eff = float(np.mean(cl_total_t))

    return ChronoPKResult(
        times_h=t_sim.tolist(),
        plasma_conc=cp.tolist(),
        auc_0_24=auc_0_24,
        cmax=cmax,
        tmax_h=tmax_h,
        cmin=cmin,
        gfr_profile=gfr_profile.tolist(),
        hepatic_flow_profile=hflow_profile.tolist(),
        cyp3a4_profile=cyp_profile.tolist(),
        dosing_time_h=dosing_time_h,
        cl_effective_L_per_h=cl_eff,
    )


def compare_dosing_times(
    dose_mg: float,
    cl_base_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    fup: float = 1.0,
    dosing_times_h: list[float] | None = None,
    route: str = "oral",
    physiology: CircadianPhysiology | None = None,
    fraction_renal: float = 0.3,
    fraction_hepatic: float = 0.7,
) -> list[ChronoPKResult]:
    """Compare PK at different dosing times to find optimal chronotherapy schedule.

    Parameters
    ----------
    dosing_times_h : list[float] | None
        Clock times to compare. Default: [6, 8, 12, 18, 22] (morning to night).
    """
    if dosing_times_h is None:
        dosing_times_h = [6.0, 8.0, 12.0, 18.0, 22.0]

    results = []
    for dt in dosing_times_h:
        r = simulate_chronopk(
            dose_mg=dose_mg,
            cl_base_L_per_h=cl_base_L_per_h,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            fup=fup,
            dosing_time_h=dt,
            route=route,
            physiology=physiology,
            fraction_renal=fraction_renal,
            fraction_hepatic=fraction_hepatic,
        )
        results.append(r)
    return results


__all__ = [
    "CircadianRhythm",
    "GFR_RHYTHM",
    "HEPATIC_FLOW_RHYTHM",
    "CYP3A4_RHYTHM",
    "GASTRIC_EMPTYING_RHYTHM",
    "REABSORPTION_RHYTHM",
    "CircadianPhysiology",
    "ChronoPKResult",
    "simulate_chronopk",
    "compare_dosing_times",
]
