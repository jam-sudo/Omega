"""Microparticle drug delivery PK — biodegradable PLGA microspheres (Phase 370).

Models biphasic drug release (burst + sustained) from biodegradable polymer
microparticles injected subcutaneously or intramuscularly, with systemic PK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# Polymer degradation parameters: (k_deg [/day], base_burst_fraction)
_POLYMER_PARAMS: dict[str, tuple[float, float]] = {
    "PLGA_50_50": (0.07, 0.20),
    "PLGA_75_25": (0.04, 0.15),
    "PLA": (0.02, 0.10),
    "PCL": (0.005, 0.05),
}

_VALID_POLYMERS = set(_POLYMER_PARAMS.keys())

# Absorption rate from depot to systemic (ka_site [/day])
_SITE_KA: dict[str, float] = {
    "sc": 2.0,
    "im": 3.0,
}

_F_DEPOT = 0.9  # fraction of released drug absorbed from depot


@dataclass(frozen=True)
class MicroparticlePKResult:
    """Result of microparticle drug delivery PK simulation."""

    drug_name: str
    polymer: str  # "PLGA_50_50", "PLGA_75_25", "PLA", "PCL"
    dose_mg: float
    particle_size_um: float

    # Release profile
    times_day: list[float]  # Time in days
    release_rate_mg_per_day: list[float]
    cumulative_released_pct: list[float]

    # Release parameters
    burst_fraction: float  # Initial burst (fraction released in first 24h)
    t50_pct_days: float  # Time to 50% release
    t90_pct_days: float  # Time to 90% release

    # Polymer degradation
    degradation_rate_per_day: float  # First-order hydrolysis rate

    # PK integration
    times_pk_day: list[float]
    c_plasma_ug_per_mL: list[float]  # Drug concentration in µg/mL
    cmax_ug_per_mL: float
    tmax_day: float
    auc_ug_day_per_mL: float

    # Bioavailability from depot
    f_depot: float  # Fraction of dose that reaches systemic circulation

    notes: str


def _compute_burst_fraction(base_burst: float, particle_size_um: float) -> float:
    """Calculate effective burst fraction accounting for particle size.

    Smaller particles have higher surface area-to-volume ratio and release
    more drug rapidly.
    """
    burst = base_burst * min(2.0, 10.0 / particle_size_um)
    return min(0.5, burst)


def simulate_microparticle_pk(
    drug_name: str,
    dose_mg: float,
    polymer: str = "PLGA_50_50",
    particle_size_um: float = 50.0,
    cl_L_per_day: float = 2.0,
    vd_L: float = 50.0,
    injection_site: str = "sc",
    t_end_days: float = 30.0,
    dt_days: float = 0.1,
) -> MicroparticlePKResult:
    """Simulate microparticle biphasic drug release and systemic PK.

    Parameters
    ----------
    drug_name:
        Name of drug being simulated.
    dose_mg:
        Total drug dose loaded in microparticles (mg).
    polymer:
        Biodegradable polymer matrix. One of "PLGA_50_50", "PLGA_75_25",
        "PLA", "PCL".
    particle_size_um:
        Mean particle diameter (µm). Smaller particles have higher burst.
    cl_L_per_day:
        Systemic clearance (L/day).
    vd_L:
        Volume of distribution (L).
    injection_site:
        Injection route: "sc" (subcutaneous) or "im" (intramuscular).
    t_end_days:
        Simulation end time (days).
    dt_days:
        Time step size (days).

    Returns
    -------
    MicroparticlePKResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_day <= 0:
        raise ValueError(f"cl_L_per_day must be > 0, got {cl_L_per_day}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if polymer not in _VALID_POLYMERS:
        raise ValueError(f"polymer must be one of {_VALID_POLYMERS}, got {polymer!r}")

    k_deg, base_burst = _POLYMER_PARAMS[polymer]
    burst_fraction = _compute_burst_fraction(base_burst, particle_size_um)

    # Drug amounts
    m_burst = burst_fraction * dose_mg  # released as impulse at t=0
    m_sustained = (1.0 - burst_fraction) * dose_mg  # sustained release depot

    # ka from depot site to systemic (not used as separate state here;
    # we assume near-complete absorption from the local depot — the
    # release rate is the rate-limiting step)
    _ka_site = _SITE_KA.get(injection_site, 2.0)

    # Time array
    times = np.arange(0.0, t_end_days + dt_days * 0.5, dt_days)
    n = len(times)

    ke = cl_L_per_day / vd_L  # elimination rate constant [/day]

    # --- Release ODE: first-order polymer degradation ---
    # m_depot[i] = remaining sustained-release drug at time i
    m_depot = np.zeros(n)
    m_depot[0] = m_sustained

    release_rates = np.zeros(n)
    # Track raw released drug (from particle matrix) for cumulative_released_pct
    cumulative_released_mg = np.zeros(n)

    # Burst at t=0 contributes to first time step release
    cumulative_released_mg[0] = m_burst  # burst released at t=0 (raw, pre-absorption)
    release_rates[0] = m_burst / dt_days if dt_days > 0 else 0.0

    for i in range(1, n):
        dm_release = k_deg * m_depot[i - 1] * dt_days
        dm_release = min(dm_release, m_depot[i - 1])
        m_depot[i] = max(0.0, m_depot[i - 1] - dm_release)
        rate = k_deg * m_depot[i - 1]  # mg/day
        release_rates[i] = rate
        cumulative_released_mg[i] = cumulative_released_mg[i - 1] + dm_release

    cumulative_pct = np.minimum(100.0, cumulative_released_mg / dose_mg * 100.0)

    # Milestone times based on raw released drug
    def _time_at_pct(target_pct: float) -> float:
        target_mg = target_pct / 100.0 * dose_mg
        for i in range(n):
            if cumulative_released_mg[i] >= target_mg:
                return float(times[i])
        return float(t_end_days)

    t50 = _time_at_pct(50.0)
    t90 = _time_at_pct(90.0)

    # --- PK: 1-cpt with release rate as drug input (absorbed fraction) ---
    # C in µg/mL = mg/L (since 1 mg/L = 1 µg/mL)
    # Burst: instant spike at t=0
    c_plasma = np.zeros(n)
    c_plasma[0] = m_burst * _F_DEPOT / vd_L  # mg/L = µg/mL

    for i in range(1, n):
        # Input = sustained release absorbed from depot
        input_rate_mg_per_day = release_rates[i - 1] * _F_DEPOT  # mg/day
        input_conc_rate = input_rate_mg_per_day / vd_L  # mg/L/day = µg/mL/day
        dC = input_conc_rate - ke * c_plasma[i - 1]
        c_plasma[i] = max(0.0, c_plasma[i - 1] + dC * dt_days)

    cmax = float(np.max(c_plasma))
    tmax_idx = int(np.argmax(c_plasma))
    tmax = float(times[tmax_idx])
    auc = float(np_trapz(c_plasma, times))

    notes = (
        f"{polymer} microparticles, d={particle_size_um} µm; "
        f"burst={burst_fraction * 100:.1f}%; k_deg={k_deg}/day; "
        f"site={injection_site}; f_depot={_F_DEPOT}"
    )

    return MicroparticlePKResult(
        drug_name=drug_name,
        polymer=polymer,
        dose_mg=dose_mg,
        particle_size_um=particle_size_um,
        times_day=times.tolist(),
        release_rate_mg_per_day=release_rates.tolist(),
        cumulative_released_pct=cumulative_pct.tolist(),
        burst_fraction=burst_fraction,
        t50_pct_days=t50,
        t90_pct_days=t90,
        degradation_rate_per_day=k_deg,
        times_pk_day=times.tolist(),
        c_plasma_ug_per_mL=c_plasma.tolist(),
        cmax_ug_per_mL=cmax,
        tmax_day=tmax,
        auc_ug_day_per_mL=auc,
        f_depot=_F_DEPOT,
        notes=notes,
    )


def compare_polymers(
    drug_name: str,
    dose_mg: float,
    cl_L_per_day: float,
    vd_L: float,
    particle_size_um: float,
) -> list[MicroparticlePKResult]:
    """Simulate all 4 polymer types and return sorted by t90 descending.

    Parameters
    ----------
    drug_name:
        Name of drug.
    dose_mg:
        Total dose (mg).
    cl_L_per_day:
        Systemic clearance (L/day).
    vd_L:
        Volume of distribution (L).
    particle_size_um:
        Mean particle diameter (µm).

    Returns
    -------
    List of MicroparticlePKResult sorted by t90_pct_days descending
    (longest-acting polymer first).
    """
    # Use a longer simulation window to allow even slow polymers to reach 90%.
    # The slowest polymer (PCL, k_deg=0.005/day) reaches 90% at ~460 days.
    # We use 500 days to ensure all polymers are fully characterised.
    _t_end_compare = 500.0

    results = []
    for polymer in _VALID_POLYMERS:
        result = simulate_microparticle_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            polymer=polymer,
            particle_size_um=particle_size_um,
            cl_L_per_day=cl_L_per_day,
            vd_L=vd_L,
            t_end_days=_t_end_compare,
            dt_days=1.0,
        )
        results.append(result)

    results.sort(key=lambda r: r.t90_pct_days, reverse=True)
    return results
