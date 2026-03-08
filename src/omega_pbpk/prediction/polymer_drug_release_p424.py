"""
Phase 424: Polymer Drug Release Prediction
Predict drug release from polymer matrix systems (PLGA microspheres, PLA nanoparticles).
Combined diffusion + erosion model. Pure Python only (stdlib: math, dataclasses).
"""

import math
from dataclasses import dataclass

# Polymer database: degradation half-lives, burst factor, diffusion factor
POLYMERS = {
    "PLGA_50_50": {"t_half_days": 28, "burst_factor": 0.15, "diffusion_factor": 0.6},
    "PLGA_75_25": {"t_half_days": 56, "burst_factor": 0.10, "diffusion_factor": 0.4},
    "PLA": {"t_half_days": 180, "burst_factor": 0.05, "diffusion_factor": 0.3},
    "PCL": {"t_half_days": 365, "burst_factor": 0.03, "diffusion_factor": 0.2},
}


@dataclass
class PolymerReleaseResult:
    """Result of polymer drug release simulation."""

    drug_name: str
    polymer_type: str
    particle_diameter_um: float
    drug_loading_pct: float
    times_days: list
    cumulative_release_pct: list
    release_rate_pct_per_day: list
    t50_days: float
    t90_days: float
    burst_release_pct: float
    release_mechanism: str
    duration_days: float
    notes: str


def _cumulative_at_t(
    t: float,
    burst_pct: float,
    k_erosion: float,
    particle_size_factor: float,
) -> float:
    """
    Compute cumulative release % at time t (days).
    Model: burst + erosion (first-order) with particle size scaling.
    """
    remaining_pct = 100.0 - burst_pct
    erosion_fraction = 1.0 - math.exp(-k_erosion * particle_size_factor * t)
    raw = burst_pct + remaining_pct * erosion_fraction
    return min(100.0, raw)


def simulate_polymer_release(
    drug_name: str,
    polymer_type: str,
    particle_diameter_um: float,
    drug_loading_pct: float,
    t_end_days: float = 90.0,
    dt_days: float = 1.0,
) -> PolymerReleaseResult:
    """
    Simulate cumulative drug release from a polymer matrix system.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    polymer_type : str
        One of: "PLGA_50_50", "PLGA_75_25", "PLA", "PCL".
    particle_diameter_um : float
        Particle diameter in micrometers (> 0).
    drug_loading_pct : float
        Drug loading as % by weight (0 < x <= 100).
    t_end_days : float
        Simulation end time (days).
    dt_days : float
        Time step (days).

    Returns
    -------
    PolymerReleaseResult
    """
    # Validation
    if polymer_type not in POLYMERS:
        raise ValueError(
            f"polymer_type '{polymer_type}' not recognized. Valid options: {list(POLYMERS.keys())}"
        )
    if particle_diameter_um <= 0:
        raise ValueError("particle_diameter_um must be > 0")
    if not (0 < drug_loading_pct <= 100):
        raise ValueError("drug_loading_pct must be in (0, 100]")

    params = POLYMERS[polymer_type]
    t_half_days = params["t_half_days"]
    burst_factor = params["burst_factor"]
    diffusion_factor = params["diffusion_factor"]

    k_erosion = 0.693 / t_half_days  # first-order erosion rate constant (1/day)

    # Particle size factor: smaller particle -> faster diffusion
    # Reference diameter: 10 um
    particle_size_factor = (10.0 / particle_diameter_um) ** 0.5

    burst_pct = burst_factor * 100.0  # convert to %

    # Build time series
    n_steps = int(round(t_end_days / dt_days)) + 1
    times_days = []
    cumulative_release_pct = []

    for step in range(n_steps):
        t = step * dt_days
        times_days.append(t)
        c = _cumulative_at_t(t, burst_pct, k_erosion, particle_size_factor)
        cumulative_release_pct.append(c)

    # Release rate: finite difference (first entry = cumulative[0] as per spec)
    release_rate_pct_per_day = [cumulative_release_pct[0]]
    for i in range(1, len(cumulative_release_pct)):
        dt = times_days[i] - times_days[i - 1]
        if dt > 0:
            rate = (cumulative_release_pct[i] - cumulative_release_pct[i - 1]) / dt
        else:
            rate = 0.0
        release_rate_pct_per_day.append(rate)

    # Burst release: cumulative at t = 1 day
    # Find the first index where t >= 1
    burst_release_pct_val = 0.0
    for i, t in enumerate(times_days):
        if t >= 1.0:
            burst_release_pct_val = cumulative_release_pct[i]
            break
    else:
        # t_end < 1, use last point
        burst_release_pct_val = cumulative_release_pct[-1]

    # t50 and t90
    t50_days = float("inf")
    t90_days = float("inf")
    for i, cum in enumerate(cumulative_release_pct):
        if cum >= 50.0 and t50_days == float("inf"):
            t50_days = times_days[i]
        if cum >= 90.0 and t90_days == float("inf"):
            t90_days = times_days[i]

    # Duration
    duration_days = t90_days if t90_days != float("inf") else t_end_days

    # Release mechanism determination
    if k_erosion * t_half_days > 1:
        release_mechanism = "erosion_dominated"
    elif diffusion_factor > 0.5:
        release_mechanism = "diffusion_dominated"
    else:
        release_mechanism = "mixed"

    notes = (
        f"Polymer: {polymer_type}, t_half={t_half_days} days, "
        f"k_erosion={k_erosion:.4f}/day. "
        f"Particle size factor: {particle_size_factor:.3f}. "
        f"Burst release: {burst_release_pct_val:.1f}% at 1 day. "
        f"t50={t50_days:.1f} days, t90={t90_days if t90_days != float('inf') else '>t_end'} days."
    )

    return PolymerReleaseResult(
        drug_name=drug_name,
        polymer_type=polymer_type,
        particle_diameter_um=particle_diameter_um,
        drug_loading_pct=drug_loading_pct,
        times_days=times_days,
        cumulative_release_pct=cumulative_release_pct,
        release_rate_pct_per_day=release_rate_pct_per_day,
        t50_days=t50_days,
        t90_days=t90_days,
        burst_release_pct=burst_release_pct_val,
        release_mechanism=release_mechanism,
        duration_days=duration_days,
        notes=notes,
    )


def compare_polymers(
    drug_name: str,
    particle_diameter_um: float,
    drug_loading_pct: float,
) -> list[PolymerReleaseResult]:
    """
    Compare all 4 polymer types for a given drug and particle size.

    Returns list of PolymerReleaseResult sorted by t50_days ascending
    (fastest release first).
    """
    results = []
    for polymer_type in POLYMERS:
        result = simulate_polymer_release(
            drug_name=drug_name,
            polymer_type=polymer_type,
            particle_diameter_um=particle_diameter_um,
            drug_loading_pct=drug_loading_pct,
        )
        results.append(result)

    results.sort(key=lambda r: r.t50_days)
    return results
