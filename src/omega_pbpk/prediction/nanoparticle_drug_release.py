"""Phase 290 — Drug Release from Nanoparticles.

Models drug release from polymeric nanoparticles via surface erosion,
diffusion, and burst release. Simulates cumulative release profile over time.
"""

from dataclasses import dataclass

VALID_MECHANISMS = {"diffusion", "erosion", "combined"}


@dataclass
class NanoparticleReleaseResult:
    """Result of nanoparticle drug release simulation."""

    drug_name: str
    dose_mg: float
    particle_diameter_nm: float
    drug_loading_pct: float
    release_mechanism: str
    burst_fraction: float
    times_h: list
    cumulative_release_pct: list
    t50_h: float | None
    t80_h: float | None
    total_released_pct: float
    release_class: str
    notes: str


def _validate_inputs(
    dose_mg: float,
    particle_diameter_nm: float,
    drug_loading_pct: float,
    k_release_per_h: float,
    burst_fraction: float,
    t_end_h: float,
    release_mechanism: str,
) -> None:
    """Validate all input parameters."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < particle_diameter_nm <= 1000):
        raise ValueError(f"particle_diameter_nm must be in (0, 1000], got {particle_diameter_nm}")
    if not (0 < drug_loading_pct <= 100):
        raise ValueError(f"drug_loading_pct must be in (0, 100], got {drug_loading_pct}")
    if k_release_per_h <= 0:
        raise ValueError(f"k_release_per_h must be > 0, got {k_release_per_h}")
    if not (0 <= burst_fraction < 1):
        raise ValueError(f"burst_fraction must be in [0, 1), got {burst_fraction}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if release_mechanism not in VALID_MECHANISMS:
        raise ValueError(
            f"release_mechanism must be one of {VALID_MECHANISMS}, got '{release_mechanism}'"
        )


def simulate_nanoparticle_release(
    drug_name: str,
    dose_mg: float,
    particle_diameter_nm: float,
    drug_loading_pct: float,
    release_mechanism: str,
    k_release_per_h: float,
    burst_fraction: float = 0.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> NanoparticleReleaseResult:
    """Simulate drug release from polymeric nanoparticles.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Total dose in mg. Must be > 0.
    particle_diameter_nm:
        Nanoparticle diameter in nm. Must be in (0, 1000].
    drug_loading_pct:
        Drug loading as % of particle mass. Must be in (0, 100].
    release_mechanism:
        One of "diffusion", "erosion", "combined".
    k_release_per_h:
        Release rate constant (1/h). Must be > 0.
    burst_fraction:
        Fraction of drug released immediately at t=0. Must be in [0, 1).
    t_end_h:
        Simulation end time in hours. Must be > 0.
    dt_h:
        Time step for Forward Euler integration. Default 0.1 h.

    Returns
    -------
    NanoparticleReleaseResult
        Dataclass with cumulative release profile and pharmacokinetic metrics.
    """
    _validate_inputs(
        dose_mg,
        particle_diameter_nm,
        drug_loading_pct,
        k_release_per_h,
        burst_fraction,
        t_end_h,
        release_mechanism,
    )

    # Initial drug amount
    m0 = dose_mg * drug_loading_pct / 100.0

    # Burst release at t=0
    m_burst = m0 * burst_fraction
    m_remaining = m0 - m_burst

    # Forward Euler simulation
    times_h = [0.0]
    cumulative_mg = [m_burst]
    m_unreleased = m_remaining

    t = 0.0
    while t < t_end_h - 1e-9:
        step = min(dt_h, t_end_h - t)

        if m_unreleased <= 0.0:
            t += step
            times_h.append(round(t, 10))
            cumulative_mg.append(m0)
            continue

        if release_mechanism == "diffusion":
            # First-order: dM/dt = k * M_unreleased
            dm = k_release_per_h * m_unreleased * step
        elif release_mechanism == "erosion":
            # Zero-order: dM/dt = k * M0 (constant rate)
            dm = k_release_per_h * m0 * step
        else:
            # Combined: average of diffusion and erosion rates
            dm_diff = k_release_per_h * m_unreleased * step
            dm_eros = k_release_per_h * m0 * step
            dm = 0.5 * (dm_diff + dm_eros)

        # Clamp to available unreleased drug
        dm = min(dm, m_unreleased)
        m_unreleased -= dm

        t += step
        times_h.append(round(t, 10))
        cumulative_mg.append(cumulative_mg[-1] + dm)

    # Convert to percent
    cumulative_release_pct = [min(100.0, val / m0 * 100.0) for val in cumulative_mg]

    total_released_pct = cumulative_release_pct[-1]

    # Find t50 and t80
    t50_h: float | None = None
    t80_h: float | None = None
    for i, pct in enumerate(cumulative_release_pct):
        if t50_h is None and pct >= 50.0:
            t50_h = times_h[i]
        if t80_h is None and pct >= 80.0:
            t80_h = times_h[i]
        if t50_h is not None and t80_h is not None:
            break

    # Release class based on t80
    if t80_h is not None:
        if t80_h < 4.0:
            release_class = "fast"
        elif t80_h <= 24.0:
            release_class = "sustained"
        else:
            release_class = "extended"
    else:
        release_class = "extended"

    notes = (
        f"M0={m0:.3f} mg (drug load {drug_loading_pct}% of {dose_mg} mg dose); "
        f"burst={burst_fraction * 100:.0f}% ({m_burst:.3f} mg); "
        f"mechanism={release_mechanism}; k={k_release_per_h:.3f}/h; "
        f"diameter={particle_diameter_nm} nm"
    )

    return NanoparticleReleaseResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_diameter_nm=particle_diameter_nm,
        drug_loading_pct=drug_loading_pct,
        release_mechanism=release_mechanism,
        burst_fraction=burst_fraction,
        times_h=times_h,
        cumulative_release_pct=cumulative_release_pct,
        t50_h=t50_h,
        t80_h=t80_h,
        total_released_pct=total_released_pct,
        release_class=release_class,
        notes=notes,
    )


def compare_mechanisms(
    drug_name: str,
    dose_mg: float,
    particle_diameter_nm: float,
    drug_loading_pct: float,
    k_release_per_h: float,
    burst_fraction: float = 0.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[NanoparticleReleaseResult]:
    """Compare all three release mechanisms, sorted by t50_h ascending.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Total dose in mg.
    particle_diameter_nm:
        Nanoparticle diameter in nm.
    drug_loading_pct:
        Drug loading as % of particle mass.
    k_release_per_h:
        Release rate constant (1/h).
    burst_fraction:
        Fraction of drug released immediately at t=0.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step for Forward Euler integration.

    Returns
    -------
    list[NanoparticleReleaseResult]
        Results for all three mechanisms sorted by t50_h ascending
        (fastest release first). Mechanisms where t50 is not reached
        within t_end_h are placed at the end.
    """
    results = []
    for mechanism in VALID_MECHANISMS:
        result = simulate_nanoparticle_release(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_diameter_nm=particle_diameter_nm,
            drug_loading_pct=drug_loading_pct,
            release_mechanism=mechanism,
            k_release_per_h=k_release_per_h,
            burst_fraction=burst_fraction,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    # Sort by t50_h ascending; None (not reached) goes to the end
    results.sort(key=lambda r: r.t50_h if r.t50_h is not None else float("inf"))
    return results
