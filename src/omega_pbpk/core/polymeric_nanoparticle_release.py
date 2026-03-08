"""
Phase 396: Drug Release from Polymeric Nanoparticles
Models drug release kinetics from polymeric nanoparticles using multiple
release mechanisms (burst, diffusion, erosion, combined).
"""

import math
from dataclasses import dataclass

_POLYMER_PARAMS = {
    "PLGA": {"burst_pct": 15.0, "k_diffusion": 0.008, "k_erosion": 0.003, "mechanism": "combined"},
    "PLA": {"burst_pct": 8.0, "k_diffusion": 0.004, "k_erosion": 0.001, "mechanism": "erosion"},
    "chitosan": {
        "burst_pct": 25.0,
        "k_diffusion": 0.015,
        "k_erosion": 0.0,
        "mechanism": "diffusion",
    },
    "PEG": {"burst_pct": 20.0, "k_diffusion": 0.020, "k_erosion": 0.0, "mechanism": "burst"},
}

_KORSMEYER_N = {
    "burst": 0.3,
    "diffusion": 0.45,
    "erosion": 0.9,
    "combined": 0.6,
}

_HIGUCHI_R2 = {
    "burst": 0.6,
    "diffusion": 0.95,
    "erosion": 0.8,
    "combined": 0.85,
}


@dataclass
class NanoparticleReleaseResult:
    drug_name: str
    polymer_type: str
    release_mechanism: str
    total_drug_mg: float
    particle_size_nm: float
    times_h: list
    cumulative_release_pct: list
    release_rate_pct_per_h: list
    t50_h: float
    t80_h: float
    burst_fraction_pct: float
    sustained_duration_h: float
    higuchi_fit_r2: float
    korsmeyer_n: float
    release_class: str
    notes: str


def _validate_inputs(total_drug_mg, particle_size_nm, drug_loading_pct, polymer_type, t_end_h):
    if total_drug_mg <= 0:
        raise ValueError(f"total_drug_mg must be > 0, got {total_drug_mg}")
    if particle_size_nm <= 0:
        raise ValueError(f"particle_size_nm must be > 0, got {particle_size_nm}")
    if not (0 < drug_loading_pct <= 100):
        raise ValueError(f"drug_loading_pct must be in (0, 100], got {drug_loading_pct}")
    if polymer_type not in _POLYMER_PARAMS:
        raise ValueError(
            f"polymer_type must be one of {list(_POLYMER_PARAMS.keys())}, got '{polymer_type}'"
        )
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")


def _find_time_for_threshold(times_h, cumulative_release_pct, threshold_pct):
    """Return interpolated time at which cumulative release reaches threshold_pct."""
    for i in range(1, len(cumulative_release_pct)):
        if cumulative_release_pct[i] >= threshold_pct:
            # Linear interpolation
            r0 = cumulative_release_pct[i - 1]
            r1 = cumulative_release_pct[i]
            t0 = times_h[i - 1]
            t1 = times_h[i]
            if r1 > r0:
                frac = (threshold_pct - r0) / (r1 - r0)
                return t0 + frac * (t1 - t0)
            return t1
    # Never reached; return end time
    return times_h[-1]


def _classify_release(burst_fraction_pct, korsmeyer_n, mechanism):
    if burst_fraction_pct > 30.0:
        return "burst_dominated"
    if mechanism == "erosion":
        return "erosion_controlled"
    return "diffusion_controlled"


def _build_notes(
    polymer_type, mechanism, burst_fraction_pct, t50_h, t80_h, particle_size_nm, release_class
):
    return (
        f"{polymer_type} nanoparticles ({particle_size_nm:.0f} nm) exhibit "
        f"{mechanism}-based release. "
        f"Initial burst: {burst_fraction_pct:.1f}% in first hour. "
        f"t50 = {t50_h:.1f} h, t80 = {t80_h:.1f} h. "
        f"Classified as '{release_class}'."
    )


def simulate_nanoparticle_release(
    drug_name,
    total_drug_mg,
    polymer_type,
    particle_size_nm=200.0,
    drug_loading_pct=10.0,
    t_end_h=720.0,
    dt_h=1.0,
):
    """Simulate drug release from polymeric nanoparticles.

    Parameters
    ----------
    drug_name : str
        Name of encapsulated drug.
    total_drug_mg : float
        Total drug mass in mg (> 0).
    polymer_type : str
        One of "PLGA", "PLA", "chitosan", "PEG".
    particle_size_nm : float
        Mean particle diameter in nm (> 0). Default 200 nm.
    drug_loading_pct : float
        Drug loading percentage (0, 100]. Default 10%.
    t_end_h : float
        Simulation end time in hours (> 0). Default 720 h (30 days).
    dt_h : float
        Time step in hours. Default 1 h.

    Returns
    -------
    NanoparticleReleaseResult
    """
    _validate_inputs(total_drug_mg, particle_size_nm, drug_loading_pct, polymer_type, t_end_h)

    params = _POLYMER_PARAMS[polymer_type]
    burst_pct = params["burst_pct"]
    k_diffusion = params["k_diffusion"]
    k_erosion = params["k_erosion"]
    mechanism = params["mechanism"]

    # Size effect: smaller particles release faster
    size_factor = (200.0 / particle_size_nm) ** 0.5

    # Combined rate constant for first-order release after burst
    k_total = (k_diffusion + k_erosion) * size_factor

    burst_fraction = burst_pct / 100.0

    # Build time series via Forward Euler
    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h = []
    cumulative_release_pct = []
    release_rate_pct_per_h = []

    # t=0: burst happens immediately
    R_frac = burst_fraction  # cumulative released fraction
    times_h.append(0.0)
    cumulative_release_pct.append(R_frac * 100.0)
    release_rate_pct_per_h.append(burst_pct)  # burst at t=0

    for step in range(1, n_steps + 1):
        t = step * dt_h
        if t > t_end_h:
            t = t_end_h

        # First-order release of remaining drug
        remaining_frac = 1.0 - R_frac
        delta_frac = k_total * remaining_frac * dt_h
        R_frac_new = min(1.0, R_frac + delta_frac)

        rate_pct_per_h = (R_frac_new - R_frac) / dt_h * 100.0

        times_h.append(t)
        cumulative_release_pct.append(R_frac_new * 100.0)
        release_rate_pct_per_h.append(rate_pct_per_h)

        R_frac = R_frac_new

        if t >= t_end_h:
            break

    # t50 and t80
    t50_h = _find_time_for_threshold(times_h, cumulative_release_pct, 50.0)
    t80_h = _find_time_for_threshold(times_h, cumulative_release_pct, 80.0)

    # Burst fraction at approximately t=1h (index 1 if dt=1, else nearest)
    if len(cumulative_release_pct) > 1:
        burst_fraction_pct = cumulative_release_pct[1]
    else:
        burst_fraction_pct = cumulative_release_pct[0]

    sustained_duration_h = t80_h

    korsmeyer_n = _KORSMEYER_N[mechanism]
    higuchi_fit_r2 = _HIGUCHI_R2[mechanism]

    release_class = _classify_release(burst_fraction_pct, korsmeyer_n, mechanism)

    notes = _build_notes(
        polymer_type, mechanism, burst_fraction_pct, t50_h, t80_h, particle_size_nm, release_class
    )

    return NanoparticleReleaseResult(
        drug_name=drug_name,
        polymer_type=polymer_type,
        release_mechanism=mechanism,
        total_drug_mg=total_drug_mg,
        particle_size_nm=particle_size_nm,
        times_h=times_h,
        cumulative_release_pct=cumulative_release_pct,
        release_rate_pct_per_h=release_rate_pct_per_h,
        t50_h=t50_h,
        t80_h=t80_h,
        burst_fraction_pct=burst_fraction_pct,
        sustained_duration_h=sustained_duration_h,
        higuchi_fit_r2=higuchi_fit_r2,
        korsmeyer_n=korsmeyer_n,
        release_class=release_class,
        notes=notes,
    )


def compare_polymers(drug_name, total_drug_mg, particle_size_nm=200.0):
    """Compare drug release profiles across all supported polymers.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    total_drug_mg : float
        Total drug mass in mg.
    particle_size_nm : float
        Mean particle diameter in nm. Default 200 nm.

    Returns
    -------
    list of NanoparticleReleaseResult
        Sorted by t50_h ascending (fastest release first).
    """
    results = []
    for polymer in _POLYMER_PARAMS:
        result = simulate_nanoparticle_release(
            drug_name=drug_name,
            total_drug_mg=total_drug_mg,
            polymer_type=polymer,
            particle_size_nm=particle_size_nm,
        )
        results.append(result)

    results.sort(key=lambda r: r.t50_h)
    return results
