"""Solid lipid nanoparticle (SLN) pharmacokinetic simulation.

SLNs are lipid-based nanocarriers that protect drug from degradation and
modify release. This module models the dual-compartment (encapsulated /
free drug) behaviour of IV-administered SLNs using forward Euler integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_LIPID_PARAMS: dict[str, dict[str, float]] = {
    "glyceryl_palmitostearate": {
        "k_release": 0.03,
        "t_half_sln": 18.0,
        "mps_uptake": 0.35,
    },
    "cetyl_palmitate": {
        "k_release": 0.05,
        "t_half_sln": 12.0,
        "mps_uptake": 0.40,
    },
    "stearic_acid": {
        "k_release": 0.08,
        "t_half_sln": 8.0,
        "mps_uptake": 0.45,
    },
    "beeswax": {
        "k_release": 0.02,
        "t_half_sln": 24.0,
        "mps_uptake": 0.30,
    },
    "carnauba_wax": {
        "k_release": 0.01,
        "t_half_sln": 36.0,
        "mps_uptake": 0.25,
    },
}


@dataclass
class SLNPKResult:
    """PK result for a solid lipid nanoparticle formulation."""

    drug_name: str
    dose_mg: float
    lipid_type: str
    particle_size_nm: float
    times_h: list = field(default_factory=list)
    c_sln_mg_L: list = field(default_factory=list)
    c_free_mg_L: list = field(default_factory=list)
    cmax_free_mg_L: float = 0.0
    tmax_free_h: float = 0.0
    auc_free_mg_h_per_L: float = 0.0
    auc_total_mg_h_per_L: float = 0.0
    encapsulated_fraction_at_24h: float = 0.0
    t_half_sln_h: float = 0.0
    mps_uptake_fraction: float = 0.0
    notes: str = ""


def simulate_sln_pk(
    drug_name: str,
    dose_mg: float,
    lipid_type: str = "glyceryl_palmitostearate",
    particle_size_nm: float = 200.0,
    logp: float = 3.0,
    vd_free_L: float = 50.0,
    cl_free_L_per_h: float = 10.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.5,
) -> SLNPKResult:
    """Simulate PK of drug loaded in solid lipid nanoparticles.

    Parameters
    ----------
    drug_name:
        Name of the encapsulated drug.
    dose_mg:
        Total administered dose in mg (IV bolus).
    lipid_type:
        SLN lipid matrix — one of:
        'glyceryl_palmitostearate', 'cetyl_palmitate', 'stearic_acid',
        'beeswax', 'carnauba_wax'.
    particle_size_nm:
        Mean particle size in nm (default 200).
    logp:
        Octanol–water partition coefficient (informational / reserved).
    vd_free_L:
        Volume of distribution of free drug (L).
    cl_free_L_per_h:
        Clearance of free drug (L/h).
    t_end_h:
        Simulation duration in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    SLNPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if particle_size_nm <= 0:
        raise ValueError(f"particle_size_nm must be > 0, got {particle_size_nm}")
    if vd_free_L <= 0:
        raise ValueError(f"vd_free_L must be > 0, got {vd_free_L}")
    if cl_free_L_per_h <= 0:
        raise ValueError(f"cl_free_L_per_h must be > 0, got {cl_free_L_per_h}")
    if lipid_type not in _LIPID_PARAMS:
        raise ValueError(
            f"Invalid lipid_type '{lipid_type}'. Must be one of {sorted(_LIPID_PARAMS)}"
        )

    params = _LIPID_PARAMS[lipid_type]
    k_release = params["k_release"]
    t_half_sln = params["t_half_sln"]
    mps_uptake_base = params["mps_uptake"]

    # Particle size effect (smaller particles → faster MPS clearance)
    size_factor = (200.0 / particle_size_nm) ** 0.3

    # Rate constants
    k_sln_elim = math.log(2.0) / t_half_sln  # SLN elimination (/h)
    k_elim_free = cl_free_L_per_h / vd_free_L  # free drug elimination (/h)

    # MPS uptake fraction (clamped to [0, 1])
    mps_uptake_fraction = min(1.0, mps_uptake_base * size_factor)

    # Initial amounts
    # Fraction cleared immediately by MPS in first pass
    a_sln = dose_mg * (1.0 - mps_uptake_fraction)
    c_free = 0.0  # free drug starts at zero

    # Storage
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    c_sln_list: list[float] = []
    c_free_list: list[float] = []

    t = 0.0
    a_sln_at_24h = 0.0
    found_24h = False

    for _i in range(n_steps):
        # Concentrations (SLN expressed per litre of plasma; Vp ≈ vd_free_L for simplicity)
        c_sln = a_sln / vd_free_L

        times.append(round(t, 6))
        c_sln_list.append(c_sln)
        c_free_list.append(c_free)

        # Record A_sln near 24 h
        if not found_24h and t >= 24.0:
            a_sln_at_24h = a_sln
            found_24h = True

        # Forward Euler
        da_sln = -(k_release + k_sln_elim) * a_sln
        dc_free = k_release * a_sln / vd_free_L - k_elim_free * c_free

        a_sln = max(0.0, a_sln + da_sln * dt_h)
        c_free = max(0.0, c_free + dc_free * dt_h)

        t += dt_h

    # If simulation ends before 24 h, use last recorded value
    if not found_24h:
        a_sln_at_24h = a_sln

    # Derived metrics — free drug
    cmax_free = max(c_free_list)
    tmax_free = times[c_free_list.index(cmax_free)]

    # Trapezoidal AUC (free)
    auc_free = 0.0
    for idx in range(1, len(times)):
        auc_free += 0.5 * (c_free_list[idx - 1] + c_free_list[idx]) * (times[idx] - times[idx - 1])

    # Trapezoidal AUC (total = SLN + free)
    total_list = [c_sln_list[i] + c_free_list[i] for i in range(len(times))]
    auc_total = 0.0
    for idx in range(1, len(times)):
        auc_total += 0.5 * (total_list[idx - 1] + total_list[idx]) * (times[idx] - times[idx - 1])

    encapsulated_fraction = a_sln_at_24h / dose_mg if dose_mg > 0 else 0.0
    encapsulated_fraction = max(0.0, min(1.0, encapsulated_fraction))

    notes = (
        f"SLN PK simulation; lipid={lipid_type}; "
        f"k_release={k_release}/h; t_half_sln={t_half_sln}h; "
        f"size_factor={size_factor:.3f}; mps_fraction={mps_uptake_fraction:.3f}; "
        f"logP={logp}"
    )

    return SLNPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        lipid_type=lipid_type,
        particle_size_nm=particle_size_nm,
        times_h=times,
        c_sln_mg_L=c_sln_list,
        c_free_mg_L=c_free_list,
        cmax_free_mg_L=cmax_free,
        tmax_free_h=tmax_free,
        auc_free_mg_h_per_L=auc_free,
        auc_total_mg_h_per_L=auc_total,
        encapsulated_fraction_at_24h=encapsulated_fraction,
        t_half_sln_h=t_half_sln,
        mps_uptake_fraction=mps_uptake_fraction,
        notes=notes,
    )


def compare_lipid_types(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[SLNPKResult]:
    """Compare SLN PK across all 5 lipid types.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose in mg.
    **kwargs:
        Additional keyword arguments forwarded to :func:`simulate_sln_pk`
        (except ``lipid_type`` which is varied automatically).

    Returns
    -------
    list[SLNPKResult]
        Five results sorted by auc_free_mg_h_per_L descending.
    """
    results = []
    for lipid in _LIPID_PARAMS:
        kw = dict(kwargs)
        kw["lipid_type"] = lipid
        results.append(simulate_sln_pk(drug_name, dose_mg, **kw))
    results.sort(key=lambda r: r.auc_free_mg_h_per_L, reverse=True)
    return results
