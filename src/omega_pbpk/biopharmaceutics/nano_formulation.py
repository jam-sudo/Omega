"""Nanoformulation drug release and pharmacokinetic modeling.

Models drug release from nanosuspensions, nanocapsules, solid lipid
nanoparticles (SLN), and polymeric nanoparticles using size-dependent
dissolution and a 2-compartment (nano depot → free drug) system.

References
----------
- Müller RH et al. Adv Drug Deliv Rev 2002;47:3-19.
- Panyam J, Labhasetwar V. Adv Drug Deliv Rev 2003;55:329-347.
- Noyes AA, Whitney WR. J Am Chem Soc 1897;19:930-934.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "NanoFormulationResult",
    "simulate_nano_formulation",
    "size_release_relationship",
]

_VALID_NANO_TYPES = ("nanosuspension", "nanocapsule", "solid_lipid_np", "polymeric_np")

# nano_type → release rate modifier (dimensionless)
_NANO_TYPE_MODIFIERS: dict[str, float] = {
    "nanosuspension": 1.0,
    "nanocapsule": 0.3,
    "solid_lipid_np": 0.5,
    "polymeric_np": 0.4,
}


@dataclass
class NanoFormulationResult:
    """Results from a nanoformulation PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Total dose (mg).
    nano_type : Type of nanoformulation.
    particle_size_nm : Particle diameter (nm).
    drug_loading_pct : Drug loading as weight percent (0-100).
    times_h : Simulation time points (h).
    c_nano_mg_L : Concentration of drug in nano depot (mg/L).
    c_free_mg_L : Systemic free-drug concentration (mg/L).
    cmax_free : Peak systemic free-drug concentration (mg/L).
    auc_free : Area under systemic free-drug curve (mg·h/L), trapezoidal.
    t90_h : Time to 90% cumulative release from nanoparticles (h).
    notes : Interpretation notes.
    """

    drug_name: str
    dose_mg: float
    nano_type: str
    particle_size_nm: float
    drug_loading_pct: float
    times_h: list[float] = field(default_factory=list)
    c_nano_mg_L: list[float] = field(default_factory=list)
    c_free_mg_L: list[float] = field(default_factory=list)
    cmax_free: float = 0.0
    auc_free: float = 0.0
    t90_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(y: list[float], x: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def _compute_k_release(particle_size_nm: float, nano_type: str) -> float:
    """Compute effective first-order release rate constant (1/h).

    Size-dependent base rate from empirical Noyes-Whitney scaling:
        k_release_base = 0.5 * (100 / particle_size_nm)

    Clamped to [0.05, 10.0] 1/h, then multiplied by nano_type modifier.
    """
    k_base = 0.5 * (100.0 / particle_size_nm)
    k_base = max(0.05, min(10.0, k_base))
    modifier = _NANO_TYPE_MODIFIERS[nano_type]
    return k_base * modifier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_nano_formulation(
    drug_name: str,
    dose_mg: float,
    nano_type: str,
    particle_size_nm: float,
    drug_loading_pct: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float = 0.05,
) -> NanoFormulationResult:
    """Simulate pharmacokinetics of a nanoformulation.

    Uses a 2-compartment forward-Euler model:
    - Nano depot: dA_nano/dt = -k_release * A_nano
    - Free drug:  dC_free/dt = (k_release * A_nano) / vd_L - (cl / vd_L) * C_free

    The initial drug mass in the nano depot is dose_mg × (drug_loading_pct/100).
    The remainder (excipient fraction) is not tracked.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Total nanoparticle dose (mg).
    nano_type : Nanoformulation type: "nanosuspension", "nanocapsule",
        "solid_lipid_np", or "polymeric_np".
    particle_size_nm : Mean particle diameter (nm). Must be >0.
    drug_loading_pct : Drug loading (% w/w). Must be in (0, 100].
    cl_L_per_h : Systemic clearance of free drug (L/h). Must be >0.
    vd_L : Volume of distribution of free drug (L). Must be >0.
    t_end_h : Simulation end time (h). Must be >0.
    dt_h : Integration step size (h). Default 0.05 h.

    Returns
    -------
    NanoFormulationResult
        Time-course arrays and summary PK metrics.

    Raises
    ------
    ValueError
        If any parameter is out of valid range or nano_type is unknown.
    """
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if nano_type not in _VALID_NANO_TYPES:
        raise ValueError(f"nano_type must be one of {_VALID_NANO_TYPES}, got '{nano_type}'")
    if particle_size_nm <= 0:
        raise ValueError(f"particle_size_nm must be positive, got {particle_size_nm}")
    if not (0 < drug_loading_pct <= 100):
        raise ValueError(f"drug_loading_pct must be in (0, 100], got {drug_loading_pct}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")

    k_release = _compute_k_release(particle_size_nm, nano_type)
    k_elim = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    drug_mass_mg = dose_mg * (drug_loading_pct / 100.0)

    # State: A_nano (mg in depot), C_free (mg/L in systemic)
    a_nano = drug_mass_mg
    c_free = 0.0

    times: list[float] = []
    c_nano_list: list[float] = []
    c_free_list: list[float] = []

    t = 0.0
    n_steps = max(1, int(math.ceil(t_end_h / dt_h)))

    t90_h: float | None = None
    initial_nano = a_nano

    for _ in range(n_steps + 1):
        times.append(t)
        # Convert depot amount to pseudo-concentration using vd for consistency
        c_nano_list.append(a_nano / vd_L)
        c_free_list.append(c_free)

        # Track t90: time when 90% of drug released from nanoparticle
        fraction_remaining = a_nano / initial_nano if initial_nano > 0 else 0.0
        if t90_h is None and fraction_remaining <= 0.10:
            t90_h = t

        if t >= t_end_h:
            break

        # Forward Euler
        da_nano = -k_release * a_nano
        dc_free = (k_release * a_nano) / vd_L - k_elim * c_free

        a_nano = max(0.0, a_nano + da_nano * dt_h)
        c_free = max(0.0, c_free + dc_free * dt_h)
        t = min(t + dt_h, t_end_h)

    if t90_h is None:
        t90_h = t_end_h  # not reached within simulation window

    cmax_free = max(c_free_list)
    auc_free = _trapz(c_free_list, times)

    modifier = _NANO_TYPE_MODIFIERS[nano_type]
    notes = (
        f"{nano_type} ({particle_size_nm:.0f} nm), "
        f"k_release={k_release:.3f} /h (modifier={modifier}), "
        f"Cmax_free={cmax_free:.3f} mg/L, AUC_free={auc_free:.2f} mg·h/L, "
        f"t90={t90_h:.2f} h"
    )

    return NanoFormulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        nano_type=nano_type,
        particle_size_nm=particle_size_nm,
        drug_loading_pct=drug_loading_pct,
        times_h=times,
        c_nano_mg_L=c_nano_list,
        c_free_mg_L=c_free_list,
        cmax_free=cmax_free,
        auc_free=auc_free,
        t90_h=t90_h,
        notes=notes,
    )


def size_release_relationship(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    sizes_nm: list[float],
    nano_type: str,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    drug_loading_pct: float = 30.0,
) -> list[dict]:
    """Evaluate AUC, Cmax, and t90 across a range of particle sizes.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Total dose (mg).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    sizes_nm : List of particle sizes to evaluate (nm). Must be non-empty.
    nano_type : Nanoformulation type (same options as simulate_nano_formulation).
    t_end_h : Simulation duration for each run (h). Default 48 h.
    dt_h : Integration step (h). Default 0.1 h.
    drug_loading_pct : Drug loading % w/w. Default 30%.

    Returns
    -------
    list[dict]
        One dict per particle size with keys:
        "particle_size_nm", "k_release_per_h", "auc_free", "cmax_free", "t90_h".

    Raises
    ------
    ValueError
        If sizes_nm is empty or any size is non-positive, or other params invalid.
    """
    if not sizes_nm:
        raise ValueError("sizes_nm must be a non-empty list")
    if any(s <= 0 for s in sizes_nm):
        raise ValueError("All particle sizes must be positive")
    if nano_type not in _VALID_NANO_TYPES:
        raise ValueError(f"nano_type must be one of {_VALID_NANO_TYPES}, got '{nano_type}'")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")

    results = []
    for size in sizes_nm:
        res = simulate_nano_formulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            nano_type=nano_type,
            particle_size_nm=size,
            drug_loading_pct=drug_loading_pct,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(
            {
                "particle_size_nm": size,
                "k_release_per_h": _compute_k_release(size, nano_type),
                "auc_free": res.auc_free,
                "cmax_free": res.cmax_free,
                "t90_h": res.t90_h,
            }
        )
    return results
