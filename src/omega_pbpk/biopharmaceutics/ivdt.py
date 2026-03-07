"""In vitro dissolution testing simulation — Phase 510."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "DissolutionTestResult",
    "simulate_usp2_dissolution",
    "simulate_usp1_dissolution",
    "compare_to_specification",
    "calculate_dissolution_efficiency",
]


@dataclass
class DissolutionTestResult:
    """Result from a USP dissolution test simulation."""

    drug_name: str
    apparatus: str  # 'USP1' or 'USP2'
    times_min: list = field(default_factory=list)
    fraction_dissolved: list = field(default_factory=list)
    conc_mg_mL: list = field(default_factory=list)
    t80_min: float = 0.0  # time to 80% dissolution (min)
    de30_pct: float = 0.0  # dissolution efficiency at 30 min (%)
    passes_q30: bool = False  # >= 75% dissolved at 30 min
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(y: list[float], x: list[float]) -> float:
    """Trapezoidal integration (pure Python)."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


def _diffusion_layer_thickness_cm(rpm: float, usp_factor: float = 1.0) -> float:
    """Compute diffusion layer thickness h (cm) via Levich hydrodynamics.

    h = usp_factor * 0.5 / sqrt(rpm)

    Parameters
    ----------
    rpm:
        Stirring speed.
    usp_factor:
        1.0 for USP2 (paddle), 0.85 for USP1 (basket — slightly faster).

    Returns
    -------
    float
        Diffusion layer thickness in cm.
    """
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    return usp_factor * 0.5 / math.sqrt(rpm)


def _surface_area_cm2(
    dose_mg: float,
    particle_size_um: float,
    density_g_cm3: float = 1.2,
) -> float:
    """Compute total particle surface area (cm²) for a dose.

    Assumes monodisperse spheres.

    Parameters
    ----------
    dose_mg:
        Total dose in mg.
    particle_size_um:
        Particle diameter (um).
    density_g_cm3:
        Particle density (g/cm³).

    Returns
    -------
    float
        Surface area in cm².
    """
    r_cm = (particle_size_um / 2.0) * 1e-4  # um → cm
    if r_cm <= 0:
        return 0.0
    vol_per_particle_cm3 = (4.0 / 3.0) * math.pi * r_cm**3
    mass_per_particle_g = vol_per_particle_cm3 * density_g_cm3
    mass_per_particle_mg = mass_per_particle_g * 1000.0
    if mass_per_particle_mg <= 0:
        return 0.0
    n_particles = dose_mg / mass_per_particle_mg
    area_per_particle_cm2 = 4.0 * math.pi * r_cm**2
    return n_particles * area_per_particle_cm2


def _run_dissolution(
    drug_name: str,
    apparatus: str,
    dose_mg: float,
    solubility_mg_mL: float,
    particle_size_um: float,
    rpm: float,
    volume_mL: float,
    t_end_min: float,
    dt_min: float,
    h_factor: float,
    diffusivity_cm2_s: float = 5e-6,
    density_g_cm3: float = 1.2,
) -> DissolutionTestResult:
    """Core Noyes-Whitney forward Euler dissolution loop."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be positive, got {solubility_mg_mL}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be positive, got {particle_size_um}")
    if volume_mL <= 0:
        raise ValueError(f"volume_mL must be positive, got {volume_mL}")
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be positive, got {dt_min}")
    if t_end_min <= 0:
        raise ValueError(f"t_end_min must be positive, got {t_end_min}")

    # Diffusivity in cm²/min
    D_cm2_min = diffusivity_cm2_s * 60.0

    # Diffusion layer thickness (cm)
    h_cm = _diffusion_layer_thickness_cm(rpm, h_factor)

    # Initial surface area
    A0_cm2 = _surface_area_cm2(dose_mg, particle_size_um, density_g_cm3)

    n_steps = max(int(t_end_min / dt_min), 1)
    times_min: list[float] = []
    fractions: list[float] = []
    concs: list[float] = []

    mass_undissolved = dose_mg  # mg
    mass_dissolved = 0.0

    # Initial number of particles (constant — surface area scales with mass_undissolved)
    r0_cm = (particle_size_um / 2.0) * 1e-4
    vol_per_particle_cm3 = (4.0 / 3.0) * math.pi * r0_cm**3
    mass_per_particle_mg = vol_per_particle_cm3 * density_g_cm3 * 1000.0
    n_particles = dose_mg / max(mass_per_particle_mg, 1e-30)

    for i in range(n_steps + 1):
        t = i * dt_min
        frac = mass_dissolved / dose_mg
        conc = mass_dissolved / volume_mL  # mg/mL
        times_min.append(t)
        fractions.append(frac)
        concs.append(conc)

        if i < n_steps and mass_undissolved > 1e-9:
            # Current radius (shrinking sphere)
            mass_per_cur = mass_undissolved / n_particles
            r_cur_cm = (mass_per_cur / (density_g_cm3 * 1000.0) * 3.0 / (4.0 * math.pi)) ** (
                1.0 / 3.0
            )
            A_cur_cm2 = n_particles * 4.0 * math.pi * r_cur_cm**2

            # Concentration in vessel (mg/mL), check non-sink
            c_vessel = conc  # mg/mL
            cs = solubility_mg_mL

            # Noyes-Whitney: dM/dt = D * A * (Cs - C) / h  [mg/min]
            driving = max(0.0, cs - c_vessel)
            rate_mg_min = D_cm2_min * A_cur_cm2 * driving / max(h_cm, 1e-6)

            dm = min(rate_mg_min * dt_min, mass_undissolved)
            mass_undissolved -= dm
            mass_dissolved += dm

    # t80: time to 80% dissolved
    t80 = t_end_min
    for t_val, frac in zip(times_min, fractions):
        if frac >= 0.80:
            t80 = t_val
            break

    # DE30: dissolution efficiency at 30 min
    # DE = (AUC under fraction_dissolved vs time, up to 30 min) / (30 * 1.0) * 100
    cutoff_min = min(30.0, t_end_min)
    t_30 = [t for t in times_min if t <= cutoff_min + 1e-9]
    f_30 = fractions[: len(t_30)]
    if len(t_30) > 1:
        auc_30 = _trapz(f_30, t_30)
        de30 = auc_30 / cutoff_min * 100.0
    else:
        de30 = fractions[0] * 100.0

    # passes_q30: >= 75% at 30 min
    # Find fraction at 30 min
    f_at_30 = fractions[-1]  # default to end
    for t_val, frac in zip(times_min, fractions):
        if t_val >= 30.0:
            f_at_30 = frac
            break
    passes_q30 = f_at_30 >= 0.75

    notes = (
        f"h={h_cm:.4f} cm; D={D_cm2_min:.4e} cm²/min; "
        f"A0={A0_cm2:.2f} cm²; n_particles={n_particles:.2e}"
    )

    return DissolutionTestResult(
        drug_name=drug_name,
        apparatus=apparatus,
        times_min=times_min,
        fraction_dissolved=fractions,
        conc_mg_mL=concs,
        t80_min=t80,
        de30_pct=de30,
        passes_q30=passes_q30,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def simulate_usp2_dissolution(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    particle_size_um: float = 50.0,
    rpm: float = 75.0,
    volume_mL: float = 900.0,
    t_end_min: float = 60.0,
    dt_min: float = 0.5,
) -> DissolutionTestResult:
    """Simulate USP apparatus II (paddle) dissolution.

    Uses Noyes-Whitney with Levich hydrodynamics:
    - h = 0.5 / sqrt(rpm)  [cm]
    - dM/dt = D * A * (Cs - C) / h

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose (mg). Must be > 0.
    solubility_mg_mL:
        Solubility in dissolution medium (mg/mL). Must be > 0.
    particle_size_um:
        Initial particle diameter (um). Must be > 0.
    rpm:
        Paddle speed (rpm). Must be > 0.
    volume_mL:
        Medium volume (mL). Must be > 0.
    t_end_min:
        Simulation duration (min). Must be > 0.
    dt_min:
        Integration time step (min). Must be > 0.

    Returns
    -------
    DissolutionTestResult
    """
    return _run_dissolution(
        drug_name=drug_name,
        apparatus="USP2",
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        particle_size_um=particle_size_um,
        rpm=rpm,
        volume_mL=volume_mL,
        t_end_min=t_end_min,
        dt_min=dt_min,
        h_factor=1.0,
    )


def simulate_usp1_dissolution(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    particle_size_um: float = 50.0,
    rpm: float = 100.0,
    volume_mL: float = 900.0,
    t_end_min: float = 60.0,
    dt_min: float = 0.5,
) -> DissolutionTestResult:
    """Simulate USP apparatus I (basket) dissolution.

    Slightly higher effective dissolution rate than USP2 (h_factor=0.85).

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Dose (mg). Must be > 0.
    solubility_mg_mL:
        Solubility in dissolution medium (mg/mL). Must be > 0.
    particle_size_um:
        Initial particle diameter (um). Must be > 0.
    rpm:
        Basket speed (rpm). Must be > 0.
    volume_mL:
        Medium volume (mL). Must be > 0.
    t_end_min:
        Simulation duration (min). Must be > 0.
    dt_min:
        Integration time step (min). Must be > 0.

    Returns
    -------
    DissolutionTestResult
    """
    return _run_dissolution(
        drug_name=drug_name,
        apparatus="USP1",
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        particle_size_um=particle_size_um,
        rpm=rpm,
        volume_mL=volume_mL,
        t_end_min=t_end_min,
        dt_min=dt_min,
        h_factor=0.85,
    )


def compare_to_specification(
    result: DissolutionTestResult,
    spec_timepoints_min: list[float],
    spec_min_pct: list[float],
) -> dict:
    """Compare a dissolution result against specification timepoints.

    Parameters
    ----------
    result:
        Dissolution test result to evaluate.
    spec_timepoints_min:
        List of timepoints (min) at which specification limits apply.
    spec_min_pct:
        Minimum fraction dissolved (0–1) required at each timepoint.
        Must have the same length as ``spec_timepoints_min``.

    Returns
    -------
    dict with keys:
        - ``passes_all`` (bool): True if all specification points are met.
        - ``timepoint_results`` (list of dicts): per-timepoint assessment with
          'time_min', 'spec_min', 'actual', 'passes'.

    Raises
    ------
    ValueError
        If ``spec_timepoints_min`` and ``spec_min_pct`` have different lengths.
    """
    if len(spec_timepoints_min) != len(spec_min_pct):
        raise ValueError("spec_timepoints_min and spec_min_pct must have the same length.")

    timepoint_results = []
    passes_all = True

    for t_spec, min_frac in zip(spec_timepoints_min, spec_min_pct):
        # Find actual fraction at this timepoint (linear interpolation)
        actual = result.fraction_dissolved[-1]  # fallback to last value
        for j, t_val in enumerate(result.times_min):
            if t_val >= t_spec:
                actual = result.fraction_dissolved[j]
                break

        passes = actual >= min_frac
        if not passes:
            passes_all = False

        timepoint_results.append(
            {
                "time_min": t_spec,
                "spec_min": min_frac,
                "actual": actual,
                "passes": passes,
            }
        )

    return {
        "passes_all": passes_all,
        "timepoint_results": timepoint_results,
    }


def calculate_dissolution_efficiency(
    times_min: list[float],
    fractions_dissolved: list[float],
    t_max_min: float = 30.0,
) -> float:
    """Calculate dissolution efficiency (DE%) up to t_max_min.

    DE% = (AUC under fraction_dissolved curve from 0 to t_max) / t_max * 100

    DE=100% when drug dissolves instantly (fraction=1 throughout).
    DE=0% when no dissolution occurs.

    Parameters
    ----------
    times_min:
        Time points (min).
    fractions_dissolved:
        Corresponding dissolution fractions (0–1).
    t_max_min:
        Upper integration limit (min). Default 30 min.

    Returns
    -------
    float
        Dissolution efficiency (%).

    Raises
    ------
    ValueError
        If inputs have different lengths or t_max_min <= 0.
    """
    if len(times_min) != len(fractions_dissolved):
        raise ValueError("times_min and fractions_dissolved must have the same length.")
    if t_max_min <= 0:
        raise ValueError(f"t_max_min must be positive, got {t_max_min}")
    if len(times_min) < 2:
        return fractions_dissolved[0] * 100.0 if fractions_dissolved else 0.0

    # Truncate to t_max_min
    t_cut: list[float] = []
    f_cut: list[float] = []
    for t_val, f_val in zip(times_min, fractions_dissolved):
        if t_val <= t_max_min + 1e-9:
            t_cut.append(t_val)
            f_cut.append(f_val)

    if len(t_cut) < 2:
        return f_cut[0] * 100.0 if f_cut else 0.0

    auc = _trapz(f_cut, t_cut)
    t_span = min(t_cut[-1], t_max_min)
    if t_span <= 0:
        return 0.0
    return auc / t_span * 100.0
