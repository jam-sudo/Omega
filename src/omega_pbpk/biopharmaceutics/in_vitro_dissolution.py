"""USP apparatus II (paddle) in vitro dissolution simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class InVitroDissolutionResult:
    drug_name: str
    apparatus: str           # "USP II" (paddle)
    medium: str              # buffer description
    pH: float
    rpm: int
    dose_mg: float
    solubility_medium_mg_mL: float
    volume_mL: float
    times_min: list[float]
    dissolved_pct: list[float]   # % dissolved over time
    t50_min: float               # time to 50% dissolution
    t80_min: float               # time to 80% dissolution
    t90_min: float               # time to 90% dissolution
    de_pct: float                # dissolution efficiency (AUC/100*t_end)
    classification: str          # rapid (>80% at 30min) / moderate / slow


def simulate_usp2_dissolution(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    particle_radius_um: float = 50.0,
    diffusivity_cm2_s: float = 5e-6,
    volume_mL: float = 900.0,
    rpm: int = 50,
    pH: float = 6.8,
    t_end_min: float = 60.0,
    dt_min: float = 0.5,
    medium: str = "PBS pH 6.8",
) -> InVitroDissolutionResult:
    """Simulate USP apparatus II dissolution (paddle method).

    Uses Noyes-Whitney with hydrodynamics:
    - Diffusion layer thickness h ≈ k * (r * mu / (rho * D * N))^0.5 (Levich approximation)
    - Simplified: h_um = h_static / (1 + k_rpm * rpm)

    Parameters
    ----------
    rpm : int
        Paddle speed (typical 50-100 rpm).
    volume_mL : float
        Dissolution medium volume (USP II: 900 mL).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if volume_mL <= 0:
        raise ValueError("volume_mL must be > 0")

    # Sink conditions: C_medium << Cs (typical USP requirement)
    # Maximum dissolved if all drug dissolved: C_max_mg_mL = dose/volume
    C_max = dose_mg / volume_mL

    # Effective diffusion layer thickness (Levich model, simplified)
    # h_static ~ 50 um, decreases with rpm
    h_static_um = 50.0
    h_um = h_static_um / (1.0 + 0.02 * rpm)

    # Convert units: D in cm²/s → um²/min
    D_um2_min = diffusivity_cm2_s * 1e8 * 60.0

    # Particle density (assume 1.2 g/cm³)
    rho_mg_um3 = 1.2 * 1000.0 / 1e12  # mg/um³
    vol_particle = (4.0 / 3.0) * 3.14159 * particle_radius_um ** 3
    mass_per_particle = vol_particle * rho_mg_um3
    n_particles = dose_mg / max(mass_per_particle, 1e-15)

    Cs_mg_um3 = solubility_mg_mL / 1e12

    n_steps = max(int(t_end_min / dt_min), 1)
    t_min = np.linspace(0.0, t_end_min, n_steps + 1)

    mass_undissolved = dose_mg
    dissolved_mg = 0.0
    dissolved_pct = np.zeros(n_steps + 1)

    for i in range(n_steps):
        if mass_undissolved <= 1e-9:
            dissolved_pct[i + 1] = 100.0
            continue

        # Current particle radius (shrinking sphere)
        mass_per_cur = mass_undissolved / n_particles
        r_cur = (mass_per_cur / rho_mg_um3 * 3.0 / (4.0 * 3.14159)) ** (1.0 / 3.0)

        # Sink condition: concentration in medium ≈ 0 (true sink)
        area_cur = n_particles * 4.0 * 3.14159 * r_cur ** 2
        diss_rate = D_um2_min * area_cur * Cs_mg_um3 / max(h_um, 0.1)

        dm = min(diss_rate * dt_min, mass_undissolved)
        mass_undissolved -= dm
        dissolved_mg += dm
        dissolved_pct[i + 1] = 100.0 * dissolved_mg / dose_mg

    # Dissolution time points
    def _t_at_pct(target: float) -> float:
        for i, pct in enumerate(dissolved_pct):
            if pct >= target:
                return float(t_min[i])
        return float(t_end_min)

    t50 = _t_at_pct(50.0)
    t80 = _t_at_pct(80.0)
    t90 = _t_at_pct(90.0)

    # Dissolution efficiency: area under % dissolved curve / (100 * t_end)
    de = float(np_trapz(dissolved_pct, t_min)) / (100.0 * t_end_min) * 100.0

    # Classification per FDA rapid dissolution guidance
    if dissolved_pct[-1] >= 85.0 and t80 <= 30.0:
        classification = "rapid"
    elif dissolved_pct[-1] >= 60.0:
        classification = "moderate"
    else:
        classification = "slow"

    return InVitroDissolutionResult(
        drug_name=drug_name,
        apparatus="USP II",
        medium=medium,
        pH=pH,
        rpm=rpm,
        dose_mg=dose_mg,
        solubility_medium_mg_mL=solubility_mg_mL,
        volume_mL=volume_mL,
        times_min=t_min.tolist(),
        dissolved_pct=dissolved_pct.tolist(),
        t50_min=t50,
        t80_min=t80,
        t90_min=t90,
        de_pct=de,
        classification=classification,
    )


def compare_rpm(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    rpms: list[int] | None = None,
) -> list[InVitroDissolutionResult]:
    """Compare dissolution profiles at multiple paddle speeds."""
    if rpms is None:
        rpms = [25, 50, 100]
    return [
        simulate_usp2_dissolution(drug_name, dose_mg, solubility_mg_mL, rpm=r)
        for r in rpms
    ]


__all__ = [
    "InVitroDissolutionResult",
    "simulate_usp2_dissolution",
    "compare_rpm",
]
