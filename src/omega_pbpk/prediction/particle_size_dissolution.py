"""Phase 179 — Particle size effect on dissolution rate and oral bioavailability.

Noyes-Whitney model with particle size-dependent diffusion layer thickness.
"""

from dataclasses import dataclass
from math import inf


@dataclass
class ParticleDissolutionResult:
    drug_name: str
    dose_mg: float
    particle_diameter_um: float
    intrinsic_solubility_mg_mL: float
    times_min: list
    dissolved_pct: list  # % of dose dissolved at each time
    t50_min: float  # time to 50% dissolution (inf if not reached)
    t90_min: float  # time to 90% dissolution (inf if not reached)
    max_dissolved_pct: float  # at end of simulation
    dissolution_limited: bool  # True if max_dissolved_pct < 90
    particle_size_class: str  # "nano"(<1um), "micro"(1-100um), "meso"(>100um)
    notes: str


def _classify_particle_size(diameter_um: float) -> str:
    if diameter_um < 1.0:
        return "nano"
    elif diameter_um <= 100.0:
        return "micro"
    else:
        return "meso"


def simulate_dissolution(
    drug_name: str,
    dose_mg: float,
    particle_diameter_um: float,
    intrinsic_solubility_mg_mL: float,
    drug_density_g_cm3: float = 1.2,
    t_end_min: float = 60.0,
    dt_min: float = 0.5,
) -> ParticleDissolutionResult:
    """Simulate dissolution using the Noyes-Whitney equation with particle size effects.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in milligrams.
        particle_diameter_um: Particle diameter in micrometers (um).
        intrinsic_solubility_mg_mL: Intrinsic aqueous solubility in mg/mL.
        drug_density_g_cm3: Drug particle density in g/cm3.
        t_end_min: Simulation end time in minutes.
        dt_min: Time step in minutes.

    Returns:
        ParticleDissolutionResult with dissolution profile.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_diameter_um <= 0:
        raise ValueError("particle_diameter_um must be > 0")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")
    if drug_density_g_cm3 <= 0:
        raise ValueError("drug_density_g_cm3 must be > 0")
    if t_end_min <= 0:
        raise ValueError("t_end_min must be > 0")

    # Constants
    D_cm2_per_s = 7.4e-6  # diffusivity in cm2/s
    D_cm2_per_min = D_cm2_per_s * 60.0  # convert to cm2/min
    volume_mL = 900.0  # dissolution volume

    # Convert units
    d_cm = particle_diameter_um * 1e-4  # um -> cm

    # Diffusion layer thickness (Hixson-Crowell modification)
    # h = 30 um for d > 30 um; h = d/2 for d <= 30 um
    if particle_diameter_um > 30.0:
        h_cm = 30.0e-4  # 30 um in cm
    else:
        h_cm = d_cm / 2.0

    Cs = intrinsic_solubility_mg_mL  # mg/mL

    # Initial conditions
    M = dose_mg  # undissolved mass in mg
    C_bulk = 0.0  # dissolved concentration mg/mL

    times_min = [0.0]
    dissolved_pct = [0.0]

    t50_min = inf
    t90_min = inf

    t = 0.0
    steps = int(t_end_min / dt_min)

    for _ in range(steps):
        if M <= 0.0 or C_bulk >= Cs:
            # Fill remaining time points with current values
            t += dt_min
            pct = min(100.0, (dose_mg - max(0.0, M)) / dose_mg * 100.0)
            times_min.append(round(t, 6))
            dissolved_pct.append(pct)
            continue

        # Surface area for spherical particles: S = 6 * M / (rho * d)
        # rho in g/cm3 = g/mL, M in mg -> convert M to g
        rho_mg_per_cm3 = drug_density_g_cm3 * 1000.0  # mg/cm3
        S_total_cm2 = 6.0 * M / (rho_mg_per_cm3 * d_cm)

        # Noyes-Whitney: dM/dt = (D * S_total / h) * (Cs - C_bulk)
        # units: [cm2/min * cm2 / cm * mg/mL] = [cm3/min * mg/mL] = mg/min
        driving_force = Cs - C_bulk
        if driving_force <= 0.0:
            driving_force = 0.0

        dM_dt = (D_cm2_per_min * S_total_cm2 / h_cm) * driving_force  # mg/min

        # Limit dissolution to available mass
        delta_M = dM_dt * dt_min
        if delta_M > M:
            delta_M = M

        M -= delta_M
        if M < 0.0:
            M = 0.0

        C_bulk += delta_M / volume_mL
        if C_bulk > Cs:
            C_bulk = Cs

        t += dt_min
        pct = (dose_mg - M) / dose_mg * 100.0

        times_min.append(round(t, 6))
        dissolved_pct.append(pct)

        if t50_min == inf and pct >= 50.0:
            t50_min = t
        if t90_min == inf and pct >= 90.0:
            t90_min = t

    max_dissolved_pct = dissolved_pct[-1] if dissolved_pct else 0.0
    dissolution_limited = max_dissolved_pct < 90.0

    particle_size_class = _classify_particle_size(particle_diameter_um)

    notes_parts = []
    if dissolution_limited:
        notes_parts.append(
            f"Dissolution limited: only {max_dissolved_pct:.1f}% dissolved in {t_end_min} min."
        )
    else:
        notes_parts.append(f"Complete dissolution achieved (t90={t90_min:.1f} min).")

    if particle_size_class == "nano":
        notes_parts.append("Nanoparticles: very high surface area, rapid dissolution.")
    elif particle_size_class == "micro":
        notes_parts.append("Microparticles: good surface area for dissolution.")
    else:
        notes_parts.append("Meso/coarse particles: consider milling to improve dissolution.")

    notes = " ".join(notes_parts)

    return ParticleDissolutionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_diameter_um=particle_diameter_um,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        times_min=times_min,
        dissolved_pct=dissolved_pct,
        t50_min=t50_min,
        t90_min=t90_min,
        max_dissolved_pct=max_dissolved_pct,
        dissolution_limited=dissolution_limited,
        particle_size_class=particle_size_class,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    intrinsic_solubility_mg_mL: float,
    diameters_um: list | None = None,
) -> list:
    """Compare dissolution across multiple particle sizes.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in milligrams.
        intrinsic_solubility_mg_mL: Intrinsic aqueous solubility in mg/mL.
        diameters_um: List of particle diameters to compare (um).

    Returns:
        List of ParticleDissolutionResult sorted by t90_min ascending (fastest first).
    """
    if diameters_um is None:
        diameters_um = [200.0, 100.0, 50.0, 10.0, 1.0]

    results = []
    for d in diameters_um:
        result = simulate_dissolution(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_diameter_um=d,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        )
        results.append(result)

    # Sort by t90_min ascending (inf values go last)
    results.sort(key=lambda r: r.t90_min)
    return results
