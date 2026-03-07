"""Polydisperse Particle Size Distribution Effect on Dissolution — Phase 366.

Models dissolution of a drug powder with a log-normal particle size
distribution using the Noyes-Whitney equation applied to 5 representative
size fractions.

The log-normal distribution is characterised by:
    d50_um : median diameter
    span   : (d90 - d10) / d50  (distribution breadth)

Five fractions with fixed weights [0.1, 0.2, 0.4, 0.2, 0.1] cover
[d10, d50/2, d50, d50*1.5, d90] of the distribution.

References
----------
- Noyes AA & Whitney WR, J Am Chem Soc. 1897;19(12):930-934.
- Hixson AW & Crowell JH, Ind Eng Chem. 1931;23(9):923-931.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PolydisperseDissoResult", "simulate_polydisperse_dissolution"]


@dataclass
class PolydisperseDissoResult:
    """Results from a polydisperse dissolution simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    d50_um : Median particle diameter (um).
    span : Distribution width = (d90-d10)/d50.
    times_min : Time points (min).
    dissolved_pct : Cumulative dissolved percentage over time (%).
    t50_min : Time to 50% dissolution (min); inf if not reached.
    t90_min : Time to 90% dissolution (min); inf if not reached.
    max_dissolved_pct : Maximum dissolved % (solubility limited).
    monodisperse_t90_min : t90 for a monodisperse d50 reference (min).
    polydisperse_faster : True when polydisperse t90 < monodisperse t90.
    dissolution_efficiency_pct : AUC(dissolved_pct) / (100 * t_end) * 100.
    notes : Summary string.
    """

    drug_name: str = ""
    d50_um: float = 0.0
    span: float = 0.0
    times_min: list = field(default_factory=list)
    dissolved_pct: list = field(default_factory=list)
    t50_min: float = float("inf")
    t90_min: float = float("inf")
    max_dissolved_pct: float = 0.0
    monodisperse_t90_min: float = float("inf")
    polydisperse_faster: bool = False
    dissolution_efficiency_pct: float = 0.0
    notes: str = ""


# Physical constants
_D_CM2_PER_MIN = 7.4e-6 * 60.0  # diffusion coefficient converted to cm^2/min
_VOLUME_ML = 900.0  # dissolution vessel volume (mL)
_UM_TO_CM = 1.0e-4  # micrometre → centimetre


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _simulate_single_fraction(
    diameter_um: float,
    mass_mg: float,
    solubility_mg_mL: float,
    drug_density_g_cm3: float,
    times_min: list,
    dt_min: float,
) -> list:
    """Simulate Noyes-Whitney dissolution for one monodisperse fraction.

    Returns a list of dissolved amounts (mg) at each time point.
    Particle shrinkage tracked via remaining mass → equivalent radius.
    """
    d_cm = diameter_um * _UM_TO_CM
    rho_mg_per_cm3 = drug_density_g_cm3 * 1000.0  # g/cm3 → mg/cm3

    m_remaining = mass_mg
    dissolved = 0.0
    dissolved_list: list[float] = []

    n_steps = len(times_min) - 1

    for i in range(n_steps + 1):
        dissolved_list.append(dissolved)

        if i >= n_steps or m_remaining <= 0.0:
            # Pad remaining steps
            while len(dissolved_list) <= n_steps:
                dissolved_list.append(dissolved)
            break

        # Current bulk concentration
        c_bulk = dissolved / _VOLUME_ML  # mg/mL

        # Check solubility limit
        if c_bulk >= solubility_mg_mL:
            dissolved_list.extend([dissolved] * (n_steps - i))
            break

        # Effective particle diameter from remaining mass
        # M = rho * (pi/6) * d^3 * N  →  d_eff = d0 * (M/M0)^(1/3)
        if mass_mg > 0:
            d_eff_cm = d_cm * (m_remaining / mass_mg) ** (1.0 / 3.0)
        else:
            d_eff_cm = d_cm

        d_eff_cm = max(d_eff_cm, 1.0e-9)

        # Diffusion layer thickness: min(r_particle, 30 um in cm)
        h_cm = min(d_eff_cm / 2.0, 30.0 * _UM_TO_CM)

        # Surface area: S = 6*M/(rho*d)
        if d_eff_cm > 0 and m_remaining > 0:
            surface_cm2 = 6.0 * m_remaining / (rho_mg_per_cm3 * d_eff_cm)
        else:
            surface_cm2 = 0.0

        # Noyes-Whitney flux: dM/dt = D * S * (Cs - Cb) / h  [mg/min]
        driving_force = max(solubility_mg_mL - c_bulk, 0.0)
        dm_dt = _D_CM2_PER_MIN * surface_cm2 * driving_force / h_cm if h_cm > 0 else 0.0

        delta_m = dm_dt * dt_min
        # Cannot dissolve more than what remains
        delta_m = min(delta_m, m_remaining)

        m_remaining -= delta_m
        m_remaining = max(m_remaining, 0.0)
        dissolved += delta_m

        # Cap at solubility
        max_dissolved = solubility_mg_mL * _VOLUME_ML
        if dissolved > max_dissolved:
            dissolved = max_dissolved

    return dissolved_list[: n_steps + 1]


def simulate_polydisperse_dissolution(
    drug_name: str,
    d50_um: float,
    span: float = 1.5,
    intrinsic_solubility_mg_mL: float = 0.1,
    dose_mg: float = 100.0,
    drug_density_g_cm3: float = 1.2,
    t_end_min: float = 60.0,
    dt_min: float = 0.5,
) -> PolydisperseDissoResult:
    """Simulate polydisperse dissolution using 5 log-normal size fractions.

    Parameters
    ----------
    drug_name : Drug identifier string.
    d50_um : Median particle diameter (um). Must be > 0.
    span : Distribution width = (d90-d10)/d50. Must be > 0.
    intrinsic_solubility_mg_mL : Aqueous solubility (mg/mL). Must be > 0.
    dose_mg : Total drug dose (mg). Must be > 0.
    drug_density_g_cm3 : Particle density (g/cm3). Default 1.2.
    t_end_min : Simulation end time (min). Default 60.
    dt_min : Forward Euler time step (min). Default 0.5.

    Returns
    -------
    PolydisperseDissoResult
    """
    # --- Validation ---
    if d50_um <= 0:
        raise ValueError("d50_um must be > 0")
    if span <= 0:
        raise ValueError("span must be > 0")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    # --- Size fractions ---
    d10 = d50_um / (1.0 + 0.5 * span)
    d90 = d50_um * (1.0 + 0.5 * span)

    fraction_diameters = [d10, d50_um / 2.0, d50_um, d50_um * 1.5, d90]
    fraction_weights = [0.1, 0.2, 0.4, 0.2, 0.1]

    n_steps = max(int(round(t_end_min / dt_min)), 1)
    times_min: list[float] = [i * dt_min for i in range(n_steps + 1)]

    # --- Simulate each fraction ---
    # dissolved_mg[i] is cumulative dissolved amount at time step i
    dissolved_mg: list[float] = [0.0] * (n_steps + 1)

    for diam, weight in zip(fraction_diameters, fraction_weights):
        fraction_mass = weight * dose_mg
        frac_dissolved = _simulate_single_fraction(
            diameter_um=diam,
            mass_mg=fraction_mass,
            solubility_mg_mL=intrinsic_solubility_mg_mL,
            drug_density_g_cm3=drug_density_g_cm3,
            times_min=times_min,
            dt_min=dt_min,
        )
        for i in range(n_steps + 1):
            dissolved_mg[i] += frac_dissolved[i]

    # Convert to percentage of dose
    dissolved_pct: list[float] = [min(d / dose_mg * 100.0, 100.0) for d in dissolved_mg]

    # --- Monodisperse reference (d50 only) ---
    mono_dissolved = _simulate_single_fraction(
        diameter_um=d50_um,
        mass_mg=dose_mg,
        solubility_mg_mL=intrinsic_solubility_mg_mL,
        drug_density_g_cm3=drug_density_g_cm3,
        times_min=times_min,
        dt_min=dt_min,
    )
    mono_pct: list[float] = [min(d / dose_mg * 100.0, 100.0) for d in mono_dissolved]

    # --- t50, t90, monodisperse_t90 ---
    def find_time_at_threshold(pct_list: list, threshold: float) -> float:
        for i, p in enumerate(pct_list):
            if p >= threshold:
                return times_min[i]
        return float("inf")

    t50 = find_time_at_threshold(dissolved_pct, 50.0)
    t90 = find_time_at_threshold(dissolved_pct, 90.0)
    mono_t90 = find_time_at_threshold(mono_pct, 90.0)

    max_dissolved = max(dissolved_pct) if dissolved_pct else 0.0

    # --- Dissolution efficiency ---
    # AUC of dissolved_pct vs time / (100 * t_end) * 100
    auc_pct = _trapezoidal_auc(times_min, dissolved_pct)
    de_pct = (auc_pct / (100.0 * t_end_min) * 100.0) if t_end_min > 0 else 0.0
    de_pct = min(de_pct, 100.0)

    polydisperse_faster = t90 < mono_t90

    notes = (
        f"Polydisperse dissolution for {drug_name}. "
        f"d50={d50_um}um, span={span:.2f}, "
        f"d10={d10:.2f}um, d90={d90:.2f}um. "
        f"t50={t50:.1f}min, t90={t90:.1f}min. "
        f"Monodisperse t90={mono_t90:.1f}min. "
        f"Polydisperse faster: {polydisperse_faster}. "
        f"DE={de_pct:.1f}%."
    )

    return PolydisperseDissoResult(
        drug_name=drug_name,
        d50_um=d50_um,
        span=span,
        times_min=times_min,
        dissolved_pct=dissolved_pct,
        t50_min=t50,
        t90_min=t90,
        max_dissolved_pct=max_dissolved,
        monodisperse_t90_min=mono_t90,
        polydisperse_faster=polydisperse_faster,
        dissolution_efficiency_pct=de_pct,
        notes=notes,
    )
