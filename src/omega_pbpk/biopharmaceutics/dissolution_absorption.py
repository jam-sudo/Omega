"""Coupled dissolution-absorption model linking in vitro dissolution to in vivo absorption.

Models the competing processes of Noyes-Whitney dissolution and first-order
absorption from the GI lumen into a one-compartment plasma PK model.
"""

from __future__ import annotations

__all__ = [
    "DissAbsResult",
    "simulate_dissolution_absorption",
    "compare_formulation_da",
]

from dataclasses import dataclass


@dataclass(frozen=True)
class DissAbsResult:
    """Result of coupled dissolution-absorption simulation."""

    formulation_name: str
    dose_mg: float
    particle_size_um: float
    times_h: list[float]
    undissolved_mg: list[float]
    dissolved_mg: list[float]
    absorbed_mg: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_L_h: float
    fraction_absorbed: float
    dissolution_rate_limited: bool  # True if absorption is dissolution-rate limited
    notes: str


def simulate_dissolution_absorption(
    formulation_name: str,
    dose_mg: float,
    particle_size_um: float,
    drug_density_g_cm3: float = 1.2,
    solubility_mg_mL: float = 1.0,
    diffusion_coeff_cm2_per_s: float = 1e-6,
    cl_L_per_h: float = 5.0,
    vd_L: float = 20.0,
    ka_per_h: float = 1.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.01,
) -> DissAbsResult:
    """Simulate coupled Noyes-Whitney dissolution + 1-compartment PK absorption.

    Uses a shrinking sphere model for dissolution surface area and forward
    Euler integration for all ODEs.

    Parameters
    ----------
    formulation_name:
        Label for the formulation.
    dose_mg:
        Administered dose in mg (> 0).
    particle_size_um:
        Mean particle diameter in micrometres (> 0).
    drug_density_g_cm3:
        Drug crystal density in g/cm³.
    solubility_mg_mL:
        Aqueous solubility in mg/mL (> 0).
    diffusion_coeff_cm2_per_s:
        Diffusion coefficient in cm²/s.
    cl_L_per_h:
        Systemic clearance in L/h (> 0).
    vd_L:
        Volume of distribution in L (> 0).
    ka_per_h:
        First-order absorption rate constant from dissolved GI pool (/h).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step in hours.

    Returns
    -------
    DissAbsResult
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if particle_size_um <= 0.0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if solubility_mg_mL <= 0.0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    # Derived constants
    radius_cm = particle_size_um * 1e-4 / 2.0  # cm
    h_diff = radius_cm / 3.0  # diffusion layer thickness (cm)
    # Convert D from cm²/s to cm²/h
    D_cm2_per_h = diffusion_coeff_cm2_per_s * 3600.0
    ke = cl_L_per_h / vd_L  # elimination rate constant (/h)
    # GI lumen volume (250 mL)
    V_gi_mL = 250.0

    # Density in mg/cm³
    density_mg_cm3 = drug_density_g_cm3 * 1000.0

    # State variables
    M_undiss = dose_mg
    M_diss = 0.0
    M_abs = 0.0
    c_plasma = 0.0  # mg/L

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    # Pre-allocate output arrays
    times: list[float] = [0.0] * (n_steps + 1)
    undissolved: list[float] = [0.0] * (n_steps + 1)
    dissolved: list[float] = [0.0] * (n_steps + 1)
    absorbed: list[float] = [0.0] * (n_steps + 1)
    c_plasma_arr: list[float] = [0.0] * (n_steps + 1)

    times[0] = 0.0
    undissolved[0] = M_undiss
    dissolved[0] = M_diss
    absorbed[0] = M_abs
    c_plasma_arr[0] = c_plasma

    for i in range(n_steps):
        # Surface area: spherical particles, scale with remaining mass^(2/3)
        if M_undiss > 1e-9:
            frac_remaining = M_undiss / dose_mg
            # Surface area per unit mass for sphere: 6 / (diameter * density)
            # diameter in cm = particle_size_um * 1e-4
            # mass to volume: M_undiss / density_mg_cm3 [cm³]
            vol_cm3 = M_undiss / density_mg_cm3
            # Assuming all particles remain spherical and uniformly shrink:
            # A = (6 / d_cm) * vol_cm3 * frac_remaining^(1/3)
            d_cm = particle_size_um * 1e-4
            surface_area_cm2 = (6.0 / d_cm) * vol_cm3 * (frac_remaining ** (1.0 / 3.0))
        else:
            surface_area_cm2 = 0.0

        # Dissolved concentration in GI lumen (mg/mL)
        cd_gi = M_diss / V_gi_mL

        # Dissolution flux: Noyes-Whitney (mg/h)
        if surface_area_cm2 > 0.0 and h_diff > 0.0:
            driving_force = max(0.0, solubility_mg_mL - cd_gi)
            diss_rate_mg = D_cm2_per_h * surface_area_cm2 * driving_force / h_diff * dt_h
            diss_rate_mg = min(diss_rate_mg, M_undiss)
        else:
            diss_rate_mg = 0.0

        # Absorption flux (mg)
        abs_rate_mg = ka_per_h * M_diss * dt_h
        abs_rate_mg = min(abs_rate_mg, M_diss)

        # Plasma PK: one-compartment
        dc_plasma = (abs_rate_mg / vd_L) - ke * c_plasma * dt_h
        c_plasma = max(0.0, c_plasma + dc_plasma)

        # Update state
        M_undiss = max(0.0, M_undiss - diss_rate_mg)
        M_diss = max(0.0, M_diss + diss_rate_mg - abs_rate_mg)
        M_abs += abs_rate_mg

        t_val = (i + 1) * dt_h
        times[i + 1] = t_val
        undissolved[i + 1] = M_undiss
        dissolved[i + 1] = M_diss
        absorbed[i + 1] = M_abs
        c_plasma_arr[i + 1] = c_plasma

    # Derived PK metrics
    cmax = max(c_plasma_arr)
    tmax = times[c_plasma_arr.index(cmax)]
    # AUC via trapezoidal rule
    auc = 0.0
    for j in range(n_steps):
        auc += 0.5 * (c_plasma_arr[j] + c_plasma_arr[j + 1]) * (times[j + 1] - times[j])

    fraction_absorbed = M_abs / dose_mg

    # Dissolution-rate limited flag
    dissolution_rate_limited = (particle_size_um > 100.0) and (solubility_mg_mL < 0.5)

    notes = (
        f"Noyes-Whitney dissolution + 1-cpt PK; "
        f"particle_size={particle_size_um} um; "
        f"solubility={solubility_mg_mL} mg/mL; "
        f"ka={ka_per_h} /h"
    )

    return DissAbsResult(
        formulation_name=formulation_name,
        dose_mg=dose_mg,
        particle_size_um=particle_size_um,
        times_h=times,
        undissolved_mg=undissolved,
        dissolved_mg=dissolved,
        absorbed_mg=absorbed,
        c_plasma_mg_L=c_plasma_arr,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_L_h=auc,
        fraction_absorbed=fraction_absorbed,
        dissolution_rate_limited=dissolution_rate_limited,
        notes=notes,
    )


def compare_formulation_da(
    formulation_name: str,
    dose_mg: float,
    particle_sizes_um: list[float],
    **kwargs,
) -> list[DissAbsResult]:
    """Compare formulations with different particle sizes.

    Returns results sorted by fraction_absorbed descending (best first).

    Parameters
    ----------
    formulation_name:
        Base name; particle size will be appended for each result.
    dose_mg:
        Administered dose in mg.
    particle_sizes_um:
        List of mean particle diameters to compare.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`simulate_dissolution_absorption`.

    Returns
    -------
    list[DissAbsResult]
        Sorted by fraction_absorbed descending.
    """
    results = [
        simulate_dissolution_absorption(
            formulation_name=f"{formulation_name}_{ps}um",
            dose_mg=dose_mg,
            particle_size_um=ps,
            **kwargs,
        )
        for ps in particle_sizes_um
    ]
    return sorted(results, key=lambda r: r.fraction_absorbed, reverse=True)
