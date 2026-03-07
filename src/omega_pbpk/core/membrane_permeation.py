"""Drug membrane permeation kinetics — Phase 475.

Models time-dependent drug permeation across a membrane (donor → acceptor),
analogous to a Franz diffusion cell experiment.

Uses Fick's first law with quasi-steady-state membrane assumption:
    Flux = D * Kp * (C_donor - C_acceptor / Kp) / L
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FranzDiffusionResult:
    """Result of a Franz diffusion cell simulation.

    Attributes:
        drug_name: Name of the drug.
        times_h: Time points (h).
        c_donor_mg_mL: Donor compartment concentration at each time (mg/mL).
        c_acceptor_mg_mL: Acceptor compartment concentration at each time (mg/mL).
        cumulative_permeated_mg: Cumulative mass permeated into acceptor (mg).
        flux_mg_cm2_h: Instantaneous flux at each time step (mg/cm²/h).
        papp_cm_h: Apparent permeability coefficient (cm/h).
        lag_time_h: Analytical lag time (h).
        steady_state_flux_mg_cm2_h: Analytical steady-state flux (mg/cm²/h).
        notes: Simulation notes.
    """

    drug_name: str
    times_h: list[float] = field(default_factory=list)
    c_donor_mg_mL: list[float] = field(default_factory=list)
    c_acceptor_mg_mL: list[float] = field(default_factory=list)
    cumulative_permeated_mg: list[float] = field(default_factory=list)
    flux_mg_cm2_h: list[float] = field(default_factory=list)
    papp_cm_h: float = 0.0
    lag_time_h: float = 0.0
    steady_state_flux_mg_cm2_h: float = 0.0
    notes: str = ""


def _validate_positive(value: float, name: str) -> None:
    """Raise ValueError if value is not strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def lag_time_analytical(
    membrane_thickness_um: float,
    diffusion_coeff_cm2_h: float,
) -> float:
    """Calculate analytical lag time: t_lag = L² / (6 * D).

    Parameters
    ----------
    membrane_thickness_um:
        Membrane thickness in micrometres (µm).
    diffusion_coeff_cm2_h:
        Diffusion coefficient in cm²/h.

    Returns
    -------
    float
        Lag time in hours.
    """
    _validate_positive(membrane_thickness_um, "membrane_thickness_um")
    _validate_positive(diffusion_coeff_cm2_h, "diffusion_coeff_cm2_h")

    # Convert thickness from µm to cm
    thickness_cm = membrane_thickness_um * 1e-4
    return (thickness_cm ** 2) / (6.0 * diffusion_coeff_cm2_h)


def steady_state_flux(
    donor_conc_mg_mL: float,
    membrane_thickness_um: float,
    diffusion_coeff_cm2_h: float,
    partition_coeff: float = 1.0,
) -> float:
    """Calculate analytical steady-state flux: J_ss = D * Kp * C_donor / L.

    Assumes sink conditions in acceptor (C_acceptor → 0).

    Parameters
    ----------
    donor_conc_mg_mL:
        Initial donor concentration (mg/mL).
    membrane_thickness_um:
        Membrane thickness (µm).
    diffusion_coeff_cm2_h:
        Diffusion coefficient (cm²/h).
    partition_coeff:
        Membrane/water partition coefficient (dimensionless).

    Returns
    -------
    float
        Steady-state flux in mg/cm²/h.
    """
    _validate_positive(donor_conc_mg_mL, "donor_conc_mg_mL")
    _validate_positive(membrane_thickness_um, "membrane_thickness_um")
    _validate_positive(diffusion_coeff_cm2_h, "diffusion_coeff_cm2_h")
    _validate_positive(partition_coeff, "partition_coeff")

    thickness_cm = membrane_thickness_um * 1e-4
    return diffusion_coeff_cm2_h * partition_coeff * donor_conc_mg_mL / thickness_cm


def calculate_permeability_coefficient(
    flux_mg_cm2_h: float,
    donor_conc_mg_mL: float,
) -> float:
    """Calculate apparent permeability coefficient: Papp = Flux / C_donor.

    Parameters
    ----------
    flux_mg_cm2_h:
        Observed flux (mg/cm²/h).
    donor_conc_mg_mL:
        Donor compartment concentration (mg/mL).

    Returns
    -------
    float
        Apparent permeability coefficient in cm/h.
    """
    if donor_conc_mg_mL <= 0.0:
        raise ValueError(f"donor_conc_mg_mL must be positive, got {donor_conc_mg_mL}")
    if flux_mg_cm2_h < 0.0:
        raise ValueError(f"flux_mg_cm2_h must be non-negative, got {flux_mg_cm2_h}")
    return flux_mg_cm2_h / donor_conc_mg_mL


def simulate_franz_diffusion(
    drug_name: str,
    dose_mg: float,
    membrane_thickness_um: float = 400.0,
    diffusion_coeff_cm2_h: float = 1e-5,
    partition_coeff: float = 1.0,
    area_cm2: float = 1.0,
    donor_volume_mL: float = 5.0,
    acceptor_volume_mL: float = 12.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> FranzDiffusionResult:
    """Simulate Franz diffusion cell experiment.

    Uses quasi-steady-state membrane assumption (Fick's first law):
        Flux = D * Kp * (C_donor - C_acceptor / Kp) / L
        dC_donor/dt  = -Flux * Area / V_donor
        dC_acceptor/dt = +Flux * Area / V_acceptor

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Total drug mass loaded into donor compartment (mg).
    membrane_thickness_um:
        Membrane thickness (µm).
    diffusion_coeff_cm2_h:
        Diffusion coefficient in membrane (cm²/h).
    partition_coeff:
        Membrane-to-water partition coefficient.
    area_cm2:
        Diffusion area (cm²).
    donor_volume_mL:
        Donor compartment volume (mL).
    acceptor_volume_mL:
        Acceptor compartment volume (mL).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    FranzDiffusionResult
    """
    # --- Input validation ---
    _validate_positive(dose_mg, "dose_mg")
    _validate_positive(membrane_thickness_um, "membrane_thickness_um")
    _validate_positive(diffusion_coeff_cm2_h, "diffusion_coeff_cm2_h")
    _validate_positive(partition_coeff, "partition_coeff")
    _validate_positive(area_cm2, "area_cm2")
    _validate_positive(donor_volume_mL, "donor_volume_mL")
    _validate_positive(acceptor_volume_mL, "acceptor_volume_mL")
    _validate_positive(t_end_h, "t_end_h")
    _validate_positive(dt_h, "dt_h")

    # Convert membrane thickness to cm
    thickness_cm = membrane_thickness_um * 1e-4

    # Permeability-like coefficient: P = D * Kp / L  [cm/h]
    perm_coeff = diffusion_coeff_cm2_h * partition_coeff / thickness_cm

    # Initial conditions
    c_donor = dose_mg / donor_volume_mL  # mg/mL
    c_acceptor = 0.0  # mg/mL
    cumulative = 0.0  # mg

    times: list[float] = []
    c_donor_list: list[float] = []
    c_acceptor_list: list[float] = []
    cumulative_list: list[float] = []
    flux_list: list[float] = []

    t = 0.0
    n_steps = max(1, round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h

        # Driving force: concentration difference across membrane
        # (accounts for back-diffusion from acceptor)
        driving_force = c_donor - c_acceptor / partition_coeff
        flux = perm_coeff * driving_force  # mg/cm²/h (can be negative if back-diffusion)
        flux = max(flux, 0.0)  # Physical constraint: net forward flux only when positive

        times.append(t)
        c_donor_list.append(c_donor)
        c_acceptor_list.append(c_acceptor)
        cumulative_list.append(cumulative)
        flux_list.append(flux)

        if step < n_steps:
            # Forward Euler update
            dc_donor = -flux * area_cm2 / donor_volume_mL
            dc_acceptor = flux * area_cm2 / acceptor_volume_mL

            c_donor = max(0.0, c_donor + dc_donor * dt_h)
            c_acceptor = max(0.0, c_acceptor + dc_acceptor * dt_h)
            cumulative += flux * area_cm2 * dt_h

    # Papp: use mean flux after lag time relative to initial donor concentration
    initial_donor_conc = dose_mg / donor_volume_mL
    lag_t = lag_time_analytical(membrane_thickness_um, diffusion_coeff_cm2_h)

    # Estimate Papp from post-lag fluxes
    post_lag_fluxes = [
        flux_list[i] for i, t in enumerate(times) if t > lag_t
    ]
    if post_lag_fluxes:
        mean_post_lag_flux = sum(post_lag_fluxes) / len(post_lag_fluxes)
        papp = calculate_permeability_coefficient(mean_post_lag_flux, initial_donor_conc)
    else:
        papp = perm_coeff  # fallback: analytical

    ss_flux = steady_state_flux(
        initial_donor_conc,
        membrane_thickness_um,
        diffusion_coeff_cm2_h,
        partition_coeff,
    )

    notes = (
        f"Franz diffusion simulation for {drug_name}. "
        f"Membrane thickness={membrane_thickness_um} µm, "
        f"D={diffusion_coeff_cm2_h:.2e} cm²/h, Kp={partition_coeff}. "
        f"Lag time={lag_t:.3f} h."
    )

    return FranzDiffusionResult(
        drug_name=drug_name,
        times_h=times,
        c_donor_mg_mL=c_donor_list,
        c_acceptor_mg_mL=c_acceptor_list,
        cumulative_permeated_mg=cumulative_list,
        flux_mg_cm2_h=flux_list,
        papp_cm_h=papp,
        lag_time_h=lag_t,
        steady_state_flux_mg_cm2_h=ss_flux,
        notes=notes,
    )
