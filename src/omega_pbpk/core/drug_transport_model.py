"""Drug transporter-mediated kinetics — Phase 505."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TransporterKineticsResult",
    "transport_rate",
    "efflux_transport_rate",
    "simulate_transporter_kinetics",
    "calculate_net_transport",
    "inhibition_constant",
]

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransporterKineticsResult:
    """Result of a transporter kinetics ODE simulation."""

    drug_name: str
    times_h: list[float]
    c_intracellular_uM: list[float]
    uptake_flux: list[float]
    efflux_flux: list[float]
    passive_flux: list[float]
    steady_state_conc_uM: float
    net_accumulation_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Core kinetics
# ---------------------------------------------------------------------------


def transport_rate(
    substrate_conc_uM: float,
    vmax_pmol_per_mg_per_min: float,
    km_uM: float,
    inhibitor_conc_uM: float = 0.0,
    ki_uM: float | None = None,
) -> float:
    """Michaelis-Menten transporter rate with optional competitive inhibition.

    Parameters
    ----------
    substrate_conc_uM:
        Substrate concentration (µM).
    vmax_pmol_per_mg_per_min:
        Maximum transport velocity (pmol/mg protein/min).
    km_uM:
        Michaelis constant for substrate (µM).
    inhibitor_conc_uM:
        Inhibitor concentration (µM); 0 if no inhibitor.
    ki_uM:
        Inhibition constant (µM); required when inhibitor_conc_uM > 0.

    Returns
    -------
    float
        Transport rate (pmol/mg protein/min).
    """
    if substrate_conc_uM < 0:
        raise ValueError("substrate_conc_uM must be >= 0")
    if vmax_pmol_per_mg_per_min < 0:
        raise ValueError("vmax must be >= 0")
    if km_uM <= 0:
        raise ValueError("km_uM must be > 0")
    if inhibitor_conc_uM < 0:
        raise ValueError("inhibitor_conc_uM must be >= 0")

    if substrate_conc_uM == 0.0:
        return 0.0

    # Competitive inhibition: apparent Km increases by (1 + I/Ki)
    if inhibitor_conc_uM > 0.0:
        if ki_uM is None or ki_uM <= 0:
            raise ValueError("ki_uM must be > 0 when inhibitor_conc_uM > 0")
        km_apparent = km_uM * (1.0 + inhibitor_conc_uM / ki_uM)
    else:
        km_apparent = km_uM

    return vmax_pmol_per_mg_per_min * substrate_conc_uM / (km_apparent + substrate_conc_uM)


def efflux_transport_rate(
    basal_conc_uM: float,
    apical_conc_uM: float,
    vmax: float,
    km: float,
    direction: str = "ba",
) -> float:
    """Direction-dependent Michaelis-Menten efflux transport rate.

    Efflux transporters (P-gp, BCRP, MRP) pump substrate from the cell
    (basolateral side) to the apical lumen in a basolateral-to-apical (B→A)
    direction.

    Parameters
    ----------
    basal_conc_uM:
        Concentration on the basolateral side (µM).
    apical_conc_uM:
        Concentration on the apical side (µM).
    vmax:
        Maximum transport velocity (same units as returned rate).
    km:
        Michaelis constant (µM).
    direction:
        'ba' (basolateral → apical, efflux) or 'ab' (apical → basolateral,
        uptake/secretion from apical side).

    Returns
    -------
    float
        Transport rate in the specified direction.
    """
    if km <= 0:
        raise ValueError("km must be > 0")
    if direction not in ("ba", "ab"):
        raise ValueError("direction must be 'ba' or 'ab'")

    if direction == "ba":
        driving_conc = basal_conc_uM
    else:
        driving_conc = apical_conc_uM

    if driving_conc <= 0:
        return 0.0

    return vmax * driving_conc / (km + driving_conc)


def simulate_transporter_kinetics(
    drug_name: str,
    initial_conc_uM: float,
    vmax_uptake: float,
    km_uptake: float,
    vmax_efflux: float = 0.0,
    km_efflux: float | None = None,
    cl_passive_per_h: float = 0.1,
    cell_volume_uL: float = 1.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.05,
) -> TransporterKineticsResult:
    """Simulate intracellular drug accumulation via active and passive transport.

    ODE: d(C_cell)/dt = (uptake_flux - efflux_flux - passive_flux) * (1/cell_volume)

    Transport rates are in pmol/mg/min; they are converted to µM/h for the ODE
    using a nominal protein density of 1 mg protein/µL cell volume.

    Parameters
    ----------
    drug_name:
        Name of the drug being modelled.
    initial_conc_uM:
        Extracellular (donor) concentration (µM); treated as constant reservoir.
    vmax_uptake:
        Maximum uptake velocity (pmol/mg/min).
    km_uptake:
        Michaelis constant for uptake (µM).
    vmax_efflux:
        Maximum efflux velocity (pmol/mg/min); 0 = no active efflux.
    km_efflux:
        Michaelis constant for efflux (µM); required when vmax_efflux > 0.
    cl_passive_per_h:
        First-order passive permeability clearance coefficient (µL/h);
        passive flux = cl_passive * C_cell (µM).
    cell_volume_uL:
        Cell volume (µL) for concentration unit conversion.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration time step (h).

    Returns
    -------
    TransporterKineticsResult
    """
    if initial_conc_uM < 0:
        raise ValueError("initial_conc_uM must be >= 0")
    if km_uptake <= 0:
        raise ValueError("km_uptake must be > 0")
    if vmax_efflux > 0 and (km_efflux is None or km_efflux <= 0):
        raise ValueError("km_efflux must be > 0 when vmax_efflux > 0")
    if cell_volume_uL <= 0:
        raise ValueError("cell_volume_uL must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Conversion factor: pmol/mg/min → µM/h
    # 1 pmol/µL = 1 µM; protein = 1 mg/µL; 1 min = 1/60 h
    # rate_uM_per_h = rate_pmol_per_mg_per_min * 60 / cell_volume_uL
    conv = 60.0 / cell_volume_uL

    _km_efflux = km_efflux if km_efflux is not None else 1.0

    times: list[float] = []
    c_intra: list[float] = []
    up_flux: list[float] = []
    ef_flux: list[float] = []
    pa_flux: list[float] = []

    t = 0.0
    c = 0.0  # intracellular concentration starts at zero

    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        # Compute fluxes at current state
        u = transport_rate(initial_conc_uM, vmax_uptake, km_uptake)
        e = transport_rate(c, vmax_efflux, _km_efflux) if vmax_efflux > 0 else 0.0
        p = cl_passive_per_h * c  # passive efflux proportional to intracellular conc

        times.append(t)
        c_intra.append(c)
        up_flux.append(u * conv)
        ef_flux.append(e * conv)
        pa_flux.append(p)

        # Forward Euler
        dc_dt = (u - e) * conv - p
        c = c + dc_dt * dt_h
        if c < 0:
            c = 0.0
        t += dt_h

    # Steady-state concentration: last time-point value
    ss_conc = c_intra[-1]

    # Net accumulation ratio: intracellular / extracellular
    net_acc = ss_conc / initial_conc_uM if initial_conc_uM > 0 else 0.0

    notes_parts = []
    if vmax_efflux > 0:
        notes_parts.append("active efflux included")
    else:
        notes_parts.append("passive transport only (no active efflux)")
    notes = "; ".join(notes_parts)

    return TransporterKineticsResult(
        drug_name=drug_name,
        times_h=times,
        c_intracellular_uM=c_intra,
        uptake_flux=up_flux,
        efflux_flux=ef_flux,
        passive_flux=pa_flux,
        steady_state_conc_uM=ss_conc,
        net_accumulation_ratio=net_acc,
        notes=notes,
    )


def calculate_net_transport(
    uptake_rate: float,
    efflux_rate: float,
    passive_rate: float,
) -> float:
    """Calculate net transport rate as uptake minus efflux minus passive loss.

    Parameters
    ----------
    uptake_rate:
        Active uptake rate (any consistent units).
    efflux_rate:
        Active efflux rate (same units).
    passive_rate:
        Passive permeation rate (same units).

    Returns
    -------
    float
        Net accumulation rate.
    """
    return uptake_rate - efflux_rate - passive_rate


def inhibition_constant(
    ic50_uM: float,
    substrate_conc_uM: float | None = None,
    km_uM: float | None = None,
) -> float:
    """Convert IC50 to Ki using the Cheng-Prusoff equation.

    Ki = IC50 / (1 + [S] / Km)

    If substrate_conc_uM or km_uM are None / zero, Ki = IC50 (no correction).

    Parameters
    ----------
    ic50_uM:
        Half-maximal inhibitory concentration (µM).
    substrate_conc_uM:
        Substrate concentration used in the IC50 assay (µM).
    km_uM:
        Michaelis constant for the substrate (µM).

    Returns
    -------
    float
        Ki (µM).
    """
    if ic50_uM <= 0:
        raise ValueError("ic50_uM must be > 0")

    if substrate_conc_uM is not None and km_uM is not None:
        if km_uM <= 0:
            raise ValueError("km_uM must be > 0")
        if substrate_conc_uM < 0:
            raise ValueError("substrate_conc_uM must be >= 0")
        return ic50_uM / (1.0 + substrate_conc_uM / km_uM)

    return ic50_uM
