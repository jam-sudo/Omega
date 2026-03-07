"""Carrier-mediated (transporter) drug transport model with saturable kinetics.

Models active + passive drug transport across a membrane using Michaelis-Menten
kinetics for the carrier-mediated component and first-order kinetics for passive
permeability. Supports both influx (uptake) and efflux transporter directions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarrierTransportResult:
    """Result of a carrier-mediated transport simulation."""

    drug_name: str
    transporter_name: str
    direction: str  # "influx" or "efflux"

    # Time-course
    times_h: list
    c_donor_mg_L: list  # donor compartment concentration
    c_acceptor_mg_L: list  # acceptor compartment concentration

    # Summary
    flux_rate_mg_per_h: float  # initial net flux rate
    apparent_permeability_cm_per_s: float
    transport_efficiency: float  # fraction transferred at t_end
    saturation_fraction: float  # C_initial / (Km + C_initial)
    influx_auc: float
    efflux_auc: float
    net_transport: str  # "influx_dominant", "efflux_dominant", or "balanced"
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i - 1] + values[i]) * dt
    return total


def simulate_carrier_transport(
    drug_name: str,
    transporter_name: str,
    c_donor_initial_mg_L: float,
    membrane_area_cm2: float = 1.0,
    vd_donor_L: float = 0.001,
    vd_acceptor_L: float = 0.001,
    vmax_mg_per_h: float = 0.1,
    km_mg_per_L: float = 1.0,
    passive_permeability_cm_per_s: float = 1e-7,
    direction: str = "influx",
    t_end_h: float = 2.0,
    dt_h: float = 0.01,
) -> CarrierTransportResult:
    """Simulate carrier-mediated drug transport across a membrane.

    Args:
        drug_name: Name of the drug.
        transporter_name: Name of the transporter protein.
        c_donor_initial_mg_L: Initial concentration in donor compartment (mg/L).
        membrane_area_cm2: Membrane surface area (cm2).
        vd_donor_L: Volume of donor compartment (L).
        vd_acceptor_L: Volume of acceptor compartment (L).
        vmax_mg_per_h: Maximum transport rate (mg/h).
        km_mg_per_L: Michaelis constant (mg/L).
        passive_permeability_cm_per_s: Passive permeability coefficient (cm/s).
        direction: "influx" (transporter moves drug donor→acceptor) or
                   "efflux" (transporter pumps drug acceptor→donor).
        t_end_h: Simulation end time (h).
        dt_h: Forward Euler time step (h).

    Returns:
        CarrierTransportResult with time-course and summary metrics.

    Raises:
        ValueError: If vmax <= 0, km <= 0, c_donor_initial < 0, or membrane_area <= 0.
    """
    if vmax_mg_per_h <= 0:
        raise ValueError(f"vmax_mg_per_h must be > 0, got {vmax_mg_per_h}")
    if km_mg_per_L <= 0:
        raise ValueError(f"km_mg_per_L must be > 0, got {km_mg_per_L}")
    if c_donor_initial_mg_L < 0:
        raise ValueError(f"c_donor_initial_mg_L must be >= 0, got {c_donor_initial_mg_L}")
    if membrane_area_cm2 <= 0:
        raise ValueError(f"membrane_area_cm2 must be > 0, got {membrane_area_cm2}")
    if direction not in ("influx", "efflux"):
        raise ValueError(f"direction must be 'influx' or 'efflux', got {direction!r}")

    # Convert passive permeability: cm/s → effective flux coefficient (mg/h per mg/L delta)
    # J_passive [mg/h] = P [cm/s] * 3600 [s/h] * A [cm2] * delta_C [mg/L] * 10
    # (unit factor 10: converts mg/L to mg/cm3 via /1000, then * area already in cm2,
    #  net: *3600 * area * 1/100 = simplified to *3600*area*10 when area in cm2, conc in mg/L)
    passive_coeff = passive_permeability_cm_per_s * 3600.0 * membrane_area_cm2 * 10.0

    # Initial conditions
    c_donor = c_donor_initial_mg_L
    c_acceptor = 0.0

    times: list[float] = []
    c_donor_trace: list[float] = []
    c_acceptor_trace: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    # Record initial state
    times.append(t)
    c_donor_trace.append(c_donor)
    c_acceptor_trace.append(c_acceptor)

    # Compute initial flux rate for summary
    if direction == "influx":
        j_active_initial = vmax_mg_per_h * c_donor / (km_mg_per_L + c_donor)
        j_passive_initial = passive_coeff * (c_donor - c_acceptor)
        flux_rate_initial = j_active_initial + j_passive_initial
    else:
        j_passive_initial = passive_coeff * (c_donor - c_acceptor)
        j_active_efflux_initial = vmax_mg_per_h * c_acceptor / (km_mg_per_L + c_acceptor)
        flux_rate_initial = j_passive_initial - j_active_efflux_initial

    # Forward Euler integration
    for _ in range(n_steps - 1):
        if direction == "influx":
            # Active transport: donor → acceptor (same direction as passive)
            j_active = vmax_mg_per_h * c_donor / (km_mg_per_L + c_donor)
            j_passive = passive_coeff * (c_donor - c_acceptor)
            j_net = j_active + j_passive

            dc_donor = -j_net / vd_donor_L
            dc_acceptor = j_net / vd_acceptor_L

        else:  # efflux
            # Passive: donor → acceptor (concentration-driven)
            # Active efflux: transporter pumps drug FROM acceptor BACK TO donor
            j_passive = passive_coeff * (c_donor - c_acceptor)
            j_active_efflux = vmax_mg_per_h * c_acceptor / (km_mg_per_L + c_acceptor)

            # Net change in donor: gains from efflux pump, loses via passive
            dc_donor = (-j_passive + j_active_efflux) / vd_donor_L
            dc_acceptor = (j_passive - j_active_efflux) / vd_acceptor_L

        c_donor = max(0.0, c_donor + dc_donor * dt_h)
        c_acceptor = max(0.0, c_acceptor + dc_acceptor * dt_h)
        t += dt_h

        times.append(t)
        c_donor_trace.append(c_donor)
        c_acceptor_trace.append(c_acceptor)

    # Summary metrics
    c_donor_final = c_donor_trace[-1]
    c_acceptor_final = c_acceptor_trace[-1]

    # Apparent permeability (Papp): amount transferred per unit area per time per initial conc
    # Amount in acceptor = C_acceptor_final * vd_acceptor_L [L] * 1000 [mL/L] → in mL ... use cm3
    # Papp [cm/s] = (C_acceptor * V_acceptor [cm3]) / (area [cm2] * C_donor0 [mg/cm3] * t_end [s])
    # Convert: V_acceptor_L * 1000 = V in mL = V in cm3
    # C_donor0 [mg/L] / 1000 = [mg/cm3]
    # t_end_s = t_end_h * 3600
    t_end_s = t_end_h * 3600.0
    if c_donor_initial_mg_L > 0 and t_end_s > 0 and membrane_area_cm2 > 0:
        amount_in_acceptor_mg = c_acceptor_final * vd_acceptor_L * 1000.0  # mg·mL/L = mg
        # Papp = amount / (area * c0_in_cm3 * t_s)
        c0_cm3 = c_donor_initial_mg_L / 1000.0  # mg/cm3
        apparent_permeability = amount_in_acceptor_mg / (membrane_area_cm2 * c0_cm3 * t_end_s)
    else:
        apparent_permeability = 0.0

    # Transport efficiency: fraction of initial drug mass that reached acceptor
    initial_mass = c_donor_initial_mg_L * vd_donor_L
    if initial_mass > 0:
        transport_efficiency = (c_acceptor_final * vd_acceptor_L) / initial_mass
        transport_efficiency = min(1.0, max(0.0, transport_efficiency))
    else:
        transport_efficiency = 0.0

    # Saturation fraction at initial donor concentration
    saturation_fraction = c_donor_initial_mg_L / (km_mg_per_L + c_donor_initial_mg_L)

    # AUC for each compartment
    influx_auc = _trapz(times, c_acceptor_trace)
    efflux_auc = _trapz(times, c_donor_trace)

    # Net transport classification
    if c_acceptor_final > c_donor_final * 0.5:
        net_transport = "influx_dominant"
    elif c_donor_final > c_acceptor_final * 2.0:
        net_transport = "efflux_dominant"
    else:
        net_transport = "balanced"

    notes = (
        f"Simulated {t_end_h:.1f}h with dt={dt_h}h. "
        f"Direction={direction}. "
        f"Vmax={vmax_mg_per_h}mg/h, Km={km_mg_per_L}mg/L. "
        f"Saturation at C0: {saturation_fraction:.1%}."
    )

    return CarrierTransportResult(
        drug_name=drug_name,
        transporter_name=transporter_name,
        direction=direction,
        times_h=times,
        c_donor_mg_L=c_donor_trace,
        c_acceptor_mg_L=c_acceptor_trace,
        flux_rate_mg_per_h=flux_rate_initial,
        apparent_permeability_cm_per_s=apparent_permeability,
        transport_efficiency=transport_efficiency,
        saturation_fraction=saturation_fraction,
        influx_auc=influx_auc,
        efflux_auc=efflux_auc,
        net_transport=net_transport,
        notes=notes,
    )


def screen_transporters(
    drug_name: str,
    c_donor_initial_mg_L: float,
    transporter_data: list[dict],
) -> list[CarrierTransportResult]:
    """Screen a drug across multiple transporters.

    Args:
        drug_name: Name of the drug.
        c_donor_initial_mg_L: Initial donor compartment concentration (mg/L).
        transporter_data: List of dicts with keys:
            - "name": transporter name (str)
            - "vmax": Vmax in mg/h (float)
            - "km": Km in mg/L (float)
            - "direction": "influx" or "efflux" (str)

    Returns:
        List of CarrierTransportResult sorted by transport_efficiency descending.
    """
    results: list[CarrierTransportResult] = []
    for td in transporter_data:
        result = simulate_carrier_transport(
            drug_name=drug_name,
            transporter_name=td["name"],
            c_donor_initial_mg_L=c_donor_initial_mg_L,
            vmax_mg_per_h=td["vmax"],
            km_mg_per_L=td["km"],
            direction=td["direction"],
        )
        results.append(result)

    results.sort(key=lambda r: r.transport_efficiency, reverse=True)
    return results
