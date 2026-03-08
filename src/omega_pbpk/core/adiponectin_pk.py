"""Phase 218 — Adiponectin PK

Models the pharmacokinetics of adiponectin-targeting biologics using a
3-compartment TMDD model via Forward Euler ODE integration.

Compartments
------------
C_drug       : biologic drug in central plasma (mg/L)
C_peripheral : drug in peripheral compartment (mg/L)
C_complex    : drug-receptor complex (mg/L)

SC depot absorption (first-order) provides input to central compartment.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_ROUTES = {"iv", "sc"}

# Fixed PK/PD parameters
KEL = 0.05  # /h  linear elimination from central
KON = 0.1  # /h/(mg/L) binding rate
KOFF = 0.01  # /h  dissociation rate
KINT = 0.05  # /h  complex internalisation/elimination
K12 = 0.1  # /h  central → peripheral
K21 = 0.05  # /h  peripheral → central
R_TOTAL = 1.0  # mg/L total receptor concentration

# SC absorption
KA_SC = 0.02  # /h  SC absorption rate
F_SC = 0.7  # SC bioavailability fraction


@dataclass
class AdiponectinPKResult:
    """Result of adiponectin PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_drug_mg_L: list[float]
    c_complex_mg_L: list[float]
    c_peripheral_mg_L: list[float]
    cmax_drug_mg_L: float
    tmax_drug_h: float
    auc_drug_mg_h_per_L: float
    receptor_occupancy_at_cmax: float
    notes: str


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return total


def simulate_adiponectin_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_L_per_h: float = 0.5,
    vd_L: float = 10.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
) -> AdiponectinPKResult:
    """Simulate adiponectin biologic PK with TMDD.

    Parameters
    ----------
    drug_name:
        Name of the biologic.
    dose_mg:
        Dose in milligrams (must be > 0).
    route:
        "iv" or "sc".
    cl_L_per_h:
        Linear clearance (L/h) — used to scale kel if desired; currently
        informational (kel is fixed per spec).
    vd_L:
        Volume of distribution (L) for initial IV concentration.
    t_end_h:
        Simulation duration in hours (default 168 h = 7 days).
    dt_h:
        Forward-Euler step size (hours).

    Returns
    -------
    AdiponectinPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in VALID_ROUTES:
        raise ValueError(f"route must be one of {sorted(VALID_ROUTES)}, got '{route}'")

    # Initial conditions
    if route == "iv":
        # Bolus: all drug in central compartment immediately
        c_drug = dose_mg / vd_L
        a_sc = 0.0  # no SC depot
        notes = f"IV bolus: C0 = {c_drug:.3f} mg/L (Vd={vd_L} L)."
    else:
        # SC: depot starts with F_SC * dose_mg, drug starts at 0
        c_drug = 0.0
        a_sc = F_SC * dose_mg  # depot amount (mg) — will be divided by Vd on absorption
        notes = (
            f"SC dose: depot = {a_sc:.1f} mg (F={F_SC}), ka={KA_SC}/h. "
            "Absorption into central compartment via first-order."
        )

    c_peripheral = 0.0
    c_complex = 0.0

    times: list[float] = []
    c_drug_list: list[float] = []
    c_complex_list: list[float] = []
    c_peripheral_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h

        times.append(t)
        c_drug_list.append(c_drug)
        c_complex_list.append(c_complex)
        c_peripheral_list.append(c_peripheral)

        if step == n_steps:
            break

        # Free receptor (simplified)
        r_free = max(0.0, R_TOTAL - c_complex)

        # SC input (mg absorbed per hour → mg/L by dividing by Vd)
        sc_input = (KA_SC * a_sc) / vd_L if route == "sc" else 0.0

        # ODEs
        d_c_drug = (
            sc_input
            - (KEL + KON * r_free) * c_drug
            + KOFF * c_complex
            - K12 * c_drug
            + K21 * c_peripheral
        )
        d_c_peripheral = K12 * c_drug - K21 * c_peripheral
        d_c_complex = KON * r_free * c_drug - KOFF * c_complex - KINT * c_complex
        d_a_sc = -KA_SC * a_sc if route == "sc" else 0.0

        # Forward Euler update
        c_drug = max(0.0, c_drug + dt_h * d_c_drug)
        c_peripheral = max(0.0, c_peripheral + dt_h * d_c_peripheral)
        c_complex = max(0.0, c_complex + dt_h * d_c_complex)
        a_sc = max(0.0, a_sc + dt_h * d_a_sc)

    # Derived PK metrics
    cmax_drug = max(c_drug_list)
    tmax_idx = c_drug_list.index(cmax_drug)
    tmax_drug = times[tmax_idx]
    auc_drug = _trapz(times, c_drug_list)

    # Receptor occupancy at Cmax: complex / R_total
    c_complex_at_cmax = c_complex_list[tmax_idx]
    receptor_occupancy_at_cmax = min(1.0, c_complex_at_cmax / R_TOTAL)

    return AdiponectinPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_drug_mg_L=c_drug_list,
        c_complex_mg_L=c_complex_list,
        c_peripheral_mg_L=c_peripheral_list,
        cmax_drug_mg_L=cmax_drug,
        tmax_drug_h=tmax_drug,
        auc_drug_mg_h_per_L=auc_drug,
        receptor_occupancy_at_cmax=receptor_occupancy_at_cmax,
        notes=notes,
    )


def assess_target_engagement(result: AdiponectinPKResult) -> dict[str, object]:
    """Assess target engagement from a simulation result.

    Parameters
    ----------
    result:
        AdiponectinPKResult from simulate_adiponectin_pk().

    Returns
    -------
    dict with keys:
        "max_receptor_occupancy_pct"   : float (0–100)
        "duration_above_50pct_h"       : float (hours above 50% occupancy)
        "clinical_relevance"           : str
    """
    # Receptor occupancy at each time point (as %)
    occupancy_pct = [min(100.0, (c / R_TOTAL) * 100.0) for c in result.c_complex_mg_L]

    max_ro_pct = max(occupancy_pct)

    # Duration above 50% occupancy (trapezoidal counting via step widths)
    times = result.times_h
    duration_50 = 0.0
    for i in range(1, len(times)):
        mid_occ = 0.5 * (occupancy_pct[i - 1] + occupancy_pct[i])
        if mid_occ >= 50.0:
            duration_50 += times[i] - times[i - 1]

    # Clinical relevance classification
    if max_ro_pct >= 80.0 and duration_50 >= 24.0:
        clinical_relevance = "High target engagement; likely therapeutic."
    elif max_ro_pct >= 50.0:
        clinical_relevance = "Moderate target engagement; efficacy possible."
    elif max_ro_pct >= 20.0:
        clinical_relevance = "Low target engagement; dose adjustment may be needed."
    else:
        clinical_relevance = "Minimal target engagement; likely sub-therapeutic."

    return {
        "max_receptor_occupancy_pct": round(max_ro_pct, 3),
        "duration_above_50pct_h": round(duration_50, 3),
        "clinical_relevance": clinical_relevance,
    }
