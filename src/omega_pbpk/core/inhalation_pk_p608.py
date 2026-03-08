"""Phase 608 — Drug Volatility and Inhalation PK

Model PK of inhaled volatile/aerosolized drugs in lungs and systemic
distribution using a 2-compartment forward Euler ODE (lung depot + central).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_MCC: float = 0.3  # mucociliary clearance rate constant (/h)
LN2: float = math.log(2)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InhalationPKResult:
    """Result of inhalation PK simulation."""

    drug_name: str
    dose_mg: float
    particle_size_um: float
    times_h: list
    c_lung_mg_g: list  # lung tissue concentration (mg per gram tissue)
    c_systemic_mg_L: list  # systemic plasma concentration
    cmax_lung_mg_g: float
    tmax_lung_h: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_lung_mg_h_per_g: float
    auc_systemic_mg_h_per_L: float
    lung_deposition_pct: float  # fraction of dose deposited in lungs (0-100)
    t_half_lung_h: float
    bioavailability_pct: float  # systemic AUC vs theoretical 100% absorbed
    notes: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _lung_deposition_fraction(particle_size_um: float) -> float:
    """Return lung deposition fraction based on MMAD (particle size)."""
    if particle_size_um < 1.0:
        return 0.10  # ultrafine — mostly exhaled
    elif particle_size_um <= 5.0:
        return 0.60  # optimal alveolar range
    elif particle_size_um <= 10.0:
        return 0.40  # tracheobronchial
    else:
        return 0.15  # oropharyngeal — largely swallowed


def _ka_lung_from_logp(logP: float) -> float:
    """Compute lung absorption rate constant from logP.

    ka_lung = 0.5 + 0.2 * logP, clamped to [0.1, 5.0] /h.
    """
    ka = 0.5 + 0.2 * logP
    return max(0.1, min(5.0, ka))


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_inhalation_pk(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    lung_weight_g: float = 1000.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> InhalationPKResult:
    """Simulate inhalation PK with lung deposition and systemic distribution.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total inhaled dose (mg).
    particle_size_um:
        Mass median aerodynamic diameter (MMAD) in micrometres.
    logP:
        Octanol-water partition coefficient (log scale).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    lung_weight_g:
        Lung tissue weight (g). Default 1000 g.
    t_end_h:
        Simulation end time (h). Default 24 h.
    dt_h:
        Integration step size (h). Default 0.05 h.

    Returns
    -------
    InhalationPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if lung_weight_g <= 0:
        raise ValueError(f"lung_weight_g must be > 0, got {lung_weight_g}")

    # --- Derived parameters ---
    dep_frac = _lung_deposition_fraction(particle_size_um)
    ka_lung = _ka_lung_from_logp(logP)
    k_elim = cl_sys_L_per_h / vd_sys_L  # systemic elimination rate (/h)
    t_half_lung = LN2 / (ka_lung + K_MCC)

    # --- Initial conditions ---
    A_lung = dose_mg * dep_frac  # amount in lung depot (mg)
    C_sys = 0.0  # systemic concentration (mg/L)

    # --- Storage ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list = []
    c_lung_list: list = []
    c_sys_list: list = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_lung_list.append(A_lung / lung_weight_g)
        c_sys_list.append(C_sys)

        # Forward Euler
        dA_lung = -(ka_lung + K_MCC) * A_lung
        dC_sys = ka_lung * A_lung / vd_sys_L - k_elim * C_sys

        A_lung = max(0.0, A_lung + dA_lung * dt_h)
        C_sys = max(0.0, C_sys + dC_sys * dt_h)
        t += dt_h

    # --- Derived PK metrics ---
    cmax_lung = max(c_lung_list)
    tmax_lung = times_h[c_lung_list.index(cmax_lung)]

    cmax_sys = max(c_sys_list)
    tmax_sys = times_h[c_sys_list.index(cmax_sys)]

    auc_lung = _trapezoidal_auc(times_h, c_lung_list)
    auc_sys = _trapezoidal_auc(times_h, c_sys_list)

    # Bioavailability: compare systemic AUC to theoretical AUC if all
    # deposited drug were absorbed systemically
    actual_dose_systemic = dose_mg * dep_frac * ka_lung / (ka_lung + K_MCC)
    theoretical_auc = actual_dose_systemic / cl_sys_L_per_h
    if theoretical_auc > 0:
        bioav_pct = min(100.0, auc_sys / theoretical_auc * 100.0)
    else:
        bioav_pct = 0.0

    notes = (
        f"Particle size {particle_size_um} um -> deposition {dep_frac * 100:.0f}%; "
        f"ka_lung={ka_lung:.3f}/h; t_half_lung={t_half_lung:.2f}h; "
        f"logP={logP:.2f}"
    )

    return InhalationPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_um=particle_size_um,
        times_h=times_h,
        c_lung_mg_g=c_lung_list,
        c_systemic_mg_L=c_sys_list,
        cmax_lung_mg_g=cmax_lung,
        tmax_lung_h=tmax_lung,
        cmax_systemic_mg_L=cmax_sys,
        tmax_systemic_h=tmax_sys,
        auc_lung_mg_h_per_g=auc_lung,
        auc_systemic_mg_h_per_L=auc_sys,
        lung_deposition_pct=dep_frac * 100.0,
        t_half_lung_h=t_half_lung,
        bioavailability_pct=bioav_pct,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Particle size comparison
# ---------------------------------------------------------------------------


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys: float,
    vd_sys: float,
    sizes: list = None,
) -> list:
    """Compare inhalation PK across different particle sizes.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total inhaled dose (mg).
    logP:
        Octanol-water partition coefficient.
    cl_sys:
        Systemic clearance (L/h).
    vd_sys:
        Systemic volume of distribution (L).
    sizes:
        List of particle sizes (µm). Default: [1.0, 3.0, 7.0, 15.0].

    Returns
    -------
    list of InhalationPKResult, sorted by auc_systemic descending.
    """
    if sizes is None:
        sizes = [1.0, 3.0, 7.0, 15.0]

    results = [
        simulate_inhalation_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_size_um=sz,
            logP=logP,
            cl_sys_L_per_h=cl_sys,
            vd_sys_L=vd_sys,
        )
        for sz in sizes
    ]

    results.sort(key=lambda r: r.auc_systemic_mg_h_per_L, reverse=True)
    return results
