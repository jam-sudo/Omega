"""Inhalation dosimetry: particle deposition, lung absorption, and systemic PK.

Models inhaled drug delivery using MMAD-based deposition fractions, then
simulates lung and GI absorption to produce plasma concentration-time profiles.

References
----------
- ICRP Publication 66 (1994) — Respiratory tract deposition model
- Labiris & Dolovich, Br J Clin Pharmacol 2003;56:588-599 — inhaled PK review
- Byron 1986 (Am Rev Respir Dis) — particle size and lung deposition
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "InhalationDosimetryResult",
    "simulate_inhalation_pk",
    "compare_particle_sizes",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LN2 = math.log(2.0)
_DEVICE_EFFICIENCY = 0.8  # fraction of nominal dose that is emitted


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class InhalationDosimetryResult:
    """Result of an inhalation dosimetry simulation."""

    drug_name: str
    nominal_dose_mg: float
    particle_size_MMAD_um: float
    emitted_dose_mg: float
    lung_deposited_fraction: float
    oropharyngeal_fraction: float
    alveolar_fraction: float
    bronchial_fraction: float
    swallowed_fraction: float
    # Plasma PK
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    # Lung metrics
    lung_depot_initial_mg: float
    lung_t_half_h: float
    systemic_bioavailability: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deposition_fractions(MMAD_um: float) -> tuple[float, float, float, float, float]:
    """Return (lung_dep, oropharyngeal, alveolar, bronchial, swallowed) fractions.

    Fractions are of the *emitted* dose.  swallowed = oropharyngeal * 0.8.
    """
    if MMAD_um <= 1.0:
        lung_dep = 0.5
    elif MMAD_um <= 5.0:
        lung_dep = 0.6 - 0.08 * (MMAD_um - 1.0)
    else:
        raw = 0.3 - 0.05 * (MMAD_um - 5.0)
        lung_dep = max(0.05, min(0.3, raw))

    oropharyngeal = 1.0 - lung_dep
    alveolar = lung_dep * 0.6
    bronchial = lung_dep * 0.4
    swallowed = oropharyngeal * 0.8
    return lung_dep, oropharyngeal, alveolar, bronchial, swallowed


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_inhalation_pk(
    drug_name: str,
    nominal_dose_mg: float,
    MMAD_um: float = 3.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral_swallowed: float = 0.3,
    ka_lung_per_h: float = 2.0,
    ka_gi_per_h: float = 0.5,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> InhalationDosimetryResult:
    """Simulate inhalation PK for a drug given as an aerosol.

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    nominal_dose_mg:
        Dose loaded into the inhaler device (mg).
    MMAD_um:
        Mass Median Aerodynamic Diameter of aerosol particles (micrometers).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    f_oral_swallowed:
        Oral bioavailability fraction of swallowed drug (GI absorption fraction).
    ka_lung_per_h:
        First-order absorption rate constant from lung depot (1/h).
    ka_gi_per_h:
        First-order absorption rate constant from GI tract (1/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    InhalationDosimetryResult
        Full PK result including deposition fractions and plasma profile.

    Raises
    ------
    ValueError
        If any core parameter is non-positive.
    """
    if nominal_dose_mg <= 0:
        raise ValueError(f"nominal_dose_mg must be > 0, got {nominal_dose_mg}")
    if MMAD_um <= 0:
        raise ValueError(f"MMAD_um must be > 0, got {MMAD_um}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    # --- Deposition fractions ---
    lung_dep, oropharyngeal, alveolar, bronchial, swallowed = _deposition_fractions(MMAD_um)

    # --- Dose partitioning ---
    emitted_dose = nominal_dose_mg * _DEVICE_EFFICIENCY
    lung_depot_initial = emitted_dose * lung_dep
    swallowed_dose = emitted_dose * swallowed

    # --- ODE state initial conditions ---
    lung_depot = lung_depot_initial  # mg in lung depot
    gi_depot = swallowed_dose  # mg in GI tract
    plasma = 0.0  # mg/L in plasma

    # --- Forward Euler integration ---
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    c_plasma: list[float] = []

    total_absorbed_lung = 0.0
    total_absorbed_gi = 0.0

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_plasma.append(plasma)

        # Rates
        rate_lung_abs = ka_lung_per_h * lung_depot  # mg/h absorbed from lung
        rate_gi_abs = ka_gi_per_h * gi_depot  # mg/h absorbed from GI
        rate_elim = ke * plasma  # mg/L/h eliminated

        # Accumulate absorbed amounts (before step update)
        total_absorbed_lung += rate_lung_abs * dt_h
        total_absorbed_gi += rate_gi_abs * f_oral_swallowed * dt_h

        # Update states (Forward Euler)
        lung_depot = lung_depot - rate_lung_abs * dt_h
        gi_depot = gi_depot - rate_gi_abs * dt_h
        plasma = (
            plasma
            + (rate_lung_abs / vd_L + rate_gi_abs * f_oral_swallowed / vd_L - rate_elim) * dt_h
        )

        # Clamp negatives due to numerical step size
        lung_depot = max(0.0, lung_depot)
        gi_depot = max(0.0, gi_depot)
        plasma = max(0.0, plasma)

    # --- PK metrics ---
    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]
    auc = _trapezoidal_auc(times, c_plasma)

    lung_t_half = _LN2 / ka_lung_per_h

    # systemic bioavailability = total absorbed / nominal dose
    total_absorbed = total_absorbed_lung + total_absorbed_gi
    systemic_bioavailability = min(1.0, total_absorbed / nominal_dose_mg)

    notes = (
        f"MMAD={MMAD_um:.1f} um; lung_dep={lung_dep:.3f}; "
        f"emitted={emitted_dose:.3f} mg; lung_t_half={lung_t_half:.2f} h"
    )

    return InhalationDosimetryResult(
        drug_name=drug_name,
        nominal_dose_mg=float(nominal_dose_mg),
        particle_size_MMAD_um=float(MMAD_um),
        emitted_dose_mg=float(emitted_dose),
        lung_deposited_fraction=float(lung_dep),
        oropharyngeal_fraction=float(oropharyngeal),
        alveolar_fraction=float(alveolar),
        bronchial_fraction=float(bronchial),
        swallowed_fraction=float(swallowed),
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=float(cmax),
        tmax_h=float(tmax),
        auc_mg_h_per_L=float(auc),
        lung_depot_initial_mg=float(lung_depot_initial),
        lung_t_half_h=float(lung_t_half),
        systemic_bioavailability=float(systemic_bioavailability),
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    nominal_dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    MMAD_values: list[float] | None = None,
) -> list[InhalationDosimetryResult]:
    """Simulate inhalation PK across multiple MMAD values and rank by lung deposition.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    nominal_dose_mg:
        Nominal dose (mg).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    MMAD_values:
        List of particle MMAD values (um) to evaluate. Defaults to [1, 2, 3, 5, 8].

    Returns
    -------
    list[InhalationDosimetryResult]
        Results sorted by lung_deposited_fraction descending.
    """
    if MMAD_values is None:
        MMAD_values = [1.0, 2.0, 3.0, 5.0, 8.0]

    results = [
        simulate_inhalation_pk(
            drug_name=drug_name,
            nominal_dose_mg=nominal_dose_mg,
            MMAD_um=mmad,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        for mmad in MMAD_values
    ]

    results.sort(key=lambda r: r.lung_deposited_fraction, reverse=True)
    return results
