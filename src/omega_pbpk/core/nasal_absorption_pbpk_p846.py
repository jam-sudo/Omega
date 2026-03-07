"""Detailed 4-Region Nasal Cavity PBPK Model — Phase 846.

Models drug absorption through four anatomical nasal regions:
  1. Anterior nasal cavity (turbulent deposition)
  2. Turbinate region (ciliated epithelium, major absorption site)
  3. Olfactory region (small particles; CNS targeting)
  4. Nasopharynx (posterior deposition, swallowing risk)

Each region has its own depot compartment with region-specific absorption
rate and mucociliary clearance loss. Absorbed drug enters a single plasma
compartment described by a one-compartment PK model.

ODEs (Forward Euler)
--------------------
For each region i (i = 0..3):
    d(depot_i)/dt = -(ka_i + kmc) * depot_i

Plasma (mg/L):
    d(C_plasma)/dt = sum_i(ka_i * depot_i) / Vd - (CL/Vd) * C_plasma

References
----------
- Djupesland PG, Drug Deliv Transl Res. 2013
- Illum L, Eur J Pharm Sci. 2003
- Gizurarson S, Am J Drug Deliv. 2012 (mucociliary clearance ~20 mm/min)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NasalAbsorptionResult",
    "simulate_nasal_absorption",
]

_KMC: float = 0.5  # mucociliary clearance rate constant (per hour)
_KA_MIN: float = 0.1
_KA_MAX: float = 5.0


@dataclass
class NasalAbsorptionResult:
    """Result of 4-region nasal absorption PBPK simulation."""

    drug_name: str
    dose_mg: float
    region_names: list
    deposition_fractions: list
    absorption_fractions: list
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    systemic_bioavailability: float
    olfactory_delivery_fraction: float
    mucociliary_clearance_rate_per_h: float
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _region_ka(region: str, logp: float) -> float:
    """Compute absorption rate constant for a named region (per hour)."""
    if region == "anterior":
        ka = 0.5 + 0.3 * logp
    elif region == "turbinate":
        ka = 1.0 + 0.5 * logp
    elif region == "olfactory":
        ka = 0.3 + 0.1 * logp
    elif region == "nasopharynx":
        ka = 0.8 + 0.3 * logp
    else:
        ka = 0.5
    return _clamp(ka, _KA_MIN, _KA_MAX)


def _region_deposition(particle_size_um: float) -> list:
    """Compute fractional deposition in each region based on particle MMAD."""
    anterior = 0.3 if particle_size_um > 10.0 else 0.2
    turbinate = 0.4 if particle_size_um <= 10.0 else 0.25
    olfactory = 0.05 if particle_size_um <= 5.0 else 0.02
    nasopharynx = 1.0 - anterior - turbinate - olfactory
    return [anterior, turbinate, olfactory, nasopharynx]


def simulate_nasal_absorption(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float = 5.0,
    logp: float = 2.0,
    mw_da: float = 350.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 6.0,
    dt_h: float = 0.05,
) -> NasalAbsorptionResult:
    """Simulate nasal drug absorption using a 4-region PBPK model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Intranasal dose (mg); must be > 0.
    particle_size_um:
        Mass median aerodynamic diameter (MMAD) of the aerosol/spray (µm); must be > 0.
    logp:
        Octanol-water partition coefficient.
    mw_da:
        Molecular weight (Da); used for notes.
    cl_L_per_h:
        Systemic clearance (L/h); must be > 0.
    vd_L:
        Volume of distribution (L); must be > 0.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    NasalAbsorptionResult

    Raises
    ------
    ValueError
        If dose_mg, cl_L_per_h, vd_L, or particle_size_um are not positive.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0; got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0; got {vd_L}")
    if particle_size_um <= 0.0:
        raise ValueError(f"particle_size_um must be > 0; got {particle_size_um}")

    region_names = ["anterior", "turbinate", "olfactory", "nasopharynx"]
    dep_fracs = _region_deposition(particle_size_um)
    kas = [_region_ka(r, logp) for r in region_names]
    ke = cl_L_per_h / vd_L

    # Initial depot amounts (mg)
    depots = [dep_fracs[i] * dose_mg for i in range(4)]
    c_plasma = 0.0

    # Track cumulative absorbed from each region
    absorbed = [0.0] * 4

    times_h: list[float] = []
    c_plasma_traj: list[float] = []

    t = 0.0
    n_steps = max(1, round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        times_h.append(t)
        c_plasma_traj.append(c_plasma)

        if step == n_steps:
            break

        # Forward Euler
        d_depots = [-(kas[i] + _KMC) * depots[i] for i in range(4)]
        absorbed_rate = sum(kas[i] * depots[i] for i in range(4))
        d_plasma = absorbed_rate / vd_L - ke * c_plasma

        # Accumulate absorbed drug (mg) from each region
        for i in range(4):
            absorbed[i] += kas[i] * depots[i] * dt_h

        # Update state
        for i in range(4):
            depots[i] = max(0.0, depots[i] + d_depots[i] * dt_h)
        c_plasma = max(0.0, c_plasma + d_plasma * dt_h)

        t += dt_h

    # PK metrics
    cmax = max(c_plasma_traj)
    tmax = times_h[c_plasma_traj.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_traj[i] + c_plasma_traj[i - 1]) * (times_h[i] - times_h[i - 1])

    # Absorption fractions per region
    abs_fracs = []
    for i in range(4):
        deposited_mg = dep_fracs[i] * dose_mg
        if deposited_mg > 0.0:
            abs_fracs.append(min(1.0, absorbed[i] / deposited_mg))
        else:
            abs_fracs.append(0.0)

    # Olfactory CNS delivery
    olfactory_delivery_fraction = dep_fracs[2] * abs_fracs[2] * 0.3

    # Systemic bioavailability
    total_absorbed_mg = sum(absorbed)
    systemic_bioavailability = min(1.0, total_absorbed_mg / dose_mg)

    # Notes
    note_parts: list[str] = []
    if particle_size_um <= 5.0:
        note_parts.append("Fine particles: enhanced olfactory deposition for CNS targeting.")
    if mw_da > 1000.0:
        note_parts.append("High MW: mucociliary clearance may limit nasal absorption.")
    if logp < 0.0:
        note_parts.append("Hydrophilic drug: limited lipophilic-driven nasal absorption.")
    if systemic_bioavailability < 0.3:
        note_parts.append("Low bioavailability: consider absorption enhancers.")
    if not note_parts:
        note_parts.append("Nasal absorption within expected range.")
    notes = " ".join(note_parts)

    return NasalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        region_names=region_names,
        deposition_fractions=dep_fracs,
        absorption_fractions=abs_fracs,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_traj,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        systemic_bioavailability=systemic_bioavailability,
        olfactory_delivery_fraction=olfactory_delivery_fraction,
        mucociliary_clearance_rate_per_h=_KMC,
        notes=notes,
    )
