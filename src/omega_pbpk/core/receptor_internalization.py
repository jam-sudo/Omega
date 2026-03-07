"""Receptor internalization kinetics under drug binding.

Models receptor downregulation via ligand-induced internalization using a
two-compartment (surface / internalized) ODE system with Forward Euler
integration.

References
----------
- Hanyaloglu AC, von Zastrow M, Annu Rev Pharmacol. 2008;48:537-568
- Bhalla US, Iyengar R, Science. 1999;283(5400):381-387 (cell signaling)
- Sorkin A, von Zastrow M, Nat Rev Mol Cell Biol. 2009;10(9):609-622
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceptorInternalizationResult:
    """Result of receptor internalization simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    receptor_name : Receptor identifier.
    initial_receptor_density : R0, initial surface receptor density (nmol/L).
    drug_conc_mg_L : Drug plasma concentration (mg/L).
    kd_receptor_nM : Receptor-drug dissociation constant (nM).
    ki_internalization_per_h : Internalization rate constant (h^-1).
    kr_recycling_per_h : Recycling rate constant (h^-1).
    ks_synthesis_per_h : Receptor synthesis rate (nmol/L/h).
    receptor_occupancy_pct : Drug-bound fraction of receptor (%) at t_end.
    internalized_fraction : Fraction of total receptor that is internalized at t_end.
    surface_receptor_pct : Surface receptor remaining as % of R0 at t_end.
    time_to_50pct_downregulation_h : Time when surface R falls to 50% R0 (h); inf if never.
    notes : Summary text.
    """

    drug_name: str
    receptor_name: str
    initial_receptor_density: float
    drug_conc_mg_L: float
    kd_receptor_nM: float
    ki_internalization_per_h: float
    kr_recycling_per_h: float
    ks_synthesis_per_h: float
    receptor_occupancy_pct: float
    internalized_fraction: float
    surface_receptor_pct: float
    time_to_50pct_downregulation_h: float
    notes: str


def simulate_receptor_internalization(
    drug_name: str,
    receptor_name: str,
    drug_conc_mg_L: float,
    drug_mw_kDa: float = 150.0,
    kd_receptor_nM: float = 10.0,
    ki_internalization_per_h: float = 0.2,
    kr_recycling_per_h: float = 0.05,
    ks_synthesis_per_h: float = 0.01,
    kdeg_per_h: float = 0.01,
    r0_nmol_per_L: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ReceptorInternalizationResult:
    """Simulate receptor internalization dynamics under sustained drug exposure.

    The model tracks surface and internalized receptor pools using Forward Euler
    integration of a two-state ODE system.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    receptor_name:
        Receptor identifier.
    drug_conc_mg_L:
        Drug plasma concentration (mg/L).  Must be >= 0.
    drug_mw_kDa:
        Drug molecular weight (kDa); used to convert mg/L to nM.
    kd_receptor_nM:
        Receptor dissociation constant (nM).  Must be > 0.
    ki_internalization_per_h:
        Internalization rate constant (h^-1).  Must be >= 0.
    kr_recycling_per_h:
        Recycling rate constant (h^-1).  Must be >= 0.
    ks_synthesis_per_h:
        Receptor synthesis rate (nmol/L/h).
    kdeg_per_h:
        Receptor degradation rate (h^-1).
    r0_nmol_per_L:
        Initial surface receptor density (nmol/L).
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    ReceptorInternalizationResult
    """
    if drug_conc_mg_L < 0:
        raise ValueError("drug_conc_mg_L must be >= 0")
    if kd_receptor_nM <= 0:
        raise ValueError("kd_receptor_nM must be > 0")
    if ki_internalization_per_h < 0:
        raise ValueError("ki_internalization_per_h must be >= 0")
    if kr_recycling_per_h < 0:
        raise ValueError("kr_recycling_per_h must be >= 0")

    # Convert drug concentration from mg/L to nM
    # nM = (mg/L) * 1e6 / (kDa * 1000)  [since 1 kDa = 1000 g/mol, 1 mg/L = 1e-3 g/L]
    drug_conc_nM = drug_conc_mg_L * 1e6 / (drug_mw_kDa * 1000.0)

    # Constant drug-bound fraction (drug_conc held constant throughout)
    drug_bound_fraction = drug_conc_nM / (kd_receptor_nM + drug_conc_nM)

    # Initial conditions
    r_surface = r0_nmol_per_L
    r_int = 0.0

    t50_h = math.inf
    threshold_50 = 0.5 * r0_nmol_per_L

    n_steps = max(1, int(round(t_end_h / dt_h)))

    for step in range(n_steps):
        # Compute derivatives
        dr_surface = (
            ks_synthesis_per_h
            + kr_recycling_per_h * r_int
            - ki_internalization_per_h * drug_bound_fraction * r_surface
            - kdeg_per_h * r_surface
        )
        dr_int = (
            ki_internalization_per_h * drug_bound_fraction * r_surface
            - kr_recycling_per_h * r_int
            - kdeg_per_h * r_int
        )

        # Forward Euler update
        r_surface_new = r_surface + dr_surface * dt_h
        r_int_new = r_int + dr_int * dt_h

        # Clamp to non-negative
        r_surface_new = max(0.0, r_surface_new)
        r_int_new = max(0.0, r_int_new)

        # Record time to 50% downregulation (first crossing)
        t_current = (step + 1) * dt_h
        if math.isinf(t50_h) and r_surface_new <= threshold_50 and r0_nmol_per_L > 0:
            t50_h = t_current

        r_surface = r_surface_new
        r_int = r_int_new

    # Final metrics
    receptor_occupancy_pct = drug_bound_fraction * 100.0

    total_receptor = r_surface + r_int
    if total_receptor > 0:
        internalized_fraction = r_int / total_receptor
    else:
        internalized_fraction = 0.0

    if r0_nmol_per_L > 0:
        surface_receptor_pct = r_surface / r0_nmol_per_L * 100.0
    else:
        surface_receptor_pct = 0.0

    notes = (
        f"{drug_name} on {receptor_name}: "
        f"occupancy={receptor_occupancy_pct:.1f}%, "
        f"surface={surface_receptor_pct:.1f}% R0, "
        f"internalized={internalized_fraction:.3f}, "
        f"t50={'inf' if math.isinf(t50_h) else f'{t50_h:.1f} h'}"
    )

    return ReceptorInternalizationResult(
        drug_name=drug_name,
        receptor_name=receptor_name,
        initial_receptor_density=r0_nmol_per_L,
        drug_conc_mg_L=drug_conc_mg_L,
        kd_receptor_nM=kd_receptor_nM,
        ki_internalization_per_h=ki_internalization_per_h,
        kr_recycling_per_h=kr_recycling_per_h,
        ks_synthesis_per_h=ks_synthesis_per_h,
        receptor_occupancy_pct=receptor_occupancy_pct,
        internalized_fraction=internalized_fraction,
        surface_receptor_pct=surface_receptor_pct,
        time_to_50pct_downregulation_h=t50_h,
        notes=notes,
    )
