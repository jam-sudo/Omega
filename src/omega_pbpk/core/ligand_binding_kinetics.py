"""Ligand-receptor binding kinetics simulation.

Simulates drug-target binding using a simple bimolecular association/dissociation
(kon/koff) model integrated with Forward Euler ODE stepping.

Key parameters:
    kon: Second-order association rate constant (nM^-1 h^-1)
    koff: First-order dissociation rate constant (h^-1)
    KD = koff / kon: Equilibrium dissociation constant (nM)

References:
    Copeland RA, Nat Rev Drug Discov, 2016 (residence time / drug-target kinetics)
    Swinney DC, Curr Top Med Chem, 2006 (kon/koff drug design)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LigandBindingResult:
    """Result of ligand-receptor binding kinetics simulation.

    Attributes:
        drug_name: Name of the drug/ligand.
        target_name: Name of the receptor/target.
        drug_conc_nM: Initial drug concentration in nM.
        receptor_conc_nM: Initial receptor concentration in nM.
        kon_per_nM_per_h: Association rate constant (nM^-1 h^-1).
        koff_per_h: Dissociation rate constant (h^-1).
        kd_nM: Equilibrium dissociation constant koff/kon (nM).
        t_half_complex_h: Half-life of drug-receptor complex = ln2/koff (h).
        complex_at_equilibrium_nM: Drug-receptor complex concentration at end of simulation (nM).
        occupancy_at_equilibrium_pct: Receptor occupancy at end of simulation (%).
        residence_time_h: Mean residence time of complex = 1/koff (h).
        selectivity_index: Ratio competing_kd/kd; 1.0 if no competitor specified.
        notes: Descriptive notes.
    """

    drug_name: str
    target_name: str
    drug_conc_nM: float
    receptor_conc_nM: float
    kon_per_nM_per_h: float
    koff_per_h: float
    kd_nM: float
    t_half_complex_h: float
    complex_at_equilibrium_nM: float
    occupancy_at_equilibrium_pct: float
    residence_time_h: float
    selectivity_index: float
    notes: str


def simulate_binding_kinetics(
    drug_name: str,
    target_name: str,
    drug_conc_nM: float,
    receptor_conc_nM: float,
    kon_per_nM_per_h: float,
    koff_per_h: float,
    t_end_h: float = 12.0,
    dt_h: float = 0.01,
    competing_kd_nM: float = 0.0,
) -> LigandBindingResult:
    """Simulate drug-receptor binding kinetics with Forward Euler integration.

    ODEs:
        dD/dt  = koff * DR - kon * D * R
        dR/dt  = koff * DR - kon * D * R
        dDR/dt = kon * D * R - koff * DR

    Initial conditions: D = drug_conc_nM, R = receptor_conc_nM, DR = 0.

    Args:
        drug_name: Name of the drug/ligand.
        target_name: Name of the receptor/target.
        drug_conc_nM: Initial free drug concentration in nM. Must be >= 0.
        receptor_conc_nM: Initial receptor concentration in nM. Must be >= 0.
        kon_per_nM_per_h: Association rate constant (nM^-1 h^-1). Must be > 0.
        koff_per_h: Dissociation rate constant (h^-1). Must be > 0.
        t_end_h: Simulation end time in hours (default 12.0).
        dt_h: Integration time step in hours (default 0.01).
        competing_kd_nM: KD of a competing ligand in nM (0 = no competitor).

    Returns:
        LigandBindingResult with equilibrium and kinetic parameters.

    Raises:
        ValueError: If drug_conc_nM < 0, receptor_conc_nM < 0, kon <= 0, or koff <= 0.
    """
    if drug_conc_nM < 0.0:
        raise ValueError(f"drug_conc_nM must be >= 0, got {drug_conc_nM}")
    if receptor_conc_nM < 0.0:
        raise ValueError(f"receptor_conc_nM must be >= 0, got {receptor_conc_nM}")
    if kon_per_nM_per_h <= 0.0:
        raise ValueError(f"kon_per_nM_per_h must be > 0, got {kon_per_nM_per_h}")
    if koff_per_h <= 0.0:
        raise ValueError(f"koff_per_h must be > 0, got {koff_per_h}")

    # Derived parameters
    kd_nM = koff_per_h / kon_per_nM_per_h
    t_half_complex_h = math.log(2.0) / koff_per_h
    residence_time_h = 1.0 / koff_per_h

    # Forward Euler integration
    D = drug_conc_nM
    R = receptor_conc_nM
    DR = 0.0

    n_steps = max(1, int(round(t_end_h / dt_h)))
    for _ in range(n_steps):
        association = kon_per_nM_per_h * D * R
        dissociation = koff_per_h * DR
        net = association - dissociation

        dD = -net
        dR = -net
        dDR = net

        D = max(0.0, D + dD * dt_h)
        R = max(0.0, R + dR * dt_h)
        DR = max(0.0, DR + dDR * dt_h)

    complex_at_eq = DR

    # Occupancy as fraction of original receptor concentration
    if receptor_conc_nM > 0.0:
        occupancy_pct = (DR / receptor_conc_nM) * 100.0
    else:
        occupancy_pct = 0.0
    occupancy_pct = min(100.0, max(0.0, occupancy_pct))

    # Selectivity index: competing_kd / kd (higher = more selective for primary target)
    if competing_kd_nM > 0.0:
        selectivity_index = competing_kd_nM / kd_nM
    else:
        selectivity_index = 1.0

    notes = (
        f"KD={kd_nM:.3f}nM; RT={residence_time_h:.2f}h; t½={t_half_complex_h:.2f}h; "
        f"occupancy={occupancy_pct:.1f}%; selectivity={selectivity_index:.2f}"
    )

    return LigandBindingResult(
        drug_name=drug_name,
        target_name=target_name,
        drug_conc_nM=drug_conc_nM,
        receptor_conc_nM=receptor_conc_nM,
        kon_per_nM_per_h=kon_per_nM_per_h,
        koff_per_h=koff_per_h,
        kd_nM=kd_nM,
        t_half_complex_h=t_half_complex_h,
        complex_at_equilibrium_nM=complex_at_eq,
        occupancy_at_equilibrium_pct=occupancy_pct,
        residence_time_h=residence_time_h,
        selectivity_index=selectivity_index,
        notes=notes,
    )


def screen_drug_concentrations(
    drug_name: str,
    target_name: str,
    concentrations_nM: list,
    receptor_conc_nM: float,
    kon_per_nM_per_h: float,
    koff_per_h: float,
) -> list:
    """Screen multiple drug concentrations and return binding kinetics results.

    Args:
        drug_name: Name of the drug/ligand.
        target_name: Name of the receptor/target.
        concentrations_nM: List of drug concentrations in nM to screen.
        receptor_conc_nM: Initial receptor concentration in nM.
        kon_per_nM_per_h: Association rate constant (nM^-1 h^-1).
        koff_per_h: Dissociation rate constant (h^-1).

    Returns:
        List of LigandBindingResult objects sorted ascending by drug_conc_nM.
    """
    results = []
    for conc in concentrations_nM:
        result = simulate_binding_kinetics(
            drug_name=drug_name,
            target_name=target_name,
            drug_conc_nM=conc,
            receptor_conc_nM=receptor_conc_nM,
            kon_per_nM_per_h=kon_per_nM_per_h,
            koff_per_h=koff_per_h,
        )
        results.append(result)

    # Sort ascending by drug concentration
    results.sort(key=lambda r: r.drug_conc_nM)
    return results
