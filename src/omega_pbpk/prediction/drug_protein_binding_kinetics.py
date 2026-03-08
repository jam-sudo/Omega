"""Phase 369 — Drug-Protein Binding Kinetics (kon/koff).

Models the kinetics of drug-protein binding (association/dissociation)
and its impact on free drug fraction using bimolecular ODE system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DrugProteinBindingKineticsResult:
    """Result of drug-protein binding kinetics simulation."""

    drug_name: str
    drug_conc_uM: float
    kd_uM: float
    times_s: list
    free_drug_uM: list
    bound_drug_uM: list
    fu_at_equilibrium: float
    t_half_binding_s: float
    t_half_dissociation_s: float
    equilibrium_time_s: float
    binding_complete: bool
    notes: str


def simulate_drug_protein_binding(
    drug_name: str,
    drug_conc_mg_L: float,
    mw_Da: float,
    protein_conc_uM: float = 600.0,
    kon_per_uM_per_s: float = 1e-3,
    koff_per_s: float = 1e-3,
    n_sites: int = 1,
    t_end_s: float = 60.0,
    dt_s: float = 0.01,
) -> DrugProteinBindingKineticsResult:
    """Simulate drug-protein binding kinetics using Forward Euler ODE integration.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    drug_conc_mg_L:
        Drug concentration in mg/L (plasma concentration).
    mw_Da:
        Molecular weight of the drug in Daltons.
    protein_conc_uM:
        Protein concentration in µM (default: 600 µM for albumin).
    kon_per_uM_per_s:
        Association rate constant in µM⁻¹·s⁻¹.
    koff_per_s:
        Dissociation rate constant in s⁻¹.
    n_sites:
        Number of binding sites per protein molecule.
    t_end_s:
        Total simulation time in seconds.
    dt_s:
        Time step for Forward Euler integration (seconds).

    Returns
    -------
    DrugProteinBindingKineticsResult
    """
    # --- Validation ---
    if drug_conc_mg_L <= 0:
        raise ValueError("drug_conc_mg_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if protein_conc_uM <= 0:
        raise ValueError("protein_conc_uM must be > 0")
    if kon_per_uM_per_s <= 0:
        raise ValueError("kon_per_uM_per_s must be > 0")
    if koff_per_s <= 0:
        raise ValueError("koff_per_s must be > 0")
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if t_end_s <= 0:
        raise ValueError("t_end_s must be > 0")

    # --- Derived parameters ---
    # Convert drug concentration from mg/L to µM
    # drug_uM = (mg/L) / (Da) * 1e6 / 1000 = (mg/L) / (g/mol) * 1000 = (mg/L * 1000) / (g/mol)
    # More precisely: drug_uM = drug_conc_mg_L / mw_Da * 1e6 / 1000
    drug_uM = drug_conc_mg_L / mw_Da * 1e6 / 1000.0

    # Equilibrium dissociation constant (µM)
    kd_uM = koff_per_s / kon_per_uM_per_s

    # Total protein binding sites (µM)
    protein_total_uM = protein_conc_uM * n_sites

    # Analytical half-lives
    t_half_binding_s = math.log(2.0) / (kon_per_uM_per_s * protein_conc_uM + koff_per_s)
    t_half_dissociation_s = math.log(2.0) / koff_per_s

    # Analytical equilibrium fu (using quadratic solution for bimolecular binding)
    # At equilibrium: kon * D_eq * P_eq = koff * DP_eq
    # Conservation: D_eq + DP_eq = drug_uM, P_eq + DP_eq = protein_total_uM
    # Quadratic: DP_eq^2 - (drug_uM + protein_total_uM + kd_uM)*DP_eq + drug_uM*protein_total_uM = 0
    a_coef = 1.0
    b_coef = -(drug_uM + protein_total_uM + kd_uM)
    c_coef = drug_uM * protein_total_uM
    discriminant = b_coef * b_coef - 4.0 * a_coef * c_coef
    discriminant = max(0.0, discriminant)
    dp_eq_analytical = (-b_coef - math.sqrt(discriminant)) / (2.0 * a_coef)
    dp_eq_analytical = max(0.0, min(dp_eq_analytical, min(drug_uM, protein_total_uM)))
    d_eq_analytical = drug_uM - dp_eq_analytical
    fu_analytical = d_eq_analytical / drug_uM if drug_uM > 0 else 1.0

    # --- Forward Euler ODE integration ---
    # States: D (free drug), P (free protein), DP (complex)
    D = drug_uM
    P = protein_total_uM
    DP = 0.0

    n_steps = max(1, int(t_end_s / dt_s))
    # Adjust dt to fit exactly
    dt_actual = t_end_s / n_steps

    times_s: list = [0.0]
    free_drug_uM: list = [D]
    bound_drug_uM: list = [DP]

    # Track when equilibrium (95% of equilibrium bound) is reached
    dp_target = 0.95 * dp_eq_analytical
    equilibrium_time_s = t_end_s  # default to end if never reached

    for i in range(1, n_steps + 1):
        # Bimolecular kinetics
        formation_rate = kon_per_uM_per_s * D * P
        dissociation_rate = koff_per_s * DP
        dDP_dt = formation_rate - dissociation_rate

        D_new = D - dDP_dt * dt_actual
        P_new = P - dDP_dt * dt_actual
        DP_new = DP + dDP_dt * dt_actual

        # Clamp to physical bounds
        DP_new = max(0.0, min(DP_new, min(drug_uM, protein_total_uM)))
        D_new = max(0.0, drug_uM - DP_new)
        P_new = max(0.0, protein_total_uM - DP_new)

        D = D_new
        P = P_new
        DP = DP_new

        t_now = i * dt_actual
        times_s.append(t_now)
        free_drug_uM.append(D)
        bound_drug_uM.append(DP)

        # Check 95% equilibrium
        if equilibrium_time_s == t_end_s and dp_eq_analytical > 0 and DP >= dp_target:
            equilibrium_time_s = t_now

    # If dp_eq_analytical is essentially 0 (very weak binding), equilibrium is immediate
    if dp_eq_analytical <= 1e-12:
        equilibrium_time_s = 0.0

    # Final fu at end of simulation
    d_final = free_drug_uM[-1]
    dp_final = bound_drug_uM[-1]
    total_final = d_final + dp_final
    fu_at_equilibrium = d_final / total_final if total_final > 1e-15 else 1.0

    # binding_complete: final fu is within 1% of analytical fu
    binding_complete = fu_at_equilibrium <= 1.01 * fu_analytical + 1e-9

    # Notes
    if dp_eq_analytical / drug_uM > 0.9:
        binding_note = "Strongly protein-bound compound (>90% bound)"
    elif dp_eq_analytical / drug_uM > 0.5:
        binding_note = "Moderately protein-bound compound (50-90% bound)"
    else:
        binding_note = "Weakly protein-bound compound (<50% bound)"

    notes = (
        f"Kd={kd_uM:.4f} uM; fu_analytical={fu_analytical:.4f}; "
        f"t_half_binding={t_half_binding_s:.4f}s; {binding_note}"
    )

    return DrugProteinBindingKineticsResult(
        drug_name=drug_name,
        drug_conc_uM=drug_uM,
        kd_uM=kd_uM,
        times_s=times_s,
        free_drug_uM=free_drug_uM,
        bound_drug_uM=bound_drug_uM,
        fu_at_equilibrium=fu_at_equilibrium,
        t_half_binding_s=t_half_binding_s,
        t_half_dissociation_s=t_half_dissociation_s,
        equilibrium_time_s=equilibrium_time_s,
        binding_complete=binding_complete,
        notes=notes,
    )
