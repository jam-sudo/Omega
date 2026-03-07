"""Drug-protein binding kinetics — albumin/AAG Langmuir ODE model with time-dependent fu.

Models time-dependent equilibration between drug and plasma proteins using
Langmuir binding kinetics (forward Euler ODE integration).

References
----------
- Barre J et al. Clin Pharmacokinet 1983; albumin binding kinetics
- Tillement JP et al. Clin Pharmacokinet 1984; AAG binding
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Protein MW database (Da)
# ---------------------------------------------------------------------------

_PROTEIN_MW_DA: dict[str, float] = {
    "albumin": 66500.0,
    "aag": 41000.0,
    "lipoprotein": 500000.0,
    "custom": 66500.0,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProteinBindingKineticsResult:
    """Result of a protein binding kinetics simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    protein_name : Protein name ("albumin", "AAG", "lipoprotein", "custom").
    initial_drug_conc_mg_L : Initial total drug concentration (mg/L).
    protein_conc_g_L : Protein concentration (g/L).
    kd_mg_per_L : Dissociation constant in mg/L (koff / kon).
    fu_equilibrium : Unbound fraction at equilibrium (C_free_final / initial).
    fu_time_dependent : List of fu at each time point.
    times_h : Simulation time points (h).
    c_free_mg_L : Free drug concentration at each time point (mg/L).
    c_bound_mg_L : Bound drug concentration at each time point (mg/L).
    t_half_binding_min : Pseudo-first-order half-life of binding (min).
    max_binding_sites_mg_per_L : Bmax — total binding capacity (mg/L).
    notes : Summary string.
    """

    drug_name: str
    protein_name: str
    initial_drug_conc_mg_L: float
    protein_conc_g_L: float
    kd_mg_per_L: float
    fu_equilibrium: float
    fu_time_dependent: list
    times_h: list
    c_free_mg_L: list
    c_bound_mg_L: list
    t_half_binding_min: float
    max_binding_sites_mg_per_L: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------


def simulate_protein_binding(
    drug_name: str,
    protein_name: str = "albumin",
    drug_conc_mg_L: float = 10.0,
    protein_conc_g_L: float = 45.0,
    kon_per_mg_per_L_per_h: float = 100.0,
    koff_per_h: float = 50.0,
    n_binding_sites: int = 1,
    drug_mw_da: float = 400.0,
    t_end_h: float = 0.1,
    dt_h: float = 0.001,
) -> ProteinBindingKineticsResult:
    """Simulate drug-protein binding kinetics using Langmuir ODE (forward Euler).

    Langmuir ODEs:
        d(C_free)/dt  = -kon * C_free * (Bmax - C_bound) + koff * C_bound
        d(C_bound)/dt =  kon * C_free * (Bmax - C_bound) - koff * C_bound

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    protein_name : str
        Plasma protein ("albumin", "AAG", "lipoprotein", "custom").
    drug_conc_mg_L : float
        Initial drug concentration (mg/L). Must be >= 0.
    protein_conc_g_L : float
        Plasma protein concentration (g/L). Must be > 0.
    kon_per_mg_per_L_per_h : float
        Association rate constant (L/mg/h). Must be > 0.
    koff_per_h : float
        Dissociation rate constant (h^-1). Must be > 0.
    n_binding_sites : int
        Number of binding sites per protein molecule (>= 1).
    drug_mw_da : float
        Drug molecular weight (Da), used for Bmax calculation.
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    ProteinBindingKineticsResult
    """
    # --- Validation ---
    if drug_conc_mg_L < 0.0:
        raise ValueError("drug_conc_mg_L must be >= 0")
    if protein_conc_g_L <= 0.0:
        raise ValueError("protein_conc_g_L must be > 0")
    if kon_per_mg_per_L_per_h <= 0.0:
        raise ValueError("kon must be > 0")
    if koff_per_h <= 0.0:
        raise ValueError("koff must be > 0")
    if n_binding_sites < 1:
        raise ValueError("n_binding_sites must be >= 1")

    protein_key = protein_name.lower()
    protein_mw_da = _PROTEIN_MW_DA.get(protein_key, 66500.0)

    # --- Derived parameters ---
    # Bmax (mg drug / L):
    #   moles of protein per L = protein_conc_g_L * 1000 (mg/L) / protein_mw_da
    #   moles of binding sites = moles_protein * n_binding_sites
    #   mg of drug bound at saturation = moles_sites * drug_mw_da
    bmax_mg_per_L = protein_conc_g_L * 1000.0 * n_binding_sites * drug_mw_da / protein_mw_da

    kd_mg_per_L = koff_per_h / kon_per_mg_per_L_per_h

    # Pseudo-first-order approximation for t_half_binding
    k_app = kon_per_mg_per_L_per_h * drug_conc_mg_L + koff_per_h
    t_half_binding_h = math.log(2.0) / k_app if k_app > 0.0 else float("inf")
    t_half_binding_min = t_half_binding_h * 60.0

    # --- Forward Euler ODE ---
    # Track only c_bound; derive c_free = drug_conc_mg_L - c_bound to enforce
    # exact drug mass conservation: c_free + c_bound = drug_conc_mg_L at all times.
    n_steps = max(int(round(t_end_h / dt_h)), 1)

    times: list[float] = []
    c_free_list: list[float] = []
    c_bound_list: list[float] = []
    fu_list: list[float] = []

    c_bound = 0.0
    t = 0.0

    # Record initial state
    c_free = float(drug_conc_mg_L) - c_bound
    times.append(t)
    c_free_list.append(c_free)
    c_bound_list.append(c_bound)
    fu_list.append(c_free / drug_conc_mg_L if drug_conc_mg_L > 0.0 else 0.0)

    for _ in range(n_steps):
        c_free = max(drug_conc_mg_L - c_bound, 0.0)
        available_sites = max(bmax_mg_per_L - c_bound, 0.0)
        dbound_dt = kon_per_mg_per_L_per_h * c_free * available_sites - koff_per_h * c_bound

        c_bound_new = c_bound + dbound_dt * dt_h
        # Clamp to physically valid range [0, min(drug_conc, bmax)]
        c_bound = min(max(c_bound_new, 0.0), min(drug_conc_mg_L, bmax_mg_per_L))

        c_free = max(drug_conc_mg_L - c_bound, 0.0)
        t += dt_h

        times.append(t)
        c_free_list.append(c_free)
        c_bound_list.append(c_bound)
        fu_list.append(c_free / drug_conc_mg_L if drug_conc_mg_L > 0.0 else 0.0)

    # fu_equilibrium: free fraction relative to initial drug dose
    fu_eq = c_free / drug_conc_mg_L if drug_conc_mg_L > 0.0 else 0.0

    notes = (
        f"Kd={kd_mg_per_L:.3f} mg/L; Bmax={bmax_mg_per_L:.2f} mg/L; "
        f"fu_eq={fu_eq:.3f}; t_half_bind={t_half_binding_min:.1f} min"
    )

    return ProteinBindingKineticsResult(
        drug_name=drug_name,
        protein_name=protein_name,
        initial_drug_conc_mg_L=drug_conc_mg_L,
        protein_conc_g_L=protein_conc_g_L,
        kd_mg_per_L=kd_mg_per_L,
        fu_equilibrium=fu_eq,
        fu_time_dependent=fu_list,
        times_h=times,
        c_free_mg_L=c_free_list,
        c_bound_mg_L=c_bound_list,
        t_half_binding_min=t_half_binding_min,
        max_binding_sites_mg_per_L=bmax_mg_per_L,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare proteins
# ---------------------------------------------------------------------------


def compare_proteins(
    drug_name: str,
    drug_conc_mg_L: float,
    drug_mw_da: float,
    protein_concentrations: list[dict],
) -> list[ProteinBindingKineticsResult]:
    """Compare drug binding across multiple plasma proteins.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    drug_conc_mg_L : float
        Initial drug concentration (mg/L).
    drug_mw_da : float
        Drug molecular weight (Da).
    protein_concentrations : list[dict]
        Each dict must contain "protein_name" (str) and "conc_g_L" (float).

    Returns
    -------
    list[ProteinBindingKineticsResult]
        Sorted by fu_equilibrium ascending (tightest binding first).
    """
    results: list[ProteinBindingKineticsResult] = []
    for entry in protein_concentrations:
        pname = str(entry["protein_name"])
        conc = float(entry["conc_g_L"])
        result = simulate_protein_binding(
            drug_name=drug_name,
            protein_name=pname,
            drug_conc_mg_L=drug_conc_mg_L,
            drug_mw_da=drug_mw_da,
            protein_conc_g_L=conc,
        )
        results.append(result)

    return sorted(results, key=lambda r: r.fu_equilibrium)


__all__ = [
    "ProteinBindingKineticsResult",
    "simulate_protein_binding",
    "compare_proteins",
]
