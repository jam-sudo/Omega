"""Saturable (concentration-dependent) plasma protein binding model.

Uses a Scatchard-type binding model to compute free and bound drug
concentrations given total drug concentration, association constant,
protein concentration, and number of binding sites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinBindingResult:
    drug_name: str
    total_conc_mg_L: float
    free_conc_mg_L: float
    bound_conc_mg_L: float
    fup: float
    binding_ratio_r: float
    percent_bound: float
    saturation_pct: float
    notes: str


def simulate_protein_binding(
    drug_name: str,
    total_conc_mg_L: float,
    n_sites: int = 1,
    ka_L_per_mg: float = 0.5,
    protein_conc_g_L: float = 40.0,
    mw_drug: float = 300.0,
    mw_protein: float = 67000.0,
) -> ProteinBindingResult:
    """Simulate saturable protein binding at a single concentration.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    total_conc_mg_L : float
        Total drug concentration in plasma (mg/L).
    n_sites : int
        Number of binding sites per protein molecule.
    ka_L_per_mg : float
        Association constant in L/mg.
    protein_conc_g_L : float
        Total protein (albumin) concentration in g/L.
    mw_drug : float
        Drug molecular weight in Da.
    mw_protein : float
        Protein molecular weight in Da.

    Returns
    -------
    ProteinBindingResult
    """
    # --- Validation ---
    if total_conc_mg_L <= 0:
        raise ValueError("total_conc_mg_L must be > 0")
    if ka_L_per_mg <= 0:
        raise ValueError("ka_L_per_mg must be > 0")
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if protein_conc_g_L <= 0:
        raise ValueError("protein_conc_g_L must be > 0")
    if mw_drug <= 0:
        raise ValueError("mw_drug must be > 0")
    if mw_protein <= 0:
        raise ValueError("mw_protein must be > 0")

    # --- Unit conversions ---
    # Ka in L/umol
    ka_umol = ka_L_per_mg * mw_drug / 1000.0
    # Protein concentration in umol/L
    p_total_umol = protein_conc_g_L / mw_protein * 1e6
    # Total drug concentration in umol/L
    total_umol = total_conc_mg_L / mw_drug * 1e6

    # --- Solve quadratic for free drug ---
    # total = free + n * P * Ka * free / (1 + Ka * free)
    # Ka * free^2 + free * (1 + n*P*Ka - total*Ka) - total = 0
    a_coeff = ka_umol
    b_coeff = 1.0 + n_sites * p_total_umol * ka_umol - total_umol * ka_umol
    c_coeff = -total_umol

    discriminant = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
    free_umol = (-b_coeff + math.sqrt(discriminant)) / (2.0 * a_coeff)

    # --- Derived quantities ---
    bound_umol = total_umol - free_umol
    fup = free_umol / total_umol
    # Clamp fup to [0, 1] for numerical safety
    fup = max(0.0, min(1.0, fup))

    free_conc_mg_L = fup * total_conc_mg_L
    bound_conc_mg_L = total_conc_mg_L - free_conc_mg_L

    binding_ratio_r = bound_umol / p_total_umol
    percent_bound = (1.0 - fup) * 100.0
    saturation_pct = binding_ratio_r / n_sites * 100.0

    notes = (
        f"Saturable binding: {drug_name} at {total_conc_mg_L:.3g} mg/L, "
        f"fup={fup:.4f}, {percent_bound:.1f}% bound, "
        f"saturation={saturation_pct:.1f}%"
    )

    return ProteinBindingResult(
        drug_name=drug_name,
        total_conc_mg_L=total_conc_mg_L,
        free_conc_mg_L=free_conc_mg_L,
        bound_conc_mg_L=bound_conc_mg_L,
        fup=fup,
        binding_ratio_r=binding_ratio_r,
        percent_bound=percent_bound,
        saturation_pct=saturation_pct,
        notes=notes,
    )


def binding_curve(
    drug_name: str,
    concentrations_mg_L: list[float],
    n_sites: int = 1,
    ka_L_per_mg: float = 0.5,
    protein_conc_g_L: float = 40.0,
    mw_drug: float = 300.0,
    mw_protein: float = 67000.0,
) -> list[ProteinBindingResult]:
    """Compute protein binding across a range of concentrations.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    concentrations_mg_L : list[float]
        List of total drug concentrations (mg/L).
    n_sites, ka_L_per_mg, protein_conc_g_L, mw_drug, mw_protein :
        See :func:`simulate_protein_binding`.

    Returns
    -------
    list[ProteinBindingResult]
    """
    if not concentrations_mg_L:
        raise ValueError("concentrations_mg_L must not be empty")

    return [
        simulate_protein_binding(
            drug_name=drug_name,
            total_conc_mg_L=c,
            n_sites=n_sites,
            ka_L_per_mg=ka_L_per_mg,
            protein_conc_g_L=protein_conc_g_L,
            mw_drug=mw_drug,
            mw_protein=mw_protein,
        )
        for c in concentrations_mg_L
    ]
