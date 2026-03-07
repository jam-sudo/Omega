"""Phase 336 — Protein Binding Saturation Model.

Models concentration-dependent protein binding saturation (nonlinear PPB)
using the exact bimolecular equilibrium quadratic formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ProteinBindingSaturationResult:
    """Result of protein binding saturation analysis."""

    drug_name: str
    total_drug_conc_mg_L: float
    total_drug_conc_uM: float
    fu_at_therapeutic: float
    saturation_index: float
    bound_drug_uM: float
    free_drug_uM: float
    conc_range_uM: list
    fu_profile: list
    concentration_at_50pct_saturation_uM: float
    binding_nonlinearity_index: float
    is_nonlinear: bool
    notes: str


def _compute_fu(
    c_total_uM: float,
    protein_conc_uM: float,
    ka_per_uM: float,
    n_binding_sites: int,
) -> tuple[float, float, float]:
    """Return (fu, bound_uM, free_uM) for given total drug concentration."""
    kd = 1.0 / ka_per_uM
    total_sites = n_binding_sites * protein_conc_uM

    # Quadratic exact solution of bimolecular equilibrium
    # Bound = (C + Kd + n*P - sqrt((C + Kd + n*P)^2 - 4*C*n*P)) / 2
    a_sum = c_total_uM + kd + total_sites
    discriminant = a_sum * a_sum - 4.0 * c_total_uM * total_sites
    # Clamp discriminant to avoid numerical issues
    discriminant = max(discriminant, 0.0)
    bound = (a_sum - math.sqrt(discriminant)) / 2.0
    # Clamp bound to valid range
    bound = max(0.0, min(bound, c_total_uM))
    free = c_total_uM - bound
    fu = free / c_total_uM if c_total_uM > 0 else 1.0
    # Clamp fu to (0, 1]
    fu = max(1e-9, min(fu, 1.0))
    return fu, bound, free


def _log_space_20(c_min: float, c_max: float) -> list:
    """Generate 20 log-spaced points from c_min to c_max (geometric steps)."""
    if c_min <= 0:
        c_min = 1e-9
    log_min = math.log(c_min)
    log_max = math.log(c_max)
    step = (log_max - log_min) / 19.0  # 20 points → 19 intervals
    return [math.exp(log_min + i * step) for i in range(20)]


def calculate_protein_binding_saturation(
    drug_name: str,
    protein_conc_uM: float,
    ka_per_uM: float,
    n_binding_sites: int,
    total_drug_conc_mg_L: float,
    mw_Da: float,
) -> ProteinBindingSaturationResult:
    """Model concentration-dependent protein binding saturation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    protein_conc_uM:
        Total protein (e.g., albumin) concentration in uM. Albumin ~ 600 uM.
    ka_per_uM:
        Association constant (uM^-1).
    n_binding_sites:
        Number of equivalent binding sites per protein molecule.
    total_drug_conc_mg_L:
        Total drug concentration (mg/L) in plasma.
    mw_Da:
        Molecular weight of the drug (Da).
    """
    # --- Input validation ---
    if protein_conc_uM <= 0:
        raise ValueError("protein_conc_uM must be > 0")
    if ka_per_uM <= 0:
        raise ValueError("ka_per_uM must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if n_binding_sites < 1:
        raise ValueError("n_binding_sites must be >= 1")
    if total_drug_conc_mg_L <= 0:
        raise ValueError("total_drug_conc_mg_L must be > 0")

    # Convert total drug concentration to uM
    # mg/L ÷ Da * 1000 → uM
    c_total_uM = (total_drug_conc_mg_L / mw_Da) * 1000.0

    # Compute fu, bound, free at therapeutic concentration
    fu_therapeutic, bound_uM, free_uM = _compute_fu(
        c_total_uM, protein_conc_uM, ka_per_uM, n_binding_sites
    )

    # Saturation index = Bound / (n * protein_conc_uM) at C_total
    total_sites = n_binding_sites * protein_conc_uM
    saturation_index = bound_uM / total_sites
    saturation_index = max(0.0, min(saturation_index, 1.0))

    # Concentration range: 20 log-spaced points from C_total/100 to C_total*100
    c_min = c_total_uM / 100.0
    c_max = c_total_uM * 100.0
    conc_range_uM = _log_space_20(c_min, c_max)

    # fu profile
    fu_profile = []
    for c in conc_range_uM:
        fu_c, _, _ = _compute_fu(c, protein_conc_uM, ka_per_uM, n_binding_sites)
        fu_profile.append(fu_c)

    # concentration_at_50pct_saturation_uM
    # 50% saturation: Bound = 0.5 * n * P
    # From quadratic: 0.5*n*P = (C + Kd + n*P - sqrt(...)) / 2
    # → C + Kd + n*P - n*P = sqrt((C + Kd + n*P)^2 - 4*C*n*P)
    # Let B50 = 0.5 * total_sites
    # B50 = (C + Kd + n*P - sqrt((C + Kd + n*P)^2 - 4*C*n*P)) / 2
    # 2*B50 = C + Kd + n*P - sqrt(...)
    # sqrt(...) = C + Kd + n*P - 2*B50 = C + Kd
    # (C + Kd)^2 = (C + Kd + n*P)^2 - 4*C*n*P
    # Expand: C^2 + 2C*Kd + Kd^2 = C^2 + 2C*Kd + Kd^2 + 2C*n*P + 2Kd*n*P + (n*P)^2 - 4C*n*P
    # 0 = -2C*n*P + 2Kd*n*P + (n*P)^2
    # 2C*n*P = 2Kd*n*P + (n*P)^2
    # 2C = 2Kd + n*P
    # C_50 = Kd + n*P/2
    kd = 1.0 / ka_per_uM
    concentration_at_50pct_saturation_uM = kd + total_sites / 2.0

    # Binding nonlinearity index: fu_high / fu_low (high=10x, low=0.1x therapeutic)
    fu_high, _, _ = _compute_fu(c_total_uM * 10.0, protein_conc_uM, ka_per_uM, n_binding_sites)
    fu_low, _, _ = _compute_fu(c_total_uM * 0.1, protein_conc_uM, ka_per_uM, n_binding_sites)
    binding_nonlinearity_index = fu_high / fu_low if fu_low > 0 else 1.0

    is_nonlinear = binding_nonlinearity_index > 1.2

    # Notes
    notes_parts = [
        f"Total drug: {c_total_uM:.3f} uM at therapeutic concentration.",
        f"fu at therapeutic: {fu_therapeutic:.4f}.",
        f"Saturation index: {saturation_index:.4f}.",
        f"50% saturation at: {concentration_at_50pct_saturation_uM:.3f} uM.",
        f"Nonlinearity index (fu_10x/fu_0.1x): {binding_nonlinearity_index:.3f}.",
        "Binding is nonlinear." if is_nonlinear else "Binding is approximately linear.",
    ]
    notes = " ".join(notes_parts)

    return ProteinBindingSaturationResult(
        drug_name=drug_name,
        total_drug_conc_mg_L=total_drug_conc_mg_L,
        total_drug_conc_uM=c_total_uM,
        fu_at_therapeutic=fu_therapeutic,
        saturation_index=saturation_index,
        bound_drug_uM=bound_uM,
        free_drug_uM=free_uM,
        conc_range_uM=conc_range_uM,
        fu_profile=fu_profile,
        concentration_at_50pct_saturation_uM=concentration_at_50pct_saturation_uM,
        binding_nonlinearity_index=binding_nonlinearity_index,
        is_nonlinear=is_nonlinear,
        notes=notes,
    )
