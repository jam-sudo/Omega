"""Phase 686 — Plasma Binding Saturation model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PlasmaBindingSaturationResult:
    """Result of plasma binding saturation simulation."""

    drug_name: str
    total_conc_mg_L: list = field(default_factory=list)
    free_conc_mg_L: list = field(default_factory=list)
    bound_conc_mg_L: list = field(default_factory=list)
    fu_list: list = field(default_factory=list)
    fu_at_therapeutic: float = 0.0
    fu_at_toxic: float = 0.0
    binding_saturated: bool = False
    bmax_mg_L: float = 0.0
    kd_mg_L: float = 0.0
    protein: str = ""
    notes: str = ""


_VALID_PROTEINS = {"albumin", "aag"}


def simulate_plasma_binding_saturation(
    drug_name: str,
    mw_Da: float,
    logP: float,
    protein: str = "albumin",
    therapeutic_conc_mg_L: float = 1.0,
    toxic_conc_mg_L: float = 10.0,
    n_points: int = 50,
) -> PlasmaBindingSaturationResult:
    """Simulate saturable plasma protein binding."""

    # Validation
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")
    if therapeutic_conc_mg_L <= 0:
        raise ValueError("therapeutic_conc_mg_L must be positive")
    if toxic_conc_mg_L <= 0:
        raise ValueError("toxic_conc_mg_L must be positive")
    if protein not in _VALID_PROTEINS:
        raise ValueError(f"protein must be one of {sorted(_VALID_PROTEINS)}, got '{protein}'")
    if n_points < 5:
        raise ValueError("n_points must be >= 5")

    # Binding parameters
    if protein == "albumin":
        bmax = 7000.0
        kd = max(0.01, 10.0 / (1.0 + max(0.0, logP) * 0.5))
    else:  # aag
        bmax = 200.0
        kd = max(0.001, 1.0 / (1.0 + max(0.0, logP) * 1.0))

    # Log-spaced concentration range
    c_min = 0.001
    c_max = 100.0 * toxic_conc_mg_L
    factor = (c_max / c_min) ** (1.0 / (n_points - 1))
    total_concs = [c_min * (factor**i) for i in range(n_points)]

    free_concs = []
    bound_concs = []
    fu_list = []

    for c_total in total_concs:
        # Solve: C_free + Bmax*C_free/(Kd+C_free) = C_total
        # Quadratic: C_free^2 + (Kd + Bmax - C_total)*C_free - Kd*C_total = 0
        b_coeff = kd + bmax - c_total
        discriminant = b_coeff * b_coeff + 4.0 * kd * c_total
        c_free = (-b_coeff + math.sqrt(discriminant)) / 2.0
        if c_free < 0:
            c_free = 0.0
        if c_free > c_total:
            c_free = c_total

        c_bound = c_total - c_free
        fu = c_free / c_total if c_total > 0 else 1.0

        free_concs.append(c_free)
        bound_concs.append(c_bound)
        fu_list.append(fu)

    # Find fu at therapeutic and toxic concentrations (closest point)
    def _closest_fu(target: float) -> float:
        best_idx = 0
        best_diff = abs(total_concs[0] - target)
        for i in range(1, len(total_concs)):
            diff = abs(total_concs[i] - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return fu_list[best_idx]

    fu_therapeutic = _closest_fu(therapeutic_conc_mg_L)
    fu_toxic = _closest_fu(toxic_conc_mg_L)

    binding_saturated = fu_toxic > 2.0 * fu_therapeutic

    notes_parts = [
        f"Protein: {protein}",
        f"Bmax: {bmax:.1f} mg/L",
        f"Kd: {kd:.4f} mg/L",
        f"fu at therapeutic ({therapeutic_conc_mg_L} mg/L): {fu_therapeutic:.4f}",
        f"fu at toxic ({toxic_conc_mg_L} mg/L): {fu_toxic:.4f}",
    ]
    if binding_saturated:
        notes_parts.append("Binding saturation detected at toxic concentrations")

    return PlasmaBindingSaturationResult(
        drug_name=drug_name,
        total_conc_mg_L=total_concs,
        free_conc_mg_L=free_concs,
        bound_conc_mg_L=bound_concs,
        fu_list=fu_list,
        fu_at_therapeutic=fu_therapeutic,
        fu_at_toxic=fu_toxic,
        binding_saturated=binding_saturated,
        bmax_mg_L=bmax,
        kd_mg_L=kd,
        protein=protein,
        notes="; ".join(notes_parts),
    )
