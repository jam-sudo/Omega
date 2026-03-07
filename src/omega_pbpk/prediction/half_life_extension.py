"""Phase 1045 — Drug Half-Life Extension Strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_STRATEGIES = {
    "PEGylation",
    "albumin_binding",
    "FcRn_recycling",
    "depot_formulation",
    "prodrug",
    "nanoparticle",
}


@dataclass(frozen=True)
class HalfLifeExtensionResult:
    drug_name: str
    strategy: str
    baseline_t_half_h: float
    extended_t_half_h: float
    extension_ratio: float
    mechanism: str
    auc_ratio: float
    dosing_frequency_reduction: float
    recommended: bool
    notes: str


def predict_half_life_extension(
    drug_name: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    strategy: str = "PEGylation",
    mw_Da: float = 300.0,
    peg_mw_kDa: float = 0.0,
    depot_release_t50_h: float = 0.0,
    prodrug_conversion_t_half_h: float = 0.0,
    albumin_affinity_uM: float = 10.0,
) -> HalfLifeExtensionResult:
    """Predict how formulation/chemical strategies extend plasma half-life."""
    if baseline_cl_L_per_h <= 0:
        raise ValueError("baseline_cl_L_per_h must be > 0")
    if baseline_vd_L <= 0:
        raise ValueError("baseline_vd_L must be > 0")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_VALID_STRATEGIES)}")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    baseline_t_half = math.log(2) * baseline_vd_L / baseline_cl_L_per_h

    # Defaults — may be overridden by depot/prodrug
    effective_cl = baseline_cl_L_per_h
    effective_vd = baseline_vd_L
    effective_t_half: float | None = None  # None means derive from cl/vd
    mechanism = "reduced_clearance"
    notes_parts: list[str] = []

    if strategy == "PEGylation":
        cl_reduction = max(0.1, 1 - 0.05 * peg_mw_kDa)
        vd_increase = 1 + 0.02 * peg_mw_kDa
        effective_cl = baseline_cl_L_per_h * cl_reduction
        effective_vd = baseline_vd_L * vd_increase
        mechanism = "PEGylation"
        notes_parts.append(
            f"PEG {peg_mw_kDa} kDa reduces CL by {(1 - cl_reduction) * 100:.0f}%"
            f" and increases Vd by {(vd_increase - 1) * 100:.0f}%."
        )

    elif strategy == "albumin_binding":
        fu_effective = albumin_affinity_uM / (albumin_affinity_uM + 600.0)
        effective_cl = baseline_cl_L_per_h * max(0.05, fu_effective)
        effective_vd = baseline_vd_L / fu_effective
        mechanism = "reduced_clearance"
        notes_parts.append(
            f"Albumin Kd={albumin_affinity_uM} µM → fu_eff={fu_effective:.4f};"
            " CL reduced, Vd increased."
        )

    elif strategy == "FcRn_recycling":
        recycling_factor = 0.3 - 0.001 * mw_Da
        recycling_factor = min(0.9, max(0.1, recycling_factor))
        effective_cl = baseline_cl_L_per_h * (1.0 - recycling_factor)
        effective_vd = baseline_vd_L
        mechanism = "recycling"
        notes_parts.append(
            f"FcRn recycling factor={recycling_factor:.3f} for MW={mw_Da} Da; CL reduced."
        )

    elif strategy == "depot_formulation":
        elim_t_half = math.log(2) * baseline_vd_L / baseline_cl_L_per_h
        release_t_half = depot_release_t50_h if depot_release_t50_h > 0 else 0.0
        effective_t_half = max(release_t_half, elim_t_half)
        effective_cl = baseline_cl_L_per_h  # unchanged
        effective_vd = baseline_vd_L
        mechanism = "depot"
        notes_parts.append(
            f"Depot release t½={depot_release_t50_h} h;"
            f" effective t½ dominated by max(release, elim)={effective_t_half:.2f} h."
        )

    elif strategy == "prodrug":
        elim_t_half = math.log(2) * baseline_vd_L / baseline_cl_L_per_h
        conv_t_half = prodrug_conversion_t_half_h if prodrug_conversion_t_half_h > 0 else 0.0
        effective_t_half = max(conv_t_half, elim_t_half)
        if effective_t_half > 0:
            effective_cl = math.log(2) * baseline_vd_L / effective_t_half
        else:
            effective_cl = baseline_cl_L_per_h
        effective_vd = baseline_vd_L
        mechanism = "prodrug_conversion"
        notes_parts.append(
            f"Prodrug conversion t½={prodrug_conversion_t_half_h} h;"
            f" effective t½={effective_t_half:.2f} h."
        )

    elif strategy == "nanoparticle":
        cl_reduction = 0.4
        vd_increase = 3.0
        effective_cl = baseline_cl_L_per_h * cl_reduction
        effective_vd = baseline_vd_L * vd_increase
        mechanism = "depot"
        notes_parts.append(
            "Nanoparticle: CL reduced 60%, Vd increased 3x via MPS uptake and tissue distribution."
        )

    # Compute extended t½
    if effective_t_half is not None:
        extended_t_half = effective_t_half
    else:
        extended_t_half = math.log(2) * effective_vd / effective_cl

    # Guard against divide-by-zero
    if baseline_t_half <= 0:
        extension_ratio = 1.0
    else:
        extension_ratio = extended_t_half / baseline_t_half

    # AUC ∝ 1/CL at same dose
    auc_ratio = baseline_cl_L_per_h / effective_cl

    dosing_frequency_reduction = extension_ratio
    recommended = extension_ratio >= 2.0

    notes_parts.append(
        f"Baseline t½={baseline_t_half:.2f} h → extended t½={extended_t_half:.2f} h"
        f" (ratio={extension_ratio:.2f}x). AUC ratio={auc_ratio:.2f}."
    )
    if recommended:
        notes_parts.append("Strategy recommended (>=2x extension).")

    return HalfLifeExtensionResult(
        drug_name=drug_name,
        strategy=strategy,
        baseline_t_half_h=baseline_t_half,
        extended_t_half_h=extended_t_half,
        extension_ratio=extension_ratio,
        mechanism=mechanism,
        auc_ratio=auc_ratio,
        dosing_frequency_reduction=dosing_frequency_reduction,
        recommended=recommended,
        notes=" ".join(notes_parts),
    )
