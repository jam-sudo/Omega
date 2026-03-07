"""Gut wall first-pass metabolism full model.

Comprehensive regional model integrating CYP3A4 activity, P-gp efflux, and
dissolution across 4 GI segments (duodenum, jejunum, ileum, colon).

References:
    Paine MF et al., J Pharmacol Exp Ther 1997.
    Yang J et al., Drug Metab Dispos 2007.
    Rostami-Hodjegan A, Tucker GT, Nat Rev Drug Discov 2007.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "GutWallFullResult",
    "simulate_gut_wall_full",
    "gut_availability_sensitivity",
]

# Segment anatomical constants
_SEGMENT_NAMES = ["duodenum", "jejunum", "ileum", "colon"]
_TRANSIT_TIMES_H = [0.5, 2.0, 1.5, 20.0]
_LENGTHS_CM = [25.0, 250.0, 350.0, 150.0]
_RADII_CM = [1.5, 1.75, 1.75, 3.0]


@dataclass(frozen=True)
class GutWallFullResult:
    """Result of comprehensive gut wall first-pass metabolism simulation."""

    drug_name: str
    dose_mg: float

    # Regional absorbed fractions
    fa_duodenum: float
    fa_jejunum: float
    fa_ileum: float
    fa_colon: float

    # Gut extraction ratios per segment (1 - fg_segment)
    eg_duodenum: float
    eg_jejunum: float
    eg_ileum: float
    eg_colon: float

    # Overall metrics
    fa_total: float
    fg_total: float
    fh: float
    f_oral_total: float

    # Metabolite fractions
    f_parent_portal: float
    f_metabolized_gut: float

    cl_gut_wall_L_per_h: float
    notes: str


def simulate_gut_wall_full(
    drug_name: str,
    dose_mg: float,
    peff_cm_per_s: float,
    solubility_mg_mL: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,  # noqa: ARG001 — accepted for API symmetry; not used in static model
    fu_plasma: float = 0.8,  # noqa: ARG001 — kept for signature completeness
    fu_gut: float = 0.5,
    cyp3a4_relative_abundance: list[float] | None = None,
    pgp_efflux_ratio: float = 1.5,
    dose_volume_mL: float = 250.0,
    hepatic_blood_flow_L_per_h: float = 90.0,
    t_end_h: float = 6.0,  # noqa: ARG001 — static model; no time integration
    dt_h: float = 0.05,  # noqa: ARG001 — static model
) -> GutWallFullResult:
    """Simulate regional gut wall first-pass metabolism.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Oral dose in mg.
    peff_cm_per_s:
        Effective intestinal permeability (cm/s).
    solubility_mg_mL:
        Aqueous solubility (mg/mL).
    cl_hepatic_L_per_h:
        Hepatic intrinsic clearance (L/h).
    vd_L:
        Volume of distribution (L) — accepted for API symmetry.
    fu_plasma:
        Fraction unbound in plasma — accepted for API symmetry.
    fu_gut:
        Fraction unbound in gut wall (default 0.5).
    cyp3a4_relative_abundance:
        Relative CYP3A4 expression per segment [duodenum, jejunum, ileum, colon].
        Default: [1.0, 2.5, 1.0, 0.1].
    pgp_efflux_ratio:
        P-gp efflux ratio (≥1). Reduces effective ka.
    dose_volume_mL:
        Volume of water co-administered (mL). Default 250 mL.
    hepatic_blood_flow_L_per_h:
        Hepatic portal blood flow (L/h). Default 90 L/h.
    t_end_h, dt_h:
        Simulation duration and step — not used in this static model.

    Returns
    -------
    GutWallFullResult
    """
    if cyp3a4_relative_abundance is None:
        cyp3a4_relative_abundance = [1.0, 2.5, 1.0, 0.1]

    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if peff_cm_per_s <= 0:
        raise ValueError("peff_cm_per_s must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if len(cyp3a4_relative_abundance) != 4:
        raise ValueError("cyp3a4_relative_abundance must have exactly 4 elements")
    if any(v < 0 for v in cyp3a4_relative_abundance):
        raise ValueError("cyp3a4_relative_abundance values must be non-negative")
    if pgp_efflux_ratio < 1.0:
        raise ValueError("pgp_efflux_ratio must be >= 1.0")

    # --- Dissolution ---
    dn = dose_mg / (solubility_mg_mL * dose_volume_mL)
    f_dissolved = min(1.0, 1.0 / max(1.0, dn))

    # --- Per-segment flow ---
    q_segment_L_per_h = hepatic_blood_flow_L_per_h / 4.0

    # --- Regional ka (permeability-based) corrected for P-gp ---
    # Convert peff from cm/s to h-based: *3600
    ka_raw = [2.0 * peff_cm_per_s * 3600.0 / _RADII_CM[i] for i in range(4)]
    ka_eff = [k / pgp_efflux_ratio for k in ka_raw]

    # --- Per-segment absorption and extraction ---
    fa_segments = []
    eg_segments = []
    clint_segments = []

    fraction_remaining = 1.0  # fraction of dose not yet absorbed

    for i in range(4):
        # Fraction absorbed from this segment
        transit_frac = 1.0 - math.exp(-ka_eff[i] * _TRANSIT_TIMES_H[i])
        fa_seg = transit_frac * f_dissolved * fraction_remaining
        fa_segments.append(fa_seg)
        fraction_remaining -= fa_seg
        fraction_remaining = max(0.0, fraction_remaining)

        # Gut wall CLint for this segment
        clint_seg = cyp3a4_relative_abundance[i] * cl_hepatic_L_per_h * 0.02
        clint_segments.append(clint_seg)

        # Well-stirred extraction ratio for this segment
        e_gut = (fu_gut * clint_seg) / (q_segment_L_per_h + fu_gut * clint_seg)
        eg_segments.append(e_gut)

    fa_total = min(1.0, sum(fa_segments))

    # --- Gut availability (fg_total): geometric mean of segment availabilities ---
    fg_segments = [1.0 - eg for eg in eg_segments]
    fg_total = 1.0
    for fg in fg_segments:
        fg_total *= fg
    fg_total = max(0.0, min(1.0, fg_total))

    # --- Hepatic availability ---
    fh = 1.0 - cl_hepatic_L_per_h / hepatic_blood_flow_L_per_h
    fh = max(0.01, min(1.0, fh))

    # --- Overall bioavailability ---
    f_oral_total = fa_total * fg_total * fh
    f_parent_portal = fa_total * fg_total
    f_metabolized_gut = fa_total * (1.0 - fg_total)

    # --- Effective gut wall CL (weighted by segment absorption fraction) ---
    total_fa_nz = max(fa_total, 1e-12)
    cl_gut_wall = sum(clint_segments[i] * (fa_segments[i] / total_fa_nz) for i in range(4))

    notes_parts = [
        f"Dn={dn:.2f}",
        f"f_dissolved={f_dissolved:.3f}",
        f"pgp_efflux_ratio={pgp_efflux_ratio:.2f}",
    ]
    if dn > 1:
        notes_parts.append("dissolution-limited absorption")

    return GutWallFullResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fa_duodenum=fa_segments[0],
        fa_jejunum=fa_segments[1],
        fa_ileum=fa_segments[2],
        fa_colon=fa_segments[3],
        eg_duodenum=eg_segments[0],
        eg_jejunum=eg_segments[1],
        eg_ileum=eg_segments[2],
        eg_colon=eg_segments[3],
        fa_total=fa_total,
        fg_total=fg_total,
        fh=fh,
        f_oral_total=f_oral_total,
        f_parent_portal=f_parent_portal,
        f_metabolized_gut=f_metabolized_gut,
        cl_gut_wall_L_per_h=cl_gut_wall,
        notes="; ".join(notes_parts),
    )


def gut_availability_sensitivity(
    drug_name: str,
    dose_mg: float,
    peff_cm_per_s: float,
    solubility_mg_mL: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    cyp3a4_fold_changes: list[float] | None = None,
) -> list[GutWallFullResult]:
    """Sensitivity analysis over CYP3A4 fold-changes.

    Scales the default CYP3A4 abundance [1.0, 2.5, 1.0, 0.1] by each fold
    change and simulates gut wall extraction. Results are returned sorted by
    fg_total descending (highest gut availability first).

    Parameters
    ----------
    drug_name, dose_mg, peff_cm_per_s, solubility_mg_mL, cl_hepatic_L_per_h, vd_L:
        Passed directly to :func:`simulate_gut_wall_full`.
    cyp3a4_fold_changes:
        List of fold-change scalars to apply. Default: [0.5, 1.0, 2.0, 5.0].

    Returns
    -------
    list[GutWallFullResult] sorted by fg_total descending.
    """
    if cyp3a4_fold_changes is None:
        cyp3a4_fold_changes = [0.5, 1.0, 2.0, 5.0]

    base_abundance = [1.0, 2.5, 1.0, 0.1]
    results = []
    for fold in cyp3a4_fold_changes:
        scaled = [v * fold for v in base_abundance]
        res = simulate_gut_wall_full(
            drug_name=drug_name,
            dose_mg=dose_mg,
            peff_cm_per_s=peff_cm_per_s,
            solubility_mg_mL=solubility_mg_mL,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            vd_L=vd_L,
            cyp3a4_relative_abundance=scaled,
        )
        results.append(res)

    results.sort(key=lambda r: r.fg_total, reverse=True)
    return results
