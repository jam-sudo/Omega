"""Gut wall extraction model — Phase 926.

Intestinal first-pass extraction model: CYP3A4 and P-gp interplay
determining gut wall bioavailability (Fg).

Steady-state well-stirred gut wall model:
  E_gut = fu_ent * CLint_gut / (Qent + fu_ent * CLint_gut)
  Fg    = 1 - E_gut

fu_ent (fraction unbound in enterocyte):
  fu_ent ≈ 1 / (1 + 10^(0.5 * logP - 1))

P-gp efflux reduces effective permeability:
  effective_Papp = Papp / (1 + pgp_efflux_ratio)

fa effective (fraction absorbed after P-gp):
  fa_eff = fa_input * (1 / (1 + pgp_efflux_ratio * 0.3))

Overall intestinal bioavailability:
  F_intestinal = fa_eff * Fg
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "GutWallExtractionResult",
    "predict_gut_wall_extraction",
    "screen_gut_extraction",
]

_VALID_LIMITING_FACTORS = frozenset(
    {"cyp3a4_metabolism", "pgp_efflux", "poor_permeability", "none"}
)


@dataclass(frozen=True)
class GutWallExtractionResult:
    """Result of gut wall extraction prediction."""

    drug_name: str
    logp: float
    pka: float
    fu_ent: float
    e_gut: float
    fg: float
    fa_input: float
    fa_effective: float
    pgp_efflux_ratio: float
    f_intestinal: float
    clint_gut_L_per_h: float
    limiting_factor: str
    notes: str


def _compute_fu_ent(logp: float) -> float:
    """Estimate fraction unbound in enterocyte from logP.

    fu_ent ≈ 1 / (1 + 10^(0.5 * logP - 1))
    """
    exponent = 0.5 * logp - 1.0
    # Clamp exponent to avoid overflow
    exponent = min(max(exponent, -30.0), 30.0)
    fu_ent = 1.0 / (1.0 + math.pow(10.0, exponent))
    # Clamp to valid range
    return min(max(fu_ent, 1e-6), 1.0)


def predict_gut_wall_extraction(
    drug_name: str,
    logp: float,
    pka: float,
    fa_input: float = 0.9,
    clint_gut_L_per_h: float = 5.0,
    pgp_efflux_ratio: float = 1.0,
    qent_L_per_h: float = 18.0,
) -> GutWallExtractionResult:
    """Predict gut wall extraction and intestinal bioavailability.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Lipophilicity (log octanol-water partition coefficient).
    pka : float
        pKa of the drug (informational; used for ionization context).
    fa_input : float
        Fraction absorbed at gut wall (0 to 1). Must be in [0, 1].
    clint_gut_L_per_h : float
        Intrinsic gut wall clearance (L/h). Must be >= 0.
    pgp_efflux_ratio : float
        P-gp efflux ratio (>= 0). Higher values indicate stronger P-gp efflux.
    qent_L_per_h : float
        Enterocyte blood flow rate (L/h). Must be > 0.

    Returns
    -------
    GutWallExtractionResult
    """
    # Validation
    if not (0.0 <= fa_input <= 1.0):
        raise ValueError("fa_input must be in [0, 1]")
    if clint_gut_L_per_h < 0:
        raise ValueError("clint_gut_L_per_h must be >= 0")
    if pgp_efflux_ratio < 0:
        raise ValueError("pgp_efflux_ratio must be >= 0")
    if qent_L_per_h <= 0:
        raise ValueError("qent_L_per_h must be > 0")

    # Compute fu_ent from logP
    fu_ent = _compute_fu_ent(logp)

    # Well-stirred gut wall model
    # E_gut = fu_ent * CLint_gut / (Qent + fu_ent * CLint_gut)
    fu_clint = fu_ent * clint_gut_L_per_h
    denom = qent_L_per_h + fu_clint
    e_gut = fu_clint / denom if denom > 0 else 0.0
    # Clamp to [0, 1]
    e_gut = min(max(e_gut, 0.0), 1.0)
    fg = 1.0 - e_gut

    # P-gp efflux reduces fa
    # fa_eff = fa_input * (1 / (1 + pgp_efflux_ratio * 0.3))
    pgp_reduction = 1.0 / (1.0 + pgp_efflux_ratio * 0.3)
    fa_effective = fa_input * pgp_reduction
    fa_effective = min(max(fa_effective, 0.0), 1.0)

    # Overall intestinal bioavailability
    f_intestinal = fa_effective * fg
    f_intestinal = min(max(f_intestinal, 0.0), 1.0)

    # Determine limiting factor
    # Thresholds for classification
    cyp3a4_dominant = e_gut > 0.3  # >30% gut wall extraction
    pgp_dominant = pgp_efflux_ratio > 2.0 and (fa_input - fa_effective) / max(fa_input, 1e-9) > 0.2
    poor_perm = fa_input < 0.3

    if cyp3a4_dominant and not pgp_dominant:
        limiting_factor = "cyp3a4_metabolism"
    elif pgp_dominant and not cyp3a4_dominant:
        limiting_factor = "pgp_efflux"
    elif poor_perm:
        limiting_factor = "poor_permeability"
    elif cyp3a4_dominant and pgp_dominant:
        # Both significant: pick the one with greater magnitude
        cyp3a4_loss = 1.0 - fg
        pgp_loss = 1.0 - pgp_reduction
        if cyp3a4_loss >= pgp_loss:
            limiting_factor = "cyp3a4_metabolism"
        else:
            limiting_factor = "pgp_efflux"
    else:
        limiting_factor = "none"

    # Build notes
    notes_parts = [
        f"fu_ent={fu_ent:.4f} (from logP={logp:.2f})",
        f"E_gut={e_gut:.4f}, Fg={fg:.4f}",
        f"fa_input={fa_input:.4f}, fa_eff={fa_effective:.4f} (P-gp ratio={pgp_efflux_ratio:.2f})",
        f"F_intestinal={f_intestinal:.4f}",
        f"CLint_gut={clint_gut_L_per_h:.2f} L/h, Qent={qent_L_per_h:.1f} L/h",
        f"Limiting factor: {limiting_factor}",
    ]
    notes = "; ".join(notes_parts)

    return GutWallExtractionResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        fu_ent=fu_ent,
        e_gut=e_gut,
        fg=fg,
        fa_input=fa_input,
        fa_effective=fa_effective,
        pgp_efflux_ratio=pgp_efflux_ratio,
        f_intestinal=f_intestinal,
        clint_gut_L_per_h=clint_gut_L_per_h,
        limiting_factor=limiting_factor,
        notes=notes,
    )


def screen_gut_extraction(
    drug_name: str,
    compounds: list[dict],
) -> list[GutWallExtractionResult]:
    """Screen multiple compounds for gut wall extraction.

    Parameters
    ----------
    drug_name : str
        Base drug name (used as prefix if compound has no 'name' key).
    compounds : list[dict]
        Each dict must have keys: logp, pka.
        Optional keys: fa_input, clint_gut_L_per_h, pgp_efflux_ratio, name.

    Returns
    -------
    list[GutWallExtractionResult]
        Results sorted by f_intestinal descending.
    """
    results: list[GutWallExtractionResult] = []
    for i, compound in enumerate(compounds):
        name = compound.get("name", f"{drug_name}_{i + 1}")
        logp = float(compound["logp"])
        pka = float(compound["pka"])
        fa_input = float(compound.get("fa_input", 0.9))
        clint_gut_L_per_h = float(compound.get("clint_gut_L_per_h", 5.0))
        pgp_efflux_ratio = float(compound.get("pgp_efflux_ratio", 1.0))

        result = predict_gut_wall_extraction(
            drug_name=name,
            logp=logp,
            pka=pka,
            fa_input=fa_input,
            clint_gut_L_per_h=clint_gut_L_per_h,
            pgp_efflux_ratio=pgp_efflux_ratio,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_intestinal, reverse=True)
    return results
