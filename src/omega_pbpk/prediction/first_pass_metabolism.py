"""Phase 1067 — Drug First-Pass Metabolism Predictor.

Predicts hepatic and intestinal first-pass metabolism using the well-stirred model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_ROUTES = {"oral", "sublingual", "rectal"}
_VALID_IONIZATION_TYPES = {"neutral", "acid", "base", "zwitterion", "amphoteric"}


@dataclass(frozen=True)
class FirstPassResult:
    """Result of first-pass metabolism prediction."""

    drug_name: str
    route: str
    fa: float  # fraction absorbed (gut lumen -> enterocyte)
    fg: float  # fraction surviving gut wall metabolism (0-1)
    fh: float  # fraction surviving hepatic first-pass (0-1)
    f_oral: float  # overall oral bioavailability = fa * fg * fh
    gut_extraction_ratio: float  # Eg = 1 - fg
    hepatic_extraction_ratio: float  # Eh = 1 - fh
    clint_gut_L_per_h: float  # intrinsic clearance in gut
    clint_hepatic_L_per_h: float  # intrinsic clearance in liver
    low_extraction_drug: bool  # True if Eh < 0.3
    high_extraction_drug: bool  # True if Eh > 0.7
    bioavailability_class: str  # "high" (>70%), "moderate" (30-70%), "low" (<30%)
    notes: str


def predict_first_pass_metabolism(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    cyp3a4_substrate: bool = False,
    cyp2d6_substrate: bool = False,
    cyp1a2_substrate: bool = False,
    pgp_substrate: bool = False,
    fu_plasma: float = 0.3,
    fu_mic: float = 0.8,  # unbound fraction in microsomes
    route: str = "oral",  # "oral", "sublingual", "rectal"
    q_hepatic_L_per_h: float = 90.0,
    q_gut_L_per_h: float = 18.0,
) -> FirstPassResult:
    """Predict first-pass metabolism using the well-stirred model.

    Args:
        drug_name: Name of the drug.
        mw_Da: Molecular weight in Daltons (must be > 0).
        logp: Octanol-water partition coefficient.
        pka: Drug pKa value.
        ionization_type: One of neutral, acid, base, zwitterion, amphoteric.
        cyp3a4_substrate: Is the drug a CYP3A4 substrate?
        cyp2d6_substrate: Is the drug a CYP2D6 substrate?
        cyp1a2_substrate: Is the drug a CYP1A2 substrate?
        pgp_substrate: Is the drug a P-glycoprotein substrate?
        fu_plasma: Unbound fraction in plasma (0, 1].
        fu_mic: Unbound fraction in microsomes (0, 1].
        route: Administration route: oral, sublingual, or rectal.
        q_hepatic_L_per_h: Hepatic blood flow in L/h (default 90).
        q_gut_L_per_h: Gut wall blood flow in L/h (default 18).

    Returns:
        FirstPassResult dataclass with all computed parameters.

    Raises:
        ValueError: If any parameter is outside the valid range.
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if fu_mic <= 0 or fu_mic > 1:
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")
    if q_gut_L_per_h <= 0:
        raise ValueError(f"q_gut_L_per_h must be > 0, got {q_gut_L_per_h}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got {route!r}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )

    # Base gut CLint from CYP3A4 (major intestinal CYP)
    clint_gut = 0.0
    if cyp3a4_substrate:
        clint_gut += 100.0 * (1 + logp * 0.05)
    clint_gut *= fu_mic

    # Base hepatic CLint from CYPs
    clint_hepatic = 1.0  # baseline (phase I non-CYP)
    if cyp3a4_substrate:
        clint_hepatic += 50.0
    if cyp2d6_substrate:
        clint_hepatic += 80.0
    if cyp1a2_substrate:
        clint_hepatic += 40.0
    clint_hepatic *= fu_mic

    # Well-stirred model
    fg_base = q_gut_L_per_h / (q_gut_L_per_h + fu_plasma * clint_gut)
    fh_base = q_hepatic_L_per_h / (q_hepatic_L_per_h + fu_plasma * clint_hepatic)

    # P-gp efflux effect on fa
    exponent = 0.3 * logp - 0.002 * mw_Da
    fa_base = min(0.95, math.pow(10.0, exponent))
    fa_base = max(0.05, fa_base)
    if pgp_substrate:
        fa_base *= 0.6

    # Route-specific adjustments
    if route == "oral":
        fa = fa_base
        fg = fg_base
        fh = fh_base
    elif route == "sublingual":
        fa = fa_base * 0.8
        fg = 1.0  # bypasses gut wall
        fh = fh_base * 0.15  # 85% bypasses portal circulation
    else:  # rectal
        fa = fa_base * 0.7
        fg = 1.0  # minimal gut wall metabolism
        fh = fh_base * 0.5  # 50% portal bypass

    # Clamp to [0, 1]
    fa = max(0.0, min(1.0, fa))
    fg = max(0.0, min(1.0, fg))
    fh = max(0.0, min(1.0, fh))

    f_oral = fa * fg * fh

    gut_extraction_ratio = 1.0 - fg
    hepatic_extraction_ratio = 1.0 - fh

    low_extraction_drug = hepatic_extraction_ratio < 0.3
    high_extraction_drug = hepatic_extraction_ratio > 0.7

    if f_oral > 0.7:
        bioavailability_class = "high"
    elif f_oral > 0.3:
        bioavailability_class = "moderate"
    else:
        bioavailability_class = "low"

    # Build notes
    note_parts = []
    if cyp3a4_substrate:
        note_parts.append("CYP3A4 substrate; significant gut wall and hepatic first-pass")
    if cyp2d6_substrate:
        note_parts.append("CYP2D6 substrate; hepatic first-pass contribution")
    if cyp1a2_substrate:
        note_parts.append("CYP1A2 substrate; hepatic first-pass contribution")
    if pgp_substrate:
        note_parts.append("P-gp substrate; efflux reduces net intestinal absorption")
    if route == "sublingual":
        note_parts.append(
            "Sublingual route bypasses gut wall; 85% portal bypass reduces hepatic extraction"
        )
    elif route == "rectal":
        note_parts.append(
            "Rectal route bypasses gut wall; 50% portal bypass reduces hepatic extraction"
        )
    if not note_parts:
        note_parts.append(
            f"Route={route}; fa={fa:.3f}, fg={fg:.3f}, fh={fh:.3f}; "
            f"bioavailability={bioavailability_class}"
        )

    notes = "; ".join(note_parts)

    return FirstPassResult(
        drug_name=drug_name,
        route=route,
        fa=fa,
        fg=fg,
        fh=fh,
        f_oral=f_oral,
        gut_extraction_ratio=gut_extraction_ratio,
        hepatic_extraction_ratio=hepatic_extraction_ratio,
        clint_gut_L_per_h=clint_gut,
        clint_hepatic_L_per_h=clint_hepatic,
        low_extraction_drug=low_extraction_drug,
        high_extraction_drug=high_extraction_drug,
        bioavailability_class=bioavailability_class,
        notes=notes,
    )
