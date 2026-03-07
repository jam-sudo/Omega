"""Comprehensive oral bioavailability model: F = fa × fg × fh.

Combines absorption (fa), gut-wall first-pass (fg), and hepatic first-pass (fh)
using mechanistic estimates from physicochemical properties.

References:
  - Rowland & Tozer, Clinical Pharmacokinetics, 5th Ed., Ch. 7
  - Amidon et al. (1995) AAPS PharmSci — BCS framework
  - Sun et al. (2004) J Pharmacol Exp Ther — Peff/Caco-2 correlation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Hepatic blood flow defaults
# ---------------------------------------------------------------------------

_QH_L_PER_H_PER_KG: float = 1.3  # ~90 L/h for 70 kg adult
_QH_DEFAULT_L_PER_H: float = 90.0
_BODY_WEIGHT_KG: float = 70.0

# CYP3A4 intestinal extraction ranges
_FG_HIGH_CYP3A4: float = 0.50  # high gut-wall extraction (fg ≈ 0.50)
_FG_MODERATE_CYP3A4: float = 0.75  # moderate extraction (fg ≈ 0.75)
_FG_LOW_CYP3A4: float = 0.95  # minimal gut-wall metabolism (fg ≈ 0.95)

# Effective intestinal permeability thresholds (cm/s) for BCS
_PEFF_HIGH_THRESHOLD: float = 2e-4  # high Peff → high permeability
_PEFF_LOW_THRESHOLD: float = 0.5e-4  # low Peff → low permeability


# ---------------------------------------------------------------------------
# Result dataclass (frozen — no mutable fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OralBioavailabilityResult:
    """Result of comprehensive oral bioavailability estimation."""

    drug_name: str
    fa: float  # fraction absorbed
    fg: float  # gut-wall survival (1 - E_gut)
    fh: float  # hepatic first-pass survival (1 - E_hepatic)
    f_absolute: float  # fa × fg × fh (clamped 0.01–1.0)
    bcs_class: int  # BCS class (1–4) provided by caller or estimated
    eh_hepatic: float  # hepatic extraction ratio
    cl_intrinsic_scaled_L_per_h: float  # CLint scaled to whole-liver (L/h)
    fu_plasma: float  # unbound fraction in plasma
    f_classification: str  # "low", "moderate", or "high"
    limiting_factor: str  # "absorption", "gut_wall", "hepatic", or "balanced"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _peff_from_properties(logP: float, psa: float) -> float:
    """Estimate effective jejunal permeability from logP and PSA.

    Simplified correlation (Sun 2004-inspired):
    log(Peff) ≈ -5.5 + 0.4*logP - 0.01*PSA  [log cm/s]
    Clamped to realistic range [1e-6, 1e-3] cm/s.
    """
    log_peff = -5.5 + 0.4 * logP - 0.01 * psa
    peff = 10.0 ** log_peff
    return float(max(1e-6, min(1e-3, peff)))


def _fa_from_bcs(bcs_class: int, logP: float, psa: float, mw: float) -> float:
    """Estimate fa from BCS class and physicochemical properties."""
    peff = _peff_from_properties(logP, psa)
    # Absorption number proxy (transit ~3.5 h, radius 1.75 cm)
    an = peff * 2.0 * 3.5 * 3600.0 / 1.75
    fa_perm = float(1.0 - math.exp(-2.0 * an))

    if bcs_class == 1:
        # High solubility, high permeability → fa limited by permeability
        return float(min(1.0, max(0.7, fa_perm)))
    elif bcs_class == 2:
        # Low solubility, high permeability → dissolution-limited
        # logP correlates with lipophilicity reducing aqueous fa
        fa_estimate = fa_perm * (1.0 / max(1.0, 1.0 + 0.1 * max(0.0, logP - 2.0)))
        return float(min(0.95, max(0.1, fa_estimate)))
    elif bcs_class == 3:
        # High solubility, low permeability
        return float(min(0.6, max(0.3, fa_perm)))
    else:
        # BCS IV: low solubility, low permeability
        return float(min(0.4, max(0.05, fa_perm * 0.5)))


def _fg_from_logP(logP: float, bcs_class: int) -> float:
    """Estimate gut-wall first-pass survival from logP and BCS class.

    Lipophilic compounds (high logP) tend to be CYP3A4 substrates with
    higher intestinal extraction.
    """
    if logP > 3.5 and bcs_class in (1, 2):
        # High CYP3A4 substrate potential
        return _FG_HIGH_CYP3A4
    elif logP > 1.5:
        # Moderate extraction
        return _FG_MODERATE_CYP3A4
    else:
        # Hydrophilic — minimal gut-wall metabolism
        return _FG_LOW_CYP3A4


def _fh_from_clint(
    cl_intrinsic_L_per_h_per_kg: float,
    qh_L_per_h_per_kg: float,
    fu_plasma: float,
    body_weight_kg: float = _BODY_WEIGHT_KG,
) -> tuple[float, float, float]:
    """Compute hepatic first-pass survival using the well-stirred model.

    EH = (CLint * fu) / (QH + CLint * fu)
    fh = 1 - EH

    Returns (fh, EH, cl_intrinsic_scaled_L_per_h).
    """
    cl_int_scaled = cl_intrinsic_L_per_h_per_kg * body_weight_kg
    qh = qh_L_per_h_per_kg * body_weight_kg
    cl_int_fu = cl_int_scaled * fu_plasma
    eh = cl_int_fu / (qh + cl_int_fu) if (qh + cl_int_fu) > 0 else 0.0
    eh = float(max(0.0, min(1.0, eh)))
    fh = 1.0 - eh
    return fh, eh, cl_int_scaled


def _f_classification(f: float) -> str:
    if f > 0.7:
        return "high"
    elif f >= 0.3:
        return "moderate"
    return "low"


def _limiting_factor(fa: float, fg: float, fh: float) -> str:
    """Identify the primary limiting step for oral bioavailability."""
    losses = {
        "absorption": 1.0 - fa,
        "gut_wall": 1.0 - fg,
        "hepatic": 1.0 - fh,
    }
    max_loss = max(losses.values())
    if max_loss < 0.1:
        return "balanced"
    worst = max(losses, key=losses.__getitem__)
    # Map internal name to output name
    name_map = {"absorption": "absorption", "gut_wall": "gut_wall", "hepatic": "hepatic"}
    return name_map[worst]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_oral_bioavailability(  # noqa: PLR0913
    drug_name: str,
    fa: float | None,
    fg: float | None,
    fh: float | None,
    cl_intrinsic_L_per_h_per_kg: float,
    qh_L_per_h_per_kg: float = _QH_L_PER_H_PER_KG,
    fu_plasma: float = 0.1,
    mw: float = 400.0,
    logP: float = 2.0,
    psa: float = 80.0,
    bcs_class: int = 1,
) -> OralBioavailabilityResult:
    """Estimate comprehensive oral bioavailability F = fa × fg × fh.

    If fa, fg, or fh are explicitly provided (not None), those values are used
    directly.  Otherwise they are estimated from physicochemical properties.

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    fa : float or None
        Fraction absorbed (0–1). None → estimated from BCS class + properties.
    fg : float or None
        Gut-wall survival (0–1). None → estimated from logP.
    fh : float or None
        Hepatic first-pass survival (0–1). None → estimated from CLint/QH.
    cl_intrinsic_L_per_h_per_kg : float
        Intrinsic hepatic clearance (L/h/kg body weight).
    qh_L_per_h_per_kg : float
        Hepatic blood flow per kg body weight (default 1.3 L/h/kg).
    fu_plasma : float
        Unbound fraction in plasma (0–1).
    mw : float
        Molecular weight (g/mol), used for BCS class estimation if needed.
    logP : float
        Octanol-water partition coefficient.
    psa : float
        Polar surface area (Å²).
    bcs_class : int
        BCS class (1–4).

    Returns
    -------
    OralBioavailabilityResult
    """
    # --- Input validation ---
    if not drug_name:
        raise ValueError("drug_name must be non-empty")
    if fa is not None and not 0.0 <= fa <= 1.0:
        raise ValueError("fa must be in [0, 1]")
    if fg is not None and not 0.0 <= fg <= 1.0:
        raise ValueError("fg must be in [0, 1]")
    if fh is not None and not 0.0 <= fh <= 1.0:
        raise ValueError("fh must be in [0, 1]")
    if cl_intrinsic_L_per_h_per_kg < 0:
        raise ValueError("cl_intrinsic_L_per_h_per_kg must be >= 0")
    if qh_L_per_h_per_kg <= 0:
        raise ValueError("qh_L_per_h_per_kg must be > 0")
    if not 0.0 < fu_plasma <= 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError("bcs_class must be 1, 2, 3, or 4")

    # --- Compute fh + EH from CLint (always needed for notes) ---
    fh_calc, eh_calc, cl_int_scaled = _fh_from_clint(
        cl_intrinsic_L_per_h_per_kg, qh_L_per_h_per_kg, fu_plasma
    )

    # --- Resolve fa ---
    if fa is None:
        fa_used = _fa_from_bcs(bcs_class, logP, psa, mw)
        fa_source = "estimated"
    else:
        fa_used = float(fa)
        fa_source = "provided"

    # --- Resolve fg ---
    if fg is None:
        fg_used = _fg_from_logP(logP, bcs_class)
        fg_source = "estimated"
    else:
        fg_used = float(fg)
        fg_source = "provided"

    # --- Resolve fh ---
    if fh is None:
        fh_used = fh_calc
        eh_used = eh_calc
        fh_source = "estimated"
    else:
        fh_used = float(fh)
        eh_used = 1.0 - fh_used
        fh_source = "provided"

    # --- Absolute F ---
    f_abs = fa_used * fg_used * fh_used
    f_abs = float(max(0.01, min(1.0, f_abs)))

    # --- Classification ---
    f_class = _f_classification(f_abs)
    limiting = _limiting_factor(fa_used, fg_used, fh_used)

    # --- Notes ---
    note_parts = [
        f"fa ({fa_source}): {fa_used:.3f}",
        f"fg ({fg_source}): {fg_used:.3f}",
        f"fh ({fh_source}): {fh_used:.3f}",
        f"EH={eh_used:.3f}",
        f"F={f_abs:.3f} ({f_class})",
        f"limiting={limiting}",
    ]
    notes = "; ".join(note_parts)

    return OralBioavailabilityResult(
        drug_name=drug_name,
        fa=fa_used,
        fg=fg_used,
        fh=fh_used,
        f_absolute=f_abs,
        bcs_class=bcs_class,
        eh_hepatic=eh_used,
        cl_intrinsic_scaled_L_per_h=cl_int_scaled,
        fu_plasma=fu_plasma,
        f_classification=f_class,
        limiting_factor=limiting,
        notes=notes,
    )


def food_effect_on_f(
    drug_name: str,
    bcs_class: int,
    logP: float,
    dissolution_rate_mg_min: float,
) -> dict:
    """Predict the effect of food on oral bioavailability.

    Food increases solubilisation for lipophilic (BCS II) drugs via bile salts
    and slows gastric emptying.  BCS I and III drugs are minimally affected.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    bcs_class : int
        BCS class (1–4).
    logP : float
        Octanol-water partition coefficient (higher → more lipophilic).
    dissolution_rate_mg_min : float
        Intrinsic dissolution rate in mg/min under fasted conditions.

    Returns
    -------
    dict
        Keys: fed_f, fasted_f, food_effect_ratio, recommendation, drug_name
    """
    # --- Input validation ---
    if not drug_name:
        raise ValueError("drug_name must be non-empty")
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError("bcs_class must be 1, 2, 3, or 4")
    if dissolution_rate_mg_min < 0:
        raise ValueError("dissolution_rate_mg_min must be >= 0")

    # --- Estimate fasted F baseline by BCS class ---
    # Rough heuristic central estimates
    bcs_fasted_f: dict[int, float] = {1: 0.80, 2: 0.40, 3: 0.70, 4: 0.20}
    fasted_f = bcs_fasted_f[bcs_class]

    # --- Food effect magnitude ---
    if bcs_class == 2:
        # Lipophilic compounds benefit most from food (bile salt solubilisation)
        if logP >= 4.0:
            food_factor = 2.5  # strong food effect
        elif logP >= 2.5:
            food_factor = 1.8  # moderate food effect
        else:
            food_factor = 1.3  # mild food effect
        # Slow dissolution rate → more benefit from food (longer GI time)
        if dissolution_rate_mg_min < 1.0:
            food_factor = min(3.0, food_factor * 1.2)
    elif bcs_class == 4:
        # Both poor solubility and permeability; food can help solubility
        if logP >= 3.5:
            food_factor = 1.5
        else:
            food_factor = 1.2
    elif bcs_class == 3:
        # High solubility, low permeability — food slows gastric emptying
        # which may slightly increase permeability contact time
        food_factor = 1.1
    else:
        # BCS I: minimal food effect
        food_factor = 1.05

    fed_f = float(min(1.0, fasted_f * food_factor))
    fasted_f = float(fasted_f)
    food_effect_ratio = float(fed_f / fasted_f) if fasted_f > 0 else 1.0

    # --- Recommendation ---
    if food_effect_ratio >= 2.0:
        recommendation = (
            f"Take {drug_name} WITH FOOD — food substantially increases absorption "
            f"(ratio={food_effect_ratio:.1f}x, likely due to bile salt solubilisation)"
        )
    elif food_effect_ratio >= 1.3:
        recommendation = (
            f"Take {drug_name} with food — moderate food benefit "
            f"(ratio={food_effect_ratio:.1f}x)"
        )
    elif food_effect_ratio <= 0.8:
        recommendation = (
            f"Take {drug_name} in fasted state — food reduces absorption "
            f"(ratio={food_effect_ratio:.1f}x)"
        )
    else:
        recommendation = (
            f"{drug_name}: minimal food effect (ratio={food_effect_ratio:.2f}x); "
            "may be taken with or without food"
        )

    return {
        "drug_name": drug_name,
        "bcs_class": bcs_class,
        "fasted_f": fasted_f,
        "fed_f": fed_f,
        "food_effect_ratio": food_effect_ratio,
        "logP": logP,
        "dissolution_rate_mg_min": dissolution_rate_mg_min,
        "recommendation": recommendation,
    }


__all__ = [
    "OralBioavailabilityResult",
    "estimate_oral_bioavailability",
    "food_effect_on_f",
]
