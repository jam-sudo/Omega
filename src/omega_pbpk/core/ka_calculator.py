"""Absorption rate constant calculator.

Compute ka from permeability, dissolution, and GI transit data using
mechanistic models (cylindrical gut, BCS framework, flip-flop detection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "KaResult",
    "ka_from_peff",
    "ka_from_dissolution",
    "effective_ka",
    "flip_flop_check",
    "absorption_window_ka",
]

# Conversion constant: 3600 s/h
_S_PER_H = 3600.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KaResult:
    """Result of effective absorption rate constant calculation."""

    peff_cm_s: float
    ka_peff_per_h: float
    dissolution_rate_mg_per_min: float
    ka_diss_per_h: float
    effective_ka_per_h: float
    bcs_class: int
    rate_limiting_step: str  # 'permeability' / 'dissolution' / 'both_fast' / 'both_slow'
    notes: str


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def ka_from_peff(peff_cm_s: float, radius_intestine_cm: float = 1.75) -> float:
    """Compute permeability-limited absorption rate constant.

    Uses the cylindrical gut model:
        ka = 2 * Peff / R

    Parameters
    ----------
    peff_cm_s:
        Effective intestinal permeability (cm/s). Must be > 0.
    radius_intestine_cm:
        Intestinal radius (cm). Default 1.75 cm.

    Returns
    -------
    float
        ka in h^-1.
    """
    if peff_cm_s <= 0:
        raise ValueError(f"peff_cm_s must be > 0, got {peff_cm_s}")
    if radius_intestine_cm <= 0:
        raise ValueError(f"radius_intestine_cm must be > 0, got {radius_intestine_cm}")

    ka_per_s = 2.0 * peff_cm_s / radius_intestine_cm
    return ka_per_s * _S_PER_H


def ka_from_dissolution(
    dissolution_rate_mg_per_min: float,
    dose_mg: float,
    solubility_mg_mL: float,
    volume_mL: float = 250.0,
) -> float:
    """Compute dissolution-limited absorption rate constant.

    k_diss = dissolution_rate / (solubility * volume)  [first-order, min^-1]
    ka_eff = min(k_diss * 60, 10.0)  [h^-1, capped at 10 h^-1]

    Parameters
    ----------
    dissolution_rate_mg_per_min:
        Rate of dissolution (mg/min). Must be > 0.
    dose_mg:
        Dose administered (mg). Must be > 0 (informational for context).
    solubility_mg_mL:
        Aqueous solubility (mg/mL). Must be > 0.
    volume_mL:
        GI volume available for dissolution (mL). Default 250 mL.

    Returns
    -------
    float
        ka in h^-1 (capped at 10 h^-1).
    """
    if dissolution_rate_mg_per_min <= 0:
        raise ValueError(
            f"dissolution_rate_mg_per_min must be > 0, got {dissolution_rate_mg_per_min}"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if volume_mL <= 0:
        raise ValueError(f"volume_mL must be > 0, got {volume_mL}")

    # First-order dissolution rate constant (min^-1)
    k_diss_per_min = dissolution_rate_mg_per_min / (solubility_mg_mL * volume_mL)
    # Convert to h^-1 and cap
    ka_per_h = min(k_diss_per_min * 60.0, 10.0)
    return ka_per_h


def effective_ka(
    peff_cm_s: float,
    dissolution_rate_mg_per_min: float,
    dose_mg: float,
    solubility_mg_mL: float,
    bcs_class: int,
) -> KaResult:
    """Compute effective ka based on BCS class.

    BCS I  : both permeability and dissolution are fast; ka = min(ka_peff, ka_diss)
    BCS II : dissolution-limited; ka = ka_diss
    BCS III: permeability-limited; ka = ka_peff
    BCS IV : both limiting; ka = min(ka_peff, ka_diss)

    Parameters
    ----------
    peff_cm_s:
        Effective intestinal permeability (cm/s). Must be > 0.
    dissolution_rate_mg_per_min:
        Rate of dissolution (mg/min). Must be > 0.
    dose_mg:
        Dose (mg). Must be > 0.
    solubility_mg_mL:
        Aqueous solubility (mg/mL). Must be > 0.
    bcs_class:
        BCS class (1, 2, 3, or 4).

    Returns
    -------
    KaResult
    """
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4, got {bcs_class}")

    ka_peff = ka_from_peff(peff_cm_s)
    ka_diss = ka_from_dissolution(dissolution_rate_mg_per_min, dose_mg, solubility_mg_mL)

    if bcs_class == 1:
        eff_ka = min(ka_peff, ka_diss)
        rate_limiting = "both_fast"
        notes = "BCS I: both fast; ka = min(ka_peff, ka_diss)"
    elif bcs_class == 2:
        eff_ka = ka_diss
        rate_limiting = "dissolution"
        notes = "BCS II: dissolution-limited"
    elif bcs_class == 3:
        eff_ka = ka_peff
        rate_limiting = "permeability"
        notes = "BCS III: permeability-limited"
    else:  # bcs_class == 4
        eff_ka = min(ka_peff, ka_diss)
        rate_limiting = "both_slow"
        notes = "BCS IV: both limited; ka = min(ka_peff, ka_diss)"

    return KaResult(
        peff_cm_s=peff_cm_s,
        ka_peff_per_h=ka_peff,
        dissolution_rate_mg_per_min=dissolution_rate_mg_per_min,
        ka_diss_per_h=ka_diss,
        effective_ka_per_h=eff_ka,
        bcs_class=bcs_class,
        rate_limiting_step=rate_limiting,
        notes=notes,
    )


def flip_flop_check(ka: float, ke: float) -> dict:
    """Detect flip-flop pharmacokinetics.

    In flip-flop PK, absorption is slower than elimination (ka < ke is normal;
    ka > ke triggers flip-flop where the apparent terminal half-life reflects
    the absorption rate, not elimination).

    Parameters
    ----------
    ka:
        Absorption rate constant (h^-1).
    ke:
        Elimination rate constant (h^-1).

    Returns
    -------
    dict with keys:
        flip_flop (bool), apparent_t_half_h (float), notes (str)
    """
    if ka <= 0:
        raise ValueError(f"ka must be > 0, got {ka}")
    if ke <= 0:
        raise ValueError(f"ke must be > 0, got {ke}")

    is_flip_flop = ka > ke
    apparent_t_half = math.log(2) / min(ka, ke)

    if is_flip_flop:
        notes = (
            f"Flip-flop PK: ka ({ka:.3f}/h) > ke ({ke:.3f}/h). "
            "Terminal phase reflects absorption, not elimination."
        )
    else:
        notes = (
            f"Normal absorption: ka ({ka:.3f}/h) <= ke ({ke:.3f}/h). "
            "Cmax occurs before elimination half-life."
        )

    return {
        "flip_flop": is_flip_flop,
        "apparent_t_half_h": apparent_t_half,
        "notes": notes,
    }


def absorption_window_ka(ka_per_h: float, transit_time_h: float = 4.0) -> dict:
    """Compute fraction absorbed within the absorption window.

    fa = 1 - exp(-ka * transit_time_h)

    Parameters
    ----------
    ka_per_h:
        Absorption rate constant (h^-1). Must be > 0.
    transit_time_h:
        Small intestinal transit time (h). Default 4 h.

    Returns
    -------
    dict with keys:
        fa (float), ka (float), transit_time_h (float), notes (str)
    """
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if transit_time_h <= 0:
        raise ValueError(f"transit_time_h must be > 0, got {transit_time_h}")

    fa = 1.0 - math.exp(-ka_per_h * transit_time_h)

    if fa >= 0.9:
        notes = f"High absorption: {fa * 100:.1f}% absorbed in {transit_time_h:.1f} h window."
    elif fa >= 0.5:
        notes = f"Moderate absorption: {fa * 100:.1f}% absorbed in {transit_time_h:.1f} h window."
    else:
        notes = (
            f"Low absorption: {fa * 100:.1f}% absorbed in {transit_time_h:.1f} h window. "
            "Consider formulation strategies to increase ka."
        )

    return {
        "fa": fa,
        "ka": ka_per_h,
        "transit_time_h": transit_time_h,
        "notes": notes,
    }
