"""First-pass effect calculator for oral bioavailability.

Calculates oral bioavailability components including gut wall and hepatic
first-pass extraction using well-stirred and parallel-tube models.

References
----------
- Rowland M (1972) Drug Metab Rev 6:51-61 (well-stirred model)
- Pang KS, Rowland M (1977) J Pharmacokinet Biopharm 5:625-653 (parallel-tube)
- Sun D et al. (2004) Pharm Res 21:1346-1351 (Caco-2 to Peff)
"""

from __future__ import annotations

import math

__all__ = [
    "hepatic_extraction_ratio",
    "gut_extraction_ratio",
    "oral_bioavailability",
    "predict_food_effect_on_bioavailability",
    "compare_routes_bioavailability",
]


def hepatic_extraction_ratio(
    clint_L_per_h: float,
    fu_plasma: float,
    qh_L_per_h: float = 90.0,
    rbp: float = 1.0,
    model: str = "well_stirred",
) -> float:
    """Calculate hepatic extraction ratio (ER).

    Parameters
    ----------
    clint_L_per_h : float
        Intrinsic clearance in L/h. Must be >= 0.
    fu_plasma : float
        Unbound fraction in plasma (0, 1]. Must be > 0 and <= 1.
    qh_L_per_h : float
        Hepatic blood flow in L/h. Default 90.0 (typical human).
    rbp : float
        Blood-to-plasma ratio. Must be > 0. Default 1.0.
    model : str
        Hepatic model: 'well_stirred' or 'parallel_tube'. Default 'well_stirred'.

    Returns
    -------
    float
        Hepatic extraction ratio in [0, 1].

    Raises
    ------
    ValueError
        If parameters violate constraints.
    """
    if clint_L_per_h < 0:
        raise ValueError(f"clint_L_per_h must be >= 0, got {clint_L_per_h}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if qh_L_per_h <= 0:
        raise ValueError(f"qh_L_per_h must be > 0, got {qh_L_per_h}")
    if rbp <= 0:
        raise ValueError(f"rbp must be > 0, got {rbp}")
    if model not in ("well_stirred", "parallel_tube"):
        raise ValueError(f"model must be 'well_stirred' or 'parallel_tube', got {model!r}")

    # Effective unbound CLint scaled by blood-to-plasma ratio
    fu_clint = fu_plasma * clint_L_per_h / rbp

    if model == "well_stirred":
        er = fu_clint / (qh_L_per_h + fu_clint)
    else:  # parallel_tube
        er = 1.0 - math.exp(-fu_clint / qh_L_per_h)

    return min(max(er, 0.0), 1.0)


def gut_extraction_ratio(
    peff_cm_s: float,
    clint_gut_L_per_h: float,
    qgut_L_per_h: float | None = None,
) -> float:
    """Calculate gut wall extraction ratio (EG).

    Parameters
    ----------
    peff_cm_s : float
        Effective intestinal permeability in cm/s. Must be >= 0.
    clint_gut_L_per_h : float
        Gut intrinsic clearance in L/h. Must be >= 0.
    qgut_L_per_h : float, optional
        Gut flow rate in L/h. If None, calculated from peff using
        Qgut = Peff * 200 cm^2 * 3600 s/h / 1000 mL/L.

    Returns
    -------
    float
        Gut wall extraction ratio in [0, 1]. fg = 1 - EG.

    Raises
    ------
    ValueError
        If parameters violate constraints.
    """
    if peff_cm_s < 0:
        raise ValueError(f"peff_cm_s must be >= 0, got {peff_cm_s}")
    if clint_gut_L_per_h < 0:
        raise ValueError(f"clint_gut_L_per_h must be >= 0, got {clint_gut_L_per_h}")

    if qgut_L_per_h is None:
        # Qgut = Peff [cm/s] * 200 [cm^2] * 3600 [s/h] / 1000 [mL/L]
        qgut_L_per_h = peff_cm_s * 200.0 * 3600.0 / 1000.0

    if qgut_L_per_h < 0:
        raise ValueError(f"qgut_L_per_h must be >= 0, got {qgut_L_per_h}")

    if qgut_L_per_h + clint_gut_L_per_h == 0:
        return 0.0

    eg = clint_gut_L_per_h / (qgut_L_per_h + clint_gut_L_per_h)
    return min(max(eg, 0.0), 1.0)


def oral_bioavailability(
    fa: float,
    fg: float,
    fh: float,
    f_lymph: float = 0.0,
) -> dict:
    """Calculate oral bioavailability from absorption and extraction components.

    F_oral = fa * fg * fh

    Parameters
    ----------
    fa : float
        Fraction absorbed from GI lumen [0, 1].
    fg : float
        Fraction escaping gut wall extraction [0, 1]. fg = 1 - EG.
    fh : float
        Fraction escaping hepatic first-pass. fh = 1 - ER.
    f_lymph : float
        Fraction absorbed via lymphatics bypassing liver [0, 1]. Default 0.0.

    Returns
    -------
    dict
        Keys: F_oral, fa, fg, fh, loss_gut_lumen, loss_gut_wall, loss_liver, notes.

    Raises
    ------
    ValueError
        If any fraction is outside [0, 1].
    """
    for name, val in [("fa", fa), ("fg", fg), ("fh", fh), ("f_lymph", f_lymph)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {val}")

    f_oral = fa * fg * fh

    notes_parts: list[str] = []
    if fa < 0.5:
        notes_parts.append("Low absorption (fa < 0.5)")
    if fg < 0.5:
        notes_parts.append("High gut wall extraction")
    if fh < 0.5:
        notes_parts.append("High hepatic first-pass")
    if not notes_parts:
        notes_parts.append("Bioavailability within acceptable range")

    return {
        "F_oral": f_oral,
        "fa": fa,
        "fg": fg,
        "fh": fh,
        "loss_gut_lumen": 1.0 - fa,
        "loss_gut_wall": 1.0 - fg,
        "loss_liver": 1.0 - fh,
        "notes": "; ".join(notes_parts),
    }


def predict_food_effect_on_bioavailability(
    f_fasted: float,
    logP: float,
    bile_acid_enhancement: bool = True,
) -> dict:
    """Predict the effect of a high-fat meal on oral bioavailability.

    High logP drugs tend to be enhanced by food due to:
    - Increased solubilization in dietary lipids
    - Bile acid secretion stimulated by fat intake

    Parameters
    ----------
    f_fasted : float
        Bioavailability in the fasted state [0, 1].
    logP : float
        Octanol-water partition coefficient.
    bile_acid_enhancement : bool
        Whether to apply bile acid enhancement for lipophilic drugs. Default True.

    Returns
    -------
    dict
        Keys: F_fasted, F_fed, food_effect_ratio, direction, recommendation.

    Raises
    ------
    ValueError
        If f_fasted not in [0, 1].
    """
    if not (0.0 <= f_fasted <= 1.0):
        raise ValueError(f"f_fasted must be in [0, 1], got {f_fasted}")

    # Fat enhancement based on logP
    if logP > 3:
        fat_enhancement = min(3.0, 1.0 + (logP - 3.0) * 0.5)
    else:
        fat_enhancement = 1.0

    # Bile acid enhancement for moderately/highly lipophilic drugs
    if bile_acid_enhancement and logP > 2:
        fat_enhancement *= 1.2

    f_fed = min(1.0, f_fasted * fat_enhancement)
    food_effect_ratio = f_fed / f_fasted if f_fasted > 0 else 1.0

    if food_effect_ratio > 1.25:
        direction = "increase"
        recommendation = "Take with a high-fat meal to improve absorption"
    elif food_effect_ratio < 0.80:
        direction = "decrease"
        recommendation = "Take in the fasted state to avoid food-induced reduction"
    else:
        direction = "no_effect"
        recommendation = "Food has minimal effect; can be taken with or without food"

    return {
        "F_fasted": f_fasted,
        "F_fed": f_fed,
        "food_effect_ratio": food_effect_ratio,
        "direction": direction,
        "recommendation": recommendation,
    }


def compare_routes_bioavailability(
    iv_auc: float,
    oral_auc: float,
    dose_iv: float,
    dose_oral: float,
) -> dict:
    """Calculate absolute oral bioavailability by comparing oral vs IV AUC.

    F_oral (%) = (AUC_oral / Dose_oral) / (AUC_iv / Dose_iv) * 100

    Parameters
    ----------
    iv_auc : float
        AUC after IV administration (any consistent units, e.g., mg*h/L).
    oral_auc : float
        AUC after oral administration (same units as iv_auc).
    dose_iv : float
        IV dose (mg). Must be > 0.
    dose_oral : float
        Oral dose (mg). Must be > 0.

    Returns
    -------
    dict
        Keys: F_oral_pct, oral_auc, iv_auc, dose_oral, dose_iv.

    Raises
    ------
    ValueError
        If doses are non-positive or AUCs are negative.
    """
    if dose_iv <= 0:
        raise ValueError(f"dose_iv must be > 0, got {dose_iv}")
    if dose_oral <= 0:
        raise ValueError(f"dose_oral must be > 0, got {dose_oral}")
    if iv_auc < 0:
        raise ValueError(f"iv_auc must be >= 0, got {iv_auc}")
    if oral_auc < 0:
        raise ValueError(f"oral_auc must be >= 0, got {oral_auc}")

    if iv_auc == 0:
        f_oral_pct = 0.0
    else:
        f_oral_pct = (oral_auc / dose_oral) / (iv_auc / dose_iv) * 100.0

    return {
        "F_oral_pct": f_oral_pct,
        "oral_auc": oral_auc,
        "iv_auc": iv_auc,
        "dose_oral": dose_oral,
        "dose_iv": dose_iv,
    }
