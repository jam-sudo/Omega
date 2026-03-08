"""Phase 519 — Dose-Response Relationship Model.

Pharmacodynamic dose-response models: Emax, sigmoid-Emax, and threshold.
Pure Python (stdlib only), no ODE needed — analytical calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class DoseResponseResult:
    """Result of a dose-response computation."""

    drug_name: str
    model_type: str  # "emax" | "sigmoid_emax" | "threshold"
    e0: float
    emax_effect: float
    ec50_mg_L: float
    hill_coeff: float
    e0_effect: float
    doses_mg: list
    cmax_values_mg_L: list
    effect_values: list
    ed50_mg: float  # dose giving 50% of max additional effect (interpolated)
    emax_dose_mg: float  # dose giving >95% Emax (interpolated); -1.0 if not reached
    therapeutic_window_ratio: float  # emax_dose / ed50; -1.0 if not available
    notes: str


# ---------------------------------------------------------------------------
# Internal model equations
# ---------------------------------------------------------------------------


def _emax_effect(c: float, e0: float, emax: float, ec50: float) -> float:
    """Emax model: E(C) = E0 + Emax * C / (EC50 + C)."""
    if c <= 0.0:
        return e0
    return e0 + emax * c / (ec50 + c)


def _sigmoid_emax_effect(c: float, e0: float, emax: float, ec50: float, n: float) -> float:
    """Sigmoid-Emax (Hill) model: E(C) = E0 + Emax * C^n / (EC50^n + C^n)."""
    if c <= 0.0:
        return e0
    cn = c**n
    ec50n = ec50**n
    return e0 + emax * cn / (ec50n + cn)


def _threshold_effect(c: float, e0: float, emax: float, ec50: float, c_thresh: float) -> float:
    """Threshold model: E=E0 below threshold, Emax model above."""
    if c <= c_thresh:
        return e0
    delta = c - c_thresh
    return e0 + emax * delta / (ec50 + delta)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def _interpolate_dose_for_effect(doses: list, effects: list, target_effect: float) -> float:
    """Linear interpolation: find dose where effect crosses target_effect.

    Returns -1.0 if the target is not crossed within the dose range.
    """
    for i in range(1, len(doses)):
        e_prev, e_curr = effects[i - 1], effects[i]
        if e_prev <= target_effect <= e_curr or e_curr <= target_effect <= e_prev:
            # linear interpolation
            if abs(e_curr - e_prev) < 1e-15:
                return doses[i - 1]
            frac = (target_effect - e_prev) / (e_curr - e_prev)
            return doses[i - 1] + frac * (doses[i] - doses[i - 1])
    return -1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_dose_response(
    drug_name: str,
    model_type: str,
    emax_effect: float,
    ec50_mg_L: float,
    f_oral: float,
    vd_L: float,
    e0_effect: float = 0.0,
    hill_coeff: float = 1.0,
    c_threshold_mg_L: float = 0.0,
    doses_mg: list | None = None,
) -> DoseResponseResult:
    """Compute dose-response curve using the specified PD model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    model_type:
        One of "emax", "sigmoid_emax", "threshold".
    emax_effect:
        Maximum pharmacodynamic effect (same units as e0_effect).
    ec50_mg_L:
        Concentration producing 50% of Emax (mg/L).
    f_oral:
        Oral bioavailability fraction (0 < f_oral <= 1).
    vd_L:
        Volume of distribution (L).
    e0_effect:
        Baseline effect (default 0).
    hill_coeff:
        Hill exponent n (default 1, used for sigmoid_emax only).
    c_threshold_mg_L:
        Threshold concentration for threshold model (default 0).
    doses_mg:
        List of doses in mg to evaluate. Default: [1, 3, 10, 30, 100, 300].

    Returns
    -------
    DoseResponseResult
    """
    # --- Validation ---
    valid_models = {"emax", "sigmoid_emax", "threshold"}
    if model_type not in valid_models:
        raise ValueError(f"model_type must be one of {valid_models}, got '{model_type}'")
    if emax_effect == 0:
        raise ValueError("emax_effect must not be zero")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    if doses_mg is None:
        doses_mg = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

    doses_mg = list(doses_mg)

    # --- Cmax approximation: F * dose / Vd ---
    cmax_values = [f_oral * d / vd_L for d in doses_mg]

    # --- Effect at each dose ---
    effects = []
    for c in cmax_values:
        if model_type == "emax":
            e = _emax_effect(c, e0_effect, emax_effect, ec50_mg_L)
        elif model_type == "sigmoid_emax":
            e = _sigmoid_emax_effect(c, e0_effect, emax_effect, ec50_mg_L, hill_coeff)
        else:  # threshold
            e = _threshold_effect(c, e0_effect, emax_effect, ec50_mg_L, c_threshold_mg_L)
        effects.append(e)

    # --- ED50 (dose giving 50% of Emax additional effect) ---
    target_50 = e0_effect + 0.5 * emax_effect
    ed50_mg = _interpolate_dose_for_effect(doses_mg, effects, target_50)

    # --- Emax dose (>95% of Emax) ---
    target_95 = e0_effect + 0.95 * emax_effect
    emax_dose_mg = _interpolate_dose_for_effect(doses_mg, effects, target_95)

    # --- Therapeutic window ratio ---
    if ed50_mg > 0 and emax_dose_mg > 0:
        twr = emax_dose_mg / ed50_mg
    else:
        twr = -1.0

    notes = (
        f"Model: {model_type}. "
        f"Cmax approximated as F*dose/Vd. "
        f"EC50={ec50_mg_L:.3f} mg/L, Emax={emax_effect:.3f}."
    )

    return DoseResponseResult(
        drug_name=drug_name,
        model_type=model_type,
        e0=e0_effect,
        emax_effect=emax_effect,
        ec50_mg_L=ec50_mg_L,
        hill_coeff=hill_coeff,
        e0_effect=e0_effect,
        doses_mg=doses_mg,
        cmax_values_mg_L=cmax_values,
        effect_values=effects,
        ed50_mg=ed50_mg,
        emax_dose_mg=emax_dose_mg,
        therapeutic_window_ratio=twr,
        notes=notes,
    )


def compare_models(
    drug_name: str,
    emax_effect: float,
    ec50_mg_L: float,
    f_oral: float,
    vd_L: float,
) -> list:
    """Run all 3 PD models and return a list of 3 DoseResponseResult objects.

    Uses hill_coeff=2 for sigmoid_emax (to distinguish from emax) and
    c_threshold=0 for threshold model.
    """
    results = []
    for model_type, extra in [
        ("emax", {}),
        ("sigmoid_emax", {"hill_coeff": 2.0}),
        ("threshold", {"c_threshold_mg_L": 0.0}),
    ]:
        r = compute_dose_response(
            drug_name=drug_name,
            model_type=model_type,
            emax_effect=emax_effect,
            ec50_mg_L=ec50_mg_L,
            f_oral=f_oral,
            vd_L=vd_L,
            **extra,
        )
        results.append(r)
    return results
